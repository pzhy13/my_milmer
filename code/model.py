import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint 
from einops.layers.torch import Rearrange
from torch_geometric.nn import GATConv # 确保安装了 torch_geometric

# --- 从 MUSE_EEG 移植并修改的类 ---

class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250, num_nodes=32): # 增加 num_nodes 参数
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        self.num_channels = num_nodes
        
        # --- 修改: 移除硬编码的 .cuda()，使用 register_buffer 以支持自动设备管理 ---
        # Create a list of tuples representing all possible edges between channels
        edge_index_list = torch.tensor([[i, j] for i in range(self.num_channels) for j in range(self.num_channels) if i != j], dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_list)

    def forward(self, x):
        # x shape: [batch, 1, channels, time] -> expected by NervFormer logic
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        return x

class NervFormerEEGModel(nn.Module):
    def __init__(self, num_channels=32, time_length=250, output_dim=768): # 参数化
        super(NervFormerEEGModel, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features=1)

        # --- 修改: 动态传递参数 ---
        self.gatnn = EEG_GAT(in_channels=time_length, out_channels=time_length, num_nodes=num_channels)

        # --- 修改: 确保卷积核尺寸不超过输入尺寸 ---
        # 原始 MUSE: (1, 25) for time, (63, 1) for space
        # 我们保留时间卷积核 (1, 25)，但空间卷积核必须匹配 num_channels
        
        time_kernel = (1, 25) if time_length >= 25 else (1, 3) # 简单的自适应保护
        pool_kernel = (1, 51) if time_length >= 51 else (1, 2)
        pool_stride = (1, 5) if time_length >= 51 else (1, 2)
        
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, time_kernel, (1, 1)),
            nn.AvgPool2d(pool_kernel, pool_stride),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)), # Dynamic Spatial Kernel
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (num_channels, 1), (1, 1)),  # Dynamic Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, time_kernel, (1, 1)),  # Temporal convolution
            nn.AvgPool2d(pool_kernel, pool_stride),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.self_attn_ts = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.self_attn_st = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.cross_attn = nn.MultiheadAttention(embed_dim=40, num_heads=8, dropout=0.75)

        self.norm1 = nn.LayerNorm(40) #d_model=40
        self.norm2 = nn.LayerNorm(40)
        self.norm3 = nn.LayerNorm(40)
        
        # --- 修改: 动态计算 Flatten 后的大小 ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, time_length)
            ts_out = self.tsconv(dummy_input).flatten(2).permute(2, 0, 1)
            # Combined features shape calculation
            # ts_out shape: [seq_len, batch, 40]
            seq_len = ts_out.shape[0]
            # combined is cat(ts, st) -> dim 0 doubles -> 2 * seq_len
            # final flatten(1) -> (batch, 2 * seq_len * 40)
            flat_dim = 2 * seq_len * 40
        
        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        # x shape input: [batch, 1, channels, time]
        x = self.instance_norm(x)
        ##### Nerv-GA#####
        # x = self.gatnn(x) # 根据需要取消注释
        ##################
        ts_features = self.tsconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        st_features = self.stconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        # Attention is applied over the 250 time steps. 
        bf_ts_features, _ = self.self_attn_ts(ts_features, ts_features, ts_features)
        bf_st_features, _ = self.self_attn_st(st_features, st_features, st_features)
        # LayerNorm
        bf_ts_features = self.norm1(bf_ts_features + ts_features)
        bf_st_features = self.norm2(bf_st_features + st_features)
        combined_features = torch.cat((bf_ts_features, bf_st_features), dim=0) # need to cat?
        cf_combined_features, _ = self.cross_attn(combined_features, combined_features, combined_features)
        final_combined_features = self.norm3(cf_combined_features + combined_features)
        final_combined_features = final_combined_features.permute(1, 0, 2).flatten(1)
        output_features = self.feed_forward(final_combined_features) 

        return output_features

# --- 原 Milmer 模型 ---

class MultiModalClassifier(nn.Module):
    def __init__(self, 
                 swin_processor, 
                 swin_model,
                 # --- 关键修改：移除了 device 参数 ---
                 # device, 
                 input_size=768, num_classes=4, 
                 num_heads=12, dim_feedforward=2048, num_encoder_layers=6, 
                 eeg_size=384, transformer_dropout_rate=0.2, cls_dropout_rate=0.1,
                 fusion_type='cross_attention',
                 instance_selection_method='amil_topk',
                 num_select = 3, num_instances=10,
                 # --- 新增参数用于 NervFormer ---
                 use_nerv_eeg=True,
                 eeg_channels=32,
                 eeg_time_len=12 # 默认值，需要根据您的实际数据长度调整 (eeg_size / eeg_channels)
                 ):
        super().__init__()
        self.transformer_dropout_rate = transformer_dropout_rate
        self.cls_dropout_rate = cls_dropout_rate
        self.fusion_type = fusion_type
        self.instance_selection_method = instance_selection_method
        # self.device = device 
        self.input_size = input_size 
        self.use_nerv_eeg = use_nerv_eeg
        self.eeg_channels = eeg_channels
        self.eeg_time_len = eeg_time_len

        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            param.requires_grad = True 

        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout_rate, batch_first=True),
            num_layers=num_encoder_layers
        )

        if self.use_nerv_eeg:
            # --- 集成 NervFormer ---
            # 注意: eeg_size 在这里可能不再直接使用，而是使用 eeg_channels 和 eeg_time_len
            self.eeg_encoder = NervFormerEEGModel(num_channels=eeg_channels, time_length=eeg_time_len, output_dim=input_size)
            # NervFormer 输出已经是 input_size，所以不需要 layernorm/proj (或者可以在里面加)
            self.layernorm = nn.Identity() # NervFormer 内部有 Norm
        else:
            self.eeg_proj = nn.Linear(eeg_size, input_size)
            self.activation = nn.ReLU()
            self.layernorm = nn.LayerNorm(eeg_size)
            
        # --- 关键修改：移除 .to(self.device)，使其在 CPU 上初始化 ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)) 
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(input_size,num_classes)

        if fusion_type == 'cross_attention':
            self.num_queries = 64
            self.query_tokens = nn.Parameter(torch.zeros(1, self.num_queries, input_size))
            nn.init.normal_(self.query_tokens, std=0.02)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=input_size,
                num_heads=num_heads,
                dropout=transformer_dropout_rate,
                batch_first=True
            )
        elif fusion_type == 'mlp':
            self.mlp_up = nn.Linear(input_size, 4*input_size)
            self.mlp_act = nn.GELU()
            self.mlp_down = nn.Linear(4*input_size, input_size)

        self.num_instances = num_instances
        self.num_select = num_select
        self.instance_weights = nn.Parameter(torch.ones(1, 10))

        self.attention_hidden_dim = 128  # 这是一个更合理的大小 (可以尝试 256)
        self.attention_V = nn.Linear(input_size * 49, self.attention_hidden_dim)
        self.attention_w = nn.Linear(self.attention_hidden_dim, 1)
        self.attention_recover = nn.Linear(input_size * 49, self.num_instances)

    def select_instances(self, images_embedding):
        # ... (select_instances 函数保持不变) ...
        if self.instance_selection_method == 'softmax':
            weights = F.softmax(self.instance_weights, dim=1)
            _, indices = torch.topk(weights, self.num_select, dim=1)
            selected_embeddings = []
            batch_size = images_embedding.size(0)
            for i in range(batch_size):
                selected = images_embedding[i, indices[0], :]
                selected_embeddings.append(selected)
            return torch.stack(selected_embeddings)
        elif self.instance_selection_method == 'amil_topk':
            batch_size = images_embedding.size(0)
            instance_features = images_embedding.view(batch_size, self.num_instances, -1)
            hidden = torch.tanh(self.attention_V(instance_features))
            weights = self.attention_w(hidden)
            weights = F.softmax(weights, dim=1)
            _, indices = torch.topk(weights, self.num_select, dim=1)
            selected_embeddings = []
            
            # --- 关键修改：从参数动态获取设备 ---
            current_device = self.attention_V.weight.device

            for i in range(batch_size):
                idx_squeeze = indices[i].squeeze()
                if idx_squeeze.numel() == 0: 
                    idx_squeeze = torch.tensor([0], device=current_device) 
                elif idx_squeeze.dim() == 0: 
                    idx_squeeze = idx_squeeze.unsqueeze(0)
                selected = images_embedding[i, idx_squeeze, :, :]
                selected_embeddings.append(selected)
            if batch_size == 1 and len(selected_embeddings) == 1:
                 return selected_embeddings[0].unsqueeze(0)
            return torch.stack(selected_embeddings)

    def forward(self, eeg_data=None, images_data=None):
        
        if eeg_data is None and images_data is None:
            raise ValueError("至少需要提供 EEG 或图像数据中的一种。")
        
        # --- 关键修改：动态获取当前设备 ---
        current_device = self.classifier.weight.device
        
        if eeg_data is not None:
            batch_size = eeg_data.size(0)
        else:
            batch_size = images_data.size(0)

        if images_data is not None:
            images_embedding_list = []
            for i in range(self.num_instances):
                image = images_data[:, i]
                # --- 关键修改：将 processor 的输出 .to(current_device) ---
                image_processed = self.img_processor(image, return_tensors="pt").to(current_device)
                with torch.no_grad(): 
                    embedding = self.swin_model(**image_processed).last_hidden_state
                images_embedding_list.append(embedding)
            
            images_embedding_stacked = torch.stack(images_embedding_list, dim=1)
            selected_embeddings = self.select_instances(images_embedding_stacked)
            selected_embeddings = selected_embeddings.view(batch_size, -1, self.input_size)
            
            if self.fusion_type == 'cross_attention':
                query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                image_features, _ = self.cross_attention(
                    query=query_tokens, key=selected_embeddings, value=selected_embeddings
                )
                images_embedding = image_features 
            elif self.fusion_type == 'mlp':
                 x = self.mlp_up(selected_embeddings)
                 x = self.mlp_act(x)
                 images_embedding = self.mlp_down(x)
            else:
                 images_embedding = selected_embeddings

            images_embedding = images_embedding + self.token_type_embeddings(
                # --- 关键修改：将新张量创建在 current_device ---
                torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=current_device)
            )
        else:
            img_embedding_dim = self.num_queries if self.fusion_type == 'cross_attention' else self.num_select * 49
            # --- 关键修改：将新张量创建在 current_device ---
            images_embedding = torch.zeros(batch_size, img_embedding_dim, self.input_size, device=current_device)
            images_embedding = images_embedding + self.token_type_embeddings(
                # --- 关键修改：将新张量创建在 current_device ---
                torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=current_device)
            )

        if eeg_data is not None:
            if self.use_nerv_eeg:
                # NervFormer 需要 4D 输入: [batch, 1, channels, time]
                # 假设 eeg_data 传入时是 [batch, channels * time] 或 [batch, features]
                # 我们需要 reshape 它。
                # 警告: 必须确保 eeg_data.size(1) == self.eeg_channels * self.eeg_time_len
                
                # 尝试自动 Reshape，如果已经是 3D (batch, channels, time)
                if eeg_data.dim() == 3:
                     eeg_reshaped = eeg_data.unsqueeze(1)
                elif eeg_data.dim() == 2:
                     eeg_reshaped = eeg_data.view(batch_size, 1, self.eeg_channels, self.eeg_time_len)
                else:
                     eeg_reshaped = eeg_data # 假设已经是 4D

                eeg_embedding = self.eeg_encoder(eeg_reshaped)
                # NervFormer output is [batch, input_size]
                # 为了与 TransformerEncoder 兼容，我们需要 seq_len 维度
                eeg_embedding = eeg_embedding.unsqueeze(1) # [batch, 1, input_size]
                
            else:
                # 旧逻辑
                eeg_data = self.layernorm(eeg_data)
                eeg_embedding = self.eeg_proj(eeg_data)
                eeg_embedding = self.activation(eeg_embedding)
                # 增加 seq 维度 [batch, 1, input_size] 如果它是 2D 的
                if eeg_embedding.dim() == 2:
                    eeg_embedding = eeg_embedding.unsqueeze(1)

            eeg_embedding = eeg_embedding + self.token_type_embeddings(
                # --- 关键修改：将新张量创建在 current_device ---
                torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=current_device)
            )
        else:
            eeg_embedding_dim = 32 
            # 如果使用 NervFormer，输出通常是 1 个 token (flattened)
            dim_len = 1 if self.use_nerv_eeg else eeg_embedding_dim
            
            # --- 关键修改：将新张量创建在 current_device ---
            eeg_embedding = torch.zeros(batch_size, dim_len, self.input_size, device=current_device)
            eeg_embedding = eeg_embedding + self.token_type_embeddings(
                 # --- 关键修改：将新张量创建在 current_device ---
                 torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=current_device)
             )

        multi_embedding = torch.cat((images_embedding, eeg_embedding), dim=1)
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        multi_embedding = torch.cat((cls_token_expanded, multi_embedding), dim=1)
        
        transformer_output = checkpoint(self.transformer_encoder, multi_embedding, use_reentrant=False)
        cls_token_output = transformer_output[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)
        x = self.classifier(cls_token_output)

        return x