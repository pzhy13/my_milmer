import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint # <--- 添加这一行

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
                 num_select = 3, num_instances=10
                 ):
        super().__init__()
        self.transformer_dropout_rate = transformer_dropout_rate
        self.cls_dropout_rate = cls_dropout_rate
        self.fusion_type = fusion_type
        self.instance_selection_method = instance_selection_method
        # --- 关键修改：移除了 self.device ---
        # self.device = device 
        self.input_size = input_size 

        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            param.requires_grad = False 

        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout_rate, batch_first=True),
            num_layers=num_encoder_layers
        )

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
        # (注意：如果这里有 .to(self.device)，也需要移除)
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
        # DataParallel 会将模型和数据分散到不同 GPU
        # 我们从一个已注册的参数 (如 classifier.weight) 获取当前所在的 GPU 设备
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
            eeg_data = self.layernorm(eeg_data)
            eeg_embedding = self.eeg_proj(eeg_data)
            eeg_embedding = self.activation(eeg_embedding)
            eeg_embedding = eeg_embedding + self.token_type_embeddings(
                # --- 关键修改：将新张量创建在 current_device ---
                torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=current_device)
            )
        else:
            eeg_embedding_dim = 32 
            # --- 关键修改：将新张量创建在 current_device ---
            eeg_embedding = torch.zeros(batch_size, eeg_embedding_dim, self.input_size, device=current_device)
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