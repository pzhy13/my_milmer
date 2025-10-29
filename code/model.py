import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalClassifier(nn.Module):
    def __init__(self, 
                 swin_processor, 
                 swin_model,
                 device, 
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
        self.device = device 
        self.input_size = input_size # 保存 input_size

        # --- 关键修改：冻结 Swin ---
        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            # 保持冻结状态以匹配训练
            param.requires_grad = False 

        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout_rate, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.eeg_proj = nn.Linear(eeg_size, input_size)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(eeg_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(self.device) 
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(input_size,num_classes)

        if fusion_type == 'cross_attention':
            # num_queries=64
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

        # attention_topk
        self.attention_V = nn.Linear(input_size * 49, input_size * 49)
        self.attention_w = nn.Linear(input_size * 49, 1)
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
            for i in range(batch_size):
                # 确保 indices[i] 不是空的并且维度正确
                idx_squeeze = indices[i].squeeze()
                if idx_squeeze.numel() == 0: # 如果为空
                    # 这种情况理论上不应该发生，但作为容错，选择第一个 instance
                    idx_squeeze = torch.tensor([0], device=self.device) 
                elif idx_squeeze.dim() == 0: # 如果只有一个元素，确保它是 1D
                    idx_squeeze = idx_squeeze.unsqueeze(0)
                selected = images_embedding[i, idx_squeeze, :, :]
                selected_embeddings.append(selected)
            # 处理 batch size 为 1 时 stack 可能失败的情况
            if batch_size == 1 and len(selected_embeddings) == 1:
                 return selected_embeddings[0].unsqueeze(0) # 保证返回 [1, num_select, 49, 768]
            return torch.stack(selected_embeddings)

    # --- 关键修改：允许 eeg_data 或 images_data 为 None ---
    def forward(self, eeg_data=None, images_data=None):
        
        # --- 输入检查和 Batch Size 确定 ---
        if eeg_data is None and images_data is None:
            raise ValueError("至少需要提供 EEG 或图像数据中的一种。")
        
        if eeg_data is not None:
            batch_size = eeg_data.size(0)
        else: # images_data is not None
            batch_size = images_data.size(0)

        # --- 处理图像模态 (如果提供) ---
        if images_data is not None:
            images_embedding_list = []
            for i in range(self.num_instances):
                image = images_data[:, i]
                image_processed = self.img_processor(image, return_tensors="pt").to(self.device)
                with torch.no_grad(): # 因为 Swin 被冻结，不需要计算梯度
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
                images_embedding = image_features # [batch_size, 64, 768]
            # ... (MLP 和 'none' 的逻辑保持不变) ...
            elif self.fusion_type == 'mlp':
                 x = self.mlp_up(selected_embeddings)
                 x = self.mlp_act(x)
                 images_embedding = self.mlp_down(x)
            else:
                 images_embedding = selected_embeddings

            # 添加图像模态类型嵌入
            images_embedding = images_embedding + self.token_type_embeddings(
                torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=self.device)
            )
        else:
            # 如果没有图像数据，创建一个零张量占位符
            # 维度需要匹配 cross-attention 输出或原始选择输出
            img_embedding_dim = self.num_queries if self.fusion_type == 'cross_attention' else self.num_select * 49 # 49 是 Swin 输出 patch 数
            images_embedding = torch.zeros(batch_size, img_embedding_dim, self.input_size, device=self.device)
            # 仍然添加模态类型嵌入，但可能意义不大
            images_embedding = images_embedding + self.token_type_embeddings(
                torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=self.device)
            )


        # --- 处理 EEG 模态 (如果提供) ---
        if eeg_data is not None:
            eeg_data = self.layernorm(eeg_data)
            eeg_embedding = self.eeg_proj(eeg_data)
            eeg_embedding = self.activation(eeg_embedding)
            # 添加 EEG 模态类型嵌入
            eeg_embedding = eeg_embedding + self.token_type_embeddings(
                torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=self.device)
            )
        else:
            # 如果没有 EEG 数据，创建一个零张量占位符
            # EEG 经过投影后的维度是 (batch_size, eeg_sequence_length, input_size)
            # 原始 EEG 序列长度是 384 / (某个下采样?) -> 投影后是 32 个 token?
            # 检查原始代码 eeg_embedding = self.eeg_proj(eeg_data)， eeg_data 是 [B, 32, 384]
            # 所以投影后是 [B, 32, 768]
            eeg_embedding_dim = 32 # 假设 EEG 投影后有 32 个 token
            eeg_embedding = torch.zeros(batch_size, eeg_embedding_dim, self.input_size, device=self.device)
            # 仍然添加模态类型嵌入
            eeg_embedding = eeg_embedding + self.token_type_embeddings(
                 torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=self.device)
             )

        # --- 融合与分类 (逻辑不变) ---
        multi_embedding = torch.cat((images_embedding, eeg_embedding), dim=1)
        # 扩展 CLS token 并连接
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
        multi_embedding = torch.cat((cls_token_expanded, multi_embedding), dim=1)
        
        multi_embedding = self.transformer_encoder(multi_embedding)

        # --- 修改：分类器的输入 ---
        # 原始代码使用 images_embedding[:, 0, :]，这在 eeg_only 时会是 0
        # 标准 Transformer 通常使用 CLS token 的输出来分类
        cls_token_output = multi_embedding[:, 0, :] # 取第一个 token (CLS token) 的输出
        
        cls_token_output = self.dropout(cls_token_output)
        x = self.classifier(cls_token_output)

        return x