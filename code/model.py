import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalClassifier(nn.Module):
    def __init__(self, 
                 # 修复：移除对全局变量的依赖
                 # 我们将 swin_processor, swin_model, 和 device 作为参数传递进来
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
        self.device = device # 保存 device 为类成员

        # 接收 SWin 模型和 processor
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(self.device) 
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(input_size,num_classes)

        if fusion_type == 'cross_attention':
            # 根据论文表4，最佳性能在 Output Size = 64 时取得
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
                selected = images_embedding[i, indices[i].squeeze(), :, :] 
                selected_embeddings.append(selected)
            
            return torch.stack(selected_embeddings)

    def forward(self, eeg_data, images_data):
        batch_size = images_data.size(0)
        
        images_embedding = []
        for i in range(self.num_instances):
            image = images_data[:, i]  # [batch_size, 224, 224, 3]
            image_data = self.img_processor(image, return_tensors="pt").to(self.device)
            embedding = self.swin_model(**image_data).last_hidden_state
            images_embedding.append(embedding)
        
        images_embedding = torch.stack(images_embedding, dim=1)
        selected_embeddings = self.select_instances(images_embedding)
        selected_embeddings = selected_embeddings.view(batch_size, -1, 768)
        
        if self.fusion_type == 'cross_attention':
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            image_features, _ = self.cross_attention(
                query=query_tokens,
                key=selected_embeddings,
                value=selected_embeddings
            )
            images_embedding = image_features  # [batch_size, 64, 768]
        
        elif self.fusion_type == 'mlp':
            x = self.mlp_up(selected_embeddings)
            x = self.mlp_act(x)
            images_embedding = self.mlp_down(x)
        
        else:
            images_embedding = selected_embeddings

        eeg_data = self.layernorm(eeg_data)
        eeg_embedding = self.eeg_proj(eeg_data)
        eeg_embedding = self.activation(eeg_embedding)

        images_embedding, eeg_embedding = (
            images_embedding + self.token_type_embeddings(torch.zeros(images_embedding.shape[0], 1, dtype=torch.long, device=self.device)),
            eeg_embedding + self.token_type_embeddings(torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=self.device))
        )

        multi_embedding = torch.cat((images_embedding, eeg_embedding), dim=1)
        multi_embedding = torch.cat((self.cls_token.expand(multi_embedding.size(0), -1, -1), multi_embedding), dim=1)
        multi_embedding = self.transformer_encoder(multi_embedding)

        cls_token_output = images_embedding[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)
        x = self.classifier(cls_token_output)

        return x