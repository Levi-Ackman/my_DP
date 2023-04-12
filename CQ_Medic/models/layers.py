import torch
import torch.nn as nn

class SelfMultiheadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8,dropout=0.1):
        super(SelfMultiheadAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.drop=nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, data_len, feature_dim]
        batch_size, data_len, feature_dim = x.shape
        
        # Compute query, key, and value projections
        q = self.query(x)  # [batch_size, data_len, feature_dim]
        k = self.key(x)  # [batch_size, data_len, feature_dim]
        v = self.value(x)  # [batch_size, data_len, feature_dim]
        
        # Reshape query, key, and value projections for multi-head attention
        q = q.view(batch_size, data_len, self.num_heads, self.head_dim)  # [batch_size, data_len, num_heads, head_dim]
        k = k.view(batch_size, data_len, self.num_heads, self.head_dim)  # [batch_size, data_len, num_heads, head_dim]
        v = v.view(batch_size, data_len, self.num_heads, self.head_dim)  # [batch_size, data_len, num_heads, head_dim]
        
        # Compute scaled dot-product attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k)  # [batch_size, num_heads, data_len, data_len]
        scores /= self.head_dim ** 0.5
        
        # Compute attention weights and apply to values
        weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, data_len, data_len]
        attended = torch.einsum('bhqk,bkhd->bqhd', weights, v)  # [batch_size, data_len, num_heads, head_dim]
        
        # Concatenate multi-head attention outputs and apply final linear layer
        out = attended.contiguous().view(batch_size, data_len, -1)  # [batch_size, data_len, feature_dim]
        out = self.fc(out)  # [batch_size, data_len, feature_dim]
        out=self.drop(out)
        
        return out

# class SelfMultiheadAttention(nn.Module):
#     def __init__(self, feature_dim, num_heads, dropout_prob=0.1):
#         super(SelfMultiheadAttention,self).__init__()
#         assert feature_dim % num_heads == 0
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
#         self.head_dim = feature_dim // num_heads
        
#         self.query_linear = nn.Linear(feature_dim, feature_dim)
#         self.key_linear = nn.Linear(feature_dim, feature_dim)
#         self.value_linear = nn.Linear(feature_dim, feature_dim)
        
#         self.dropout = nn.Dropout(dropout_prob)
#         self.output_linear = nn.Linear(feature_dim, feature_dim)
        
#     def forward(self, x):
#         batch_size, data_len, feature_dim = x.shape
        
#         # Split input into num_heads separate "heads"
#         x = x.view(batch_size, data_len, self.num_heads, self.head_dim)
#         x = x.permute(0, 2, 1, 3)  # [batch_size, num_heads, data_len, head_dim]
        
#         # Apply separate linear transformations to each head
#         query = self.query_linear(x)  # [batch_size, num_heads, data_len, head_dim]
#         key = self.key_linear(x)  # [batch_size, num_heads, data_len, head_dim]
#         value = self.value_linear(x)  # [batch_size, num_heads, data_len, head_dim]
        
#         # Compute attention scores using dot product of query and key
#         scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, data_len, data_len]
#         scores /= (self.head_dim ** 0.5)  # Scale by sqrt(d_k)
#         attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, data_len, data_len]
#         attention_weights = self.dropout(attention_weights)
        
#         # Apply attention weights to value vectors
#         context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, data_len, head_dim]
#         context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, data_len, num_heads, head_dim]
#         context = context.view(batch_size, data_len, self.feature_dim)  # [batch_size, data_len, feature_dim]
        
#         # Apply final linear transformation to aggregated context vectors
#         output = self.output_linear(context)  # [batch_size, data_len, feature_dim]
        
#         return output

class FeedforwardLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout_prob=0.1):
        super(FeedforwardLayer,self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size, data_len, feature_dim]
        x = self.fc1(x)  # [batch_size, data_len, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, data_len, feature_dim]
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_prob=0.1):
        super(TransformerBlock,self).__init__()
        self.self_attention = SelfMultiheadAttention(input_dim, num_heads, dropout_prob)
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedforward = FeedforwardLayer(input_dim, hidden_dim, dropout_prob)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        # x shape: [batch_size, data_len, input_dim]
        residual = x
        x = self.self_attention(x)  # [batch_size, data_len, input_dim]
        x = self.dropout(x)
        x = self.norm1(x + residual)  # [batch_size, data_len, input_dim]
        
        residual = x
        x = self.feedforward(x)  # [batch_size, data_len, input_dim]
        x = self.dropout(x)
        x = self.norm2(x + residual)  # [batch_size, data_len, input_dim]
        
        return x


class TransformerTimeSeriesForecast(nn.Module):
    def __init__(self, feature_dim, pre_len,num_heads, hidden_dim, num_layers, dropout_prob=0.1,expansion=4,data_len=30):
        super(TransformerTimeSeriesForecast,self).__init__()
        self.num_layers = num_layers

        self.proj=nn.Linear(feature_dim,hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, int(hidden_dim*expansion), dropout_prob)
            for _ in range(num_layers)
        ])
        self.pre_len=pre_len
        self.fc = nn.Linear(data_len*hidden_dim, pre_len) # [batch_size, data_len*hidden_dim] -> [batch_size, pre_len]

        # self.activation=nn.Sigmoid()
        # self.activation=nn.Tanh()
        
    def forward(self, x):
        # x shape: [batch_size, data_len, feature_dim]
        batch_size, data_len, feature_dim=x.shape

        x=self.proj(x)  # [batch_size, data_len, hidden_dim]
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)  # [batch_size, data_len, hidden_dim]
        # Extract the last time step as the prediction
        x = x.contiguous().view(batch_size,-1)  # [batch_size, data_len*hidden_dim]
        pre = self.fc(x)  # [batch_size, pre_len]

        # pre=self.activation(pre)

        return pre


