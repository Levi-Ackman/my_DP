import torch
import  torch.nn as nn
import torch.nn as nn

class BiLSTMBlock(nn.Module):
    def __init__(self,input_dim, output_dim, num_layers=1, dropout=0.2):
        super(BiLSTMBlock, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # define the BiLSTM layer
        self.bilstm = nn.LSTM(input_dim, output_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # define the LayerNorm layer
        self.layernorm = nn.LayerNorm(output_dim * 2)
        
        # define the LeakyReLU activation function
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        
        # define the dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x has shape [batch_size, data_len, feature_dim]
        
        # pass the input tensor through the BiLSTM layer
        out, _ = self.bilstm(x)
        # out has shape [batch_size, data_len, output_dim*2]
        
        # apply LayerNorm
        out = self.layernorm(out)
        # out has shape [batch_size, data_len, output_dim*2]
        
        # apply LeakyReLU
        out = self.leakyrelu(out)
        # out has shape [batch_size, data_len, output_dim*2]
        
        # apply dropout
        out = self.dropout(out)
        # out has shape [batch_size, data_len, output_dim*2]
        
        # project the output to the desired output dimension
        out = nn.Linear(self.output_dim*2, self.output_dim)(out)
        # out has shape [batch_size, data_len, output_dim]
        
        return out

class FCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=5,dropout=0.2):
        super(FCNBlock, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # define a stack of convolutional layers with LeakyReLU activation and dropout
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1) for _ in range(num_layers)])
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        
        # define the LayerNorm layer
        self.layernorm = nn.LayerNorm(input_dim)
        
        # define a final projection layer to output the desired dimension
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        
    def forward(self, x):
        # x has shape [batch_size, data_len, feature_dim]
        
        # transpose the input tensor to shape [batch_size, feature_dim, data_len]
        x = x.transpose(1, 2)
        
        # pass the input tensor through the stack of convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.leakyrelu(x)
            x = self.dropout(x)
        # x has shape [batch_size, input_dim, data_len]
        
        # apply LayerNorm
        x = self.layernorm(x.transpose(1, 2))
        # x has shape [batch_size, input_dim, data_len]
        x=x.transpose(1, 2)
        # transpose the output tensor to shape [batch_size, data_len, output_dim]
        out = self.proj(x).transpose(1, 2)
        # out has shape [batch_size, data_len, output_dim]
        
        return out
    
class CrossModalityMultiAttentionHead(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.2):
        super(CrossModalityMultiAttentionHead, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # define Q, K, and V linear transformations
        self.Q_linear = nn.Linear(input_dim, input_dim)
        self.K_linear = nn.Linear(input_dim, input_dim)
        self.V_linear = nn.Linear(input_dim, input_dim)

        # define output linear transformation
        self.out_linear = nn.Linear(input_dim, input_dim)

        # define dropout layer and LayerNorm layer
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, q,k,v):
        # x has shape [batch_size, data_len, feature_dim]
        # split x into Q, K, and V
        Q = self.Q_linear(q)
        K = self.K_linear(k)
        V = self.V_linear(v)

        # split Q, K, and V into num_heads
        batch_size, data_len, _ = q.size()
        split_size = self.input_dim // self.num_heads

        assert self.input_dim % self.num_heads==0, 'please rechoose your heads,make sure the split head_dim sum up to input_dim !'


        Q = Q.view(batch_size, data_len, self.num_heads, split_size).transpose(1, 2)  # [batch_size, num_heads, data_len, split_size]
        K = K.view(batch_size, data_len, self.num_heads, split_size).transpose(1, 2)  # [batch_size, num_heads, data_len, split_size]
        V = V.view(batch_size, data_len, self.num_heads, split_size).transpose(1, 2)  # [batch_size, num_heads, data_len, split_size]

        # compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim ** 0.5)  # [batch_size, num_heads, data_len, data_len]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # compute weighted sum of values
        weighted_values = torch.matmul(attn, V)  # [batch_size, num_heads, data_len, split_size]

        # concatenate the num_heads back into feature_dim
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, data_len, -1)  # [batch_size, data_len, input_dim]

        # pass the concatenated values through output linear layer and LayerNorm
        out = self.out_linear(weighted_values)
        out = self.dropout(out)
        out = self.layernorm(q + out)

        return out

class FeedforwardLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedforwardLayer,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.layer_norm=nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # x shape: [batch_size, data_len, feature_dim]
        res=x
        x = self.fc1(x)  # [batch_size, data_len, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, data_len, feature_dim]
        x = self.dropout(x)

        x=self.layer_norm(x+res)

        return x
    


class LSTM_FCN_2_Atten_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8,dropout=0.1):
        super(LSTM_FCN_2_Atten_Block,self).__init__()
        self.lstm_block=BiLSTMBlock(input_dim=input_dim,output_dim=hidden_dim,num_layers=1)
        self.fcn_block=FCNBlock(input_dim=input_dim,output_dim=hidden_dim,dropout=dropout)
        
        self.cross_attn_block_1=CrossModalityMultiAttentionHead(
            input_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
           )
        self.cross_attn_block_2=CrossModalityMultiAttentionHead(
            input_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
           )
        self.feedf_block=FeedforwardLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
            )
        self.proj=nn.Sequential(
            nn.Linear(hidden_dim*2,input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout)
           )

    def forward(self,x):
        # x shape: [batch_size, data_len, feature_dim]
        x1=self.lstm_block(x)
        x2=self.fcn_block(x)

        lstm_2_fcn=self.cross_attn_block_1(x1,x2,x2)
        fcn_2_lstm=self.cross_attn_block_2(x2,x1,x1)

        cross_attn_out=self.proj( torch.cat([lstm_2_fcn,fcn_2_lstm],dim=-1) )

        ff_out=self.feedf_block(cross_attn_out)

        return ff_out



class LSTM_FCN_2_Atten(nn.Module):
    def __init__(self, data_len,pre_len,input_dim, hidden_dim,expansion=4,num_layers=2, num_heads=8,dropout=0.1):
        super(LSTM_FCN_2_Atten,self).__init__()
        self.proj_layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
          )
        self.ls_fcn_2_crs_layers=nn.ModuleList(
            [
               LSTM_FCN_2_Atten_Block(
                input_dim=hidden_dim, 
                hidden_dim=int(hidden_dim*expansion),
                num_heads=num_heads,
                dropout=dropout,
               ) for _ in range(num_layers)
            ]
          )
        self.pre_head=nn.Linear(hidden_dim*data_len, pre_len)

    def forward(self,x):
        batch_size,_,_=x.shape
        feature=self.proj_layers(x)
        for layer in self.ls_fcn_2_crs_layers:
            feature=layer(feature)

        feature=feature.contiguous().view(batch_size,-1)  # [batch_size, data_len*hidden_dim]
        
        # feature=feature.contiguous().view(batch_size,1,-1)  # [batch_size,1, data_len*hidden_dim]
        
        pred=self.pre_head(feature)

        return pred

class MOCO_LSTM_FCN_2_Atten(nn.Module):
    def __init__(self, data_len,pre_len,input_dim, hidden_dim,expansion=4,num_layers=2, num_heads=8,dropout=0.1):
        super(MOCO_LSTM_FCN_2_Atten,self).__init__()
        self.proj_layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
          )
        self.ls_fcn_2_crs_layers=nn.ModuleList(
            [
               LSTM_FCN_2_Atten_Block(
                input_dim=hidden_dim, 
                hidden_dim=int(hidden_dim*expansion),
                num_heads=num_heads,
                dropout=dropout,
               ) for _ in range(num_layers)
            ]
          )
        self.representaion_pro=nn.Linear(hidden_dim*data_len, hidden_dim)

    def forward(self,x):
        batch_size,_,_=x.shape
        feature=self.proj_layers(x)
        for layer in self.ls_fcn_2_crs_layers:
            feature=layer(feature)

        feature=feature.contiguous().view(batch_size,-1)  # [batch_size, data_len*hidden_dim]
        
        # feature=feature.contiguous().view(batch_size,1,-1)  # [batch_size,1, data_len*hidden_dim]
        
        # pred=self.pre_head(feature)

        return feature









def test_bilstm():
   pass

def test_fcn():
    pass

def test_cross():
    pass
