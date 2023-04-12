'''created in 2023/1/4
by Levi Ack'''
__author__='Levi-Ack'

import torch
import math
import torch.nn as nn
import copy
from torch.autograd import Variable
from torch.functional import F
import torch
import math
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
from setting import LAYERS,D_MODEL,D_FF,DROPOUT,H_NUM,TGT_VOCAB,SRC_VOCAB

from setting import DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
A over view of the the Encoder :
[
    Embedding + PositionalEncoding 
        X (batch_size, seq_len, embed_dim)
    self attention, attention mask 
        X_attention = SelfAttention(Q,K,V)
    Layer Normalization, Residual
        X_attention = LayerNorm(X_attention)
        X_attention = X + X_attention
    Feed Forward
        X_hidden = Linear(Activate(Linear(X_attention)))
    Layer Normalization, Residual
        X_hidden = LayerNorm(X_hidden)
        X_hidden = X_attention + X_hidden  
]
"""

'''difinition of each model'''
class Embedding(nn.Module):
    def __init__(self,embed_dim,vocab_size):
        super(Embedding,self).__init__()
        self.embedder=nn.Embedding(vocab_size,embed_dim)
        self.Linear_prject=nn.Linear(embed_dim,embed_dim)
        self.embed_dim=embed_dim
    '''
    here we first embed the input_tensor 
    and do a linear_project as the oirginal paper
    (to create a intermedite variable a which will be used to creat Q/K/V metrics latter)
    '''
    def forward(self,x):
        return self.Linear_prject(self.embedder(x))
class PositionnalEmbedding(nn.Module):
    '''
    the position_code generate by position_embedding is fixed in the entire training and validating process 
    once been created
    '''
    def __init__(
        self,
        embed_dim,
        dropout=0.1,
        max_len=5000,
    ):
        '''
        explaination over the attribute:
        embed_dim: Dimension of the input tensor after input_embedding
        dropout: dropout rate
        max_len: max_length of each sentences
        '''
        super(PositionnalEmbedding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        # here we fisrt create a tensor with shape[max_len, max_len] to store the position vector
        # which will be used to add to the input tensor
        # each row is a position-code of a word
        positon_code=torch.zeros(max_len,embed_dim,device=DEVICE)

        # here we make a position vector with shape(max_len,1)
        position=torch.arange(0,max_len,device=DEVICE).unsqueeze(1)
        
        # 使用exp log 实现sin/cos公式中的分母 div_term.shape([embed_dim])
        '''
        here when we use x=torch.arange(start,end,2) we will end up with getting the even sequence of the tensor x'=torch.arange(start,end)
        for example, x1=x=torch.arange(0,10) x1=[0,1,2,..9],  x2=x1=x=torch.arange(0,10,2),x2=[0,2,4,6,8,10]
        '''
        div_term=torch.exp(torch.arange(0,embed_dim,2,device=DEVICE)*(-math.log(10000.0)/embed_dim))
        
        # padding the [max_len,embed_dim] matrics
        ''' 
        the function of x[:,0::2] and x[:,1::2] is divice the tensor to 2i,2i+1,namely divide the tensor to a odd one and even one
        把数组按照某个维度的index 分成奇序列和偶序列,或者说 2i 和 2i+1
        '''
        ##由于position.shape=[max_len,1], div_term.shape =[embed_dim] 所以 equal to positon_code.shape[max_len,embed_dim]
        positon_code[:,0::2]=torch.sin(position *div_term)
        positon_code[:,1::2]=torch.cos(position *div_term)
        '''
        we should notice that the torch.sin/cos/exp are all point-wised 
        that's say the function will be used seperately on each elment
        '''
        positon_code=positon_code.unsqueeze(0) #在第一维度增加一个维度，将tensor转化为三维tensor
        
        self.register_buffer('positon_code',positon_code)

    def forward(self,x):
        # x=nn.parameter(self.positon_code[:,:x.size(1)],requires_grad=False) a alternative way
        x=x+Variable(self.positon_code[:,:x.size(1)],requires_grad=False) #skipping connection
           
        return x

class PositionnalEmbedding_1(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionnalEmbedding_1, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.zeros(max_len, d_model, device=DEVICE)
        self.position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0,d_model,2,device=DEVICE) *
                             (-math.log(10000.0) / d_model))
    def forward(self,x):
        pe=self.pe
        position=self.position
        div_term=self.div_term
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return x

def attention(query,key,value,mask=None,dropout=None):
    '''
    attributes
    ----
     h代表每个多少个头,这里使矩阵之间的乘积,对所有的head进行同时计算
    :param query:  [batch_size, h, sequence_len, embedding_dim]
    :param key:    [batch_size, h, sequence_len, embedding_dim]
    :param value:  [batch_size, h, sequence_len, embedding_dim]
    :param mask: decoder 使用,用于mask部分输入信息,只做局部 attention
    :param dropout:
    ----
    '''
    dim_k=query.size(-1) # record the last dim of the query,namely the embed_dim
    '''
    将key的最后两个维度互换才能和query矩阵相乘
    将相乘结果除以一个归一化的值
    由query和key 矩阵计算得到的scores矩阵维度: batch_size,h,sequence_len, sequence_len
    scores矩阵中的每一行代表着长度为sequence_len的句子中每个单词与其他所有单词的相似度
    '''
    scores=torch.matmul(query,key.transpose(-2,-1)/math.sqrt(dim_k))
    # 归一化 ，归一化之前要进行mask操作，防止填充的0字段影响
    if mask is not None:
        # 将填充的0值变为很小的负数，根据softmax exp^x的图像，当x的值很小的时候，值很接近0
        # 补充，当值为0时， e^0=1 故不能为1
        scores = scores.masked_fill(mask==0,-1e9)  # mask掉mask矩阵中元素为0的位置scores矩阵中对应位置的值
        '''python example:
        >>> t = torch.randn(3,2)
        >>> t
        tensor([[-0.9180, 8],
                [ 1,      3],
                [ 4,  1.1607]])
        >>> m = torch.randint(0,2,(3,2))
        >>> m
        tensor([[0, 1],
                [1, 1],
                [1, 0]])
        >>> m == 0
        tensor([[ True, False],
                [False, False],
                [False,  True]])
        >>> t.masked_fill(m == 0, -1e9)
        tensor([[-1.0000e+09, 8],
                [ 1,          3],
                [ 4, -1.0000e+09]])
        '''
    attention=F.softmax(scores,dim=-1) ##按照每一行，softmax，计算attention 
    if dropout is not None:
        attention=dropout(attention)

    # 返回注意力和 value的乘积，以及注意力矩阵 attention
    # [batch_size,h,sequence_len, sequence_len] * [batch_size, h, sequence_len, embed_dim]
    # 返回结果矩阵维度 [batch_size,h,sequence_len,embed_dim]
    return torch.matmul(attention, value), attention


class MultiHeaderAttention_1(nn.Module):
    '''
    首先初始化三个权重矩阵 W_Q, W_K, W_V
    将embedded の x 与 权重矩阵相乘,生成Q,K,V
    将Q K V按照head数划分为不同的张量q k v
    对不同的qkv,分别利用attention计算Attention(softmax(QK^T/sqrt(d_model))V)
    输出 [batch_size, sequence_len, embed_dim]最后将张量进行拼接
    '''
    def __init__(self, heads, embed_dim, dropout=0.1):
        """
        :param h: head的数目
        :param d_model: embed的维度
        :param dropout:
        """
        super(MultiHeaderAttention_1, self).__init__()
        self.model_name='MultiHeaderAttention_1'
        ##确保embed_dim是可以被头数heads整除的！
        assert embed_dim % heads == 0 ,'embed_dim is undivisible'

        self.head_dim = embed_dim // heads
        self.heads=heads
        self.embed_dim=embed_dim

        self.query=nn.Linear(embed_dim,embed_dim)
        self.key=nn.Linear(embed_dim,embed_dim)
        self.value=nn.Linear(embed_dim,embed_dim)

        self.fc_out=nn.Linear(embed_dim,embed_dim)

        self.attention=None ##记录attention，便于查看/返回
        self.dropout=nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):
        # cause the attention_score have the shape of [batch_size,heads,sequence_len, sequence_len]
        # so we need do a transformation a match the mask-matrix and score-matrix
        if mask is not None:
            mask=mask.unsqueeze(1) # [batch_size or num_samples,seq_len,seq_len]->[batch_size or num_simples, 1, seq_len,seq_len]
        
        num_batch_size=query.size(0)
        seq_len=query.size(1)

        # query,key,value dimension= [batch_size,seq_len,embed_dim] ->[batch_size,seq_len,embed_dim]
        #use linear_lay to get three matrics
        query=self.query(query)
        key=self.key(key)
        value=self.value(value)

        # head_dim=embed_dim/heads
        # query维度(transpose之后)：[batch_size,seq_len,embed_dim] ->[batch_size,seq_len,heads,head_dim] ->[batch_size, heads, seq_len, head_dim]
        query=query.view(num_batch_size,-1,self.heads,self.head_dim).transpose(1,2)
        key=key.view(num_batch_size,-1,self.heads,self.head_dim).transpose(1,2)
        value=value.view(num_batch_size,-1,self.heads,self.head_dim).transpose(1,2)
        
        # 对query key value 计算 attention
        # attention 返回最后的x 和 atten weight
        x,self.attention=attention(query,key,value,mask=mask,dropout=self.dropout)

        # 将多个头的注意力矩阵concat起来
        # 输入：x shape: [batch_size, heads, sequence_len, embed_dim/heads(or head_dim)]
        # 输出：x shape: [batch_size, sequence_len, embed_dim]
        x=x.transpose(1,2).contiguous().view(num_batch_size,-1,self.embed_dim)
        #调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        
        return self.fc_out(x)  # [batch_size, sequence_len, embed_dim]

class MultiHeaderAttention(nn.Module):
    """
    首先初始化三个权重矩阵 W_Q, W_K, W_V
    将embed x 与 权重矩阵相乘，生成Q,K,V
    将Q K V分解为多个head attention
    对不同的QKV，计算Attention（softmax(QK^T/sqrt(d_model))V）

    输出 batch_size, sequence_len, embed_dim
    """
    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: head的数目
        :param d_model: embed的维度
        :param dropout:
        """
        super(MultiHeaderAttention, self).__init__()
        self.model_name='MultiHeaderAttention'
        assert d_model % h == 0

        self.d_k = d_model // h   # 每一个head的维数
        self.h = h  # head的数量

        # 定义四个全连接函数 WQ,WK,WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存attention结果
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # query维度(transpose之后)：batch_size, h, sequence_len, embedding_dim/h
        query, key, value = [l(x).view(nbatches, -1,self.h, self.d_k).transpose(1,2)
                             for l,x in zip(self.linears, (query, key, value))]
        # 对query key value 计算 attention
        # attention 返回最后的x 和 atten weight
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将多个头的注意力矩阵concat起来
        # 输入：x shape: batch_size, h, sequence_len, embed_dim/h(d_k)
        # 输出：x shape: batch_size, sequence_len, embed_dim
        x = x.transpose(1,2).contiguous().view(nbatches, -1,self.h*self.d_k)


        return self.linears[-1](x)  # batch_size, sequence_len, embed_dim

class PositionwiseFeedForward(nn.Module):
    """对每个position采取相同的操作
    初始化参数：
        embed_dim: embedding维数
        expansion: 线性层隐层网络单元膨胀率
        dropout
    前向传播参数: input: dimension= [batch_size, sequence_len, embed_dim]
                 output: dimension= [batch_size, sequence_len, embed_dim]
    """
    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.embed_dim=embed_dim
        self.expansion=expansion
        self.Linear_1=nn.Linear(embed_dim,embed_dim*self.expansion)
        self.Linear_2=nn.Linear(embed_dim*self.expansion,embed_dim)
        self.droput=nn.Dropout(dropout)
        self.activation=nn.GELU()
    def forward(self,x):
        return self.Linear_2(self.droput(self.activation(self.Linear_1(x))))
class SublayerConnection(nn.Module):
    """
    sublayerConnection把Multi-Head Attention和Feed Forward层连在一起
    组合LayerNorm和Residual
    """
    def __init__(self, LayerNorm_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.norm=nn.LayerNorm(LayerNorm_dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublyer):
        # 返回Layer Norm 和残差连接后结果
        return x + self.dropout(sublyer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, LayerNorm_dim, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.LayerNorm_dim=LayerNorm_dim

        self.layers=nn.ModuleList(
            [
             SublayerConnection(LayerNorm_dim,dropout) for _ in range(2)
            ])
    def forward(self,x,mask):
        # 将embedding输入进行multi head attention
        # 得到 attention之后的结果
        """
        'lambda x:self.... '是一个函数对象,参数是x
        用例: t=lambda x: x+20
        then use t(10),the return is '30'
        """
        x = self.layers[0](x, lambda x: self.self_attn(x,x,x,mask))

        return self.layers[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Encoder, self).__init__()
        self.layers=nn.ModuleList(
        [
            layer for _ in range(num_layers)
        ])
        self.norm=nn.LayerNorm(layer.LayerNorm_dim)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    Decoder的第二个attention层会使用Encoder的输出作为K,V,第一个attention层的输出作为Q
    """
    def __init__(self, LayerNorm_dim, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.LayerNorm_dim = LayerNorm_dim
         # Self-Attention
        self.self_attn = self_attn  ##第一个attention层使用自身输入作为Q/K/V

        # 第二个attention 使用Encoder的输出作为K,V,第一个attention层的输出作为Q
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # decoder 有三次残差连接
        self.layers=nn.ModuleList(
            [
                SublayerConnection(LayerNorm_dim,dropout) for _ in range(3)
            ])
    def forward(self,x,memory,src_mask,trg_mask):
        m=memory ##memory 来自encoder输出
        # self-attention的q，k和v均为decoder hidden or decoder inputs
        x=self.layers[0](x,lambda x:self.self_attn(x,x,x,trg_mask))

        # context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.layers[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.layers[2](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(Decoder, self).__init__()
        
        self.layers=nn.ModuleList(
            [
                layer for _ in range(num_layers)
            ])
        self.norm=nn.LayerNorm(layer.LayerNorm_dim)
    def forward(self,x,memory,src_mask,trg_ask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理 遮盖结束标志'.'的影响
        和一个对输出的attention mask + subsequent mask处理 上三角遮盖未来信息
        """
        for layer in self.layers:
            x=layer(x,memory,src_mask,trg_ask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(Transformer, self).__init__()
        self.encoder=encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator=generator

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self,trg,memory,src_mask,trg_mask):
        return self.decoder(self.trg_embed(trg),memory,src_mask,trg_mask)

    def forward(self,src,trg,src_mask,trg_mask):
        ##encoder输出将作为语境信息（context-information）输入decoder
        out=self.decode(trg,self.encode(src,src_mask), src_mask,trg_mask)
        
        return out

class Generator(nn.Module):
    ## used to get the tgt_vocab_size tensor
    def __init__(self, embed_dim, tgt_vocab_size):
        super(Generator, self).__init__()
        # 将decode后的结果，先进入一个全连接层变为词典大小的向量
        self.project=nn.Linear(embed_dim,tgt_vocab_size)

    def forward(self,x):

        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.project(x),dim=-1)

class LabelSmoothing(nn.Module):
    """标签平滑处理"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        """
        损失函数是KLDivLoss,那么输出的y值得是log_softmax
        具体请看pytorch官方文档,KLDivLoss的公式
        """
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

"""损失函数"""
class SimpleLossCompute:
    """
    简单的计算损失和进行参数反向传播更新训练的函数
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


"""优化器"""
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].embed_dim, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def clones(module, N):
    """
    将传入的module深度拷贝N份
    参数不共享
    :param module: 传入的模型 ex:nn.Linear(d_model, d_model)
    :param N: 拷贝的N份
    :return: nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def make_model(
    src_vocab_size,
    trg_vocab_size,
    num_layers=LAYERS,
    embed_dim=D_MODEL,
    expansion=4,
    heads=H_NUM,
    dropout=DROPOUT):
    print('use model_1')
    
    c=copy.deepcopy
    # 实例化Attention对象
    attn=MultiHeaderAttention_1(heads,embed_dim).to(DEVICE)
    # 实例化FeedForward对象
    feed_forward = PositionwiseFeedForward(embed_dim, expansion, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position_enbedder= PositionnalEmbedding(embed_dim, dropout=dropout).to(DEVICE)

    print('used {}'.format(attn.model_name))
    
    # 实例化Transformer模型对象
    model=Transformer(
        Encoder(EncoderLayer(embed_dim,c(attn),c(feed_forward),dropout).to(DEVICE),num_layers).to(DEVICE),
        Decoder(DecoderLayer(embed_dim, c(attn), c(attn), c(feed_forward), dropout).to(DEVICE), num_layers).to(DEVICE),
        nn.Sequential(Embedding(embed_dim,src_vocab_size).to(DEVICE),c(position_enbedder)).to(DEVICE),
        nn.Sequential(Embedding(embed_dim,trg_vocab_size).to(DEVICE),c(position_enbedder)).to(DEVICE),
        Generator(embed_dim,trg_vocab_size)
        ).to(DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)

if __name__ == "__main__":
    # # def forward(self,src,trg,src_mask,trg_mask):
    # print('test begin')
    # transformer=make_model(512,512,3,512,4,8)

    # src=torch.LongTensor([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
    # trg=torch.LongTensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
    # print('input-tensor-shape{}'.format(trg.shape))
    # x=transformer(src,trg,src_mask=None,trg_mask=None)
    # print(f' out put of transformer: {x}  output_shape:{x.shape}')

    print('model_test begin: \n')
    x1=torch.LongTensor([2,10,3,1,0,2,3,6,0])
    model_1=Embedding(512,20)
    x1_fater=model_1(x1)
    print('original tensor before Embedding is {} after Embedding is {}'.format(x1.shape,model_1(x1).shape))

    x1=torch.LongTensor([2,10,3,1,0,2,3,6,0])
    model_1=PositionnalEmbedding(512)
    x1_fater=model_1(x1)
    print('original tensor before Embedding is {} after Embedding is {}'.format(x1.shape,model_1(x1).shape))











        
