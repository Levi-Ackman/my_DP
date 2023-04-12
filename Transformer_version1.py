'''
a from_scratch implementaion of the prestigious transformer model
in the paper'Attention is all you need, a more detailed look of the model canb be find in'https://zhuanlan.zhihu.com/p/340149804'
this code is more or less a repreduction of the 
'https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py'
'''
__author__='Levi-Ack'

import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads=heads
        self.head_dim=embed_size//heads

        '''when the embedding_size is not divisible ,raise a error'''
        assert(self.head_dim*heads==embed_size), 'embedding size needs to be divisible by heads'
        
        ## here we use the follow three Linear layer to finish the WA=Q/K/V
        ## function, i.e. to get the three metircs 
        ## one take_home message is that the all three transformation dosen't change the dim of the input tensor at all
        ## so the entire process is highly controlabel
        self.values=nn.Linear(embed_size,embed_size)
        self.keys=nn.Linear(embed_size,embed_size)
        self.queries=nn.Linear(embed_size,embed_size)
        self.fc_out=nn.Linear(embed_size,embed_size)
    '''
    here we input three tensor(Q/K/V) to complete the self attenttion process
    a attribute here is mask,and this one is uss to avoid the decoder's input
    get future imformation
    to put it a simple way,when we transfer a sentence[1,2,3,4,5], when we predict 2, we don't
    want it to use the imformation of 3,4,5, cause that will be cheat,
    so when we do attention of 2,we need to mask the imformation of 3,4,5,,so only the information of 1 and 2 are used
    the way to do so is narratived more detailedly in'https://zhuanlan.zhihu.com/p/340149804'
    check it for a explicit look, here we can simply remember that we do it by simply make the attention metrix which we
    get 'by (Q*K)/scale' as a subsequence metrix,or'上三角矩阵，上三角全0'
    '''
    def forward(self,values,keys,query,mask):
        # record the number of trainning smaples
        N=query.shape[0]
        value_length,key_length,query_length=values.shape[1],keys.shape[1],query.shape[1]
        '''use the linear_layer to get three Metrics for the follow attention_step'''
        values=self.values(values) # (N, value_len, embed_size)
        keys=self.keys(keys) # (N, key_length, embed_size)
        queries =self.queries(query) # (N, query_length, embed_size)

        '''here we split the embeddings into different pieces to apply the multi_head_attention'''
        values=values.reshape(N,value_length,self.heads,self.head_dim)
        keys=keys.reshape(N,key_length,self.heads,self.head_dim)
        queries =queries .reshape(N,query_length,self.heads,self.head_dim)
        ''' 
        we use the 'torch.einsum' to do the matri_mul for us,i.e. the for query*keys for each training example
        with every other training example don't be confused by einsum
        here we give a simple example: let's say we have two tensor x([2,5,8,6]),y([2,3,8,6])--(N, key_length, heads,head_dim)
        when we apply 'torch.einsum("nqhd,nkhd->nhqk", [x, y])' we get a new tensor of[2,8,5,3]
        so by the upper function we can finish the 'query*keys' very conveniently
        '''
        energy =torch.einsum('nqhd,nkhd->nhqk',[queries ,keys])
        '''
        after the upper mulitple, we get a new tnesor energy,which can be 
        easily used to multi with Values matrix
        '''
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        if mask is not None:
            energy=energy.masked_fill(mask==0,float('-1e20'))
        '''
        Normalize energy values,so that they sum to 1. Also divide by scaling factor for better stability
        '''
        attention=torch.softmax((energy/self.embed_size**(1/2)),dim=3)
        '''do softmax at the last dimension'''
        # attention shape: (N, heads, query_len, key_len)

        '''here we do Atten*V to get the finial output and we also reshape the tensor to original shape
        namly the contacate in multihead_attention'''
        out=torch.einsum('nhqk,nvhd->nqhd',[attention,values])
        out=out.reshape(N,query_length,self.heads*self.head_dim)
         # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), 
        # then we reshape and flatten the last two dimensions.-- flatten more or less like a contacet
        
        out=self.fc_out(out)
        # here the Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
'''build a transformer bolck,which will be reused N times in encoder,and can be used as part of the decoder'''
class Transformerblock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(Transformerblock,self).__init__()
        self.attention=SelfAttention(embed_size,heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        ''' 
        one thing deserving notice is that either the self_attention(specificly multi_head) or the feed_forward layer
        dosen't change the shape of the input tensor at all
        so the whole process is very brief and controllable
        '''
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )

        self.dropout=nn.Dropout(dropout)
    def forward(self,value,key,query,mask):
        atttion=self.attention(value,key,query,mask)
        ## add a skip connection here,run through layer_normalization and finally dropout
        ''' we should take care of the dropout,make sure it apply after the layer_norm'''
        x=self.dropout(self.norm1(atttion+query))
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(forward+x))
        return out


''' 
a brief introduction of the attributes:
src_vocab_size   -- number of sentence in every batch
embed_size       -- dimension of each signle word in every input sentences
num_layers       -- how many bolck was used
heads            -- number of heads in multi-head-attention
device           -- device(cuda/cpu)
forward_expansion -- forward_expansion rate of the feed_forward_layer
dropout          -- dropout_rate
max_len          -- max_length of the sentence
'''
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        '''
        the function of 'nn.Embedding' is embed  each word in a sentence 
        into a tensor ,like if we apply nn.Embedding(10,20) to a sentence(10) which have 10 words
        then we get a tensor of (10,20),so each word are embded into a 20_dim tensor
        a more detailed look please check 
        'https://blog.csdn.net/qq_35812205/article/details/125303611?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167266548416800222899620%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167266548416800222899620&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-125303611-null-null.142^v68^pc_rank_34_queryrelevant25,201^v4^add_ask,213^v2^t3_esquery_v1&utm_term=nn.embedding&spm=1018.2226.3001.4187'
        '''

        self.word_embedding=nn.Embedding(src_vocab_size,embed_size) ##embed the whole input
        self.position_embedding=nn.Embedding(max_len,embed_size)  ##make a position embedding,which can be reused multi time to add to the input tensor
        self.layeers=nn.ModuleList(
            [
                Transformerblock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion,
                )
                for _ in range(num_layers)
            ])
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        N, seq_length = x.shape
        '''
        the function of torch.arange(0,seq_length).expand(N,seq_length) can be basically explained as:
        x=torch.arange(0,10).expand(10,10)
        then we have a x which is like:
        x=[
            [0,1,2,3.....9,10],
            [0,1,2,3.....9,10],
            ...
            [0,1,2,3.....9,10],
            ]
        and x.shape()=[10,10]
        '''
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        ''' input plus the postiion_embeding to finish the position_embedding'''
        out=self.dropout((self.word_embedding(x)+self.position_embedding(positions)))

        # In the Encoder the query, key, value are all the same, they all get by the linear_projection of the input_tensor
        # it's in the decoder this will change. This might look a bit odd in this case.
        for layer in self.layeers:
            out=layer(out,out,out,mask)
        
        return out

''' 
the decoder are more or less like the decoder,only a few changes
'''
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.norm=nn.LayerNorm(embed_size)
        self.attention=SelfAttention(embed_size,heads)
        self.transformer_block=Transformerblock(embed_size,heads,dropout,forward_expansion)
        self.dropout=nn.Dropout(dropout)
    '''
    here we need to notice that the imformation those two multi_attention process in Decoder needed
    are different,namely the(Q,K,V,mask),first take the decoder_input as (Q/K/V),and the second take Q,K from encoder as its Q,K,and V is the output of the first multi_atten layer
    '''
    def forward(self,x,value,key,src_mask,trg_mask):
        attention=self.attention(x,x,x,trg_mask)
        query=self.dropout(self.norm(attention+x))

        out=self.transformer_block(value,key,query,src_mask)

        return out
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        foraward_expansion,
        dropout,
        device,
        max_len,
    ):
        super().__init__()
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_len,embed_size)

        self.layers=nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,foraward_expansion,dropout,device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out=nn.Linear(embed_size,trg_vocab_size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,enc_out,src_mask,trg_mask):
        N,seq_len=x.shape
        positions=torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        x=self.dropout((self.word_embedding(x)+self.position_embedding(positions)))

        for layer in self.layers:
            x=layer(x,enc_out,enc_out,src_mask,trg_mask)
        out=self.fc_out(x)

        return out
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        device='cpu',
        max_len=100,
        ):
        super(Transformer,self).__init__()
        self.encoder=Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len,
        )

        self.decoder=Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len,
        )

        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx

        self.device=device
    # src_mask are used to mask the '.'in enc_input,so it should have the same dim of the last_dim of enc_input
    def make_src_mask(self,src):
        src_mask=(src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    # trg_mask are used to mask the attention matrix, so they should have the same dim
    def make_trg_mask(self,trg):
        N,trg_len=trg.shape
        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)
        return trg_mask.to(self.device)

    def forward(self,src,trg):
        src_mask=self.make_src_mask(src)
        trg_mask=self.make_trg_mask(trg)
        enc_src=self.encoder(src,src_mask)
        out=self.decoder(trg,enc_src,src_mask,trg_mask)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device name:',device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)   #[num_sentences,num_words]
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)   #[num_sentences,num_words]

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 81
    trg_vocab_size = 81
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1]) ##cause we always wanna the input one word beyond output, so the translating can work
    print('input_tensor{}  after transformer:  {}'.format(trg.shape,out.shape))

    x1=torch.rand(2,24,512)  #[num_sentences,num_words,embed_dim]
    model_1=SelfAttention(512,8).to(device)
    y1=model_1(x1,x1,x1,mask=None)
    print('input_tensor{}  after SelfAttention:  {}'.format(x1.shape,y1.shape))

    x2=torch.rand(2,24,512) #[num_sentences,num_words,embed_dim]
    model_2=Transformerblock(512,8,0.1,4).to(device)
    y2=model_2(x2,x2,x2,mask=None)
    print('input_tensor{}  after Transformerblock:  {}'.format(x2.shape,y2.shape))


    x3=torch.LongTensor(([1,2,3,4,5],[2,4,6,8,0])) #[num_sentences,num_words]
    model_3=Encoder(81,512,4,8,device,4,0.1,100).to(device)

    y3=model_3(x3,mask=None)
    print('input_tensor{}  after Encoder:  {}'.format(x3.shape,y3.shape))

    x4=torch.LongTensor(([1,2,3,4,5],[2,4,6,8,0]))  #[num_sentences,num_words]
    model_4=Decoder(81,512,3,8,4,0.1,device,100).to(device)

    y4=model_4(x,y3,None,None)
    print('input_tensor{}  after Decoder:  {}'.format(x4.shape,y4.shape))


















