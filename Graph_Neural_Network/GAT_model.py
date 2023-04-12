import torch
import torch.nn as nn
import torch.functional as F
from torch_geometric.nn import TransformerConv,TopKPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap

torch.manual_seed(42)

class GNN(nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size=model_params['model_embedding_size']
        n_heads=model_params['model_attention_heads']
        self.n_layers=model_params['model_layers']
        dropout_rate=model_params['model_dropout_rate']
        top_K_ratio=model_params['model_top_K_ratio']
        self.top_K_every_n=model_params['model_top_K_every_n']
        dense_nerons=model_params['model_dense_nerons']
        edge_dim=model_params['model_edge_dim']

        self.conv_layers=nn.ModuleList([])
        self.transf_layers=nn.ModuleList([])
        self.pooling_layers=nn.ModuleList([])
        self.bn_layers=nn.ModuleList([])
        
        #a single Graph_Attention_Layer:
        self.conv1=TransformerConv(feature_size,
                                   embedding_size,
                                   heads=n_heads,
                                   dropout=dropout_rate,
                                   edge_dim=edge_dim,
                                   beta=True)
        #used for tranfer the result after n attention heads to shape of the ori_tensor
        self.transf1=nn.Linear(embedding_size*n_heads,
                               embedding_size)
        self.bn1=nn.BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))
            self.transf_layers.append(nn.Linear(embedding_size*n_heads,embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(embedding_size))
            ''' after a given step, we performer a top_k pooling to prune the graph size'''
            if i % self.top_K_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size,
                                                       ratio=top_K_ratio))
        ''' the classification_head uesd applied to the final layers for graph_level_cls'''
        self.lin1=nn.Linear(embedding_size*2,dense_nerons)  ## the '*2* was because we will use the combination of gap and gmp
        self.lin2=nn.Linear(dense_nerons,int(dense_nerons/2))  #convergence the tensor to a given state
        ''' cause this is a binary classification case, so we can easily output a single value between [0,1]'''
        self.lin3=nn.Linear(int(dense_nerons/2),1)  
        
        ''' # or we can use the follow linear_layer with cross_entopy loss_fn
        self.lin3=nn.Linear(int(dense_nerons/2),2) 
        '''
    def forward(self,x,edge_index,edge_attr,batch_index):
        '''
        x is the input embedding of each graph node
        edge_index stand for the adjacent matric
        edge_attr stand for embedding of each edge
        batch_index stand for the batch_size used for gap and gmp
        '''
        ## initiate stage
        x= self.conv1(x,edge_index,edge_attr)
        x=nn.ReLU(self.transf1(x))
        x=self.bn1(x)

        ## this is used to hold the intermedate graph_representation
        ## cause when we conduct graph_level_cls we will use 
        # the whole state information in each stage to get a holistic view
        global_representaion=[]

        for i in range(self.n_layers):
            x=self.conv_layers[i](x,edge_index,edge_attr)
            x=nn.ReLU(self.transf_layers[i](x))
            x=self.bn_layers[i](x)

            #always aggregate layers information after perform top_k pooling
            if i%self.top_K_every_n ==0 or i ==self.n_layers:
                x,edge_index,edge_attr,batch_index,_,_=self.pooling_layers[int(i/self.top_K_every_n)](
                    x,edge_index,edge_attr,batch_index
                )

                global_representaion.append(torch.cat(  [gmp(x,batch_index),gap(x,batch_index)], dim=1 ))
        ## 
        x= sum(global_representaion)

        x=nn.ReLU(self.lin1(x))
        x=nn.Dropout(x,p=0.8)
        x=nn.ReLU(self.lin2(x))
        x=nn.Dropout(x,p=0.8)

        x=self.lin3(x)
        # x=nn.Sigmoid(x) ## use this when we applied cross_entropy

        return x










