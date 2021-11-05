import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
########### for GCN ##############
from torch.nn import init
####### for GAT and HAN ##########
#import dgl
#import dgl.function as fn
#from dgl.nn import GATConv

######################## Accessory Functions ##############################

def tensor_to_string(t, index = False):
    AA_dict = {0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'Q', 6:'E', 7:'G', 8:'H', 9:'I', 10:'L',
               11:'K', 12:'M', 13:'F', 14:'P', 15:'S', 16:'T', 17:'W', 18:'Y', 19:'V', 20:'!'}
    if not index:
        t = t.max(dim=-1)[1]
    s = []
    for batch in t:
        seq = ''
        for i in batch:
            seq += AA_dict[int(i)]
        s.append(seq)
    return s

############################# GCN ####################################################

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, 
            dropout=0.0, bias=True, hetero=False, share_weight = False, channel_num = 1):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.hetero = hetero
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.share_weight = share_weight # whether the weight of different channels are the same
        if share_weight: 
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))#.cuda())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))#.cuda())
            else:
                self.bias = None
        else:
            self.weight = nn.Parameter(torch.FloatTensor(channel_num, input_dim, output_dim))#.cuda())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(channel_num, 1, output_dim))#.cuda())
            else:
                self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=3)
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, feature_dim, hidden_dim, embedding_dim, num_layers, 
                 concat=True, bn=True, dropout=0.0, bias = True, channel_num=1):
        super(GcnEncoderGraph, self).__init__()

        # feature_dim: dimension of the node feature vectors
        # hidden_dim: dimension of the node hidden vectors
        # embedding_dim: dimension of the node embedding_vectors
        # num_layers: number of the hidden layers

        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
  
        self.channel_num = channel_num
        self.bias = bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(feature_dim, hidden_dim, embedding_dim, num_layers, 
                                                                                  add_self, normalize=True, dropout=dropout)

        self.act = nn.ReLU()

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=False, dropout=0.0):

        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                                normalize_embedding=normalize, bias=self.bias, channel_num = self.channel_num)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, channel_num = self.channel_num) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, channel_num = self.channel_num)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' 
        Batch normalization of 3D tensor x
        '''
        shape = x.shape
        x = torch.reshape(x,(-1,shape[-2],shape[-1]))
        bn_module = nn.BatchNorm1d(x.size()[1])#.cuda()
        return bn_module(x).reshape(shape)

    def forward(self, x, adj, **kwargs):

        out_all = []
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
      
        out_all.append(x)

        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out_all.append(x)

        out = self.conv_last(x,adj)
        out_all.append(out)

        if self.concat:
            output = [torch.cat(list(tensor.transpose(0,1)), dim =-1) for tensor in out_all]
            output = torch.cat(output,dim=-1)
        else:
            output = torch.cat(list(out.transpose(0,1)),dim=-1)
         
        return output

############################# GAT ####################################################

class GAT(nn.Module):
    """
    Based on https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
    """
    def __init__(self,
                 #g,
                 num_layers,  ### number of layers
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers - 1  # SZ add
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, self.num_hidden_layers):  # SZ modify
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, x, adj, **kwargs): # SZ modify
        g = None# change the format of the graphs
        h = inputs # node features
        for l in range(self.num_hidden_layers):
            h = self.gat_layers[l](g, h).flatten(1)  # SZ modify
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)  # SZ modify
        return logits          

############################# HAN ####################################################

"""
Based on https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model.py
"""

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
 
####################### Graph-level Embedding ##################################

class GraphLevelEmbedding(nn.Module):
    def __init__(self, label_num = 1080, pred_hidden_dims=[4000], pooling='max', act=nn.ReLU(), CUDA = False, # for prediction 
                 ### for GNN
                 model='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, num_layers=3, channel_num=5, 
                 concat=False, bn=True, dropout=0.0, bias=True, 
                 ### for RNN (if consider the sequence)
                 seq_embed=False, rnn_model='LSTM', rnn_bias=True, rnn_hidden_size=64, rnn_num_layers=3, 
                 seq_cat=True, seq_GNN=False, bidirectional=True, rnn_dropout=0.01):
        #************************************************************************************************
        super(GraphLevelEmbedding, self).__init__()
     
        # label_num: number of classes
        # pred_hidden_dims: dimension of the hidden vectors
        # seq_cat: whether concatenate the node embedding to the amino acid for RNN
        # seq_GNN: whether utilize GNN to embed the sequence 

        self.act = act
        self.CUDA = CUDA

        #### GNN for node embedding ###
        self.model = model
        self.pooling = pooling
        self.channel_num = channel_num
        self.concat = concat

        if seq_embed and seq_GNN:
            self.gnn_in_dim = feature_dim + rnn_hidden_size
        else:
            self.gnn_in_dim = feature_dim

        if model == 'GCN':
            self.GNN = GcnEncoderGraph(feature_dim = self.gnn_in_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim, 
                                       num_layers = num_layers, concat=concat, bn = bn, dropout=dropout, bias=bias, channel_num=channel_num)
        else:
            print('Error! No GNN model named %s!'%model)
            return None

        if concat:
            self.gnn_out_dim = (hidden_dim * (num_layers - 1) + embedding_dim) * channel_num
        else:
            self.gnn_out_dim = embedding_dim * channel_num

        ### Seq embedding ###
        self.seq_embed = seq_embed
        self.rnn_model = rnn_model
        self.seq_cat = seq_cat
        self.seq_GNN = seq_GNN
        self.bidirectional = bidirectional
        self.rnn_hidden_size = rnn_hidden_size

        if seq_embed:
            if seq_cat and (not seq_GNN):
                seq_feature_dim = 21 + self.gnn_out_dim
            else:
                seq_feature_dim = 21

            if seq_GNN:
                self.pred_input_dim = self.gnn_out_dim
            else:
                self.pred_input_dim = self.gnn_out_dim + rnn_hidden_size

            if rnn_model == 'LSTM': 
                self.RNN = nn.LSTM(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout, 
                                   batch_first=True, bidirectional = bidirectional)
            elif rnn_model == 'GRU':
                self.RNN = nn.GRU(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout,
                                  batch_first=True, bidirectional = bidirectional)
            elif rnn_model == 'RNN':
                self.RNN = nn.RNN(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout,
                                  batch_first=True, bidirectional = bidirectional)
            else:
                print('Error! No RNN model named %s!'%rnn_model)

        else:
            self.pred_input_dim = self.gnn_out_dim

        ### feedforward part ###
        self.label_num = label_num

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_num) #, num_aggs=self.num_aggs)

        self.softmax = nn.Softmax(dim = -1)
        ### Initialize the weights ###
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_num): 
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_num)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_num))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def forward(self, x, adj, seq=None, seq_mask=None, **kwargs):
        ### GNN
       
        if (not self.seq_GNN) or (not self.seq_embed):
            node_vec = self.GNN(x, adj)
        #print(x)
        #print(adj)

        if self.seq_embed:
            seq_shape = seq.shape
            batch_size = seq_shape[0]
            node_num = seq_shape[1]
            seq_len = seq_shape[2]
            aa_dim = seq_shape[3]

            if self.seq_cat and (not self.seq_GNN):
                node_vec_repeat = node_vec.repeat(1,1,seq_len).reshape(batch_size, node_num, seq_len, self.gnn_out_dim)
                seq = torch.cat((node_vec_repeat, seq),-1)
                aa_dim += self.gnn_out_dim 

            seq = seq.reshape(-1, seq_len, aa_dim)

            seq_emb, _  = self.RNN(seq)
            seq_emb = seq_emb[:,-1,:]
            if self.bidirectional:
                seq_emb = seq_emb[:, :self.rnn_hidden_size] + seq_emb[: ,self.rnn_hidden_size:] 
            
            seq_emb = seq_emb.reshape(-1, node_num, self.rnn_hidden_size)
            seq_emb = seq_emb * seq_mask # the embedding of the padding nodes should be padding

            if self.seq_GNN:
                seq_emb = seq_emb.repeat(1,self.channel_num,1).reshape(batch_size, self.channel_num, node_num, self.rnn_hidden_size)
                x = torch.cat((x,seq_emb),-1)
                node_vec = self.GNN(x, adj)
            else:
                node_vec = torch.cat((node_vec, seq_emb),-1)
        
        #print(node_vec.shape)

        if self.pooling == 'max':
            node_vec,_ = torch.max(node_vec, dim=-2)
        elif self.pooling == 'sum':
            node_vec,_ = torch.sum(node_vec, dim=-2)
        elif self.pooling == 'mean':
            node_vec,_ = torch.mean(node_vec, dim=-2)
        else:
            print('Error! No pooling method named %s!'%self.pooling)
        #print(node_vec)
        ypred = self.pred_model(node_vec)
        return ypred
 
    def hierarchy_arrange(self, soft_vec, arrange_index):
        """
        Change the softmax vector from one level to another.
        """
        shape = soft_vec.shape
        out_vec = torch.zeros(shape[0],len(arrange_index)).float()
        if self.CUDA:
            out_vec = out_vec.cuda()
        for i,idx in enumerate(arrange_index):
            out_vec[:,i] = torch.sum(soft_vec[:,:idx],dim=-1)
            soft_vec = soft_vec[:,idx:]
        return out_vec 

    def loss(self, pred, label, loss_type='softmax', lambdas = [1,1,1], weight = None, arrange_index = None): 
        # softmax + CE

        # weight: tensor of list of tensors 
        # arrange_index: help tp arrange the softmax vector
        if loss_type == 'softmax':
            if type(weight) == torch.Tensor and (weight >= 0).any():
                loss_all = torch._C._nn.nll_loss(F.log_softmax(pred,dim=-1), label, weight = weight)
                return torch.mean(loss_all)
            else:
                return F.cross_entropy(pred, label, reduction='mean')
        elif loss_type == 'hierarchy':
            prob_family = self.softmax(pred)
            prob_sf = self.hierarchy_arrange(prob_family, arrange_index[0])
            prob_fold = self.hierarchy_arrange(prob_family, arrange_index[1])
            prob_class = self.hierarchy_arrange(prob_family, arrange_index[2])

            #print(prob_family, prob_family.dtype)
            #print(prob_class, prob_class.dtype)

            if type(weight) == torch.Tensor and (weight >= 0).any():
                cross_entropy_1 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_family), label[:,0], weight = weight[0]))
                cross_entropy_2 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_sf), label[:,1], weight = weight[1]))
                cross_entropy_3 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_fold), label[:,2], weight = weight[2]))
                cross_entropy_4 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_class), label[:,3], weight = weight[3]))
            else:
                cross_entropy_1 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_family), label[:,0]))
                cross_entropy_2 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_sf), label[:,1]))
                cross_entropy_3 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_fold), label[:,2]))
                cross_entropy_4 = torch.mean(torch._C._nn.nll_loss(torch.log(prob_class), label[:,3]))

            #print(prob_family)
            #print(weight[3])
            #print(torch._C._nn.nll_loss(torch.log(prob_family), label[:,0], weight = weight[0]))

            return cross_entropy_1 + cross_entropy_2 * lambdas[0] + cross_entropy_3 * lambdas[1] + cross_entropy_4 * lambdas[2] 

###### Joint Embedding ###

class GraphLevelJointEmbedding(GraphLevelEmbedding):
    def __init__(self, label_num, label_mat, pred_hidden_dims=[4000], pooling='max', act=nn.ReLU(), CUDA = True, # for prediction and joint embedding
                 ### for GNN
                 model='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, num_layers=3, channel_num=5,
                 concat=False, bn=True, dropout=0.0, bias=True,
                 ### for RNN (if consider the sequence)
                 seq_embed=False, rnn_model='LSTM', rnn_bias=True, rnn_hidden_size=64, rnn_num_layers=3,
                 seq_cat=True, seq_GNN=False, bidirectional=True, rnn_dropout=0.01):
        #************************************************************************************************
        super(GraphLevelJointEmbedding, self).__init__()

        # label_num: number of classes
        # pred_hidden_dims: dimension of the hidden vectors
        # seq_cat: whether concatenate the node embedding to the amino acid for RNN
        # seq_GNN: whether utilize GNN to embed the sequence 

        self.act = act
        self.CUDA = CUDA

        #### GNN for node embedding ###
        self.model = model
        self.pooling = pooling
        self.channel_num = channel_num
        self.concat = concat

        if seq_embed and seq_GNN:
            self.gnn_in_dim = feature_dim + rnn_hidden_size
        else:
            self.gnn_in_dim = feature_dim

        if model == 'GCN':
            self.GNN = GcnEncoderGraph(feature_dim = self.gnn_in_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim,
                                       num_layers = num_layers, concat=concat, bn = bn, dropout=dropout, bias=bias, channel_num=channel_num)
        else:
            print('Error! No GNN model named %s!'%model)
            return None

        if concat:
            self.gnn_out_dim = (hidden_dim * (num_layers - 1) + embedding_dim) * channel_num
        else:
            self.gnn_out_dim = embedding_dim * channel_num

        ### Seq embedding ###
        self.seq_embed = seq_embed
        self.rnn_model = rnn_model
        self.seq_cat = seq_cat
        self.seq_GNN = seq_GNN
        self.bidirectional = bidirectional
        self.rnn_hidden_size = rnn_hidden_size

        if seq_embed:
            if seq_cat and (not seq_GNN):
                seq_feature_dim = 21 + self.gnn_out_dim
            else:
                seq_feature_dim = 21

            if seq_GNN:
                self.pred_input_dim = self.gnn_out_dim
            else:
                self.pred_input_dim = self.gnn_out_dim + rnn_hidden_size

            if rnn_model == 'LSTM':
                self.RNN = nn.LSTM(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout,
                                   batch_first=True, bidirectional = bidirectional)
            elif rnn_model == 'GRU':
                self.RNN = nn.GRU(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout,
                                  batch_first=True, bidirectional = bidirectional)
            elif rnn_model == 'RNN':
                self.RNN = nn.RNN(seq_feature_dim, rnn_hidden_size, num_layers = rnn_num_layers, bias = rnn_bias, dropout = rnn_dropout,
                                  batch_first=True, bidirectional = bidirectional)
            else:
                print('Error! No RNN model named %s!'%rnn_model)

        else:
            self.pred_input_dim = self.gnn_out_dim

        ### feedforward part ###
        self.label_num = label_num
        self.label_mat = label_mat
        self.label_weight = nn.Parameter(torch.FloatTensor(label_mat.shape[0], self.pred_input_dim))
        if CUDA:
            self.label_mat = self.label_mat.cuda()
            self.label_weight = self.label_weight.cuda()

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_num) #, num_aggs=self.num_aggs)

        self.softmax = nn.Softmax(dim = -1)
        ### Initialize the weights ###
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
      
    def forward(self, x, adj, seq=None, seq_mask=None, **kwargs):
        ### GNN

        if (not self.seq_GNN) or (not self.seq_embed):
            node_vec = self.GNN(x, adj)

        if self.seq_embed:
            seq_shape = seq.shape
            batch_size = seq_shape[0]
            max_node_num = seq_shape[1]
            seq_len = seq_shape[2]
            aa_dim = seq_shape[3]

            if self.seq_cat and (not self.seq_GNN):
                node_vec_repeat = node_vec.repeat(1,1,seq_len).reshape(batch_size, max_node_num, seq_len, self.gnn_out_dim)
                seq = torch.cat((node_vec_repeat, seq),-1)
                aa_dim += self.gnn_out_dim

            seq = seq.reshape(-1, seq_len, aa_dim)

            seq_emb, _  = self.RNN(seq)
            seq_emb = seq_emb[:,-1,:]
            if self.bidirectional:
                seq_emb = seq_emb[:, :self.rnn_hidden_size] + seq_emb[: ,self.rnn_hidden_size:]

            seq_emb = seq_emb.reshape(-1, max_node_num, self.rnn_hidden_size)
            seq_emb = seq_emb * seq_mask # the embedding of the padding nodes should be padding

            if self.seq_GNN:
                seq_emb = seq_emb.repeat(1,self.channel_num,1).reshape(batch_size, self.channel_num, max_node_num, self.rnn_hidden_size)
                x = torch.cat((x,seq_emb),-1)
                node_vec = self.GNN(x, adj)
            else:
                node_vec = torch.cat((node_vec, seq_emb),-1)

        label_embed = torch.matmul(self.label_mat, self.label_weight)
        label_embed = label_embed.expand(batch_size, -1, -1)
        joint_embed = torch.matmul(node_vec, label_emb.transpose(-2,-1))
        joint_embed = self.softmax(joint_embed)
        joint_embed = self.GNN_joint(joint_embed, embed)

        #if self.pooling == 'max':
        #    node_vec,_ = torch.max(node_vec, dim=-2)
        #elif self.pooling == 'sum':
        #    node_vec,_ = torch.sum(node_vec, dim=-2)
        #elif self.pooling == 'mean':
        #    node_vec,_ = torch.mean(node_vec, dim=-2)
        #else:
        #    print('Error! No pooling method named %s!'%self.pooling)
        
        ypred = self.pred_model(node_vec)
        return ypred


