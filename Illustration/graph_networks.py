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

############################### For antibody design #####################################

###### GNN Container ######

class EncoderGNN(nn.Module):
    """
    Encoder for the structure graph.
    """
    def __init__(self, feature_dim, hidden_dim = 512, seq_dim = 64, embedding_dim = 64, num_layers = 3,
                 gnn_kind = 'GCN', concat=False, bn=True, dropout=0.1, bias = True, channel_num=11, USE_CUDA=True):
        super(EncoderGNN, self).__init__()

        ### architecture ###
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.channel_num = channel_num
        ### versions ###
        self.gnn_kind = gnn_kind
        self.concat = concat
        self.bn = bn
        self.bias = bias
        self.USE_CUDA = USE_CUDA
        ### GNN module
        if gnn_kind == 'GCN':
            self.gnn = GcnEncoderGraph(feature_dim = feature_dim, hidden_dim = hidden_dim, seq_dim = seq_dim,
                                       embedding_dim = embedding_dim, num_layers = num_layers,
                                       concat = concat, bn = bn, dropout = dropout,
                                       bias = bias, channel_num = channel_num, CUDA = USE_CUDA)
            self.gnn_out_dim = self.gnn.out_dim
        else:
            print('Error! No GNN module named "%s"!'%gnn_kind)
            quit()

    def forward(self, feat, seq_feat, adj):
        output = self.gnn(feat, seq_feat, adj)
        return output

####################### Sequence Encoder ####################

class EncoderRNN(nn.Module):
    """
    Encoder (can be RNN, GRU, LSTM or Transformer) for the antigen and antibody sequences.
    """
    def __init__(self, input_size = 22, hidden_size = 512, with_linear = False, output_size = 32, n_layers=3, dropout_p=0.1,
                 rnn_kind = 'LSTM', bidirectional=True, USE_CUDA = True, avoid_padding = True):
        super(EncoderRNN, self).__init__()

        ### architecture ###
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        ### versions ###
        self.rnn_kind = rnn_kind
        self.bidirectional = bidirectional
        self.USE_CUDA = USE_CUDA
        self.avoid_padding = avoid_padding
        self.with_linear = with_linear
        ### RNN module
        if rnn_kind == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
        elif rnn_kind == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
        elif rnn_kind == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
        elif rnn_kind == 'Transformer':
            pass
        else:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()
        ### Linear layer
        if with_linear:
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, seq, seq_sele_idx, seq_length_array):
        """
        seq: seq_num x seq_len x aa_dim
        seq_sele_idx: [[batch_idx,...],[chain_idx,...]]
        seq_length_array: list or tensor of sequence length, length = seq_num
        """
        batch_size, seq_num_batch, seq_len, aa_dim = seq.shape
        seq = seq[seq_sele_idx]
        seq_num = seq.shape[0]
        seq_length_array = [i for i in seq_length_array if i > 0]
        # whether avoid padding
        if self.avoid_padding:
            seq = pack_padded_sequence(seq, seq_length_array, batch_first=True, enforce_sorted=False)
            output_pack, hidden = self.rnn(seq)
            output_pack, _ = pad_packed_sequence(output_pack, batch_first = True)
            if self.bidirectional:
                output_pack = output_pack[:,:,:self.hidden_size] + output_pack[:,:,self.hidden_size:] # Sum bidirectional outputs
            output = torch.zeros([seq_num, seq_len, self.hidden_size])
            if self.USE_CUDA:
                output = output.cuda()
            max_seq_len = output_pack.shape[1]
            output[:,:max_seq_len,:] = output_pack
        else:
            output, hidden = self.rnn(seq)
            # whether bidirectional
            if self.bidirectional:
                output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:] # Sum bidirectional outputs
        if self.with_linear:
            output = output.reshape(-1, self.hidden_size)
            output = self.linear(output)

        output_new = torch.zeros(batch_size, seq_num_batch, seq_len, output.shape[-1])
        if self.USE_CUDA:
            output_new = output_new.cuda()
        output_new[seq_sele_idx] = output
        return output_new

###########################################################
# Generative Module
###########################################################

class Decoder(nn.Module):
    """
    Predict the residue given the node embedding and the sequence embedding.
    """
    def __init__(self, input_size = 100, output_size = 20, # for linear layer
                 num_heads = 4, dropout = 0.1, kdim = 512, vdim = 512,
                 USE_CUDA = True):
        super(Decoder, self).__init__()

        ### architecture ###
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.kdim = kdim
        self.vdim = vdim
        self.USE_CUDA = USE_CUDA

        ### Linear layer
        self.linear = nn.Linear(input_size, output_size)
        ### attention layer
        self.attention = nn.MultiheadAttention(embed_dim = input_size, num_heads = num_heads, dropout = dropout, batch_first = True,
                                               kdim = kdim, vdim = vdim)
        self.linear_attn = nn.Linear(input_size, output_size)
        ### softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node_emb, seq_emb, mask, key_padding_mask=None, attn_mask=None, SAMPLE = 'multinomial', k = None):
        """
        node_emb: batch_size x gnn_dim (input size)
        seq_emb: batch_size x seq_len x rnn_dim
        mask: batch_size x 1
        """
        mask = mask.reshape(-1,1)
        if self.USE_CUDA:
            mask = mask.cuda()
        pred = self.linear(node_emb).squeeze() # batch_size x out_dim

        if len(attn_mask.shape) >= 3:
            batch_size, len_1, len_2 = attn_mask.shape
            attn_mask = attn_mask.repeat(1, self.num_heads, 1).reshape(-1, len_1, len_2)

        H, A = self.attention(node_emb, seq_emb, seq_emb, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        H = self.linear_attn(H.squeeze()) # batch_size x out_dim
        pred = pred + H
        pred = pred.squeeze()
        aa_onehot = self.sample(pred) # batch_size x out_dim
        pred = pred * mask
        aa_onehot = aa_onehot * mask
        aa = onehot_to_char(aa_onehot)
        return aa, aa_onehot, pred

    def sample(self, output, SAMPLE = 'multinomial', k = None):
        """
        SAMPLE: top-k, max, multinomial
        """
        if torch.isnan(output).any():
            print('NaN exist!', output)
            output[torch.isnan(output)] = 1
        if SAMPLE == 'max' or k == 1:  # Sample top value only
            top_i = output.data.topk(1)[1]
        else:
            output = self.softmax(output)
            if SAMPLE == 'multinomial' or k is None or k >= self.output_size:
                top_i = torch.multinomial(output, 1)
            elif SAMPLE == 'top-k':  # top-k sampling
                top_v, top_k = output.data.topk(k)
                top_v_sele = torch.multinomial(top_v, 1)
                top_i = torch.gather(top_k, -1, top_v_sele)
            else:
                print('Error! No sampling method named %s!'%SAMPLE)
                return None
        ### construct the one-hot encoding sequence
        next_resi = torch.zeros(output.shape[0], self.output_size)
        if self.USE_CUDA:
            next_resi = next_resi.cuda()
        next_resi = next_resi.scatter_(1,top_i,1)
        return next_resi

###########################################################
# Final Model
###########################################################

class CDR_Generator(nn.Module):
    """
    Encoder (can be RNN, GRU, LSTM or Transformer) for the antigen and antibody sequences.
    """
    def __init__(self, gen_version = 'CDR-parallel', with_ag_seq = False, USE_CUDA = True, cdr_num = 6, avoid_padding = True,
                 ### GNN ###
                 gnn_kind = 'GCN', feat_dim = 4, gnn_hidden_size = 512, gnn_emb_size = 20, gnn_layers = 3,
                 concat=False, bn=True, dropout_gnn=0.1, bias = True, channel_num=11,
                 ### RNN ###
                 rnn_kind = 'LSTM', input_size = 22, rnn_hidden_size = 512, emb_size = 64, #encode_size = 32,
                 rnn_layers=3, dropout_rnn=0.1, bidirectional=True,
                 ### Decoder ###
                 output_size = 20, num_heads = 4, dropout_decode = 0.1):
        super(CDR_Generator, self).__init__()
        """
        gen_version: CDR-parallel, HL-parallel, Sequential
        with_ag_seq: whether consider the antigen sequences
        """
        ### versions ###
        self.gen_version = gen_version
        self.with_ag_seq = with_ag_seq
        self.rnn_kind = rnn_kind
        self.gnn_kind = gnn_kind
        self.USE_CUDA = USE_CUDA
        self.cdr_num = cdr_num # maximum # of CDRs to be designed
        self.avoid_padding = avoid_padding
        ### architecture ###
        ## GNN
        self.feat_dim = feat_dim
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_emb_size = gnn_emb_size
        self.gnn_layers = gnn_layers
        self.dropout_gnn = dropout_gnn
        self.channel_num = channel_num
        self.concat = concat
        self.bn = bn
        self.bias = bias
        ## embedding layer
        self.input_size = input_size
        self.emb_size = emb_size
        ## RNN
        self.rnn_hidden_size = rnn_hidden_size
        self.emb_size = emb_size
        #self.encode_size = encode_size
        self.rnn_layers = rnn_layers
        self.dropout_rnn = dropout_rnn
        self.bidirectional = bidirectional
        ## Decoder
        self.output_size = output_size
        self.num_heads = num_heads
        self.dropout_decode = dropout_decode

        ### embedding layer
        if self.avoid_padding:
            self.embed = nn.Embedding(input_size, emb_size, padding_idx = -1)
        else:
            self.embed = nn.Embedding(input_size, emb_size)
        ### GNN module
        self.gnn = EncoderGNN(feature_dim = feat_dim, hidden_dim = gnn_hidden_size, seq_dim = emb_size,
                              embedding_dim = gnn_emb_size, num_layers = gnn_layers,
                              gnn_kind = 'GCN', concat = concat, bn = bn,
                              dropout = dropout_gnn, bias = bias, channel_num = channel_num, USE_CUDA = USE_CUDA)
        self.gnn_out_dim = self.gnn.gnn_out_dim
        ### RNN module
        self.seq_encoder = EncoderRNN(input_size = emb_size, hidden_size = rnn_hidden_size, with_linear = False, output_size = None,
                                      n_layers = rnn_layers, dropout_p = dropout_rnn,
                                      rnn_kind = rnn_kind, bidirectional = bidirectional, USE_CUDA = USE_CUDA, avoid_padding = avoid_padding)
        ### Decoder
        self.decoder = Decoder(input_size = self.gnn_out_dim, output_size = output_size,
                               num_heads = num_heads, dropout = dropout_decode, kdim = rnn_hidden_size, vdim = rnn_hidden_size,
                               USE_CUDA = USE_CUDA)


    def forward(self, seq_ag, seq_ab, feat, adj,
                graph_mask, seq_sele_ag, ag_indexes, seq_sele_ab, ab_indexes, cdr_mask, seq_len_ab, seq_len_ag = None, step = None):
        """
        Inputs:
            seq_ag (antigen sequence): if with_ag_seq, complete antigen sequences, batch_size x max_chain_num_ag x max_seq_len_ag x aa_dim (22)
                                       else, sum of ag_node_num x emb_dim
            seq_ab (masked antibody sequence (HL sequences or CDR sequences)): batch_size x 2 x max_seq_len_ab x aa_dim
                    if consider NonCDR regions, HL sequences;
                    else, masked HL sequences.
                    Paddings are represented as [0, 0, ..., 0, 1], masks are represented as [0, 0, ..., 1, 0].
            feat: node feature matrix, batch_size x channel_num x max_node_num x feat_dim
            adj: adjacency tensor, batch_size x channel_num x max_node_num x max_node_num
        For shape:
            graph_mask: tensor of the graph mask, negative for ag, positive for ab, 0 for padding, batch_size x max_node_num x 1
            ag_indexes: list of epitope index, [[batch_idx,...], [chain_idx,...], [resi_idx,...]], 3 x # of residues in the batch
            ab_indexes: list of cdr index, similar to ag_indexes
            cdr_mask: tensor of bullin points, batch_size x 6 x max_cdr_len x 1
            For sequence embedding:
                seq_len_ag: tensor of sequence length of antigens, batch_size x max_chain_num_ag (0 for unexisting chains)
                seq_len_ab: tensor of chain amounts of antibodies, batch_size x 2
        """
        ### shape
        batch_size, HL_num, max_seq_len_ab, aa_dim = seq_ab.shape
        max_chain_num_ag = seq_ag.shape[1]
        #max_seq_len_ag = seq_ag.shape[2]
        max_node_num = feat.shape[-2]
        ab_seq_num = batch_size * HL_num
        seq_cdr_num = int(self.cdr_num / HL_num)
        tar_num = batch_size * self.cdr_num
        max_cdr_len = cdr_mask.shape[-2]
        if step is None:
            step = max_cdr_len
        ### feat_mat
        feat = feat.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
        ### masks
        graph_mask_emb = graph_mask.repeat(1,1,self.emb_size)
        graph_mask = graph_mask.repeat(1,1,self.gnn_out_dim)
        key_padding_mask = torch.ones(batch_size, HL_num, max_seq_len_ab)
        for i,len_ab_batch in enumerate(seq_len_ab):
            for j,len_ab_chain in enumerate(len_ab_batch):
                key_padding_mask[i,j,:int(len_ab_chain)] = 0
        key_padding_mask = key_padding_mask == 0
        key_padding_mask = key_padding_mask.repeat(1,1,seq_cdr_num).reshape(tar_num,-1) # mask: tar_num x max_seq_len_ab
        attn_mask = key_padding_mask.unsqueeze(-2) # mask: tar_num x 1 x max_seq_len_ab
        if self.USE_CUDA:
            key_padding_mask = key_padding_mask.cuda()
            attn_mask = attn_mask.cuda()
        ### antigen embedding (seq embedding)
        emb_ag = self.embed(seq_ag.max(dim = -1)[1])
        if self.with_ag_seq:
            seq_feat_ag = emb_ag[ag_indexes] # sum of ag_node_num x emb_dim
            #seq_ag_emb = self.seq_encoder(emb_ag.reshape(-1, max_seq_len_ag, self.emb_size),
            seq_ag_emb = self.seq_encoder(emb_ag, seq_sele_ag, seq_len_ag.reshape(-1)) # batch_size x max_chain_num x max_seq_len_ag x rnn_dim
        else:
            seq_feat_ag = emb_ag # sum of ag_node_num x emb_dim
        ### outputs initialization
        profile = torch.zeros(batch_size, self.cdr_num, max_cdr_len, self.output_size)
                  # batch_size x 6 (max_cdr_num) x max_cdr_len x 20 (output dim)
        cdr_onehot = torch.zeros(batch_size, self.cdr_num, max_cdr_len, aa_dim)
                  # batch_size x 6 (max_cdr_num) x max_cdr_len x 22 (aa dim)
        if self.USE_CUDA:
            profile = profile.cuda()
            cdr_onehot = cdr_onehot.cuda()
        cdr_onehot[:,:,:,-2] = 1
        cdrs = [] # list of cdr sequences: batch_size x max_cdr_num (for the untargeted cdr, the generated sequence is empty)
        for b in range(batch_size):
            cdrs.append(['','','','','',''])

        ### iterative generation
        for idx in range(step):
            cdr_mask_temp = cdr_mask[:,:,idx] # (batch_size x 6 (max_cdr_num)) x 1: mask the cdrs that were already completed
            if not cdr_mask_temp.any():
                continue

            if self.USE_CUDA:
                cdr_mask_temp = cdr_mask_temp.cuda()
            ### antibody embedding
            emb_ab = self.embed(seq_ab.max(dim = -1)[1]) # batch_size x 2 x max_seq_len_ab x emb_dim
            seq_feat_ab = emb_ab[ab_indexes] # sum of ab_node_num x emb_dim
            seq_ab_emb = self.seq_encoder(emb_ab, seq_sele_ab, seq_len_ab.reshape(-1)) # batch_size x 2 x max_seq_len_ab x rnn_dim
            seq_ab_emb = seq_ab_emb.repeat(1,1,seq_cdr_num,1).reshape(tar_num, max_seq_len_ab, -1)
                         # tar_num (batch_size*max_cdr_num) x max_seq_len_ab x rnn_dim

            ### graph embedding
            seq_feat = torch.zeros(batch_size, max_node_num, self.emb_size)
            if self.USE_CUDA:
                seq_feat = seq_feat.cuda()
            seq_feat[graph_mask_emb < 0] = seq_feat_ag.reshape(-1)
            seq_feat[graph_mask_emb > 0] = seq_feat_ab.reshape(-1)
            seq_feat = seq_feat.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
            gnn_emb = self.gnn(feat, seq_feat, adj)  # batch_size x max_node_num x gnn_dim

            ### select the target nodes
            tar_node_emb = torch.zeros(tar_num, 1, self.gnn_out_dim) # tar_num x 1 x gnn_dim
            if self.USE_CUDA:
                tar_node_emb = tar_node_emb.cuda()
            tar_node_emb[cdr_mask_temp.reshape(-1)] = gnn_emb[graph_mask == idx+1].reshape(-1, 1, self.gnn_out_dim)

            ### predict the amino acid and make the sequence
            aa, aa_onehot, prob = self.decoder(tar_node_emb,  # tar_num x 1 x gnn_dim
                                               seq_ab_emb,    # tar_num x max_seq_len_ab x rnn_dim
                                               cdr_mask_temp,      # batch_size x 6 (max_cdr_num) x 1
                                               key_padding_mask = key_padding_mask * cdr_mask_temp.reshape(-1,1),
                                               attn_mask = attn_mask * cdr_mask_temp.reshape(-1,1,1))
            # aa: list of AA with length of tar_num (batch_size*max_cdr_num)
            # aa_one_hot: tar_num x output_size
            # prob: probability, tar_num x output_size

            ### make up the masked positions
            profile[:,:,idx,:] = prob.reshape(batch_size, self.cdr_num, -1)
            cdr_onehot[:,:,idx,-2] = 0
            cdr_onehot[:,:,idx,:self.output_size] = aa_onehot.reshape(batch_size, self.cdr_num, -1)
            seq_ab[ab_indexes] = cdr_onehot[cdr_mask.repeat(1,1,1,aa_dim)].reshape(-1, aa_dim)
            cdrs = self.cdr_makeup(cdrs, aa)
        return profile, cdrs, seq_ab

    def cdr_makeup(self, cdrs, aa):
        idx = 0
        for i,batch in enumerate(cdrs):
            for j,seq in enumerate(batch):
                cdrs[i][j] += aa[idx]
                idx += 1
        return cdrs

    def loss(self, profile, target, cdr_mask):
        """
        profile: output before the softmax layer, batch_size x 6 (max_cdr_num) x max_cdr_len x 20 (output dim)
        target: groudtruth, batch_size x 6 (max_cdr_num) x max_cdr_len
        cdr_mask: tensor of bullin points, batch_size x 6 x max_cdr_len x 1
        """
        profile = profile[cdr_mask.repeat(1,1,1,self.output_size)].reshape(-1,self.output_size)
        target = target[cdr_mask.squeeze()].reshape(-1)
        loss = F.cross_entropy(profile, target)
        return loss
