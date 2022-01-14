import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np
import random
import copy

######################## Accessory Functions ##############################

def onehot_to_char(H):
    """
    transform the tensor into string
    H: onehot encoding matrix
    """
    AA_dict = {0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'Q', 6:'E', 7:'G', 8:'H', 9:'I', 10:'L',
               11:'K', 12:'M', 13:'F', 14:'P', 15:'S', 16:'T', 17:'W', 18:'Y', 19:'V'}
    max_val, idx = torch.max(H, dim = -1)
    s = []
    for i,key in enumerate(idx):
        if max_val[i] < 1:
            s.append('')
        else:
            s.append(AA_dict[int(key)])
    return s

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

def tensor_to_string_2(t, index = False, MAX_SAMPLE = 'top-k', k = 3):
    """
    transform the tensor into string
    sampling method: top-k, max, multinomial
    dimension of t: seq_num x seq_len x aa_dim(21) 
    """

    AA_dict = {0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'Q', 6:'E', 7:'G', 8:'H', 9:'I', 10:'L',
               11:'K', 12:'M', 13:'F', 14:'P', 15:'S', 16:'T', 17:'W', 18:'Y', 19:'V', 20:'!'}
    if not index:
        if MAX_SAMPLE == 'max' or k == 1:
            t = t.max(dim=-1)[1]
        else:
            if k >= 21 or MAX_SAMPLE == 'multinomial':
                t = [torch.multinomial(batch, 1).reshape(-1) for batch in t]
            elif MAX_SAMPLE == 'top-k':  # top-k sampling
                top_v, top_k = t.data.topk(k)
                top_v_sele = [torch.multinomial(batch, 1) for batch in top_v]
                t = [torch.gather(top_k[i], -1, batch).reshape(-1) for i, batch in enumerate(top_v_sele)]
            else:
                print('Error! No sampling method named %s!'%MAX_SAMPLE)
                return None
    s = []
    for batch in t:
        seq = ''
        for i in batch:
            seq += AA_dict[int(i)]
        s.append(seq)
    return s


def seq_final(tensor, node_num, MAX_SAMPLE = 'top-k', k = 3):
    batch_size = node_num.shape[0]
    #print(batch_size)
    seq = []
    seq_complete = []
    start = 0
    for i in range(batch_size):
        end = start + node_num[i]
        seq_tensor = tensor[start:end]
        start = end
        seq_piece = tensor_to_string_2(seq_tensor, index = False, MAX_SAMPLE = MAX_SAMPLE, k = k)
        seq.append(seq_piece)
        seq_all = ''.join([s.split('!')[0] for s in seq_piece])
        seq_complete.append(seq_all)
    return seq, seq_complete


###########################################################
# GNN Module
###########################################################

####################### GCN ###############################

# GCN basic operation
class GraphConv(nn.Module):
    """
    Graph Convolutional Layer.
    """
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, 
            dropout=0.0, bias=True, share_weight = False, channel_num = 1, CUDA = False):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        #self.hetero = hetero
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.share_weight = share_weight # whether the weight of different channels are the same
        if share_weight: 
            self.weight = torch.FloatTensor(input_dim, output_dim)
            if CUDA:
                self.weight = self.weight.cuda()
            self.weight = nn.Parameter(self.weight)
            if bias:
                self.bias = torch.FloatTensor(output_dim)
                if CUDA:
                    self.bias = self.bias.cuda()
                self.bias = nn.Parameter(self.bias)
            else:
                self.bias = None
        else:
            self.weight = torch.FloatTensor(channel_num, input_dim, output_dim)
            if CUDA:
                self.weight = self.weight.cuda()
            self.weight = nn.Parameter(self.weight)
            if bias:
                self.bias = torch.FloatTensor(channel_num, 1, output_dim)
                if CUDA:
                    self.bias = self.bias.cuda()
                self.bias = nn.Parameter(self.bias)
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

#**********************************************************

class GraphConvolNetwork(nn.Module):
    """
    Node-wise graph embedding module.
    """
    def __init__(self, feature_dim, hidden_dim, embedding_dim, num_layers,
                 concat=False, bn=True, dropout=0.0, bias = True, channel_num=1):
        super(GraphConvolNetwork, self).__init__()

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

        self.hidden_out_dim = hidden_dim * channel_num
        if concat:
            self.out_dim = (hidden_dim * (num_layers - 1) + embedding_dim) * channel_num
        else:
            self.out_dim = channel_num * embedding_dim

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

####################### GNN Container ###############################

class GNN_Container(nn.Module):
    def __init__(self, label_num = 1080, pred_hidden_dims=[4000], pooling='max', act=nn.ReLU(), CUDA = False, # for prediction 
                 ### for GNN
                 model='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, num_layers=3, channel_num=5,
                 concat=False, bn=True, dropout=0.0, bias=True,
                 ### for RNN (if consider the sequence)
                 seq_embed=False, rnn_model='LSTM', rnn_bias=True, rnn_hidden_size=64, rnn_num_layers=3,
                 seq_cat=True, seq_GNN=False, bidirectional=True, rnn_dropout=0.01):
        #************************************************************************************************
        super(GNN_Container, self).__init__()

        # label_num: number of classes
        # pred_hidden_dims: dimension of the hidden vectors
        # seq_cat: whether concatenate the node embedding to the amino acid for RNN
        # seq_GNN: whethe GNN to embed the sequence 

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

        if type(model) == str:
            if model == 'GCN':
                self.GNN = GraphConvolNetwork(feature_dim = self.gnn_in_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim,
                                              num_layers = num_layers, concat=concat, bn = bn, dropout=dropout, bias=bias, channel_num=channel_num)
            else:
                print('Error! No GNN model named %s!'%model)
                return None
        else:
            self.GNN = model

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

        if self.pooling is not None:
            if self.pooling == 'max':
                node_vec,_ = torch.max(node_vec, dim=-2)
            elif self.pooling == 'sum':
                node_vec = torch.sum(node_vec, dim=-2)
            elif self.pooling == 'mean':
                node_vec = torch.mean(node_vec, dim=-2)
            else:
                print('Error! No pooling method named %s!'%self.pooling)
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

            return cross_entropy_1 + cross_entropy_2 * lambdas[0] + cross_entropy_3 * lambdas[1] + cross_entropy_4 * lambdas[2]

###########################################################
# Generative Model
###########################################################

####################### Encoder ###########################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size = 11, seq_cat = True, skip_connect = False,
                 n_layers=1, dropout_p=0.1, bidirectional=True, language_model = None, rnn_kind = 'LSTM', USE_CUDA = True):
        super(EncoderRNN, self).__init__()

        ### versions ###
        self.seq_cat = seq_cat  # whether cat the sequence for seq embed
        self.bidirectional = bidirectional
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        self.skip_connect = skip_connect
        ### architecture ###
        self.input_size = input_size
        self.condition_size = condition_size

        if seq_cat:
            self.rnn_input_size = input_size + condition_size
        else:
            self.rnn_input_size = input_size

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        if skip_connect:
            self.latent_size = hidden_size
        else:
            self.latent_size = condition_size + hidden_size

        if language_model is not None:
            self.rnn = language_model
        else:
            if rnn_kind == 'GRU':
                self.rnn = nn.GRU(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            elif rnn_kind == 'LSTM':
                self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            elif rnn_kind == 'RNN':
                self.rnn = nn.RNN(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            else:
                print('Error! No RNN module named "%s"!'%rnn_kind)
                quit()
        self.o2p = nn.Linear(self.latent_size, output_size * 2)

    def forward(self, seq, condition):
        batch_size, seq_len, aa_dim = seq.shape
        if self.seq_cat: # cat the condition for each residue
           cond_repeat = condition.repeat(1,1,seq_len).reshape(batch_size, seq_len, self.condition_size)
           seq = torch.cat((cond_repeat, seq),-1)

        output, hidden = self.rnn(seq)
        output = output[:,-1,:] # Take only the last value
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[: ,self.hidden_size:] # Sum bidirectional outputs

        if not self.skip_connect:
            output = torch.cat([output, condition], dim = -1)

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)  # mean and log-variance
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if self.USE_CUDA:
            eps = eps#.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

####################### Decoder ###########################

class DecoderRNN(nn.Module):
    """
    Sequence generator for the next residue
    """
    def __init__(self, input_size, hidden_size, output_size=21, feedback_size=11, skip_connect=False, cond_dim_skip=None,
                 n_layers=3, dropout_p=0.1, rnn_kind = 'LSTM', USE_CUDA=True):
        super(DecoderRNN, self).__init__()

        ### versions ###
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        if not rnn_kind in ['GRU','LSTM','RNN']:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()
        self.skip_connect = skip_connect
        ### architecture ###
        self.input_size = input_size
        self.rnn_input_size = input_size
        self.feedback_size = feedback_size
        self.hidden_size = hidden_size
        self.output_size = output_size # dimension of the residue vector

        if skip_connect:
            #self.rnn_input_size = input_size + output_size
            self.skip_dim = cond_dim_skip + hidden_size
            self.n_layers = 3 # skip connection only support for 3 layers
            if rnn_kind == 'GRU':
                self.rnn_1 = nn.GRU(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.GRU(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.GRU(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
            elif rnn_kind == 'LSTM':
                self.rnn_1 = nn.LSTM(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.LSTM(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.LSTM(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
            elif rnn_kind == 'RNN':
                self.rnn_1 = nn.RNN(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.RNN(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.RNN(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)

            #self.out = nn.Linear(hidden_size + input_size, output_size) # for the output of each step

        else:
            #self.rnn_input_size = input_size + condition_size + output_size
            self.n_layers = n_layers
            if rnn_kind == 'GRU':
                self.rnn = nn.GRU(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)
            elif rnn_kind == 'LSTM':
                self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)
            elif rnn_kind == 'RNN':
                self.rnn = nn.RNN(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)

        self.out = nn.Linear(hidden_size + input_size - feedback_size, output_size) # for the output of each step

    def forward(self, z, resi_info, hidden, condi_1 = None, condi_2 = None):
        """
        z: noise       hidden: previous hidden state
        cond_1, cond_2: for skip connection
        """
        inputs = torch.cat((z.unsqueeze(1), resi_info), -1)
        if self.skip_connect:
            condi_dim = condi_1.shape[-1]
            output_1, hidden_1 = self.rnn_1(inputs, hidden[0])
            output_1 = torch.cat((output_1, condi_2.reshape(-1,1,condi_dim)), -1)
            output_2, hidden_2 = self.rnn_2(output_1, hidden[1])
            output_2 = torch.cat((output_2, condi_1.reshape(-1,1,condi_dim)), -1)
            output, hidden_3 = self.rnn_3(output_2, hidden[2])
            output = output.squeeze(1)
            output = torch.cat((output, z), -1)
            hidden = [hidden_1, hidden_2, hidden_3]
        else:
            output, hidden = self.rnn(inputs, hidden)
            output = output[:,-1,:]  # only consider the final output
            output = torch.cat((z,output), -1)

        output = self.out(output)
        return output, hidden

#**********************************************************

class Sequential_Generator(nn.Module):
    """
    Autoregressive model for sequence generation on the complete element.
    """
    def __init__(self, input_size, hidden_size = 64, skip_connect=False, cond_dim_skip=None, USE_CUDA=False, teacher_forcing = True,
                 ### for RNN
                 rnn_kind = 'LSTM', output_size=21, condition_size=11, n_layers=3, dropout_p=0.1,
                 ### for GNN
                 gnn_embed = True, gnn_kind = 'GCN', gnn_layer_num = 2, gnn_hidden_dim = 64, gnn_emb_dim = 10, channel_num = 5):
        super(Sequential_Generator, self).__init__()

        ### version ###
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        if not rnn_kind in ['GRU','LSTM','RNN']:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()
        self.skip_connect = skip_connect
        ### architecture ###
        self.input_size = input_size # size of the input noise
        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.output_size = output_size # dimension of the residue vector
        self.rnn_kind = rnn_kind

        if skip_connect:
            self.z2h_dim = input_size
        else:
            self.z2h_dim = input_size + condition_size

        if rnn_kind == 'LSTM':
            self.z2h = nn.Linear(self.z2h_dim, hidden_size * 2) # for initial hidden status
        else:
            self.z2h = nn.Linear(self.z2h_dim, hidden_size) # for initial hidden status

        self.gnn_embed = gnn_embed
        self.teacher_forcing = teacher_forcing
        self.embed = nn.Embedding(output_size + 1, hidden_size)  # embedding of the residues (20 aa + padding + initial)

        if gnn_embed:
            self.channel_num = channel_num
            if gnn_kind == 'GCN':
                #self.gnn_in_dim = self.output_size + self.condition_size
                self.gnn_in_dim = hidden_size
                self.GNN = GraphConvolNetwork(feature_dim = self.gnn_in_dim, hidden_dim = gnn_hidden_dim, embedding_dim = gnn_emb_dim,
                                              num_layers = gnn_layer_num, channel_num=channel_num)
            else:
                print('Error! No GNN model named %s!'%gnn_kind)
                quit()
            self.resi_info_dim = self.GNN.out_dim
        else:
            self.resi_info_dim = hidden_size

        if skip_connect:
            self.rnn_input_size = self.resi_info_dim + input_size
        else:
            self.rnn_input_size = self.resi_info_dim + input_size + condition_size

        self.n_layers = n_layers
        self.step_decoder = DecoderRNN(input_size=self.rnn_input_size, hidden_size=hidden_size,
                                       output_size=output_size, feedback_size=self.resi_info_dim,
                                       skip_connect=skip_connect, cond_dim_skip=cond_dim_skip, n_layers=n_layers, dropout_p=dropout_p,
                                       rnn_kind = rnn_kind, USE_CUDA=USE_CUDA)

    def generate(self, condition, mask, Adj = None, n_steps = 35, temperature = 1, condi_1 = None, condi_2 = None, MAX_SAMPLE = 'top-k', k = 3):
        batch_size, max_node_num, cond_dim = condition.shape
        condition = condition.reshape(-1, cond_dim)
        num_seq = condition.shape[0]
        z = Variable(torch.randn(num_seq, self.input_size))
        if self.USE_CUDA:
            z = z.cuda()

        if not self.skip_connect:
            z = torch.cat((z,condition),-1) # the new noise contain the condition

        outputs = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # empty output tensor
        Indexes = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # for one-hot encoding sequence
        ### initial input ###
        resi_info = Variable(self.output_size * torch.ones(num_seq, 1)) # initial input
        hidden = self.z2h(z).unsqueeze(1).repeat(1, self.n_layers, 1).transpose(0,1).contiguous()  # initial hidden status
        if self.USE_CUDA:
            outputs = outputs.cuda()
            resi_info = resi_info.cuda()
            hidden = hidden.cuda()
            Indexes = Indexes.cuda()
        ### initial embedding
        resi_info = self.embed(resi_info.long())
        resi_info = F.relu(resi_info)
        if self.gnn_embed:
            x = resi_info.reshape(batch_size, max_node_num, -1)
            x = x * mask
            x = x.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
            resi_info = self.GNN(x, Adj)
        resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        if self.rnn_kind == 'LSTM':
            if self.skip_connect:
                hidden = [tuple([hi.unsqueeze(0).contiguous() for hi in torch.chunk(hid, 2, dim=-1)]) for hid in hidden]
            else:
                hidden = torch.chunk(hidden, 2, dim=-1)
                hidden = tuple([hid.contiguous() for hid in hidden])
        ### generate the outputs step by step ###
        for i in range(n_steps):
            output, hidden = self.step_decoder(z, resi_info, hidden, condi_1, condi_2)
            outputs[:,i,:] = output
            next_resi, top_i = self.sample(output, num_seq, temperature, MAX_SAMPLE = MAX_SAMPLE, k = k)
            Indexes[:,i,:] = next_resi # one-hot encoding sequence

            resi_info = self.embed(next_resi.max(dim = -1)[1])
            resi_info = F.relu(resi_info)
            if self.gnn_embed:
                x = resi_info.reshape(batch_size, max_node_num, -1)
                x = x * mask
                x = x.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
                resi_info = self.GNN(x, Adj)
            resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        outputs = outputs.reshape(batch_size, max_node_num, n_steps, self.output_size) * mask.unsqueeze(-1)
        Indexes = Indexes.reshape(batch_size, max_node_num, n_steps, self.output_size) * mask.unsqueeze(-1)
        return outputs, Indexes

    def forward_vae(self, z, condition, mask, node_num, Adj = None, n_steps = 35, temperature = 1, condi_1 = None, condi_2 = None,
                    ground_truth = None, MAX_SAMPLE = 'top-k', k = 3):
        batch_size, max_node_num, cond_dim = condition.shape
        ### change the shape for separate elements
        condition = condition.reshape(-1, cond_dim)
        num_seq = condition.shape[0]
        if ground_truth is not None:
            ground_truth = ground_truth.reshape(num_seq, -1, self.output_size).long()
        z = z.reshape(num_seq, -1)
        ### initial preparation
        if not self.skip_connect:
            z = torch.cat((z,condition),-1) # the new noise contain the condition
        outputs = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # empty output tensor
        resi_info = Variable(self.output_size * torch.ones(num_seq, 1)) # initial input
        hidden = self.z2h(z).unsqueeze(1).repeat(1, self.n_layers, 1).transpose(0,1).contiguous()  # initial hidden status
        if self.USE_CUDA:
            outputs = outputs.cuda()
            resi_info = resi_info.cuda()
            hidden = hidden.cuda()
        ### initial embedding
        resi_info = self.embed(resi_info.long())
        resi_info = F.relu(resi_info)
        if self.gnn_embed:
            x = resi_info.reshape(batch_size, max_node_num, -1)
            x = x * mask
            x = x.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
            resi_info = self.GNN(x, Adj)
        resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        if self.rnn_kind == 'LSTM':
            if self.skip_connect:
                hidden = [tuple([hi.unsqueeze(0).contiguous() for hi in torch.chunk(hid, 2, dim=-1)]) for hid in hidden]
            else:
                hidden = torch.chunk(hidden, 2, dim=-1)
                hidden = tuple([hid.contiguous() for hid in hidden])

        ### generate the outputs step by step ###
        for i in range(n_steps):
            output, hidden = self.step_decoder(z, resi_info, hidden, condi_1, condi_2)
            outputs[:,i,:] = output
            next_resi, top_i = self.sample(output, num_seq, temperature, MAX_SAMPLE = MAX_SAMPLE, k = k)

            if self.teacher_forcing and (random.random() < temperature):
                resi_info = self.embed(ground_truth[:,i,:].max(dim = -1)[1])
            else:
                resi_info = self.embed(next_resi.max(dim = -1)[1])
            resi_info = F.relu(resi_info)

            if self.gnn_embed:
                #x = torch.cat((output, condition), -1).reshape(batch_size, max_node_num, -1)
                x = resi_info.reshape(batch_size, max_node_num, -1)
                x = x * mask
                x = x.repeat(1, self.channel_num, 1).reshape(batch_size, self.channel_num, max_node_num, -1)
                resi_info = self.GNN(x, Adj)
            resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        outputs = outputs.reshape(batch_size, max_node_num, n_steps, self.output_size) *  mask.unsqueeze(-1)
        outputs = torch.cat([outputs[i][:node_num[i]] for i in range(batch_size)])

        return outputs

    def sample(self, output, num_seq, temperature, MAX_SAMPLE = 'top-k', k = 3):
        """
        sampling method: top-k, max, multinomial 
        """
        if MAX_SAMPLE == 'max' or k == 1:  # Sample top value only
            top_i = output.data.topk(1)[1]
        else:
            output_dist = output.data.view(num_seq, -1).div(temperature)
            output_dist = F.softmax(output_dist, dim = -1)
            if k >= self.output_size or MAX_SAMPLE == 'multinomial':
                top_i = torch.multinomial(output_dist, 1)
            elif MAX_SAMPLE == 'top-k':  # top-k sampling
                top_v, top_k = output_dist.data.topk(k)
                top_v_sele = torch.multinomial(top_v, 1)
                top_i = torch.gather(top_k, -1, top_v_sele)
            else:
                print('Error! No sampling method named %s!'%MAX_SAMPLE)
                return None

        ### construct the one-hot encoding sequence
        next_resi = torch.zeros(num_seq, self.output_size)
        if self.USE_CUDA:
            next_resi = next_resi.cuda()
        next_resi = next_resi.scatter_(1,top_i,1)
        return next_resi, top_i

###################### Container ##########################

class c_text_VAE(nn.Module):
    def __init__(self,input_size=21,encoder_hidden_size=64, decoder_hidden_size=64, encoder_output_size=32, condition_size=11, cond_dim_skip=None,
                 seq_cat=True, n_encoder_layers=3, n_decoder_layers=3, dropout_p=0.1, bidirectional=True, language_model = None,rnn_kind = 'LSTM',
                 channel_num = 5, USE_CUDA = True, skip_connection = False, gnn_embed = True, teacher_forcing = True):
        super(c_text_VAE, self).__init__()
        """
        input_size: 20 + 1 (AA & padding)
        output_size: dimension of z
        """
        self.USE_CUDA = USE_CUDA
        self.encoder_output_size = encoder_output_size
        self.encoder = EncoderRNN(input_size, encoder_hidden_size, encoder_output_size, condition_size, seq_cat, skip_connection,
                                  n_encoder_layers, dropout_p, bidirectional, language_model, rnn_kind, USE_CUDA)

        self.decoder = Sequential_Generator(encoder_output_size, hidden_size = decoder_hidden_size, skip_connect=skip_connection,
                                            cond_dim_skip=cond_dim_skip,
                                            USE_CUDA=USE_CUDA, condition_size=condition_size, teacher_forcing = teacher_forcing,
                                            rnn_kind = rnn_kind, output_size=21, n_layers=n_decoder_layers, dropout_p=dropout_p, # for RNN
                                            gnn_embed = gnn_embed, channel_num = channel_num) # for GNN

    def forward(self, inputs_raw, condition, mask, node_num, Adj = None, n_steps = 35, temperature = 1.0, condi_1=None, condi_2=None,
                MAX_SAMPLE = 'top-k', k = 3):
        batch_size, max_node_num, seq_len, aa_dim = inputs_raw.shape
        ### for condition shape
        seq_encoder = torch.cat([inputs_raw[i][:node_num[i]] for i in range(batch_size)])
        cond_encoder = torch.cat([condition[i][:node_num[i]] for i in range(batch_size)])
        mu, logvar, z = self.encoder(seq_encoder, cond_encoder)
        ### for decoder
        z_new = torch.zeros(batch_size, max_node_num, self.encoder_output_size)
        if self.USE_CUDA:
            z_new = z_new.cuda()
        start = 0
        for i in range(batch_size):
            end = start + int(node_num[i])
            z_new[i,:node_num[i],:] = z[start:end]
            start = end
        decoded = self.decoder.forward_vae(z_new, condition, mask, node_num, Adj, n_steps, temperature, condi_1, condi_2, inputs_raw,
                                           MAX_SAMPLE = MAX_SAMPLE, k = k)
        return mu, logvar, z, decoded

    def generator(self, condition, mask, node_num, Adj = None, n_steps = 35, temperature = 1.0, condi_1 = None, condi_2 = None,
                  MAX_SAMPLE = 'top-k', k = 3):
        batch_size = condition.shape[0]
        decoded, Indexes = self.decoder.generate(condition, mask, Adj, n_steps, temperature, condi_1, condi_2, MAX_SAMPLE = MAX_SAMPLE, k = k)
        decoded = torch.cat([decoded[i][:node_num[i]] for i in range(batch_size)])
        Indexes = torch.cat([Indexes[i][:node_num[i]] for i in range(batch_size)])
        return decoded, Indexes

#**********************************************************

class VAE_Container(nn.Module):
    def __init__(self, CUDA = False, seq_len = 35, aa_dim = 21, version = 'text-VAE', padding_weight = 0.1,
                 ### for struc_GNN
                 gnn_model = None, gnn_kind='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, gnn_num_layers=3, gnn_out_dim = 100, channel_num=5,
                 concat=False, bn=True, dropout=0.0, bias=True,
                 ### for cVAE 
                 cvae_hidden_size=[512,256,128,16],
                 ### for ctext-VAE
                 seq_cat = True, skip_connection = False, gnn_embed = True, teacher_forcing = True, language_model = None, rnn_kind='LSTM'):
        #************************************************************************************************
        super(VAE_Container, self).__init__()

        self.CUDA = CUDA
        self.version = version
        self.skip_connect = skip_connection

        #### GNN for node embedding ###

        self.gnn_kind = gnn_kind
        self.channel_num = channel_num
        self.concat = concat

        if skip_connection and version == 'text-VAE':
            self.gnn_num_layers=3 # skip connection only support for 3 layers
        else:
            self.gnn_num_layers = gnn_num_layers

        if gnn_model is not None:
            self.GNN = gnn_model
            self.gnn_out_dim = gnn_out_dim
        else:
            if gnn_kind == 'GCN':
                self.GNN = GraphConvolNetwork(feature_dim = feature_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim,
                                              num_layers = self.gnn_num_layers, concat=concat, bn = bn, dropout=dropout, bias=bias, channel_num=channel_num)
            else:
                print('Error! No GNN model named %s!'%gnn_kind)
                quit()
            self.gnn_out_dim = self.GNN.out_dim

        self.cond_dim_skip = hidden_dim * channel_num

        ### Seq embedding ###

        self.seq_len = seq_len
        self.aa_dim = aa_dim

        if version == 'cVAE':
            self.seq_dim = self.seq_len * self.aa_dim
            self.cvae_hidden_size = cvae_hidden_size
            self.model = cVAE(self.seq_dim, self.gnn_out_dim, cvae_hidden_size)

        elif version == 'text-VAE':
            self.rnn_kind = rnn_kind
            self.seq_cat = seq_cat
            self.cond_dim_skip = hidden_dim * channel_num
            self.gnn_embed = gnn_embed
            self.teacher_forcing = teacher_forcing

            self.cond_dim_skip = hidden_dim * channel_num

            self.model = c_text_VAE(input_size=aa_dim, condition_size=self.gnn_out_dim, seq_cat=seq_cat, language_model = language_model,
                                    rnn_kind = rnn_kind, USE_CUDA = CUDA, skip_connection = skip_connection, cond_dim_skip = self.cond_dim_skip,
                                    gnn_embed = gnn_embed, teacher_forcing = teacher_forcing, channel_num=channel_num)

        ### for loss ###
        self.loss_weight = torch.ones(aa_dim)
        self.loss_weight[-1] = padding_weight # force to not generate paddings
        self.criterion = nn.CrossEntropyLoss()
        if CUDA:
            self.criterion = self.criterion.cuda()
        ### Initialize the weights ###
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, seq, node_num, mask=None, n_steps = 35, temperature = 1.0, MAX_SAMPLE = 'top-k', k = 3, **kwargs):
        batch_size, max_node_num, max_seq_len, resi_dim  = seq.shape
        ### structure embedding
        stru_cond = self.GNN(x, adj)
        ###
        if self.version == 'cVAE':
            ### remove padding nodes
            seq = seq.reshape(batch_size, max_node_num, self.seq_dim)
            seq = torch.cat([seq[i][:node_num[i]] for i in range(batch_size)])
            cond = torch.cat([stru_cond[i][:node_num[i]] for i in range(batch_size)])
            #***********************
            mu, sig, z, out = self.model(seq, cond)
            out = out.reshape(-1, self.seq_len, self.aa_dim)
        elif self.version == 'text-VAE':
            if self.skip_connect:
                cond_1 = torch.cat(list(cond_all[0].transpose(0,1)),dim=-1)
                cond_2 = torch.cat(list(cond_all[1].transpose(0,1)),dim=-1)
            else:
                cond_1 = None
                cond_2 = None
            mu, sig, z, out = self.model(seq, stru_cond, mask, node_num, adj, n_steps, temperature, condi_1 = cond_1, condi_2 = cond_2,
                                         MAX_SAMPLE = MAX_SAMPLE, k = k)
        return mu, sig, z, out

    def generator(self, x, adj, node_num, mask=None, n_steps = 35, temperature = 1, MAX_SAMPLE = 'top-k', k = 3):
        batch_size, channel_num, max_node_num, cond_dim  = x.shape
        ### structure embedding
        stru_cond  = self.GNN(x, adj)
        ###
        if self.version == 'cVAE':
            ### remove padding nodes
            cond = torch.cat([stru_cond[i][:node_num[i]] for i in range(batch_size)])
            #***********************
            out = self.model.generator(cond)
            out = out.reshape(-1, self.seq_len, self.aa_dim)
            out_dist = F.softmax(out, dim=-1)
            seq, seq_complete = seq_final(out_dist, node_num, MAX_SAMPLE = MAX_SAMPLE, k = k)
        elif self.version == 'text-VAE':
            if self.skip_connect:
                cond_1 = torch.cat(list(cond_all[0].transpose(0,1)),dim=-1)
                cond_2 = torch.cat(list(cond_all[1].transpose(0,1)),dim=-1)
            else:
                cond_1 = None
                cond_2 = None
            out, Indexes = self.model.generator(stru_cond, mask, node_num, adj, n_steps, temperature, condi_1 = cond_1, condi_2 = cond_2,
                                       MAX_SAMPLE = MAX_SAMPLE, k = k)
            seq, seq_complete = seq_final(Indexes, node_num, MAX_SAMPLE = 'max')
        return out, seq, seq_complete

    def vae_loss(self, mu, sig, decoded, seq, node_num, habits_lambda, kld_weight = 0.05):
        ### remove padding nodes
        batch_size = seq.shape[0]
        seq = seq[:,:,:35,:] #?
        seq = torch.cat([seq[i][:node_num[i]] for i in range(batch_size)])
        groud_truth = torch.max(seq, dim = -1)[1].reshape(-1)
        #***********************
        decoded = decoded.reshape(-1,self.aa_dim)
        ce = self.criterion(decoded, groud_truth[:decoded.shape[0]])
        #ce = 0
        KLD = (-0.5 * torch.sum(sig - torch.pow(mu, 2) - torch.exp(sig) + 1, 1)).mean().squeeze()
        clamp_KLD = torch.clamp(KLD.mean(), min=habits_lambda).squeeze()
        loss = ce + clamp_KLD * kld_weight
        return loss, ce, KLD

