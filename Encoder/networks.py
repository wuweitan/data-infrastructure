import sklearn
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
########### for GCN ##############
from torch.nn import init
import random
####### for GAT and HAN ##########
import dgl
import dgl.function as fn
from dgl.nn import GATConv

from scipy.sparse import csr_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix
########### for RNN ##############
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

######################## Accessory Functions ##############################

channel_dict_heter = {0: ('node', 'sequential', 'node'),
                      1: ('node', 'beta_parallel', 'node'),
                      2: ('node', 'beta_antiparallel', 'node'),
                      3: ('node', 'alpha_packing', 'node'),
                      4: ('node', 'other', 'node')}

def Adjacency_to_edge_index(A_matrix, CUDA=True):
    """Transform the adjacency matrices into edge index format.

    Args:
        A_matrix (numpy.array or torch.Tensor): The adjacency tensor.
        CUDA (bool): Whether assign the matrix to the GPU.

    Returns: 
        torch.Tensor: The adjacency matrix in edge index format.
    """

    result = from_scipy_sparse_matrix(csr_matrix(A_matrix))[0]
    if CUDA:
        return torch.from_numpy(result[0].numpy()).cuda(), torch.from_numpy(result[1].numpy()).cuda()
    else:
        return torch.from_numpy(result[0].numpy()), torch.from_numpy(result[1].numpy())


def Tensor_to_DGL(A_tensor, CUDA=True):
    """Transform the graph into the DGL format. 

    Args:
        A_matrix (numpy.array or torch.Tensor): The adjacency tensor.
        CUDA (bool): Whether assign the matrix to the GPU.

    Returns: 
        DGL.graph: The graph in DGL format.
    """

    edges_heter = {}
    channel_num = A_tensor.shape[0]
    if channel_num == 1:
        kind = ('node', 'all', 'node')
        edges_heter[kind] = Adjacency_to_edge_index(A_tensor[i], CUDA=CUDA)
    else:
        for i,A_mat in enumerate(A_tensor):
            kind = channel_dict_heter[i]
            edges_heter[kind] = Adjacency_to_edge_index(A_mat, CUDA=CUDA)
    g_heter = dgl.heterograph(edges_heter)
    if g_heter.num_nodes() != A_tensor.shape[-1]:
        print('Error! Node amounts do not match! (%d and %d)'%(g_heter.num_nodes(), A_tensor.shape[-1]))
    return g_heter

############################# GCN ####################################################

# GCN basic operation
class GraphConv(nn.Module):
    """GCN layer.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        channel_num: Number of channels.
        dropout: Dropout probability.
    """
    def __init__(self, input_dim, output_dim, add_self=True, normalize_embedding=False, 
            dropout=0.0, bias=True, share_weight = False, channel_num = 1):
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
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
            else:
                self.bias = None
        else:
            self.weight = nn.Parameter(torch.FloatTensor(channel_num, input_dim, output_dim).cuda())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(channel_num, 1, output_dim).cuda())
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
    """GCN layer.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        channel_num: Number of channels.
        dropout: Dropout probability.
    """

    def __init__(self, feature_dim, hidden_dim, embedding_dim, num_layers, 
                 concat=False, bn=True, dropout=0.0, bias = True, channel_num=1):
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
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x).reshape(shape)

    def forward(self, x, adj, node_num = None, **kwargs):

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
         
        return output, out_all
          
############################# GAT ####################################################

class GAT(nn.Module):
    def __init__(self,
                 channel_num = 1,
                 num_layers = 3,  ### number of layers
                 in_dim = 768,
                 num_hidden = 128,
                 num_out = 64,
                 heads = 4,
                 activation = nn.ReLU(),
                 feat_drop = 0.0,
                 attn_drop = 0.0,
                 negative_slope = 0.2,
                 residual = False,
                 allow_zero_in_degree=True):
        super(GAT, self).__init__()
        #if 0 in g.in_degrees():
        #    self.g = dgl.add_self_loop(g)
        self.channel_num = channel_num
        self.num_layers = num_layers
        self.num_hidden_layers = num_layers - 1
        self.num_out = num_out
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.out_dim = channel_num * num_out

        if type(heads) == int:
            heads = [heads] * num_layers

        self.gat_layers_all = []
        for i in range(channel_num):
            # input projection (no residual)
            gat_layers = []
            gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree = allow_zero_in_degree).cuda())
            # hidden layers
            for l in range(1, self.num_hidden_layers):  # SZ modify
                # due to multi-head, the in_dim = num_hidden * num_heads
                gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree = allow_zero_in_degree).cuda())
            # output projection
            gat_layers.append(GATConv(
                num_hidden * heads[-2], num_out, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree = allow_zero_in_degree).cuda())
            self.gat_layers_all.append(gat_layers.copy())

    def forward(self, x, adj, node_nums, **kwargs):
        batch_size, channel_num, max_node_num, feat_dim = x.shape
        ### prepare the feature matrix
        x = torch.cat([x[i][:, :node_nums[i], :] for i in range(batch_size)], 1)[0]
        ### transform adj to DGL graph
        adj = adj.cpu()
        graph = dgl.batch([Tensor_to_DGL(A_tensor[:,:node_nums[i],:node_nums[i]]) for i,A_tensor in enumerate(adj)])
        ###
        logits_all = []
        for i, g_kind in enumerate(graph.etypes):  # for different channels
            h = x.clone()
            #print(graph.device)
            #print(h.device)
            for l in range(self.num_hidden_layers):  # for different layers
                h = self.gat_layers_all[i][l](graph[g_kind], h).flatten(1)
            # output projection
            logits = self.gat_layers_all[i][-1](graph[g_kind], h).mean(1) # node_num * out_dim
            logits_shaped = Variable(torch.zeros(batch_size, max_node_num, self.num_out)).cuda()  # batch_size * max_node_num * out_dim
            start = 0
            for j, n_num in enumerate(node_nums):
                end = start + n_num
                logits_shaped[j, :n_num, :] = logits[start:end]
                start = end
            logits_all.append(logits_shaped)
        logits = torch.cat(logits_all, -1) # batch_size * max_node_num * out_dim
        return logits, logits_all

############################# HAN ####################################################

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
    """HAN layer.

    Args:
        num_meta_paths: Number of homogeneous graphs generated from the metapaths.
        in_size: Input feature dimension
        out_size: Output feature dimension
        layer_num_heads: Number of attention heads
        dropout: Dropout probability
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
        """
        Args:
            gs (list[DGLGraph]): List of graphs.
            h (torch.Tensor): Input features.
        
        Returns:
            torch.Tensor: The output features.
        """
        semantic_embeddings = []
        #for i, g in enumerate(gs):
        #    semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        for i, gt in enumerate(gs.etypes):
            semantic_embeddings.append(self.gat_layers[i](gs[gt], h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.out_dim = out_size
        if type(num_heads) == int:
            num_heads = [num_heads] * num_heads

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, x, adj, node_nums):
        batch_size, channel_num, max_node_num, feat_dim = x.shape
        ### prepare the feature matrix
        x = torch.cat([x[i][:, :node_nums[i], :] for i in range(batch_size)], 1)[0]
        ### transform adj to DGL graph
        adj = adj.cpu()
        graph = dgl.batch([Tensor_to_DGL(A_tensor[:,:node_nums[i],:node_nums[i]]) for i,A_tensor in enumerate(adj)])
        ### forward
        for gnn in self.layers:
            x = gnn(graph, x)
        result = self.predict(x)

        logits = Variable(torch.zeros(batch_size, max_node_num, self.out_dim)).cuda()
        start = 0
        for j, n_num in enumerate(node_nums):
            end = start + n_num
            logits[j, :n_num, :] = result[start:end]
            start = end

        return logits, None
 
###########################################################
# For VAE absed Model
###########################################################

###### Encoder ######

def create_sinusoidal_embeddings(nb_p, dim, E, USE_CUDA = True):  # positional embedding
    theta = np.array([
        [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for p in range(nb_p)
    ])
    E.detach_()
    E.requires_grad = False
    E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
    E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
    if USE_CUDA:
        E = E.cuda()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size = 11, seq_cat = True, skip_connect = False,
                 n_layers=3, dropout_p=0.1, bidirectional=True, rnn_kind = 'LSTM', USE_CUDA = True, encode_emb = True, MLP_layer = 3, ignore_padding = True,  
                 heads_num=4, max_position_embeddings = 36, layer_norm_eps = 1e-5):
        super(EncoderRNN, self).__init__()

        ### versions ###
        self.seq_cat = seq_cat  # whether cat the sequence for seq embed
        self.bidirectional = bidirectional
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        self.skip_connect = skip_connect
        self.encode_emb = encode_emb
        self.ignore_padding = ignore_padding
        ### architecture ###
        if rnn_kind == 'Transformer':
            input_size += 1   # for CLAS token
        self.input_size = input_size
        #print(input_size)
        self.condition_size = condition_size

        if seq_cat: ### concatenate the sequences with the conditions
            if encode_emb:
                self.embed = nn.Embedding(self.input_size, hidden_size)
                self.rnn_input_size = hidden_size + condition_size
            else:
                self.rnn_input_size = input_size + condition_size
        else:
            if encode_emb:
                self.embed = nn.Embedding(self.input_size, hidden_size)
                self.rnn_input_size = hidden_size
            else:
                self.rnn_input_size = input_size

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.MLP_layer = MLP_layer # num of MLP layers

        if rnn_kind == 'GRU':
            self.rnn = nn.GRU(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            self.rnn_out_size = hidden_size
        elif rnn_kind == 'LSTM':
            self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            self.rnn_out_size = hidden_size
        elif rnn_kind == 'RNN':
            self.rnn = nn.RNN(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=bidirectional, batch_first = True)
            self.rnn_out_size = hidden_size
        elif rnn_kind == 'Transformer':
            ### position embedding
            self.position_embeddings = nn.Embedding(max_position_embeddings, self.rnn_input_size)
            create_sinusoidal_embeddings(
                nb_p=max_position_embeddings,
                dim=self.rnn_input_size,
                E=self.position_embeddings.weight,
                USE_CUDA = USE_CUDA
            )
            encoder_layer = nn.TransformerEncoderLayer(d_model = self.rnn_input_size, dim_feedforward = hidden_size, nhead=heads_num, 
                                                       layer_norm_eps = layer_norm_eps, batch_first = True, dropout=dropout_p)
            encoder_norm = nn.LayerNorm(normalized_shape = self.rnn_input_size, eps=layer_norm_eps)
            self.rnn = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)
            self.rnn_out_size = self.rnn_input_size
        else:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()

        if rnn_kind == 'Transformer': 
            self.latent_size = self.rnn_input_size + condition_size
        elif skip_connect:
            self.latent_size = hidden_size
        else:
            self.latent_size = condition_size + hidden_size

        if MLP_layer == 1:
            self.o2p = nn.Linear(self.latent_size, output_size * 2)
        elif MLP_layer > 1:
            mlp_mu = [nn.Linear(self.latent_size, hidden_size), nn.ReLU()]
            mlp_sigma = [nn.Linear(self.latent_size, hidden_size), nn.ReLU()]
            for i in range(MLP_layer - 2):
                mlp_mu.append(nn.Linear(hidden_size, hidden_size))
                mlp_mu.append(nn.ReLU())

                mlp_sigma.append(nn.Linear(hidden_size, hidden_size))
                mlp_sigma.append(nn.ReLU())

            mlp_mu.append(nn.Linear(hidden_size, output_size))
            mlp_sigma.append(nn.Linear(hidden_size, output_size))
            self.mu_out = nn.Sequential(*mlp_mu)
            self.sigma_out = nn.Sequential(*mlp_sigma)
        else:
            print('Error! There should be at least 1 layer for residue mapping!')
            quit()

    def forward(self, seq, condition, seq_length_array = None, key_padding_mask=None):
        # seq: seq_num x seq_len x aa_dim
        # condition: seq_num x condi_dim
        # seq_length_array: seq_num x 1
        # key_padding_mask: seq_num x max_seq_len, 0 for valid tokens
        if self.rnn_kind == 'Transformer':
            seq = torch.nn.functional.pad(seq, (0, 1, 1, 0))  # add CLAS token
            seq[:,0,-1] = 1
            #print(key_padding_mask[:2])
            key_padding_mask = torch.nn.functional.pad(key_padding_mask, (1, 0)) 
            #print(key_padding_mask[:2])
        batch_size, seq_len, aa_dim = seq.shape
        if self.encode_emb:
           seq = self.embed(seq.max(dim = -1)[1]) 
        if self.seq_cat: # cat the condition for each residue
           cond_repeat = condition.repeat(1,1,seq_len).reshape(batch_size, seq_len, self.condition_size)
           seq = torch.cat((cond_repeat, seq),-1)

        if self.rnn_kind == 'Transformer': 
            seq = self.posi_embedding(seq)
            if self.ignore_padding:
                 output = self.rnn(seq, src_key_padding_mask = key_padding_mask)  # seq_num x seq_len x rnn_input_size
                 output[key_padding_mask != 0] = 0
            else:
                 output = self.rnn(seq)
            output = output[:,0,:]  ### apply the embedding of the CLAS token
            # output: seq_num x 1 x rnn_input_size
        else:
            if self.ignore_padding:
                seq = pack_padded_sequence(seq, seq_length_array.cpu(), batch_first=True, enforce_sorted=False)
                output, hidden = self.rnn(seq)
                output, _ = pad_packed_sequence(output, batch_first = True)
                output = output[torch.arange(batch_size), seq_length_array.long() - 1]
            else:
                output, hidden = self.rnn(seq)
                output = output[:,-1,:] # Take only the last value
            if self.bidirectional:
                output = output[:, :self.hidden_size] + output[: ,self.hidden_size:] # Sum bidirectional outputs
            # output: seq_num x 1 x hidden_size

        if not self.skip_connect:
            output = torch.cat([output, condition], dim = -1) 

        if self.MLP_layer == 1:
            ps = self.o2p(output)
            mu, logvar = torch.chunk(ps, 2, dim=1)  # mean and log-variance
        else:
            mu = self.mu_out(output)
            logvar = self.sigma_out(output)
            
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if self.USE_CUDA:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def posi_embedding(self, seq):
        seq_length = seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(seq[:,:,0])
        position_embeddings = self.position_embeddings(position_ids)
        return seq + position_embeddings

###### Decoder ######

class DecoderRNN(nn.Module):
    ################################################################
    # sequence generator for the next residue
    ################################################################
    def __init__(self, input_size, hidden_size, output_size=21, feedback_size=11, skip_connect=False, cond_dim_skip=None,
                 n_layers=3, dropout_p=0.1, rnn_kind = 'LSTM', USE_CUDA=True, MLP_layer = 1,
                 heads_num=4, layer_norm_eps = 1e-5, max_position_embeddings = 36):
        super(DecoderRNN, self).__init__()

        ### versions ###
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        if not rnn_kind in ['GRU','LSTM','RNN','Transformer']:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()
        self.skip_connect = skip_connect
        ### architecture ###
        self.rnn_input_size = input_size  # feadback (hidden_dim) for Transformer and (z + c) + feadback for RNNs
        self.feedback_size = feedback_size
        self.hidden_size = hidden_size
        self.output_size = output_size # dimension of the residue vector
        self.MLP_layer = MLP_layer # num of MLP layers

        if skip_connect:
            self.skip_dim = cond_dim_skip + hidden_size
            self.n_layers = 3 # skip connection only support for 3 layers
            if rnn_kind == 'GRU':
                self.rnn_1 = nn.GRU(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.GRU(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.GRU(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size
            elif rnn_kind == 'LSTM':
                self.rnn_1 = nn.LSTM(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.LSTM(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.LSTM(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size
            elif rnn_kind == 'RNN':
                self.rnn_1 = nn.RNN(self.rnn_input_size, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_2 = nn.RNN(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_3 = nn.RNN(self.skip_dim, hidden_size, 1, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size
            else:
                print('Error!')
                quit()

        else:
            self.n_layers = n_layers       
            if rnn_kind == 'GRU':
                self.rnn = nn.GRU(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size + input_size - feedback_size
            elif rnn_kind == 'LSTM':
                self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size + input_size - feedback_size
            elif rnn_kind == 'RNN':
                self.rnn = nn.RNN(self.rnn_input_size, hidden_size, n_layers, dropout=dropout_p, batch_first = True)
                self.rnn_out_size = hidden_size + input_size - feedback_size
            elif rnn_kind == 'Transformer':
                ### position embedding
                self.position_embeddings = nn.Embedding(max_position_embeddings, self.rnn_input_size)
                create_sinusoidal_embeddings(
                    nb_p=max_position_embeddings,
                    dim=self.rnn_input_size,
                    E=self.position_embeddings.weight,
                    USE_CUDA = USE_CUDA
                )
                ### decoder layers
                decoder_layer = nn.TransformerDecoderLayer(d_model = self.rnn_input_size, dim_feedforward = hidden_size, nhead=heads_num,
                                                           layer_norm_eps = layer_norm_eps, batch_first = True, dropout=dropout_p)
                decoder_norm = nn.LayerNorm(normalized_shape = self.rnn_input_size, eps=layer_norm_eps)
                self.rnn = nn.TransformerDecoder(decoder_layer, n_layers, decoder_norm)
                self.rnn_out_size = self.rnn_input_size * 2
            else:
                print('Error! No RNN module named "%s"!'%rnn_kind)
                quit()

        if MLP_layer == 1:
            self.out = nn.Linear(self.rnn_out_size, output_size) # for the output of each step
        elif MLP_layer > 1:
            mlp = [nn.Linear(self.rnn_out_size, hidden_size), nn.ReLU()]
            for i in range(MLP_layer - 2):
                mlp.append(nn.Linear(hidden_size, hidden_size))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(hidden_size, output_size))
            self.out = nn.Sequential(*mlp)
        else:
            print('Error! There should be at least 1 layer for residue mapping!')
            quit()            

    def forward(self, z, resi_info, hidden = None, condi_1 = None, condi_2 = None, tgt_mask = None):
        """
        z: noise       
        hidden: previous hidden state
        cond_1, cond_2: for skip connection
        """
        if self.rnn_kind == 'Transformer':
            resi_info = self.posi_embedding(resi_info)
            output = self.rnn(tgt = resi_info,
                              tgt_mask = tgt_mask,
                              memory = z)
            z = z.repeat(1, output.shape[-2], 1)
            output = torch.cat((z, output), -1)
            hidden = None
        else:
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

    def posi_embedding(self, seq):
        seq_length = seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(seq[:,:,0])
        position_embeddings = self.position_embeddings(position_ids)
        return seq + position_embeddings


class Sequential_Generator(nn.Module):
    ################################################################
    # autoregressive for sequence generation on the complete element 
    ################################################################
    def __init__(self, input_size, hidden_size = 64, skip_connect=False, cond_dim_skip=None, USE_CUDA=True, teacher_forcing = True, MLP_layer = 1,
                 ### for RNN
                 rnn_kind = 'LSTM', output_size=21, condition_size=11, n_layers=3, dropout_p=0.1,
                 ### for Transformer
                 heads_num=4, layer_norm_eps = 1e-5, max_position_embeddings = 36): 
        super(Sequential_Generator, self).__init__()

        ### version ###
        self.USE_CUDA = USE_CUDA
        self.rnn_kind = rnn_kind
        if not rnn_kind in ['GRU','LSTM','RNN','Transformer']:
            print('Error! No RNN module named "%s"!'%rnn_kind)
            quit()
        self.skip_connect = skip_connect
        ### architecture ###
        self.input_size = input_size # size of the input noise
        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.output_size = output_size # dimension of the residue vector
        self.rnn_kind = rnn_kind

        if rnn_kind == 'Transformer' or not skip_connect:
            self.z2h_dim = input_size + condition_size
        else:
            self.z2h_dim = input_size

        if rnn_kind == 'LSTM':
            self.z2h = nn.Linear(self.z2h_dim, hidden_size * 2) # for initial hidden status
        else:
            self.z2h = nn.Linear(self.z2h_dim, hidden_size) # for initial hidden status

        self.teacher_forcing = teacher_forcing
        self.embed = nn.Embedding(output_size + 1, hidden_size)  # embedding of the residues (20 aa + padding + initial)
        self.resi_info_dim = hidden_size

        if rnn_kind == 'Transformer':
            self.rnn_input_size = self.resi_info_dim
        elif skip_connect:
            self.rnn_input_size = self.resi_info_dim + input_size 
        else:
            self.rnn_input_size = self.resi_info_dim + input_size + condition_size

        self.n_layers = n_layers
        self.step_decoder = DecoderRNN(input_size=self.rnn_input_size, hidden_size=hidden_size, 
                                       output_size=output_size, feedback_size=self.resi_info_dim, 
                                       skip_connect=skip_connect, cond_dim_skip=cond_dim_skip, 
                                       n_layers=n_layers, dropout_p=dropout_p, 
                                       rnn_kind = rnn_kind, USE_CUDA=USE_CUDA, MLP_layer = MLP_layer,
                                       heads_num=heads_num, layer_norm_eps = layer_norm_eps, max_position_embeddings = max_position_embeddings)

    def generate(self, condition, mask, Adj = None, n_steps = 35, temperature = 1, condi_1 = None, condi_2 = None, MAX_SAMPLE = 'top-k', k = 3):
        batch_size, max_node_num, cond_dim = condition.shape
        condition = condition.reshape(-1, cond_dim)
        num_seq = condition.shape[0]
        ### noise and condition ### 
        noise = Variable(torch.randn(num_seq, self.input_size))
        if self.USE_CUDA:
            noise = noise.cuda()
        if not self.skip_connect:
            z = torch.cat((noise,condition),-1) # the new noise contain the condition
        else:
            z = noise
        ### initialization ###
        outputs = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # empty output tensor
        Indexes = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # for one-hot encoding sequence
        resi_info = Variable(self.output_size * torch.ones(num_seq, 1)) # initial input
        if self.USE_CUDA:
            outputs = outputs.cuda()
            resi_info = resi_info.cuda()
            Indexes = Indexes.cuda()

        if self.rnn_kind == 'Transformer':
            z = self.z2h(z).unsqueeze(1)
            ### initial embedding
            resi_info = self.embed(resi_info.long())
            resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim) # tgt sequence: batch_size x 1 x (hidden_dim)
            ### autoregressive generation
            for i in range(n_steps):
                output, _ = self.step_decoder(resi_info = resi_info,
                                              tgt_mask = None, #tgt_mask,
                                              z = z)
                output = output[:, -1, :]
                outputs[:,i,:] = output
                next_resi, top_i = self.sample(output, num_seq, temperature, MAX_SAMPLE = MAX_SAMPLE, k = k)
                Indexes[:,i,:] = next_resi # one-hot encoding sequence
                resi_info = torch.cat([resi_info, self.embed(next_resi.max(dim = -1)[1]).reshape(num_seq, 1, self.resi_info_dim)], 1) # num_seq x (i+1) x emb_dim* 
                
        else:
            hidden = self.z2h(z).unsqueeze(1).repeat(1, self.n_layers, 1).transpose(0,1).contiguous()  # initial hidden status
            if self.USE_CUDA:
                hidden = hidden.cuda()
            ### initial embedding
            resi_info = self.embed(resi_info.long())
            resi_info = F.relu(resi_info)
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
                resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        ### reshape
        outputs = outputs.reshape(batch_size, max_node_num, n_steps, self.output_size) * mask.unsqueeze(-1)
        Indexes = Indexes.reshape(batch_size, max_node_num, n_steps, self.output_size) * mask.unsqueeze(-1)
        return outputs, Indexes, noise

    def forward_vae(self, z, condition, mask, node_num, Adj = None, n_steps = 35, temperature = 1, condi_1 = None, condi_2 = None, 
                    ground_truth = None, MAX_SAMPLE = 'top-k', k = 3):
        batch_size, max_node_num, cond_dim = condition.shape
        ### change the shape for separate elements
        condition = condition.reshape(-1, cond_dim)
        num_seq = condition.shape[0]
        if ground_truth is not None:
            ground_truth = ground_truth.reshape(num_seq, -1, self.output_size).long()
        z = z.reshape(num_seq, -1)
        # initial input
        resi_info = Variable(self.output_size * torch.ones(num_seq, 1)) # initial input: num_seq x 1
        if self.USE_CUDA:
            resi_info = resi_info.cuda()
        ### initial embedding
        resi_info = self.embed(resi_info.long())
        resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim) # num_seq x 1 x emb_dim*

        ### for transformer ###
        if self.rnn_kind == 'Transformer':
            z = torch.cat((z,condition),-1) # the new noise contain the condition
            z = self.z2h(z).unsqueeze(1)
            tgt_emb = self.embed(ground_truth.max(dim = -1)[1])  # groudtruth embedding: num_seq x seq_len x emb_dim*
            tgt_emb = torch.cat([resi_info, tgt_emb], 1)
            tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(tgt_emb.size(-2))
            if self.USE_CUDA:
                tgt_mask = tgt_mask.cuda() 

            outputs, _ = self.step_decoder(resi_info = tgt_emb,
                                           tgt_mask = tgt_mask,
                                           z = z)
            outputs = outputs[:,:-1,:] #num_seq x seq_len x emb_dim*

        ### for RNNs ###
        else:
            if not self.skip_connect:
                z = torch.cat((z,condition),-1) # the new noise contain the condition
            ### initial preparation
            resi_info = F.relu(resi_info)
            outputs = Variable(torch.zeros(num_seq, n_steps, self.output_size)) # empty output tensor
            hidden = self.z2h(z).unsqueeze(1).repeat(1, self.n_layers, 1).transpose(0,1).contiguous()  # initial hidden status
            if self.USE_CUDA:
                outputs = outputs.cuda()
                hidden = hidden.cuda()

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
                resi_info = resi_info.reshape(num_seq, 1, self.resi_info_dim)

        ### reshape
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

