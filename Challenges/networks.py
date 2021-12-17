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
                 concat=True, bn=True, dropout=0.0, bias = True, channel_num=1):
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

        if type(model) == str:
            if model == 'GCN':
                self.GNN = GcnEncoderGraph(feature_dim = self.gnn_in_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim,
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
                self.GNN = GcnEncoderGraph(feature_dim = self.gnn_in_dim, hidden_dim = gnn_hidden_dim, embedding_dim = gnn_emb_dim,
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
            resi_info, _ = self.GNN(x, Adj)
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
                resi_info, _ = self.GNN(x, Adj)
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
            resi_info, _ = self.GNN(x, Adj)
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
                resi_info, _ = self.GNN(x, Adj)
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
                 gnn_model = None, gnn_kind='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, gnn_num_layers=3, channel_num=5,
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
        else:
            if gnn_kind == 'GCN':
                self.GNN = GcnEncoderGraph(feature_dim = feature_dim, hidden_dim = hidden_dim, embedding_dim = embedding_dim,
                                           num_layers = self.gnn_num_layers, concat=concat, bn = bn, dropout=dropout, bias=bias, channel_num=channel_num)
            else:
                print('Error! No GNN model named %s!'%gnn_kind)
                quit()

        self.cond_dim_skip = hidden_dim * channel_num
        self.gnn_out_dim = self.GNN.out_dim

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
        stru_cond, cond_all = self.GNN(x, adj)
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
        stru_cond, cond_all = self.GNN(x, adj)
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

###########################################################
# Antibody Design Model
###########################################################

################### GNN Encoder ###########################

class GcnEncoderGraph_Antibody(nn.Module):
    def __init__(self, feature_dim, hidden_dim, seq_dim, embedding_dim, num_layers,
                 concat=False, bn=True, dropout=0.0, bias = True, channel_num=1, CUDA=True):
        super(GcnEncoderGraph_Antibody, self).__init__()

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
        self.CUDA = CUDA

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(feature_dim, hidden_dim, seq_dim, embedding_dim, num_layers,
                                                                                  add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()

        self.hidden_out_dim = hidden_dim * channel_num
        if concat:
            self.out_dim = (hidden_dim * (num_layers - 1) + embedding_dim) * channel_num
        else:
            self.out_dim = channel_num * embedding_dim

        ### Initialize the weights ###
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, seq_dim, embedding_dim, num_layers, add_self, normalize=False, dropout=0.0):

        conv_first = GraphConv(input_dim=input_dim + seq_dim, output_dim=hidden_dim, add_self=add_self,
                                normalize_embedding=normalize, bias=self.bias, channel_num = self.channel_num, CUDA = self.CUDA)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim + seq_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, channel_num = self.channel_num, CUDA = self.CUDA)
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim + seq_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, channel_num = self.channel_num, CUDA = self.CUDA)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' 
        Batch normalization of 3D tensor x
        '''
        shape = x.shape
        x = torch.reshape(x,(-1,shape[-2],shape[-1]))
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.CUDA:
            bn_module = bn_module.cuda()
        return bn_module(x).reshape(shape)

    def forward(self, x, seq_feat, adj, **kwargs):
        out_all = []
        ### 1-st layer
        x = self.conv_first(torch.cat([x,seq_feat],-1), adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all.append(x)
        ### hidden layers
        for i in range(self.num_layers-2):
            x = self.conv_block[i](torch.cat([x,seq_feat],-1),adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out_all.append(x)
        ### last layer
        out = self.conv_last(torch.cat([x,seq_feat],-1),adj)
        out_all.append(out)
        ### cat the outputs
        if self.concat:
            output = [torch.cat(list(tensor.transpose(0,1)), dim =-1) for tensor in out_all]
            output = torch.cat(output,dim=-1)
        else:
            output = torch.cat(list(out.transpose(0,1)),dim=-1)
        return output

#**********************************************************

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
            self.gnn = GcnEncoderGraph_Antibody(feature_dim = feature_dim, hidden_dim = hidden_dim, seq_dim = seq_dim,
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

##################### Sequence Encoder ####################

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

############### Generative Module #########################

class Decoder(nn.Module):
    """
    Predict the residue given the node embedding and the sequence embedding.
    """
    def __init__(self, input_size = 100, output_size = 20, # for linear layer
                 num_heads = 4, dropout = 0.1, kdim = 512, vdim = 512,  
                 USE_CUDA = True, with_attn = True):
        super(Decoder, self).__init__()

        ### architecture ###
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.kdim = kdim
        self.vdim = vdim
        self.USE_CUDA = USE_CUDA
        self.with_attn = with_attn
 
        ### Linear layer
        self.linear = nn.Linear(input_size, output_size)
        ### attention layer
        if self.with_attn:
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

        if self.with_attn:
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

###################### Container ##########################

class CDR_Generator(nn.Module):
    """
    Encoder (can be RNN, GRU, LSTM or Transformer) for the antigen and antibody sequences.
    """
    def __init__(self, gen_version = 'CDR-parallel', with_ag_seq = False, with_ab_seq = True, USE_CUDA = True, cdr_num = 6, avoid_padding = True, 
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
        self.with_ab_seq = with_ab_seq
        self.rnn_kind = rnn_kind
        self.gnn_kind = gnn_kind
        self.USE_CUDA = USE_CUDA
        if self.gen_version == 'sequential': # sequentially generate HCDR1, HCDR2, ..., LCDR3
            self.cdr_num = 1
        elif self.gen_version == 'HL-parallel': # sequentially generate CDR1, CDR2, CDR3, parallel for HL chains
            self.cdr_num = 2
        else:
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
                               USE_CUDA = USE_CUDA, with_attn = self.with_ab_seq)    


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
                                               batch_size x 1 x max_cdr_ab_sum x 1 (sequential)
                                               batch_size x 2 x max_cdr_chain_sum x 1 (HL_parallel)
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
        if self.gen_version == 'sequential' or self.gen_version == 'HL-parallel':
            seq_cdr_num = 1
        else:
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
        if self.gen_version == 'sequential' or self.gen_version == 'HL-parallel':
            key_padding_mask = torch.ones(batch_size, HL_num, max_seq_len_ab)
            for i,len_ab_batch in enumerate(seq_len_ab):
                for j,len_ab_chain in enumerate(len_ab_batch):
                    key_padding_mask[i,j,:int(len_ab_chain)] = 0
            key_padding_mask = key_padding_mask == 1
            key_padding_mask = key_padding_mask.reshape(tar_num,-1) # mask: tar_num x max_seq_len_ab
        else:
            key_padding_mask = torch.ones(batch_size, HL_num, max_seq_len_ab) 
            for i,len_ab_batch in enumerate(seq_len_ab):
                for j,len_ab_chain in enumerate(len_ab_batch):
                    key_padding_mask[i,j,:int(len_ab_chain)] = 0
            key_padding_mask = key_padding_mask == 1
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
            cdr_batch = []
            for cdr_idx in range(self.cdr_num):
                cdr_batch.append('')
            cdrs.append(cdr_batch)

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
            if self.gen_version == 'sequential':
                aa, aa_onehot, prob = self.decoder(tar_node_emb,  # tar_num x 1 x gnn_dim 
                                                   seq_ab_emb,    # tar_num x max_seq_len_ab x rnn_dim
                                                   cdr_mask_temp,      # batch_size x 6 (max_cdr_num) x 1
                                                   key_padding_mask = key_padding_mask * cdr_mask_temp.reshape(-1,1),
                                                   attn_mask = attn_mask * cdr_mask_temp.reshape(-1,1,1)) 
            else:
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
                                                  batch_size x 1 x max_cdr_ab_sum x 20 (sequential)
                                                  batch_size x 2 x max_cdr_chain_sum x 20 (HL_parallel)
        target: groudtruth, batch_size x 6 (max_cdr_num) x max_cdr_len
                            batch_size x 1 x max_cdr_ab_sum (sequential)
                            batch_size x 2 x max_cdr_chain_sum (HL_parallel)
        cdr_mask: tensor of bullin points, batch_size x 6 x max_cdr_len x 1
                                           batch_size x 1 x max_cdr_ab_sum x 1 (sequential)
                                           batch_size x 2 x max_cdr_chain_sum x 1 (HL_parallel)
        """
        profile = profile[cdr_mask.repeat(1,1,1,self.output_size)].reshape(-1,self.output_size)
        target = target[cdr_mask.squeeze()].reshape(-1) 
        loss = F.cross_entropy(profile, target)
        return loss

