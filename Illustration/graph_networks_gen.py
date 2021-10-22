import torch
import torch.nn.functional as F
########### for GCN ##############
import torch.nn as nn
from torch.nn import init
import numpy as np
import random

from torch.autograd import Variable

######################## Accessory Functions ##############################

def tensor_to_string(t, index = False, MAX_SAMPLE = 'top-k', k = 3):
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
        seq_piece = tensor_to_string(seq_tensor, index = False, MAX_SAMPLE = MAX_SAMPLE, k = k)
        seq.append(seq_piece)
        seq_all = ''.join([s.split('!')[0] for s in seq_piece])
        seq_complete.append(seq_all)
    return seq, seq_complete

############################# GCN ####################################################

# GCN basic operation
class GraphConv(nn.Module):
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
         
        return output, out_all
          
 
###########################################################
# For VAE absed Model
###########################################################

#****************** for cVAE ******************************

class cVAE(nn.Module):
    def __init__(self, input_size, condition_size, hidden_sizes):
        super().__init__()

        self.input_size = input_size
        self.condition_size = condition_size
        self.hidden_sizes = hidden_sizes

        ### for encoder 
        self.fc = torch.nn.Linear(input_size + condition_size, hidden_sizes[0])  # 1-st layer
        self.BN = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.BN1 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.BN2 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        ### for decoder
        self.fc4 = torch.nn.Linear(hidden_sizes[3] + condition_size, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size) # last layer

    def sample_z(self,x_shape, mu, log_var):
        # Using reparameterization trick to sample from a gaussian    
        eps = Variable(torch.randn(x_shape, self.hidden_sizes[-1]))#.cuda()
        return mu + torch.exp(log_var / 2) * eps

    def encoder(self, x, condition):
        x = torch.cat((x,condition),1)
        # Layer 0
        out1 = self.fc(x)
        out1 = F.relu(self.BN(out1))
        # Layer 1
        out2 = self.fc1(out1)
        out2 = F.relu(self.BN1(out2))
        # Layer 2
        out3 = self.fc2(out2)
        out3 = F.relu(self.BN2(out3))
        # Layer 3 - mu
        mu   = self.fc3_mu(out3)
        # layer 3 - sig
        sig  = F.softplus(self.fc3_sig(out3))
        return mu, sig

    def decoder(self, z, condition):
        # add the condition
        sample = torch.cat((z, condition),1)
        # Layer 4
        out4 = self.fc4(sample)
        out4 = F.relu(self.BN4(out4))
        # Layer 5
        out5 = self.fc5(out4)
        out5 = F.relu(self.BN5(out5))
        # Layer 6
        out6 = self.fc6(out5)
        out6 = F.relu(self.BN6(out6))
        # Layer 7
        out7 = F.sigmoid(self.fc7(out6))
        return out7

    def forward(self, x, condition):
        ########### Encoder ##############
        mu, sig = self.encoder(x, condition)
        ########### Decoder #############
        # sample from the distro
        z = self.sample_z(x.size(0),mu, sig)
        # add the condition
        out7 = self.decoder(z, condition)
        return mu, sig, z, out7

    def generator(self, condition):
        ########### Decoder #############        
        num_seq = condition.size()[0]
        z = Variable(torch.randn(num_seq, self.hidden_sizes[-1]))#.cuda()
        out7 = self.decoder(z, condition)
        return out7

#****************** for text-VAE ****************************** 

###### Encoder ######

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size = 11, seq_cat = True, skip_connect = False,
                 n_layers=1, dropout_p=0.1, bidirectional=True, rnn_kind = 'LSTM', USE_CUDA = True):
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

###### Decoder ######

class DecoderRNN(nn.Module):
    ################################################################
    # sequence generator for the next residue
    ################################################################
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


class Sequential_Generator(nn.Module):
    ################################################################
    # autoregressive for sequence generation on the complete element 
    ################################################################
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

###### Container ######

class c_text_VAE(nn.Module):
    def __init__(self, input_size=21, encoder_hidden_size=64, decoder_hidden_size=64, encoder_output_size=32, condition_size=11, cond_dim_skip=None,
                 seq_cat=True, n_encoder_layers=3, n_decoder_layers=3, dropout_p=0.1, bidirectional=True, rnn_kind = 'LSTM', channel_num = 5,
                 USE_CUDA = True, skip_connection = False, gnn_embed = True, teacher_forcing = True):
        super(c_text_VAE, self).__init__()
        """
        input_size: 20 + 1 (AA & padding)
        output_size: dimension of z
        """
        self.USE_CUDA = USE_CUDA
        self.encoder_output_size = encoder_output_size
        self.encoder = EncoderRNN(input_size, encoder_hidden_size, encoder_output_size, condition_size, seq_cat, skip_connection,
                                  n_encoder_layers, dropout_p, bidirectional, rnn_kind, USE_CUDA)

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

#************************************** Overall Holder ************************************************************

class VAE_Container(nn.Module):
    def __init__(self, CUDA = False, seq_len = 35, aa_dim = 21, version = 'text-VAE', padding_weight = 0.1,
                 ### for struc_GNN
                 gnn_kind='GCN', feature_dim=11, hidden_dim=100, embedding_dim=20, gnn_num_layers=3, channel_num=5, 
                 concat=False, bn=True, dropout=0.0, bias=True,
                 ### for cVAE 
                 cvae_hidden_size=[512,256,128,16],
                 ### for ctext-VAE
                 seq_cat = True, skip_connection = False, gnn_embed = True, teacher_forcing = True, rnn_kind='LSTM'):
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

            self.model = c_text_VAE(input_size=aa_dim, condition_size=self.gnn_out_dim, seq_cat=seq_cat, rnn_kind = rnn_kind,
                                    USE_CUDA = CUDA, skip_connection = skip_connection, cond_dim_skip = self.cond_dim_skip, 
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
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

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


