#######################################################################################
#  SZhu add
#  help to load the data into the correct format
#######################################################################################

import numpy as np
import pickle
import torch

import torch_geometric
from scipy.sparse import csr_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import networkx as nx
import torch.utils.data
import random
from math import ceil

import os
import pickle

####################### path helper ######################

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

def make_path(path):
    """
    Create the directories in then path if they do not exist.
    Will take the all the substrings in the path split by '/' as a directory.
    """
    if path.startswith('/'):
        begin = '/'
    else:
        begin = ''
    substring = path.strip('/').split('/')
    current_path = begin
    for dire in substring:
        current_path += dire + '/'
        if not os.path.exists(current_path):
            os.mkdir(current_path)
    return 0

################# graph helper #################################

def Adjacency_to_edge_index(A_matrix):
    """
    Transform the adjacency matrices into edge index format.
    """
    if type(A_matrix) == list:
        multiple = True
        result = []
        for A in A_matrix:
            result.append(Adjacency_to_edge_index(A)[0])
    else:
        multiple = False
        while(len(A_matrix.shape) > 2):
            A_matrix = A_matrix[0]
        result = from_scipy_sparse_matrix(csr_matrix(A_matrix))[0]
    return result, multiple

def to_nxGragh(X, A):
    adj_dim = A.shape[0]
    G_all = []
    for idx in range(adj_dim):
        G = nx.Graph()
        node_num = X.shape[0]
        for i in range(node_num):
            G.add_node(i,feat=X[i])
        for i in range(node_num):
            for j in range(i+1,node_num):
                if A[idx,i,j] != 0:
                    G.add_edge(i, j, weight= A[idx,i,j])
                    G.add_edge(j, i, weight= A[idx,j,i])
        G_all.append(G)
    return G_all

################################ Data Loader ####################################

def assign_weight(n_fea, weight_dict, ignore_missing = False):
    if ignore_missing:
        result = []
        for fea in n_fea:
             if fea[0] == 1 or fea[1] == 1:
                 result.append(0)
             else:
                 result.append(weight_dict[tuple(fea)])
        result = np.array(result)
    else:
        result = np.array([weight_dict[tuple(fea)] for fea in n_fea])
    return result

def Gen_dataloading(Seq_file, X_file, A_file, seq_mask_dim = 21,features = ['len', 'ss', 'level'], weight_dict = None,
                batch_size=64, normalize=False, shuffle=False,max_SSE=None,max_nodes=60,num_workers=1,
                set_kind = 'training', ignore_missing = False, test_num=None):
    """
    Load the data from the files.
    Seq_file: file of the embedded sequences
    X_file: file of the node features
    A_file: file of the adjacency tensors

    max_SSE: maximum length of the SSE to be considered (Ignore the proteins with longer SSEs. Consider all SSEs if None.)
    test_num: if not None, randomly select <test_num> samples (for code test)
    """

    fea_sele = [0,1]
    for fea in set(features):
        if fea == 'len':
            fea_sele += [2,3,4]
        elif fea == 'ss':
            fea_sele += [5,6,7]
        elif fea == 'level':
            fea_sele += [8,9,10]
    fea_sele = sorted(fea_sele)

    with open(Seq_file,'rb') as handle_S, open(X_file,'rb') as handle_X, open(A_file,'rb') as handle_A:
        S_all = pickle.load(handle_S) 
        X_all = [n_fea[:,fea_sele] for n_fea in pickle.load(handle_X)]
        A_all = pickle.load(handle_A)

    feature_num = int(X_all[0].shape[-1])
    length = len(X_all)

    if weight_dict is not None and len(features) == 3:
        weight_dict = dict_load(weight_dict)[set_kind]
        weight_all = [assign_weight(n_fea, weight_dict, ignore_missing) for n_fea in X_all]
        print('Node-wise weights loaded!')
    else:
        weight_all = [None for n_fea in X_all]

    if len(A_all[0].shape) == 2:
        channel_num = 1
        add_dim = True
    elif len(A_all[0].shape) == 3:
        channel_num = A_all[0].shape[0]
        add_dim = False
    else:
        print('Error! Dimension of the adjacency matrix/tensor should be 2 or 3, not %d!'%len(A_all[0].shape))
        return 1

    if (len(A_all) != length):
        print('Error! The sizes of the data do not match! (X:%d, A:%d)'%(X_all.shape[0],A_all.shape[0]))
        return 1
    else:
        Data = []

        if test_num:
            print('Randomly loading %d samples for code test...'%test_num) 
            if test_num > length:
                select_idx = random.choices(range(length),k = test_num)
            else:
                select_idx = random.sample(range(length),test_num)
        else:
            print('Loading the original %d samples...'%length)
            select_idx = range(length)

        for i in select_idx:
            S = S_all[i] 
            SSE_len_max = max([seq_ele.shape[0] for seq_ele in S])
            if max_SSE and SSE_len_max > max_SSE:
                continue # ignore the proteins with too long sequence elements
 
            X = X_all[i]
            A = A_all[i]   

            if A.shape[-1] > max_nodes:
                continue # ignore the graphs whose amount of nodes are larger than max_nodes
     
            if add_dim:
                A = np.expand_dims(A,axis=0) 
    
            graph = to_nxGragh(X,A)
            W = weight_all[i]
            Data.append((S,graph,W))
            #Data.append((S,graph,W))

            sample_len = len(Data)

        dataset_sampler = GraphSampler(Data, normalize=normalize, seq_mask_dim = seq_mask_dim, max_num_nodes=max_nodes, max_SSE=max_SSE)
        print('Graph sampling completed. %d graphs sampled.'%(sample_len))
        print('Size of the sampler: %d'%(dataset_sampler.__sizeof__()))
        dataset_loader = torch.utils.data.DataLoader(dataset_sampler,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)
        return dataset_loader, length, sample_len, feature_num, channel_num

############## Dataloader_helper (modified from diffPool) ################

def node_iter(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict

class GraphSampler(torch.utils.data.Dataset):
    ''' 
    Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, normalize=True, seq_mask_dim = 21, max_num_nodes=0, max_SSE=None):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.seq = []
        self.seq_mask_all = []
        self.weights = [] 
        self.channel_num = len(G_list[0][1])
        
        #self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G[1][0].number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        ### max element sequence length (SZ add)
        self.max_ele_seq_lengths = max([max([se_eb.shape[0] for se_eb in G[0]]) for G in G_list])
        print('max sequence element length: %d'%self.max_ele_seq_lengths)
        if max_SSE:
            self.padding_length = max_SSE
        else:
            self.padding_length = self.max_ele_seq_lengths
        print('Padding to the length of %d.'%self.padding_length)

        self.feat_dim = node_dict(G_list[0][1][0])[0]['feat'].shape[0]

        for sample in G_list:
            ### adjacency matrices ###
            G_all = sample[1]
 
            node_num = G_all[0].number_of_nodes()
            if node_num > self.max_num_nodes:
                continue # ignore the graphs whose amount of nodes are larger than max_nodes
 
            adj_tensor = []
            for G in G_all:
                adj = np.array(nx.to_numpy_matrix(G))
                if normalize and adj.shape[-1] > 1:
                    #sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                    sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj + np.eye(adj.shape[-1]), axis=-1, dtype=float).squeeze())) #
                    #sqrt_deg = np.array([np.diag(v) for v in sqrt_deg])
                    adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
                adj_tensor.append(adj)
            adj_tensor = np.array(adj_tensor)
            G = G_all[0]

            ### sequence ###
            g_seq = sample[0]
            node_num = len(g_seq)
            seq_emb = np.zeros((self.max_num_nodes, self.padding_length, seq_mask_dim), dtype=float)
            seq_mask = np.zeros((self.max_num_nodes,1), dtype=float)

            for i in range(node_num):
                seq_mask[i] = 1
                seq_len = g_seq[i].shape[0]
                seq_emb[i,:seq_len,:20] = g_seq[i]
                seq_emb[i,seq_len:,20] = 1
 
            self.seq.append(seq_emb)
            self.seq_mask_all.append(seq_mask)

            ### add the info into the lists
            self.adj_all.append(adj_tensor)
            self.len_all.append(node_num)
            
            # feat matrix: max_num_nodes x feat_dim
            f = np.zeros((self.channel_num, self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[:,i,:] = node_dict(G)[u]['feat']
            self.feature_all.append(f)   

            wgt = np.zeros(self.max_num_nodes)
            if sample[2] is not None:
                wgt[:node_num] = sample[2]

            self.weights.append(wgt)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        adj_dim = adj.shape[0]
        num_nodes = adj.shape[1]
        adj_padded = np.zeros((adj_dim, self.max_num_nodes, self.max_num_nodes))
        adj_padded[:, :num_nodes, :num_nodes] = adj

        return {'seq':self.seq[idx].copy(),
                'seq_mask':self.seq_mask_all[idx].copy(),
                'adj':adj_padded,
                'feats':self.feature_all[idx].copy(),
                'num_nodes': num_nodes,
                'weights': self.weights[idx].copy()}


