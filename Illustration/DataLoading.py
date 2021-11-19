#######################################################################################
#  SZhu add
#  help to load the data into the correct format
#######################################################################################

import numpy as np
import pickle
import torch

#import torch_geometric
from scipy.sparse import csr_matrix
#from torch_geometric.utils.convert import from_scipy_sparse_matrix

import networkx as nx
import torch.utils.data
import random
from math import ceil

import os
import pickle
from google_drive_downloader import GoogleDriveDownloader as gdd

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

def to_nxGragh(X,A,Y,heter=False):
    adj_dim = A.shape[0]
    G_all = []
    for idx in range(adj_dim):
        G = nx.Graph()
        if type(Y) == np.ndarray: # hierarchical label
            G.graph['label'] = Y.astype(int)
        else:
            G.graph['label'] = int(Y)
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

def Dataloader(database = 'SCOPe_debug', path = '../Datasets/SCOPe/', task = 'Discriminative', batch_size = 16, seq_mask_dim = 21, 
               normalize=False, shuffle=[True,False,False]):

    if not database in ['SCOPe_debug']:
        print('Error! No database named %s!'%database)
        return None
    else:
        if not path.endswith('/'):
            path += '/'
        zip_path = path + '%s.zip'%database
        #download database
        if not os.path.exists(zip_path):
            print('Downloading the database...')
            if database == 'SCOPe_debug':
                file_id = '1BFsBdQzLiRKmc1lDOZBwREcCnfg4EiRU'
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=zip_path, unzip=True)
        else:
            print('The database %s has already been downloaded.'%database)
        # load data
        task_title = task[:3] 
        datasets = {}
        stat_dict = {}
        print()
        print('Database: %s'%database)
        print('Task: %s'%task)
        print('Shuffle: %s'%' '.join([str(char) for char in shuffle]))
        for j,sk in enumerate(['training', 'validation','test']):
            if database == 'SCOPe_debug':
                Seq_file = path + '%s_Seq_TOPS+-RCSB_%s_debug.list'%(task_title,sk)
                X_file = path + '%s_X_TOPS+-RCSB_%s_debug.list'%(task_title,sk)
                A_file = path + '%s_Adj_TOPS+-RCSB_dire_hbond_%s_debug.list'%(task_title,sk)
                Y_file = None
                data_loader, length, class_num = dataloading(Seq_file, X_file, A_file, Y_file, label = None, 
                                                             seq_mask_dim = 21, batch_size=batch_size, normalize=normalize, shuffle=shuffle[j]) 
                datasets[sk] = data_loader
                stat_dict[sk] = length
        for sk in ['training', 'validation','test']:
            print('%s: %d samples'%(sk, stat_dict[sk]))
        print('Batch size: %d'%batch_size)
        return datasets['training'], datasets['validation'], datasets['test']


def dataloading(Seq_file, X_file, A_file, Y_file,label = None, seq_mask_dim = 21,features = ['len', 'ss', 'level'],  
                batch_size=64,normalize=False,shuffle=False,max_SSE=35,max_nodes=60,num_workers=1,test_num=None):
    """
    Load the data from the files.
    Seq_file: file of the embedded sequences
    X_file: file of the node features
    A_file: file of the adjacency tensors
    Y_file: file of the labeld

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

    if Y_file is None:
        Y_all = np.random.choice(np.arange(1080), size = 100, replace = True)
    else:
        Y_all = np.loadtxt(Y_file)

    if label == 'hierarchy':
        class_num = int(max(Y_all[:,-1]))
    else:
        class_num = int(max(Y_all))
    feature_num = int(X_all[0].shape[-1])
    length = len(X_all)
    if len(A_all[0].shape) == 2:
        channel_num = 1
        add_dim = True
    elif len(A_all[0].shape) == 3:
        channel_num = A_all[0].shape[0]
        add_dim = False
    else:
        print('Error! Dimension of the adjacency matrix/tensor should be 2 or 3, not %d!'%len(A_all[0].shape))
        return 1

    if (len(A_all) != length) or (Y_all.shape[0] != length):
        print('Error! The sizes of the data do not match! (X:%d, A:%d, Y:%d)'%(X_all.shape[0],A_all.shape[0],Y_all.shape[0]))
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
            #print('Loading the original %d samples...'%length)
            select_idx = range(length)

        for i in select_idx:
            S = S_all[i] 
            SSE_len_max = max([seq_ele.shape[0] for seq_ele in S])
            if max_SSE and SSE_len_max > max_SSE:
                continue # ignore the proteins with too long sequence elements
 
            X = X_all[i]
            Y = Y_all[i]
            A = A_all[i]   

            if A.shape[-1] > max_nodes:
                continue # ignore the graphs whose amount of nodes are larger than max_nodes
     
            if add_dim:
                A = np.expand_dims(A,axis=0) 
    
            graph = to_nxGragh(X,A,Y)
            Data.append((S,graph))
            #Data.append((S,graph,W))

            sample_len = len(Data)

        dataset_sampler = GraphSampler(Data, normalize=normalize, seq_mask_dim = seq_mask_dim, max_num_nodes=max_nodes, max_SSE=max_SSE)
        #print('Graph sampling completed. %d graphs sampled.'%(sample_len))
        #print('Size of the sampler: %d'%(dataset_sampler.__sizeof__()))
        dataset_loader = torch.utils.data.DataLoader(dataset_sampler,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     num_workers=num_workers)
        return dataset_loader, sample_len, class_num

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
        #self.weights = [] 
        self.channel_num = len(G_list[0][1])
        
        #self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G[1][0].number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        ### max element sequence length (SZ add)
        self.max_ele_seq_lengths = max([max([se_eb.shape[0] for se_eb in G[0]]) for G in G_list])
        #print('max sequence element length: %d'%self.max_ele_seq_lengths)
        if max_SSE:
            self.padding_length = max_SSE
        else:
            self.padding_length = self.max_ele_seq_lengths
        #print('Padding to the length of %d.'%self.padding_length)

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
            self.label_all.append(G.graph['label'] - 1) # May 23rd
            
            # feat matrix: max_num_nodes x feat_dim
            f = np.zeros((self.channel_num, self.max_num_nodes, self.feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[:,i,:] = node_dict(G)[u]['feat']
            self.feature_all.append(f)   

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
                'label':self.label_all[idx],
                'num_nodes': num_nodes}

######################## For antibody design ######################################

class ClusterSampler(torch.utils.data.Dataset):
    '''
    Load Data for training: one sample for each cluster
    '''
    def __init__(self, data_dict, debug = False, sele_num = 100):
        '''
        Data_dictionary: set -> cluster -> sample -> features
        '''
        self.clusters_list = list(data_dict.keys())
        if debug:
            cluster_num = sele_num
            self.clusters_list = self.clusters_list[:cluster_num]
            #print('Debugging...')
            #print('%s clusters loaded...'%cluster_num)
        else:
            cluster_num = len(self.clusters_list)
            #print('Training...')
            #print('%s clusters loaded...'%cluster_num)

        self.seq_ag_all = []
        self.seq_ab_all = []
        self.seq_ab_noCDR = []
        self.seq_ab_mask = []
        self.cdr_ground_all = []
        self.feat_all = []
        self.adj_all  = []= []
        self.graph_idx_mask_all = []
        self.ag_indexes_all = []
        self.ab_indexes_all = []
        self.cdr_mask_all = []
        self.seq_len_ab_all = []
        self.seq_len_ag_all = []
        self.epi_size_all = []
        self.para_size_all = []
        self.weight_all = []

        sample_idx = 0
        self.idx_dict = {}
        for cluster in self.clusters_list:
            self.idx_dict[cluster] = []
            for sample in data_dict[cluster].keys():
                self.idx_dict[cluster].append(sample_idx)
                sample_idx += 1

                self.seq_ag_all.append(data_dict[cluster][sample]['seq_ag_onehot'])
                self.seq_ab_all.append(data_dict[cluster][sample]['seq_ab_onehot'])
                self.seq_ab_noCDR.append(data_dict[cluster][sample]['seq_ab_onehot_noCDR'])
                self.seq_ab_mask.append(data_dict[cluster][sample]['seq_ab_onehot_masked'])
                self.cdr_ground_all.append(data_dict[cluster][sample]['cdr_groundtruth'])
                self.feat_all.append(data_dict[cluster][sample]['feat'])
                self.adj_all.append(data_dict[cluster][sample]['adj'])
                self.graph_idx_mask_all.append(data_dict[cluster][sample]['graph_idx_mask'])
                self.ag_indexes_all.append(data_dict[cluster][sample]['ag_indexes'])
                self.ab_indexes_all.append(data_dict[cluster][sample]['ab_indexes'])
                self.cdr_mask_all.append(data_dict[cluster][sample]['cdr_mask'])
                self.seq_len_ab_all.append(data_dict[cluster][sample]['seq_len_ab'])
                self.seq_len_ag_all.append(data_dict[cluster][sample]['seq_len_ag'])
                self.epi_size_all.append(data_dict[cluster][sample]['epitope_size'])
                self.para_size_all.append(data_dict[cluster][sample]['paratope_size'])
                self.weight_all.append(data_dict[cluster][sample]['weight'])

        print('%d clusters and %d samples loaded.'%(cluster_num, sample_idx))

    def __len__(self):
        return len(self.clusters_list)

    def __getitem__(self, idx):
        cluster_sele = self.clusters_list[idx]
        sample_idx = np.random.choice(self.idx_dict[cluster_sele])

        return {'seq_ag_onehot': self.seq_ag_all[sample_idx].copy(),
                'seq_ab_onehot': self.seq_ab_all[sample_idx].copy(),
                'seq_ab_onehot_noCDR': self.seq_ab_noCDR[sample_idx].copy(),
                'seq_ab_onehot_masked': self.seq_ab_mask[sample_idx].copy(),
                'cdr_groundtruth': self.cdr_ground_all[sample_idx].copy(),
                'feat': self.feat_all[sample_idx].copy(),
                'adj': self.adj_all[sample_idx].copy(),
                'graph_idx_mask': self.graph_idx_mask_all[sample_idx].copy(),
                'ag_indexes': self.ag_indexes_all[sample_idx].copy(),
                'ab_indexes': self.ab_indexes_all[sample_idx].copy(),
                'cdr_mask': self.cdr_mask_all[sample_idx].copy(),
                'seq_len_ab': self.seq_len_ab_all[sample_idx].copy(),
                'seq_len_ag': self.seq_len_ag_all[sample_idx].copy(),
                'epitope_size': self.epi_size_all[sample_idx],
                'paratope_size': self.para_size_all[sample_idx],
                'weight': self.weight_all[sample_idx]}
