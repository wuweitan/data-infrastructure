import sys
import errno
import os
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import re
import string
import random

import networkx as nx # for graph similarity

import pdb_helper
import ProteinGraph_helper

sys.path.append("../DataDownLoad/")

from DataDownLoad import download_helper

import pickle
import torch

import torch_geometric
from scipy.sparse import csr_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import torch.utils.data
import random
from math import ceil

import os
from ProteinGraph_helper import to_nxGragh

#*******************************************************************

class SingleProteinInfo(ProteinGraph_helper.PDB_information):
    """
    Read a pdb file and extract the information for further process
    """
    def __init__(self, pdb_file, **kwargs):
        self.data_dir = pdb_file # path of pdb file

    def rcsb_graphql_query(self, pdb, chain): # perform rcsb graphql query, detailed attributes if you need more, please see https://data.rcsb.org/data-attributes.html
        '''
        input: pdb id and chain
        output: alignment query information
        '''
        info = json.loads(requests.post("https://data.rcsb.org/graphql",
                                        json={"query": "query($instance_ids:String!){polymer_entity_instances(instance_ids:[$instance_ids]){polymer_entity{rcsb_polymer_entity_align{aligned_regions{ref_beg_seq_id}}}rcsb_polymer_entity_instance_container_identifiers{auth_to_entity_poly_seq_mapping}}}",
                                        "variables": {"instance_ids":pdb+'.'+chain}}).text)
        return info


class ProteinGraph(ProteinGraph_helper.Protein_Graph):
    """Calculate the features for protein graphs construction
    element-wise / residue-wise / atom-wise

    :param pdb_file: path of the pdb file 
    :type pdb_file: string
    :param ss_ref: reference secondary structure sequence, defaults to None 
    :type ss_ref: string, optional    
    :param ss_ref: reference amino acids sequence, defaults to None 
    :type ss_ref: string, optional
    :param ss_kind: secondary structure version (3 or 8), default t o3
    :type ss_kind: int, optional
    :param treat_as_missing: whether take the residues with missing information as missing residues (or raise Errors), defaults to True 
    :type ss_ref: booling, optional
    """
    def __init__(self, pdb_file, ss_ref=None, sequence_ref=None, ss_kind=3, treat_as_missing=True, **kwargs):
        self.data_dir = pdb_file # path of pdb file
        self.pdb_info = SingleProteinInfo(pdb_file)

        self.protein_dict = self.pdb_infoprotein_dict
        self.index_dict = self.pdb_infoindex_dict

        self.ss_ref = ss_ref
        self.sequence_ref = sequence_ref
        self.ss_kind = ss_kind
        self.treat_as_missing = treat_as_missing

class Datasets(object):
    """Process on the benchmark datasets: SCOPe, Pfam, CATH, self-defined

    :param dataset: name of the database, defaults to "SCOPe" 
    :type dataset: string, optional
    :param path: path of the stored database
    :type path: version, optional
    :param path: version of the database
    :type path: string, optional
    :param structure: whether consider the structure samples, defaults to True 
    :type structure: booling, optional
    :param sequence: whether consider the sequence samples, defaults to True 
    :type sequence: booling, optional
    """
    def __init__(self, dataset = 'SCOPe', path='../dataset/', version = '2.07',
                 structure = True, sequence = True, **kwargs):
        """
        version: only of SCOPe
        """
        self.dataset = dataset
        if not path.endswith('/'):
            path += '/'
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        if dataset in ['SCOPe', 'CATH']:
            path = path + dataset + '/'
            if not os.path.exists(path):
                os.mkdir(path)
         
            if dataset == 'SCOPe':

                if structure:
                    with open(SCOPe_FILE,'r') as scope_file:
                        lines = scope_file.readlines()
                        length = len(lines)
                    for i in range(length):
                        line = lines[i]
                        if line[0] == '>':

                            seq_num += 1

                            if i != 0:
                                if pdb_id != 1:
                                    dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
                                    download_helper.fasta_write(title,seq,seq_file_name,record_type = 'w',line_length = 60)
                                    if dl_result == 0:
                                        stru_num += 1
                            title = line[1:].strip('\n')
                            pdb_id,fold,info = download_helper.read_SCOPe_title(title)

                            pdb_file_name = STRU_PATH + line[1:8] + '.pdb'
                            if seq:
                                seq_file_name = SEQ_PATH + line[1:8] + '.fasta'
                                seq = ''
                        elif seq:
                            seq += line.strip('\n')
                    if pdb_id != 1:
                        dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
                        if seq:
                            download_helper.fasta_write(title,seq,seq_file_name,record_type = 'w',line_length = 60)
                        if dl_result == 0:
                            stru_num += 1

                    if seq:
                        print('%d sequences in the list.'%seq_num)
                    print('%d structures have been downloaded'%stru_num)

            elif dataset == 'CATH':
                pass

        else:
            print('Processing a self-defined dataset in %s ...'%path)
            file_list = os.listdir(path)

    def data_info(self, data_structure = 'dict')
        '''
        Read the pdb information of each sample
        '''
        if data_structure = 'dict':
            self.info_all = {}
        else:
            self.info_all = []
        for pdb in self.pdb_list:
            file_path = ''
            if data_structure = 'dict':
                self.info_all[pdb] = SingleProteinInfo(file_path)
            else:
                self.info_all.append(SingleProteinInfo(file_path))

    def protein_graph(self, graph_type = '')
        '''
        Construct the graph descriptions of each sample
        '''
        if data_structure = 'dict':
            self.info_all = {}
        else:
            self.info_all = []
        for pdb in self.pdb_list:
            file_path = ''
            if data_structure = 'dict':
                self.info_all[pdb] = ProteinGraph(file_path)
            else:
                self.info_all.append(ProteinGraph(file_path))

 
    def seq_filter(self, seq_list, abnormal_char = ['X'])
        '''
        Discard the sequence with certain charactrers
        '''
        new_list = []
        for seq in seq_list: 
            fag = True
            for char in abnormal_char:
                if char in seq:
                    flag = False
                    break
            if flag:
                new_list.append(seq)
        return new_list

    def struct_cluster(self, **kwargs)
        pass


    def data_split(self, dataset, split_method = 'rough', ratio = [0.7, 0.15, 0.15], shuffle = True):
        '''
        Split the database into training, validation and test sets

        :param dataset: A processed dataset
        :type dataset: list
        :param split_method: data splitting method
        :type split_method: str, optional
        :param ratio: data splitting ratio
        :type ratio: list, optional
        :param shuffle: whether shuffle the original dataset
        :type shuffle: bool, optional
        :return: 3 lists of the split datasets
        :rtype: lists
        '''
        size = len(list(dataset))
        train_size = round(size * ratio[0])
        vali_size = round(size * ratio[1])
        test_size = size - train_size - vali_size
        if shuffle:
            numpy.random.shuffle(dataset)

        if split_method == 'rough':
            train_set = dataset[:train_size]
            vali_set = dataset[train_size : train_size + vali_size]
            test_set = dataset[train_size + vali_size :]
        elif split_method == 'cluster':
            pass
        elif plit_method == 'label':
            pass
        else:         
            print('Error! No splitting method named %s!'%split)
            return None, None, None
        return train_set, vali_set, test_set

    def Dataloader(self,seq_mask_dim = 21,features = ['len', 'ss', 'level'],
                batch_size=64, normalize=False, shuffle=False,max_SSE=None,max_nodes=60,num_workers=1,
                set_kind = 'training', ignore_missing = False):
        self.dataloader = Dataloading(self.Seq_file, self.X_file, self.A_file, 
                seq_mask_dim = 21,features = ['len', 'ss', 'level'], weight_dict = self.weight_dict,
                batch_size=64, normalize=False, shuffle=False,max_SSE=None,max_nodes=60,num_workers=1,
                set_kind = 'training', ignore_missing = False)

#******************************* DataLoader *******************************************

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

def Dataloading(Seq_file, X_file, A_file, seq_mask_dim = 21,features = ['len', 'ss', 'level'], weight_dict = None,
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
        self.DGL_Graph = []

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
            channel_num = adj_tensor.shape[0]
            version = 'Homogenous' if channel_num == 1 else 'Heterogenous'
            self.DGL_Graph.append(ProteinGraph_helper.to_dglGraph(adj_tensor, version = version, edge_type = [str(i) dor i in range(channel_num)]))

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
                'weights': self.weights[idx].copy(),
                'dgl_graph': dgl.batch(self.DGL_Graph[idx].copy())}
