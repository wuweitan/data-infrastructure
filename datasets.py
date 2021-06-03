import numpy as np
import scipy.sparse as sps
from utils_compound import read_graph
import requests
import json


class dataset_protComp_pair:
    def __init__(self, ds_name, download=False, root='./datasets/'):
        self.ds_name = ds_name
        self.root = root
        if download:
            self.download()

    def download(self): # download function
        raise NotImplementedError

    def get_seqAndSmile(self): # quiry protein sequence and ligand smile
        raise NotImplementedError

    def get_processedData(self): # quiry processed data
        raise NotImplementedError

    def protSeq_conversion(self, seq, len_max=1500): # convert protein sequence to tokenized sequence
        '''
        input: protein sequence
        output: protein tokenized sequence
        '''
        assert len(seq) <= len_max, 'Sequence length must less than ' + str(len_max) + '.'

        token_dict = {'_PAD':0, '_GO':1, '_EOS':2, '_UNK':3, 'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'Q': 9, 'E': 10, 'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17, 'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28}
        prot_x = np.zeros(len_max)
        for n, c in enumerate(seq):
            prot_x[n] = token_dict[c]
        return prot_x

    def compSmile_conversion(self, smile, len_max=100): # convert ligand smile to compound graph
        '''
        input: compound smile
        output: compound graph
        '''
        comp_x, comp_adj = read_graph(smile, len_max)
        assert not comp_x is None, 'This smile is not convertable.'
        return comp_x, comp_adj

    def rcsb_graphql_query(self, pdb, chain): # perform rcsb graphql query, detailed attributes if you need more, please see https://data.rcsb.org/data-attributes.html
        '''
        input: pdb id and chain
        output: alignment query information
        '''
        info = json.loads(requests.post("https://data.rcsb.org/graphql", json={"query": "query($instance_ids:String!){polymer_entity_instances(instance_ids:[$instance_ids]){polymer_entity{rcsb_polymer_entity_align{aligned_regions{ref_beg_seq_id}}}rcsb_polymer_entity_instance_container_identifiers{auth_to_entity_poly_seq_mapping}}}", "variables": {"instance_ids":pdb+'.'+chain}}).text)
        return info


class dataset_deepaffinity(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_deepaffinity, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='ic50_train'): # {ic50, ec50, ki, kd} * {train, test, er, ion_channel, gpcr, tyrosine_kinase}
        pass

    def get_processedData(self, subset='ic50_train'): # quiry processed data
        '''
        input: subset name, from {ic50, ec50, ki, kd} * {train, test, er, ion_channel, gpcr, tyrosine_kinase}
        output: processed data, prot_x, prot_len, comp_x, comp_adj, comp_len, aff
        '''
        subset1, subset2 = subset.split('_')
        root = self.root + subset1 + '/processed_data/'

        prot_x = np.load(root + 'prot_x_' + subset2 + '.npy')
        prot_len = np.load(root + 'prot_len_' + subset2 + '.npy')
        comp_x = np.load(root + 'comp_x_' + subset2 + '.npy').reshape((-1, 100, 43))
        comp_adj = sps.load_npz(root + 'comp_adj_' + subset2 + '.npz')
        comp_len = np.load(root + 'comp_len_' + subset2 + '.npy')
        aff = np.load(root + 'aff_' + subset2 + '.npy')

        return prot_x, prot_len, comp_x, comp_adj, comp_len, aff


class dataset_deeprelations(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_deeprelations, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='train'): # train, val, test, unseen_protein, unseen_compound, unseen_both
        pass

    def get_processedData(self, subset='train'): # quiry processed data
        '''
        input: subset name, from {train, val, test, unseenProt, unseenComp, unseenBoth}
        output: processed data, prot_x, prot_adj, prot_len, comp_x, comp_adj, comp_len, aff
        '''
        root = self.root + 'deeprelations/processed_data/'

        prot_x = np.load(root + 'prot_x_' + subset + '.npy')
        prot_x = self.prot_x.reshape((-1, 1000))
        prot_x = np.concatenate((self.prot_x, np.zeros((self.prot_x.shape[0], 500))), axis=1) # protein pad to length 1500

        prot_len = np.load(root + 'prot_len_' + subset + '.npy').squeeze()

        comp_x = np.load(root + 'comp_x_' + subset + '.npy')
        comp_x = np.concatenate((self.comp_x, np.zeros((self.comp_x.shape[0], 44, 43))), axis=1).astype(np.float32) # compound pad to length 100

        comp_adj = np.load(root + 'comp_adj_' + subset + '.npy')
        comp_adj = np.concatenate((self.comp_adj, np.zeros((self.comp_adj.shape[0], 44, 56))), axis=1)
        comp_adj = np.concatenate((self.comp_adj, np.zeros((self.comp_adj.shape[0], 100, 44))), axis=2).astype(np.float32)
        comp_adj[:, np.arange(100), np.arange(100)] = 1 # add self loop

        comp_len = np.load(root + 'comp_len_' + subset + '.npy').squeeze()

        aff = np.load(root + 'aff_' + subset + '.npy').squeeze().astype(np.float32)

        cont = np.load(root + 'cont_' + subset + '.npy').squeeze()
        cont = np.concatenate((self.cont, np.zeros((self.cont.shape[0], 500, 56))), axis=1)
        cont = np.concatenate((self.cont, np.zeros((self.comp_adj.shape[0], 1500, 44))), axis=2).astype(np.float32)

        prot_adj = sps.load_npz(root + 'prot_adj_' + subset + '.npz')

        return prot_x, prot_adj, prot_len, comp_x, comp_adj, comp_len, aff, comt


class dataset_platinum(dataset_protComp_pair):
    def __init__(self, ds_name, download=False, root='./datasets/'):
        super(dataset_platinum, self).__init__(ds_name, download, root)

    def download(self):
        pass

    def get_seqAndSmile(self, subset='ki'): # quiry raw data
        pass

    def get_processedData(self, subset='ki'): # quiry processed data
        '''
        input: subset name, from {train, val, test, unseenProt, unseenComp, unseenBoth}
        output: processed data, prot_x, prot_len, comp_x, comp_adj, comp_len, aff
        '''
        root = self.root + 'platinum/processed_data/'

        prot_x = np.load(root + 'prot_x_' + subset + '.npy')
        prot_len = np.load(root + 'prot_len_' + subset + '.npy')
        comp_x = np.load(root + 'comp_x_' + subset + '.npy').reshape((-1, 100, 43))
        comp_adj = sps.load_npz(root + 'comp_adj_' + subset + '.npz')
        comp_len = np.load(root + 'comp_len_' + subset + '.npy')
        aff = np.load(root + 'aff_' + subset + '.npy')

        return prot_x, prot_len, comp_x, comp_adj, comp_len, aff

