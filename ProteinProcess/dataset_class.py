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
    """
    Calculate the features for protein graphs construction
    element-wise / residue-wise / atom-wise
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
    """
    Process on the benchmark datasets: SCOPe, Pfam, CATH, self-defined
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


