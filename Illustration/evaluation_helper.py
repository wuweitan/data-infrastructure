import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

import argparse
import os
import time
import random
import pickle

import torch.nn.functional as F
#from tqdm import tqdm

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

matrix_blosum62 = matlist.blosum62

############################## Accessory Functions #######################################

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

def cdr_seq_recover(cdr_groundtruth, cdr_mask):
    """
    Recover the cdr vectors to sequences.
    """
    AA_dict = {0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'Q', 6:'E', 7:'G', 8:'H', 9:'I', 10:'L',
               11:'K', 12:'M', 13:'F', 14:'P', 15:'S', 16:'T', 17:'W', 18:'Y', 19:'V'}
    cdr_seq_true = []
    cdr_shape = cdr_groundtruth.shape
    cdr_mask = cdr_mask.reshape(cdr_shape)
    cdr_groundtruth = (1 + cdr_groundtruth) * cdr_mask - 1
    cdr_groundtruth = cdr_groundtruth.numpy()
    for i,cdr_batch in enumerate(cdr_groundtruth):
        cdr_seq_true.append([])
        for seq_vec in cdr_batch:
            seq = ''
            for char in seq_vec:
                #print(char)
                if char == -1:
                    break
                else:
                    seq += AA_dict[int(char)]
            cdr_seq_true[i].append(seq) 
    return cdr_seq_true

############################## Data Loader #######################################
    
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

class DataSampler(torch.utils.data.Dataset):
    ''' 
    Load Data for testing: dirctly load the samples, weights included
    '''
    def __init__(self, data_dict, debug = False, sele_num = 100):
        '''
        Data_dictionary: set -> cluster -> sample -> features                
        '''
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
        self.cluster_all = []

        sample_idx = 0
        for cluster in data_dict.keys():
            for sample in data_dict[cluster].keys():
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
                self.cluster_all.append(cluster)
                sample_idx += 1
                if debug and sample_idx >= sele_num:
                    break
            if debug and sample_idx >= sele_num:
                break

        print('%d sample loaded.'%sample_idx)

    def __len__(self):
        return len(self.seq_ag_all)

    def __getitem__(self, sample_idx):
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
                'weight': self.weight_all[sample_idx],
                'cluster': self.cluster_all[sample_idx]}

############################## Evaluation Functions #######################################

def cross_entropy(profile, target, cdr_mask, reduction = 'none'):
    """
    profile: output before the softmax layer, batch_size x 6 (max_cdr_num) x max_cdr_len x 20 (output dim)
    target: groudtruth, batch_size x 6 (max_cdr_num) x max_cdr_len
    cdr_mask: tensor of bullin points, batch_size x 6 x max_cdr_len x 1
    reduction: reduction method of cross_entropy
    """
    output_size = profile.shape[-1]
    profile = profile[cdr_mask.repeat(1,1,1,output_size)].reshape(-1,output_size)
    target = target[cdr_mask.squeeze()].reshape(-1)
    ce = F.cross_entropy(profile, target, reduction = reduction)
    return ce

def sequence_indentity(seq_1, seq_2, version = 'BLAST', matrix = matrix_blosum62):
    '''Calculate the identity between two sequences
    :param seq_1, seq_2: protein sequences
    :type seq_1, seq_2: str
    :param version: squence identity version
    :type version: str, optional
    :return: sequence identity
    :rtype: float
    '''
    l_x = len(seq_1)
    l_y = len(seq_2)
    X = seq_1.upper()
    Y = seq_2.upper()
    if version == 'BLAST':
        alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
        max_iden = 0
        for i in alignments:
            same = 0
            for j in range(i[-1]):
                if i[0][j] == i[1][j] and i[0][j] != '-':
                    same += 1
            iden = float(same)/float(i[-1])
            if iden > max_iden:
                max_iden = iden
        identity = max_iden
    elif version == 'Gap_exclude':
        l = min(l_x,l_y)
        alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
        max_same = 0
        for i in alignments:
            same = 0
            for j in range(i[-1]):
                if i[0][j] == i[1][j] and i[0][j] != '-':
                    same += 1
            if same > max_same:
                max_same = same
        identity = float(max_same)/float(l)
    else:
        print('Error! No sequence identity version named %s!'%version)
    return identity

def model_evaluation(model, data_path, model_path = None, title = 'Test', set_kind = 'Test', epoch_list = [1],
                     batch_size = 16, num_workers = 1,
                     seq_path = None, result_path = None,
                     USE_CUDA = True, Multi_GPU = False, with_NonCDR = True, with_AgSeq = False, debug = False):

    if seq_path is not None:
        if not seq_path.endswith('/'):
            seq_path += '/'
        seq_dict_path = seq_path + 'Seq_%s_%s_dict.pickle'%(title, set_kind)
        seq_file_path = seq_path + 'Seq_%s_%s.txt'%(title,set_kind)
        if os.path.exists(seq_dict_path):
            seq_dict = dict_load(seq_dict_path)
        else:
            seq_dict = {}
    else:
        seq_dict = {}

    if result_path is not None:
        if not result_path.endswith('/'):
            result_path += '/'
        result_dict_path = result_path + 'Result_%s_%s_dict.pickle'%(title,set_kind)
        result_file_path = result_path + 'Result_%s_%s.txt'%(title,set_kind)
        if os.path.exists(result_dict_path):
            result_dict = dict_load(result_dict_path)
        else:
            result_dict = {}
    else:
        result_dict = {}
            
    print('Data loading...')

    iteration_idx = 0
    data_dict = dict_load(data_path)
    #train_set = ClusterSampler(data_dict, debug = debug)
    train_set = DataSampler(data_dict, debug = debug) 
    dataloader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    if debug:
        print('Debugging...')
    else:
        print('Evaluating...')

    if USE_CUDA:
        model = model.cuda()
        #if Multi_GPU:
        #    model = nn.DataParallel(model)

    for epoch in epoch_list:
        print('Epoch %d:'%(epoch))

        if (epoch in result_dict.keys()):
            print('Processed before.')
            continue
        else:
            seq_dict[epoch] = {}
            result_dict[epoch] = {}

        ### Load the model

        if model_path is not None:
            checkpoint_path = model_path + 'model_%d.pickle'%epoch
            if not os.path.exists(checkpoint_path):
                print('Checkpoint for Epoch %d does not exist.'%epoch)
                continue
            model.load_state_dict(torch.load(checkpoint_path))
            print('Model successfully loaded.')  

        ### evaluation  

        model.eval()
        ce_all = []
        ce_list_all = []
        ppl_all = []
        iden_dict = {}
        iden_all = []
        iden_batch_all = []

        start_time = time.time()

        ### whether record the sequences ###
        if (seq_path is not None): # record sequence 
            with open(seq_file_path,'a') as eval_f:
                eval_f.write('Epoch %d:\n'%(epoch))

        ### iterations ###
        for batch_idx,data in enumerate(dataloader):
            iteration_idx += 1
            ### load the batch data ###
            seq_ag = Variable(data['seq_ag_onehot'].float(), requires_grad=False)
            if with_NonCDR:
                seq_ab = Variable(data['seq_ab_onehot_noCDR'].float(), requires_grad=False)
            else:
                seq_ab = Variable(data['seq_ab_onehot_masked'].float(), requires_grad=False)
            feat = Variable(data['feat'].float(), requires_grad=False)
            adj = Variable(data['adj'].float(), requires_grad=False)
            cdr_groundtruth = data['cdr_groundtruth']
            
            ag_indexes = [[],[],[]]
            ab_indexes = [[],[],[]]
            epi_size = data['epitope_size'].numpy()
            para_size = data['paratope_size'].numpy()
            batch_size = epi_size.shape[0]
            for i in range(batch_size):
                ## antigen
                ag_indexes[0] += [i] * epi_size[i]
                ag_indexes[1] += list(data['ag_indexes'][i,0,:epi_size[i]].numpy())
                ag_indexes[2] += list(data['ag_indexes'][i,1,:epi_size[i]].numpy())
                ## antigen
                ab_indexes[0] += [i] * para_size[i]
                ab_indexes[1] += list(data['ab_indexes'][i,0,:para_size[i]].numpy())
                ab_indexes[2] += list(data['ab_indexes'][i,1,:para_size[i]].numpy())

            graph_idx_mask = Variable(data['graph_idx_mask'], requires_grad=False)
            cdr_mask = Variable(data['cdr_mask'], requires_grad=False)
            seq_len_ab = data['seq_len_ab']
            seq_len_ag = data['seq_len_ag']

            seq_sele_ag = [[],[]]
            for i,len_ag_batch in enumerate(seq_len_ag):
                for j,len_ag_chain in enumerate(len_ag_batch):
                    if len_ag_chain != 0:
                        seq_sele_ag[0].append(i)
                        seq_sele_ag[1].append(j)
            seq_sele_ab = [[],[]]
            for i,len_ab_batch in enumerate(seq_len_ab):
                for j,len_ab_chain in enumerate(len_ab_batch):
                    if len_ab_chain != 0:
                        seq_sele_ab[0].append(i)
                        seq_sele_ab[1].append(j)

            if not with_AgSeq:
                seq_ag = seq_ag[ag_indexes]
            
            if USE_CUDA:
                seq_ag = seq_ag.cuda()
                seq_ab = seq_ab.cuda()
                feat = feat.cuda()
                adj = adj.cuda()
                #cdr_groundtruth = cdr_groundtruth.cuda()

            cluster = list(data['cluster'].numpy())

            ### forward ###

            profile, cdrs, seq_ab = model(seq_ag, seq_ab, feat, adj,
                                          graph_idx_mask, seq_sele_ag, ag_indexes, seq_sele_ab, ab_indexes, cdr_mask, seq_len_ab, seq_len_ag)
            print('CDR seq:', cdrs)            
            profile = profile.detach().cpu()
             
            ### evaluation metric calculation ###    
            ## cross_entropy & perplexity
            ce_list = cross_entropy(profile, cdr_groundtruth, cdr_mask, reduction = 'none')
           
            ce_list_all += list(ce_list)
            if len(ce_list) > 0:
                ce = torch.mean(ce_list)   # cross-entropy
                ce_all.append(float(ce))              
                ppl = torch.exp(ce)        # perplexity
                ppl_all.append(float(ppl))
            else:
                ce = np.nan
                ppl = np.nan

            ## sequence identity
            cdr_groundtruth = cdr_groundtruth.cpu()
            cdr_seq_true = cdr_seq_recover(cdr_groundtruth, cdr_mask)
            seq_iden_list = []
            for i,seq_batch in enumerate(cdr_seq_true):
                clus = cluster[i]
                if not clus in iden_dict.keys():
                    iden_dict[clus] = []
                for j,c_seq in enumerate(seq_batch): 
                    if c_seq != '':
                        iden = sequence_indentity(cdrs[i][j], c_seq)
                        iden_dict[clus].append(iden)
                        seq_iden_list.append(iden)

            iden_all += seq_iden_list
            iden_ave_temp = np.mean(seq_iden_list)
            iden_batch_all.append(iden_ave_temp)

            print('Cross_entropy: %.4f'%float(ce), '\tPerplexity: %.4f'%float(ppl), '\tAAR: %.4f'%float(iden_ave_temp))

            ### record the sequences ###
            if seq_path is not None:
                with open(seq_file_path,'a') as eval_f:
                    for seq_batch in cdrs:
                        eval_f.write('HCDR_1:%s\tHCDR_2:%s\tHCDR_3:%s\n'%(seq_batch[0],seq_batch[1], seq_batch[2]))
                        eval_f.write('LCDR_1:%s\tLCDR_2:%s\tLCDR_3:%s\n'%(seq_batch[3],seq_batch[4], seq_batch[5]))
            ### record the results ###
            if result_path is not None:
                with open(result_file_path,'a') as eval_f:
                    eval_f.write('%.4f\t%.4f\n'%(float(ce), float(ppl)))

            ### iteration end  ###

        ### record the loss ###
        aver_ce = np.mean(ce_list_all)
        aver_ce_batch = np.mean(ce_all)
        aver_ppl = np.exp(aver_ce)
        aver_ppl_batch = np.mean(ppl_all)
        aver_iden = np.mean(iden_all)
        aver_iden_batch = np.mean(iden_batch_all) 
        end_time = time.time()

        print('Average-CE: %.4f  Average-PPL: %.4f'%(aver_ce, aver_ppl))
        print('Average-AAR: %.4f  Average-AAR (over batches): %.4f'%(aver_iden, aver_iden_batch))
        print('Evaluation time: %.4fs'%(end_time - start_time))

        result_dict[epoch]['CE_all_list'] = ce_list_all
        result_dict[epoch]['CE_aver'] = aver_ce
        result_dict[epoch]['CE_batch_aver_list'] = ce_all
        result_dict[epoch]['CE_batch_aver'] = aver_ce_batch
        result_dict[epoch]['PPL_aver'] = aver_ppl
        result_dict[epoch]['PPL_batch_aver_list'] = ppl_all
        result_dict[epoch]['PPL_batch_aver'] = aver_ppl_batch
        result_dict[epoch]['iden_all_dict'] = iden_dict
        result_dict[epoch]['iden_aver'] = aver_iden
        result_dict[epoch]['iden_batch_aver_list'] = iden_batch_all
        result_dict[epoch]['iden_batch_aver'] = aver_iden_batch

        if result_path is not None:
            with open(result_file_path,'a') as eval_f:
                #eval_f.write('Average-CE: %.4f  Average-PPL: %.4f  Evaluation time: %.4fs\n\n'%(aver_ce, aver_ppl, end_time - start_time))
                eval_f.write('Average-CE: %.4f  Average-PPL: %.4f\n'%(aver_ce, aver_ppl))
                eval_f.write('Average-AAR: %.4f  Average-AAR (over batches): %.4f\n'%(aver_iden, aver_iden_batch))
                eval_f.write('Evaluation time: %.4fs\n'%(end_time - start_time))

        ### save the sequences and results ### 
        if (not debug) and (seq_path is not None):
            _save = dict_save(seq_dict, seq_dict_path)
        if (not debug) and (result_path is not None):
            _save = dict_save(result_dict, result_dict_path)

    return seq_dict, result_dict

