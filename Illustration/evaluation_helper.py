import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F

import argparse
import os
import time
import random
import pickle

import shutil

import networks
import DataLoading

#from torch_geometric.data import DataLoader
#from torch_geometric.nn.inits import uniform
#from torch_geometric.nn import global_mean_pool
from copy import deepcopy

#from tqdm import tqdm

from Bio.PDB.DSSP import dssp_dict_from_pdb_file
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

######################################## Discriminative #######################################

def Disc_evaluate(dataset, model, batch_size=64, label=None, arrange_index=None, hierarchy_dict=None, max_num_examples=None, weight = None):
    model.eval()

    labels = []
    if label != 'hierarchy':
        preds = []
        weight_list = None
        if type(weight) != type(None):
            weight_list = []
    else:
        weight_list = [None, None, None, None]
        if type(weight) != type(None):
            weight_list = [[],[],[],[]]

        if not arrange_index:
            print('Error! No input of the arrange index!')
            return None
        if not hierarchy_dict:
            print('Error! No input of the hierarchy dictionary!')
            return None
        preds = {}
        softmax = nn.Softmax(dim=-1)
        label_index_dict = {'family':0, 'super-family':1, 'fold':2, 'class':3}

    for batch_idx, data in enumerate(dataset):
        seq = Variable(data['seq'].float(), requires_grad=False)#.cuda()
        seq_mask = Variable(data['seq_mask'].float(), requires_grad=False)#.cuda()
        adj = Variable(data['adj'].float(), requires_grad=False)#.cuda()
        h0 = Variable(data['feats'].float())#.cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()

        #print(labels)

        ypred = model(h0, adj, seq, seq_mask)

        if label != 'hierarchy':
            labels.append(data['label'].long().numpy())
            if type(weight) != type(None):
                weight_list += [float(weight[int(i) - 1]) for i in data['label']]

            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
        else:
            ypred_fam = softmax(ypred)
            ypred_sf = hierarchy_arrange(ypred_fam, arrange_index[0])
            ypred_fold = hierarchy_arrange(ypred_fam, arrange_index[1])
            ypred_class = hierarchy_arrange(ypred_fam, arrange_index[2])

            _, fam_indices = torch.max(ypred_fam, 1)
            _, sf_indices = torch.max(ypred_sf, 1)
            _, fold_indices = torch.max(ypred_fold, 1)
            _, class_indices = torch.max(ypred_class, 1)

            if batch_idx == 0:
                labels = data['label'].long().numpy()
                preds['from_softmax'] = np.array([np.array(fam_indices.cpu()), np.array(sf_indices.cpu()), np.array(fold_indices.cpu()), np.array(class_indices.cpu())])
                preds['from_family'] = []
            else:
                labels = np.vstack([labels,data['label'].long().numpy()])
                preds['from_softmax'] = np.hstack([preds['from_softmax'],
                                        np.array([np.array(fam_indices.cpu()), np.array(sf_indices.cpu()), np.array(fold_indices.cpu()), np.array(class_indices.cpu())])])
            for fam_idx in fam_indices:
                preds['from_family'].append(hierarchy_dict[int(fam_idx) + 1])

            if type(weight) != type(None):
                weight_list[0] += [float(weight[0][int(i) - 1]) for i in data['label'][:,0]]
                weight_list[1] += [float(weight[1][int(i) - 1]) for i in data['label'][:,1]]
                weight_list[2] += [float(weight[2][int(i) - 1]) for i in data['label'][:,2]]
                weight_list[3] += [float(weight[3][int(i) - 1]) for i in data['label'][:,3]]

        if max_num_examples is not None:
            if (batch_idx+1) * batch_size > max_num_examples:
                break

    if type(weight) != type(None):
        weight_list = np.array(weight_list)

    if label != 'hierarchy':
        labels = np.hstack(labels)
        preds = np.hstack(preds)

        result = {#'prec': metrics.precision_score(labels, preds, average='macro', sample_weight = weight_list),
                  #'recall': metrics.recall_score(labels, preds, average='macro', sample_weight = weight_list),
                  'acc': metrics.accuracy_score(labels, preds, sample_weight = weight_list)}
                  #'F1': metrics.f1_score(labels, preds, average="micro", sample_weight = weight_list)}
    else:
        preds['from_family'] = np.array(preds['from_family']).T

        result = {'from_softmax':{},'from_family':{}}

        result['from_softmax']['family'] = {#'prec': metrics.precision_score(labels[:,0], preds['from_softmax'][0], average='macro', 
                                            #                                sample_weight = weight_list[label_index_dict['family']]),
                                            #'recall': metrics.recall_score(labels[:,0], preds['from_softmax'][0], average='macro',
                                            #                               sample_weight = weight_list[label_index_dict['family']]),
                                            'acc': metrics.accuracy_score(labels[:,0], preds['from_softmax'][0],
                                                                          sample_weight = weight_list[label_index_dict['family']])}
                                            #'F1': metrics.f1_score(labels[:,0], preds['from_softmax'][0], average="micro",
                                            #                       sample_weight = weight_list[label_index_dict['family']])}
        result['from_family']['family'] = result['from_softmax']['family']
        for pred_kind in ['from_softmax','from_family']:
            for level in ['super-family','fold','class']:
                result[pred_kind][level] = {#'prec': metrics.precision_score(labels[:,label_index_dict[level]], 
                                            #                                preds[pred_kind][label_index_dict[level]], average='macro',
                                            #                                sample_weight = weight_list[label_index_dict[level]]),
                                            #'recall': metrics.recall_score(labels[:,label_index_dict[level]], 
                                            #                               preds[pred_kind][label_index_dict[level]], average='macro',
                                            #                               sample_weight = weight_list[label_index_dict[level]]),
                                            'acc': metrics.accuracy_score(labels[:,label_index_dict[level]],
                                                                          preds[pred_kind][label_index_dict[level]],
                                                                          sample_weight = weight_list[label_index_dict[level]])}
                                            #'F1': metrics.f1_score(labels[:,label_index_dict[level]], 
                                            #                       preds[pred_kind][label_index_dict[level]], average='macro',
                                            #                       sample_weight = weight_list[label_index_dict[level]])} 
    return result

def accu_print(result, label, kind, epoch, best_result):
    if label != 'hierarchy':
        if result['acc'] > best_result['acc']:
            best_result['acc'] = result['acc']
            best_result['epoch'] = epoch
        print('%s accuracy: %.4f'%(kind,result['acc']))
    else:
        if result['from_softmax']['family']['acc'] > best_result['acc']:
            best_result['acc'] = result['from_softmax']['family']['acc']
            best_result['epoch'] = epoch
        for pred_kind in ['from_softmax','from_family']:
            text = '%s accuracy (%s):'%(kind, pred_kind)
            for level in ['family','super-family','fold','class']:
                text += '%s: %.4f '%(level,result[pred_kind][level]['acc'])
            print(text)
    return best_result

def accu_save(result, label, kind, epoch, total_time, all_time, accu_path):
    accu_file = open(accu_path,'a')
    if label != 'hierarchy':
        accu_file.write('{val:<5}\t'.format(val = epoch))
        accu_file.write('{val:<10}\t'.format(val = kind))
        accu_file.write('{val:<25}\t'.format(val = result['acc']))
        accu_file.write('{val:<25}\t'.format(val = total_time))
        accu_file.write('{val:<25}\n'.format(val = all_time))
    else:
        for pred_kind in ['from_softmax','from_family']:
            accu_file.write('{val:<5}\t'.format(val = epoch))
            accu_file.write('{val:<25}\t'.format(val = kind + '(%s)'%pred_kind))
            for level in ['family','super-family','fold','class']:
                accu_file.write('{val:<25}\t'.format(val = result[pred_kind][level]['acc']))
            accu_file.write('{val:<25}\t'.format(val = total_time))
            accu_file.write('{val:<25}\n'.format(val = all_time))
    accu_file.close()
    return 0

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

############################## Generative Model #################################################

def perplexity(output, target):
    return torch.exp(F.cross_entropy(output, target, reduction = 'none'))

def Gen_evaluation(model, dataset, temperature = 1.0, seq_len = 35, MAX_SAMPLE = 'top-k', k = 3):

    start_time = time.time()

    model.eval()

    seq_all = []
    seq_complete_all = []
    iden_all = []
    ppl_all = []

    for batch_idx, data in enumerate(dataset):
        ### load the batch data ###
        seq_true = Variable(data['seq'].float(), requires_grad=False)#.cuda()
        seq_mask = Variable(data['seq_mask'].float(), requires_grad=False)#.cuda()
        adj = Variable(data['adj'].float(), requires_grad=False)#.cuda()

        h0 = Variable(data['feats'].float())#.cuda()
        batch_num_nodes = data['num_nodes'].int().numpy()

        target = []
        seq_nature = []
        for i,seq_tr in enumerate(seq_true):
            target += list(torch.max(seq_tr[:batch_num_nodes[i]], -1)[1].numpy().reshape(-1))
            seq_nature += seq_recover(seq_tr[:batch_num_nodes[i]])
        target = torch.Tensor(target).long()

        #if balance:
        #    batch_size = batch_num_nodes.shape[0]
        #    ele_weight = torch.cat([data['weights'][i][:batch_num_nodes[i]] for i in range(batch_size)]).float()#.cuda()
        #    ele_weight = ele_weight / ele_weight.shape[0]
        #else:
        #    ele_weight = None

        ### forward ###

        out, seq, seq_complete = model.generator(h0, adj, batch_num_nodes, seq_mask, n_steps = seq_len, temperature = temperature, MAX_SAMPLE = MAX_SAMPLE, k=k)
        seq_all += seq
        seq_complete_all += seq_complete

        ### evaluation

        ppl = perplexity(out.reshape(-1, out.shape[-1]), target)
        ppl_all += list(ppl)

        i = 0
        for prot in seq:
            for ele_s in prot:
                ele_s = ele_s.split('!')[0]
                if len(ele_s) != 0:
                    iden_all.append(sequence_indentity(ele_s, seq_nature[i]))
                i += 1

    return seq_all, seq_complete_all, iden_all, ppl_all

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

