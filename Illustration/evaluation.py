import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from tensorboardX import SummaryWriter

import argparse
import os

import shutil
import time

import graph_networks
import DataLoading 

from tqdm import tqdm
#from torch_geometric.data import DataLoader
#from torch_geometric.nn.inits import uniform
#from torch_geometric.nn import global_mean_pool
from copy import deepcopy

from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add

matrix = matlist.blosum62

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

######################################## Generative #######################################

def seq_recover(tensor_list):
    """
    recover the tensor to element-wise strings
    input: list of tensors or arrays
    """
    AA_dict = {0:'A', 1:'R', 2:'N', 3:'D', 4:'C', 5:'Q', 6:'E', 7:'G', 8:'H', 9:'I', 10:'L',
               11:'K', 12:'M', 13:'F', 14:'P', 15:'S', 16:'T', 17:'W', 18:'Y', 19:'V'}

    seq_ele = []
    for node in tensor_list:
        node = np.argmax(node, axis = -1)
        s = ''
        for i in node:
            if i == 20:
                break
            s += AA_dict[int(i)]
        seq_ele.append(s)
    return seq_ele

###### perplexity ######

def perplexity(output, target):
    return torch.exp(F.cross_entropy(output, target, reduction = 'none'))

###### identity ######

def sequence_indentity(seq_1, seq_2, version = 'BLAST'):
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

###### Pipeline ######

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
