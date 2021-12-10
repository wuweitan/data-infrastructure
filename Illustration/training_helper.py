import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import sklearn.metrics as metrics

import argparse
import os
import time
import random
import pickle
#from tqdm import tqdm

import networks
import DataLoading

#from torch_geometric.data import DataLoader
#from torch_geometric.nn.inits import uniform
#from torch_geometric.nn import global_mean_pool
from copy import deepcopy

#from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist

#matrix = matlist.blosum62

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

##################################################################################
# For Discriminative Model
##################################################################################

def hierarchy_arrange(soft_vec, arrange_index):
    """
    Change the softmax vector from one level to another.
    """
    shape = soft_vec.shape
    out_vec = torch.zeros(shape[0],len(arrange_index)).float()
    for i,idx in enumerate(arrange_index):
        out_vec[:,i] = torch.sum(soft_vec[:,:idx],dim=-1)
        soft_vec = soft_vec[:,idx:]
    return out_vec

def evaluate(dataset, model, batch_size=64, label=None, arrange_index=None, hierarchy_dict=None, max_num_examples=None, weight = None):
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

def discriminative_train(dataset, model, num_epochs, val_dataset=None, test_dataset=None, batch_size=16, max_num_examples=None,
                         eva_period=1, save_period=1, hierarchy_dict=None, model_path = None, log_path=None,
                         loss_kind='softmax', learning_rate=0.001, clip=2.0, lambdas=[1,1,1], weight=None, arrange_index=None):

    ######################## Training log files #######################################
    if log_path:
        if not log_path.endswith('/'):
            log_path += '/'
        loss_all_file = open(log_path + 'loss_all.txt','w')
        loss_ave_file = open(log_path + 'loss_ave.txt','w')
        accu_path = log_path + 'accuracy.txt'
        accu_file = open(accu_path,'w')

        accu_file.write('{val:<5}\t'.format(val = 'epoch'))
        if loss_kind != 'hierarchy':
            accu_file.write('{val:<25}\t'.format(val = 'accuacy'))
            accu_file.write('{val:<10}\t'.format(val = 'set'))
        else:
            accu_file.write('{val:<25}\t'.format(val = 'set'))
            accu_file.write('{val:<25}\t'.format(val = 'family_accuacy'))
            accu_file.write('{val:<25}\t'.format(val = 'super-family_accuacy'))
            accu_file.write('{val:<25}\t'.format(val = 'fold_accuacy'))
            accu_file.write('{val:<25}\t'.format(val = 'class_accuacy'))
        accu_file.write('{val:<25}\t'.format(val = 'train_time'))
        accu_file.write('{val:<25}\n'.format(val = 'total_time'))

        loss_all_file.close()
        loss_ave_file.close()
        accu_file.close()

    else:
        accu_path = None

    ######################## Training setting #######################################

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate)

    if type(weight) != type(None): # weights of the samples for balanced data

        weight_dict = {}
        if loss_kind == 'hierarchy':
            for sk in weight.keys():
                #weight_dict[sk] = [torch.Tensor(w).float().cuda() for w in weight[sk]]
                weight_dict[sk] = [torch.Tensor(w).float() for w in weight[sk]]
        else:
            for sk in weight.keys():
                weight_dict[sk] = torch.Tensor(weight[sk]).float()#.cuda()
        weight = weight_dict
    else:
        weight = {'training':None, 'validation': None, 'test':None}

    train_result_all = {}
    best_train_result = {'epoch': 0, 'acc': 0}

    if val_dataset is not None:
        val_result_all = {}
        best_val_result = {'epoch': 0, 'acc': 0}
    if test_dataset is not None:
        test_result_all = {}
        best_test_result = {'epoch': 0, 'acc': 0}

    ######################## Training process #######################################

    for epoch in range(num_epochs):

        epoch += 1

        avg_loss = 0.0
        epoch_start_time = time.time()
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()

            seq = Variable(data['seq'].float(), requires_grad=False)#.cuda()
            seq_mask = Variable(data['seq_mask'].float(), requires_grad=False)#.cuda()
            adj = Variable(data['adj'].float(), requires_grad=False)#.cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False)#.cuda()
            label = Variable(data['label'].long())#.cuda()
            #weight = Variable(data['weight'].float(), requires_grad=False)#.cuda()

            ypred = model(h0, adj, seq, seq_mask)
            #loss = model.loss(ypred, label = label, loss_type = loss_kind, lambdas = lambdas, weight = weight['training'], arrange_index = arrange_index)
            loss = model.loss(ypred, label = label, loss_type = loss_kind, lambdas = lambdas, arrange_index = arrange_index)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            ###### record training loss ##########

            loss = loss.cpu()

            if log_path:  # record all the loss (SZ)
                loss_all_file = open(log_path + 'loss_all.txt','a')
                loss_all_file.write(str(float(loss)) + '\n')
                loss_all_file.close()

            avg_loss += loss

        avg_loss /= batch_idx + 1
        total_time = time.time() - epoch_start_time
        print('Average loss: %f'%avg_loss)

        ###### record average training loss ######

        if log_path:  # record average loss (SZ)
            loss_ave_file = open(log_path + 'loss_ave.txt','a')
            loss_ave_file.write(str(float(avg_loss)) + '\n')
            loss_ave_file.close()

        ###### evaluate the model ######

        if epoch == 1 or epoch % eva_period == 0:
            result = evaluate(dataset, model, batch_size, label=loss_kind, arrange_index=arrange_index,
                              hierarchy_dict=hierarchy_dict, max_num_examples = max_num_examples, weight = weight['training'])
            train_result_all[epoch] = result
            best_train_result = accu_print(result, loss_kind, 'Training', epoch, best_train_result)

            if val_dataset is not None:
                val_result = evaluate(val_dataset, model, batch_size, label=loss_kind, arrange_index=arrange_index,
                                      hierarchy_dict=hierarchy_dict, max_num_examples = max_num_examples, weight = weight['validation'])
                val_result_all[epoch] = val_result
                best_val_result = accu_print(val_result, loss_kind, 'Validation', epoch, best_val_result)

            if test_dataset is not None:
                test_result = evaluate(test_dataset, model, batch_size, label=loss_kind, arrange_index=arrange_index,
                                       hierarchy_dict=hierarchy_dict, max_num_examples = max_num_examples, weight = weight['test'])
                test_result_all[epoch] = test_result
                best_test_result = accu_print(test_result, loss_kind, 'Test', epoch, best_test_result)

            all_time = time.time() - epoch_start_time

            ###### record the accuracy ######

            if log_path:
                _acc_save = accu_save(result, loss_kind, 'Training', epoch, total_time, all_time, accu_path)

                if val_dataset is not None:
                    _acc_save = accu_save(val_result, loss_kind, 'Validation', epoch, total_time, all_time, accu_path)
                if test_dataset is not None:
                    _acc_save = accu_save(test_result, loss_kind, 'Test', epoch, total_time, all_time, accu_path)
        else:
            all_time = time.time() - epoch_start_time

        ##### save the model ######

        if model_path and epoch % save_period == 0:
            _model_save = model_save(model, model_path + 'model_dict_%d.pickle'%epoch)
            _model_save = model_save(optimizer, model_path + 'optimizer_dict_%d.pickle'%epoch)

        print('Training time for Epoch %d: %.4f s'%(epoch, total_time))
        print('Total time for Epoch %d: %.4f s'%(epoch, all_time))

    results_all = {'training': train_result_all}
    print('Best training result: %.4f (epoch %d)'%(best_train_result['acc'],best_train_result['epoch']))

    if val_dataset is not None:
        results_all['validation'] = val_result_all
        print('Best validation result: %.4f (epoch %d)'%(best_val_result['acc'],best_val_result['epoch']))
    if test_dataset is not None:
        results_all['test'] = test_result_all
        print('Best test result: %.4f (epoch %d)'%(best_test_result['acc'],best_test_result['epoch']))

    return model, optimizer, results_all

def model_save(model, model_path):
    _dict_save = DataLoading.dict_save(model.state_dict(), model_path)
    return 0

##################################################################################
# For Generative Model
##################################################################################

def VAE_training(model, train_set, Epoch_NUM = 5, learning_rate = 0.0001,
                 clip = 2.0, kld_weight = 0, kld_max = 1.0, kld_start_inc = 0, kld_inc = 0.0001, habits_lambda = 1.0,
                 loss_file = None, log_file = None, eval_file = None, eval_inter = 1, model_path = None, save_inter = 1, balance = False,
                 temperature = 1, temperature_min = 0.1, temperature_dec = 0.0001, seq_len = 35, MAX_SAMPLE = 'top-k', k = 3):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### record files
    if log_file:
        with open(log_file,'w') as log_f:
            log_f.write('Average-Loss\tAverage-CE\tAverage-KLD\n')
    if loss_file:
        with open(loss_file,'w') as loss_f:
            loss_f.write('Overall-Loss\tCross-Entropy\tKL-Divergence\n')
    if eval_file:
        eval_f = open(eval_file,'w')
        eval_f.close()

    iteration_idx = 0

    for epoch in range(1,Epoch_NUM + 1):
        print('Epoch %d:'%epoch)

        start_time = time.time()

        model.train()

        loss_all = []
        ce_all = []
        KLD_all = []

        for batch_idx, data in enumerate(train_set):
            ### load the batch data ###
            seq = Variable(data['seq'].float(), requires_grad=False)#.cuda()
            seq_mask = Variable(data['seq_mask'].float(), requires_grad=False)#.cuda()
            adj = Variable(data['adj'].float(), requires_grad=False)#.cuda()

            h0 = Variable(data['feats'].float())#.cuda()
            batch_num_nodes = data['num_nodes'].int().numpy()

            if balance:
                batch_size = batch_num_nodes.shape[0]
                ele_weight = torch.cat([data['weights'][i][:batch_num_nodes[i]] for i in range(batch_size)]).float()#.cuda()
                ele_weight = ele_weight / ele_weight.shape[0]
            else:
                ele_weight = None

            iteration_idx += 1

            optimizer.zero_grad()

            ### forward ###

            mu, sig, z, out = model(h0, adj, seq, batch_num_nodes, seq_mask, n_steps = seq_len, temperature = temperature, MAX_SAMPLE = MAX_SAMPLE, k=k)
            if temperature > temperature_min:
                    temperature -= temperature_dec

            ### loss calculation ###     

            #print(out.shape, seq.shape, batch_num_nodes)
            #loss, ce, KLD = model.vae_loss(mu, sig, out, seq, batch_num_nodes, habits_lambda, seq_len, kld_weight, ele_weight)
            loss, ce, KLD = model.vae_loss(mu, sig, out, seq, batch_num_nodes, habits_lambda, kld_weight)

            ### record ###

            if loss_file:
                loss_f = open(loss_file,'a')
                loss_f.write('%f\t%f\t%f\n'%(float(loss), float(ce), float(KLD)))
                loss_f.close()
            loss_all.append(float(loss))
            ce_all.append(float(ce))
            KLD_all.append(float(KLD))

            ### gradient clip ###

            loss.backward()
            ec = nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if iteration_idx > kld_start_inc and kld_weight < kld_max:
                kld_weight += kld_inc

            ### iteration end  ###

        aver_loss = np.mean(loss_all)
        aver_ce = np.mean(ce_all)
        aver_kld = np.mean(KLD_all)
        print('Average-Loss: %.4f\tAverage-CE: %.4f\tAverage-KLD: %.4f'%(aver_loss, aver_ce, aver_kld))

        train_end_time = time.time()
        print('Training time: %.4fs'%(train_end_time - start_time))

    return model, optimizer

##################################################################################
# For Antibody Design
##################################################################################

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
                'weight': self.weight_all[sample_idx]}

############################## Training Functions #######################################

def model_training(model, data_path, optimizer = None, Epoch_start = 0, Epoch_end = 500, 
                   batch_size = 16, num_workers = 1, learning_rate = 0.0001, clip = 2,
                   iter_file = None, epoch_file = None, seq_file = None, eval_inter = 1, model_path = None, save_inter = 1,
                   with_NonCDR = True, with_AgSeq = False, USE_CUDA = True, debug = False, 
                   cdr_node_features = 'all', ordered_feature_list = ['NBRradii', 'Charge', 'SASA', 'Dist']):

    if Epoch_start == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ### record files
        if not debug:
            if iter_file is not None and os.path.exists(iter_file):
                os.remove(iter_file)
            if epoch_file is not None and os.path.exists(epoch_file):
                os.remove(epoch_file)
            if seq_file is not None and os.path.exists(seq_file):
                os.remove(seq_file)
        else:
            iter_file = None
            epoch_file = None
            seq_file = None
            model_path = None

    print('Data loading...')

    iteration_idx = 0
    data_dict = dict_load(data_path)
    train_set = ClusterSampler(data_dict, debug = debug)
    #train_set = DataSampler(data_dict, debug = debug) #
    dataloader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    if debug:
        print('Debugging...')
    else:
        print('Training...')

    for epoch in range(Epoch_start, Epoch_end):
        print('Epoch %d:'%(epoch+1))
        model.train()
        loss_all = []
        start_time = time.time()

        ### whether record the sequences ###
        if (seq_file is not None) and (epoch == 0 or (epoch+1) % eval_inter == 0): # record sequence 
            record_flag = True
            with open(seq_file,'a') as eval_f:
                eval_f.write('Epoch %d:\n'%(epoch + 1))
        else:
            record_flag = False

        ### iterations ###
        for batch_idx,data in enumerate(dataloader):
            iteration_idx += 1
            optimizer.zero_grad()
            ### load the batch data ###
            ### inputs
            seq_ag = Variable(data['seq_ag_onehot'].float(), requires_grad=False)
            if with_NonCDR:
                seq_ab = Variable(data['seq_ab_onehot_noCDR'].float(), requires_grad=False)
            else:
                seq_ab = Variable(data['seq_ab_onehot_masked'].float(), requires_grad=False)
            feat = Variable(data['feat'].float(), requires_grad=False)
            adj = Variable(data['adj'].float(), requires_grad=False)
            cdr_groundtruth = data['cdr_groundtruth']
           
            ### masks and indexes 
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

            ### applied Ag seq part
            if not with_AgSeq:
                seq_ag = seq_ag[ag_indexes]
            ### CDR node features
            if cdr_node_features is None or cdr_node_features == 'None':
                feat = feat * (graph_idx_mask > 0) 
            elif cdr_node_features != 'all':
                feat_mask = graph_idx_mask > 0
                for i,fea_kind in enumerate(ordered_feature_list):
                    if not fea_kind in cdr_node_features:
                        feat[:,:,i:i+1] = feat[:,:,i:i+1] * feat_mask
            
            if USE_CUDA:
                seq_ag = seq_ag.cuda()
                seq_ab = seq_ab.cuda()
                feat = feat.cuda()
                adj = adj.cuda()
                cdr_groundtruth = cdr_groundtruth.cuda()

            ### forward ###

            profile, cdrs, seq_ab = model(seq_ag, seq_ab, feat, adj,
                                          graph_idx_mask, seq_sele_ag, ag_indexes, seq_sele_ab, ab_indexes, cdr_mask, seq_len_ab, seq_len_ag)
            #print('CDR seq:', cdrs)            
 
            ### loss calculation ###     
            loss = model.loss(profile, cdr_groundtruth, cdr_mask)
            print(loss)
            loss_all.append(float(loss))            
            if iter_file is not None:
                loss_f = open(iter_file,'a')
                loss_f.write('%f\n'%(float(loss)))           
                loss_f.close() 

            ### backward & gradient clip ###
 
            if not torch.isnan(loss).any():
                loss.backward()
                ec = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            ### record the sequences ###
            if record_flag:
                with open(seq_file,'a') as eval_f:
                    for seq_batch in cdrs:
                        eval_f.write('HCDR_1:%s\tHCDR_2:%s\tHCDR_3:%s\n'%(seq_batch[0],seq_batch[1], seq_batch[2]))
                        eval_f.write('LCDR_1:%s\tLCDR_2:%s\tLCDR_3:%s\n'%(seq_batch[3],seq_batch[4], seq_batch[5]))

            ### iteration end  ###

        if record_flag:
            with open(seq_file,'a') as eval_f:
                eval_f.write('\n')

        ### record the loss ###
        aver_loss = np.mean(loss_all)
        train_end_time = time.time()
        print('Average-Loss: %.4f  Training time: %.4fs'%(aver_loss, train_end_time - start_time))
        if epoch_file is not None:
            loss_f = open(epoch_file,'a')
            loss_f.write('%f\n'%(float(aver_loss)))                               
            loss_f.close()

        ### save the model ### 
        if (not debug) and (epoch == 0 or (epoch+1) % save_inter == 0) and (model_path is not None):
            _model_save = torch.save(model.state_dict(), model_path + 'model_%d.pickle'%(epoch + 1))
            _model_save = torch.save(optimizer.state_dict(), model_path + 'optimizer_%d.pickle'%(epoch + 1))

    print('Training (%d epochs and %d iterations) completed!'%(epoch+1,iteration_idx))

    return model, optimizer

