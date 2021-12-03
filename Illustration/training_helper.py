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
#from tqdm import tqdm

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

