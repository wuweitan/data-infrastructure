import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
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

######################################## Pretrain #######################################

class graphcl(nn.Module):
    """
    Origin from https://github.com/Shen-Lab/GraphCL
    """
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        #self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, seq, x, adj, seq_mask):
    #def forward_cl(self, x, edge_index, edge_attr, batch):
        #x = self.gnn(x, edge_index, edge_attr)
        #x = self.pool(x, batch)
        x = self.gnn(seq, x, adj, seq_mask)  # SZ change
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def Pretrain(args, model, device, dataset, optimizer):
    """
    Origin from https://github.com/Shen-Lab/GraphCL
    """
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        #x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        #x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        x1 = model.forward_cl(seq, x, adj, seq_mask)
        x2 = model.forward_cl(seq, x, adj, seq_mask)

        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)

######################################## Evaluation Function #######################################

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

############################# Training Function #############################################

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

def VAE_training(model, train_set, Epoch_NUM = 5, learning_rate = 0.0001, 
                 clip = 2.0, kld_weight = 0, kld_max = 1.0, kld_start_inc = 0, kld_inc = 0.0001, habits_lambda = 1.0
                 loss_file = None, log_file = None, eval_file = None, eval_inter = 1, model_path = None, save_inter = 1, balance = False,
                 temperature = None, temperature_min = None, temperature_dec = None, seq_len = 35, MAX_SAMPLE = 'top-k', k = 3):

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

            loss, ce, KLD = model.vae_loss(mu, sig, out, seq, batch_num_nodes, habits_lambda, seq_len, kld_weight, ele_weight)

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
            ec = nn.utils.clip_grad_norm(model.parameters(), clip)
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
