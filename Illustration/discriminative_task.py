import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse
import os
import shutil
import time

import graph_networks
import DataLoading 
import model_helper

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

######################################## Set Path and Arguments #######################################

parser = argparse.ArgumentParser()

######################## set data path ########################

parser.add_argument('--data_dire', type=str, default='../../Data/Datasets/Protein_graphs/', help='path of the merged data')
parser.add_argument('--log_path', type=str, default='../Training_log/Discriminative/', help='path of the training log')
parser.add_argument('--model_path', type=str, default='../Model_checkpoints/Discriminative/', help='path of the trained models')

######################## Versions #############################

parser.add_argument('--model', type=str, default='GCN', help='which GNN model, GCN')
#parser.add_argument('--graph_version', type=str, default='PROTEINS_undire_homo', help='which version of the graph')
parser.add_argument('--graph_version', type=str, default='TOPS+-RCSB_dire_hbond', help='which version of the graph')
parser.add_argument('--features', type=str, nargs='*', default=['len', 'ss', 'level'], help='which features to be considered')

parser.add_argument('--balance_weight', type=str, default='../../Data/Datasets/Filtered_data/weight_list.pickle', help='Path of the weight of the samples. None for unbalanced data.')

parser.add_argument('--seq_embed', type=int, default=0, help='whether introduce sequence embedding')
parser.add_argument('--seq_cat', type=int, default=1, help='whether cat sequence for embedding')
parser.add_argument('--seq_GNN', type=int, default=0, help='whether apply GNN on sequence embedding')

parser.add_argument('--cuda', type=str, default='0', help='CUDA')
parser.add_argument('--label', type=str, default='fold', help='which hierarchy of labels')

parser.add_argument('--test_num', type=int, default=None, help='Load a batch of data to test the code.')

################ Training setting #############################

parser.add_argument('--max_nodes', type=int, default=60, help='maximum number of nodes')
parser.add_argument('--max_SSE', type=int, default=35, help='maximum length of nodes')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='amount of training epochs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to load data')
parser.add_argument('--eva_period', type=int, default=1, help='period to evaluate the model while training')
parser.add_argument('--save_period', type=int, default=1, help='period to save the model while training')
parser.add_argument('--eva_max_num', type=int, default=None, help='maximum evaluating sample while training')

### for hierarchical loss

parser.add_argument('--loss_weights', type=float, nargs='*', default=[1.0, 1.0, 1.0], help='weights of the cross_entropies in the hierarchical loss')
parser.add_argument('--arrange_index', type=str, default='../../Data/Datasets/Filtered_data/Pred_Arrange_Index.list', help='path of the arrange index file for hierarchical labels')
parser.add_argument('--label_map_dict', type=str, default='../../Data/Datasets/Filtered_data/family_label_map_dict.pickle', help='path of the label mapping dictionary')


################ Hyper-parameters #############################

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')

### for GNN ###

parser.add_argument('--bn', type=int, default=1, help='whether batch normalization')
parser.add_argument('--bias', type=int, default=1, help='whether introduce bias in GNN')
parser.add_argument('--dropout', type=float, default=0.0, help='GNN dropout rate')
parser.add_argument('--readout_pooling', type=str, default='max', help='readout pooling method')
parser.add_argument('--concat', type=int, default=0, help='whether concatenate the results from each layer')
parser.add_argument('--normalize', type=int, default=1, help='whether normalize the adjacency tensors')
### dimensions
parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
parser.add_argument('--output_dim', type=int, default=20, help='output dimension')
### layers
parser.add_argument('--num_gc_layers', type=int, default=3, help='number of graph convolution layers')
parser.add_argument('--fnn_layers', nargs='*', default=[4000], help='graph-level feedforward NN layers')

### for RNN ###

parser.add_argument('--rnn_bias', type=int, default=1, help='whether introduce bias in RNN')
parser.add_argument('--rnn_dropout', type=float, default=0.0, help='RNN dropout rate')
### dimensions ###
parser.add_argument('--seq_hidden_dim', type=int, default=20, help='sequence embedding hidden dimension')
### layers ###
parser.add_argument('--num_rnn_layers', type=int, default=3, help='number of RNN layers')

args = parser.parse_args()

cuda = args.cuda

### Data path ###

data_dire = args.data_dire
log_path = args.log_path
model_path = args.model_path

if not data_dire.endswith('/'):
    data_dire += '/'
if not log_path.endswith('/'):
    log_path += '/'
if not model_path.endswith('/'):
    model_path += '/'

### version ###

method = args.model
graph_version = args.graph_version
features = list(set(args.features))
balance_weight = args.balance_weight

seq_embed = bool(args.seq_embed)
seq_cat = bool(args.seq_cat)
seq_GNN = bool(args.seq_GNN)

label = args.label

test_num = args.test_num

# channel
graph_type = graph_version.split('_')[0]
if graph_type == 'PROTEINS':
    if 'heter' in graph_version:
        channel_num = 2
    else:
        channel_num = 1
else:
    channel_num = 5
# feature
input_dim = 2 + len(features) * 3
# label
if label == 'hierarchy':
    num_classes = 4304
    loss_kind = 'hierarchy'
else:
    loss_kind = 'softmax'
    if label == 'class':
        num_classes = 6
    elif label == 'fold':
        num_classes = 1080
    elif label == 'super-family':
        num_classes = 1820
    elif label == 'family':
        num_classes = 4304
    else:
        print('Error! No label kind named %s'%label)
        quit()

### hyper-parameter ###

learning_rate = args.lr
max_nodes_num = args.max_nodes
max_SSE = args.max_SSE
batch_size = args.batch_size
epoch_num = args.epochs
clip = args.clip
num_workers = args.num_workers
bn = bool(args.bn)
bias = bool(args.bias)
rnn_bias = bool(args.rnn_bias)
dropout = args.dropout
rnn_dropout = args.rnn_dropout
readout_pooling = args.readout_pooling
concat = bool(args.concat)
normalize = bool(args.normalize)

hidden_dim = args.hidden_dim
seq_hidden_dim = args.seq_hidden_dim
output_dim = args.output_dim

num_gc_layers = args.num_gc_layers
num_rnn_layers = args.num_rnn_layers
fnn_layers = args.fnn_layers

### training setting ###

eva_period = args.eva_period
save_period = args.save_period
eva_max_num = args.eva_max_num
loss_weights = args.loss_weights

if loss_kind == 'hierarchy':
    arrange_index = DataLoading.dict_load(args.arrange_index)
    label_map_dict = DataLoading.dict_load(args.label_map_dict) 
else:
    arrange_index = None
    label_map_dict = None

### make path ###

features_new = []
for f in ['len','ss','level']:
    if f in features:
        features_new.append(f)
features = features_new

title = '-'.join(features) + '_' + '-'.join([str(i) for i in loss_weights]) + '_'
if seq_embed:
    title += 'Seq'
    if seq_GNN:
        title += '-GNN'
    elif seq_cat:
        title += '-Cat'
    title += '_'
#if concat:
#    title += 'concat_'
#if not bn:
#    title += 'no-bn_'
#if not bias:
#    title += 'no-bias_'

if dropout != 0:
    title += 'Drop-%f_'%dropout

if normalize:
    title += 'normalize_'
if balance_weight != None and balance_weight != 'None':
    balance_weight = DataLoading.dict_load(balance_weight)['Discriminative']
    if label != 'hierarchy':
        label_idex = {'class':3, 'fold':2, 'super-family':1, 'family':0}
        balance_weight_new = {}
        for skind in balance_weight.keys():
            balance_weight_new[skind] = balance_weight[skind][label_idex[label]]
        balance_weight = balance_weight_new
    title += 'balance_'
    print('Applying balanced data...')
else:
    balance_weight = None
    print('Applying unbalanced data...')

# export scalar data to JSON for external processing
log_path = os.path.join(log_path, 'log_%s_%s_%s_%s%d'%(method, graph_version, label, title, epoch_num))
log_path += '/'
print('Log path:',log_path)

if test_num:
    CODE_TEST = True
    log_path = None
else:
    CODE_TEST = False
    if os.path.isdir(log_path):
        print('Remove existing log dir: %s'%log_path)
        shutil.rmtree(log_path)
    DataLoading.make_path(log_path)

model_path += 'model_%s_%s_%s_%s%d/'%(method, graph_version, label, title, epoch_num)
print('Model path:',model_path)
if not CODE_TEST:
    if os.path.isdir(model_path):
        print('Remove existing model dir: %s'%model_path)
        shutil.rmtree(model_path)
    DataLoading.make_path(model_path)
else:
    model_path = None

os.environ['CUDA_VISIBLE_DEVICES'] = cuda
print('CUDA', cuda)

print('****************************************************************')

######################################## Load Data #######################################

print('Graph Version: %s'%graph_version)

train_seq_file = data_dire + 'Dis_Seq_%s_training.list'%graph_type
train_fea_file = data_dire + 'Dis_X_%s_training.list'%graph_type
train_adja_file = data_dire + 'Dis_Adj_%s_training.list'%graph_version
train_label_file = data_dire + 'Y_training_%s.txt'%label

vali_seq_file = data_dire + 'Dis_Seq_%s_validation.list'%graph_type
vali_fea_file = data_dire + 'Dis_X_%s_validation.list'%graph_type
vali_adja_file = data_dire + 'Dis_Adj_%s_validation.list'%graph_version
vali_label_file = data_dire + 'Y_validation_%s.txt'%label

test_seq_file = data_dire + 'Dis_Seq_%s_test.list'%graph_type
test_fea_file = data_dire + 'Dis_X_%s_test.list'%graph_type
test_adja_file = data_dire + 'Dis_Adj_%s_test.list'%graph_version
test_label_file = data_dire + 'Y_test_%s.txt'%label

print('Training set:')
print('Node features: %s'%train_fea_file)
print('Adjaceny matrix: %s'%train_adja_file)
print('Labels: %s'%train_label_file)
print()

print('Validation set:')
print('Node features: %s'%vali_fea_file)
print('Adjaceny matrix: %s'%vali_adja_file)
print('Labels: %s'%vali_label_file)
print()

print('Test set:')
print('Node features: %s'%test_fea_file)
print('Adjaceny matrix: %s'%test_adja_file)
print('Labels: %s'%test_label_file)
print()

train_dataset, train_num, train_sele_num, tra_class_num, tra_feat_dim, tra_chan_num = DataLoading.dataloading(train_seq_file,
                                                                                                              train_fea_file,
                                                                                                              train_adja_file,
                                                                                                              train_label_file,
                                                                                                              label = label,
                                                                                                              features = features,
                                                                                                              batch_size = batch_size,
                                                                                                              max_nodes = max_nodes_num,
                                                                                                              max_SSE = max_SSE,
                                                                                                              num_workers = num_workers,
                                                                                                              normalize = normalize, shuffle=True,
                                                                                                              test_num = test_num)
val_dataset, val_num, val_sele_num, val_class_num, val_feat_dim, val_chan_num = DataLoading.dataloading(vali_seq_file,
                                                                                                        vali_fea_file, 
                                                                                                        vali_adja_file, 
                                                                                                        vali_label_file,
                                                                                                        label = label,
                                                                                                        features = features,
                                                                                                        batch_size = batch_size,
                                                                                                        max_nodes= max_nodes_num,
                                                                                                        max_SSE = max_SSE,
                                                                                                        num_workers = num_workers,
                                                                                                        normalize = normalize, shuffle=False,
                                                                                                        test_num = test_num)
test_dataset, test_num, test_sele_num, test_class_num, test_feat_dim, test_chan_num = DataLoading.dataloading(test_seq_file,
                                                                                                              test_fea_file, 
                                                                                                              test_adja_file, 
                                                                                                              test_label_file,
                                                                                                              label = label,
                                                                                                              features = features,
                                                                                                              batch_size = batch_size,
                                                                                                              max_nodes = max_nodes_num,
                                                                                                              max_SSE = max_SSE,
                                                                                                              num_workers = num_workers,
                                                                                                              normalize = normalize, shuffle=False,
                                                                                                              test_num = test_num)

if input_dim != tra_feat_dim or input_dim != val_feat_dim or input_dim != test_feat_dim:
    print('Error! Input dimension do not match!')
    quit()
if channel_num != tra_chan_num or channel_num != val_chan_num or channel_num != test_chan_num:
    print('Error! Amounts of channels do not match!')
    quit()

print('%d (%d) training samples loaded.'%(train_sele_num,train_num))
print('%d (%d) validation samples loaded.'%(val_sele_num,val_num))
print('%d (%d) test samples loaded.'%(test_sele_num,test_num))
print('****************************************************************')

######################################## Training #######################################

print('Method: %s'%method)

all_vals = []
    
if method == 'GCN':
    model = graph_networks.GraphLevelEmbedding(label_num = num_classes, pred_hidden_dims=fnn_layers, pooling=readout_pooling,
            # for GNN
            model=method, feature_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=output_dim, num_layers=num_gc_layers, channel_num=channel_num,
            concat=concat, bn=bn, dropout=dropout, bias=bias,
            # for RNN
            seq_embed = seq_embed, rnn_bias = rnn_bias, rnn_hidden_size = seq_hidden_dim, rnn_num_layers = num_rnn_layers, 
            seq_cat = seq_cat, seq_GNN = seq_GNN, rnn_dropout = rnn_dropout).cuda()
else:
    print('Error! No method named %s!'%args.method)
    quit()

print('Training...')

trained_model, optimizer, results = model_helper.discriminative_train(train_dataset, model, epoch_num, val_dataset=val_dataset, test_dataset=None,
                                                           batch_size = batch_size, learning_rate = learning_rate, clip = clip, eva_period = eva_period,
                                                           save_period = save_period, log_path=log_path, max_num_examples = eva_max_num, 
                                                           arrange_index=arrange_index, loss_kind = loss_kind, lambdas = loss_weights,  
                                                           weight = balance_weight, hierarchy_dict = label_map_dict, model_path = model_path)
print('Training completed.') 
print('Save the model and calculating the test accuracy...')
if model_path: 
    _model_save = model_helper.model_save(model, model_path + 'model_dict_%d.pickle'%epoch_num)
    _model_save = model_helper.model_save(optimizer, model_path + 'optimizer_dict_%d.pickle'%epoch_num)

if log_path:
    _save_dict = DataLoading.dict_save(results, log_path + 'Training_results.pickle')

if test_dataset is not None:
    test_result = model_helper.evaluate(test_dataset, trained_model, batch_size=batch_size, label=loss_kind, arrange_index=arrange_index, hierarchy_dict=label_map_dict, max_num_examples=eva_max_num)
    if log_path:
        _save_dict = DataLoading.dict_save(test_result, log_path + 'Test_result.pickle')

    if loss_kind != 'hierarchy':
        for k in test_result.keys():
            print('Test %s:'%k,test_result[k])
    else:
        for pred_kind in ['from_softmax','from_family']:
            print(pred_kind)
            for level in ['family','super-family','fold','class']:
                text = ''
                for k in test_result[pred_kind][level].keys():
                    text += '%s: %.4f  '%(k,test_result[pred_kind][level][k])
                print('%s: %s'%(level,text))
            



