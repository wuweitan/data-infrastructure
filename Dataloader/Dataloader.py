import numpy as np
import pickle
import sklearn
import dgl
from scipy.sparse import csr_matrix

import torch
import torch_geometric
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import networkx as nx
import torch.utils.data
import random
from math import ceil

import os
import pickle

from dataset import SCOPe_graphs # can add more datasets

def Dataloader(dataset = 'SCOPe_graph', data_path = 'datasets/', batch_size = 64, shuffle = True, numworkers = 1,
               ### For SCOPe_graph:
               normalize=True, seq_mask_dim = 21, max_num_nodes=60, max_SSE=None, DGL)
    if dataset == 'SCOPe_graph':
        data_sampler = SCOPe_graphs(data_path, normalize, seq_mask_dim, max_num_nodes, max_SSE, DGL) 
    else:
        print('Error! No dataset named %s!'%dataset)
        return None

    dataset_loader = torch.utils.data.DataLoader(data_sampler,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers)
    return dataset_loader

