import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import re
import string
import random

import networkx as nx # for graph similarity

import pdb_helper



class DataFilter():
    """
    Sequence-based evaluation
    """
    def __init__(self, dataset, **kwargs):
        self.origin_set = dataset
        pass

    def seq_filter(self, **kwargs)
        pass

    def struct_filter(self, **kwargs)
        pass


class DataSplit():
    """
    Evaluation metrics for general predictions
    """
    def __init__(self, dataset, **kwargs):
        self.origin_set = dataset

    def data_split(self, dataset, ratio = [0.7, 0.15, 0.15]):
        pass
