import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import re
import string
import random

import networkx as nx # for graph similarity

import pdb_helper


def sequence_indentity(self, seq_1, seq_2, version = 'BLAST'):
    return None


class Sequence():
    """
    Sequence-based evaluation
    """
    def __init__(self, model, **kwargs):
        self.model = model
        pass

    def seq_gen(self, **kwargs)
        pass


class Accuracy():
    """
    Evaluation metrics for general predictions
    """
    def __init__(self, model, **kwargs):
        self.model = model
