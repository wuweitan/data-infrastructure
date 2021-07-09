import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist #SZ add

matrix = matlist.blosum62

import re
import string
import random

import networkx as nx # for graph similarity

import pdb_helper


def sequence_indentity(self, seq_1, seq_2, version = 'BLAST'):
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
        max_iden = 
        for i in alignments:
            same = 0
            for j in xrange(i[-1]):
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
            for j in xrange(i[-1]):
                if i[0][j] == i[1][j] and i[0][j] != '-':
                    same += 1
            if same > max_same:
                max_same = same
        identity = float(max_same)/float(l)
    return identity

def TM_score(self, pdb_1, pdb_2):
    
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_2" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_2")
        tms_1 = out_1[k_1-8:k_1-1]
    else:
        return None

    command_2 = './TMalign ' + pdb_2 + ' ' + pdb_1 + ' -a'
    output_2 = os.popen(command_2)
    out_2 = output_2.read()
    if "(if normalized by length of Chain_2" in out_2:
        k_2 = out_2.index("(if normalized by length of Chain_2")
        tms_2 = out_2[k_2-8:k_2-1]
    else:
        return None
    return (float(tms_1) + float(tms_2))/2

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
