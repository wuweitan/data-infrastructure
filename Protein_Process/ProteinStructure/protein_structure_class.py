import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import re
import string
import random

import networkx as nx # for graph similarity

import pdb_helper

class SingleProteinStructure():
    """
    Read a pdb file and extract the information for further process
    """
    def __init__(self, pdb_file, **kwargs):
        self.data_dir = pdb_file


class ProteinGraph():
    """
    Calculate the features for protein graphs construction
    element-wise / residue-wise / atom-wise
    """
    def __init__(self, pdb_info, **kwargs):
        self.data_info = pdb_info
