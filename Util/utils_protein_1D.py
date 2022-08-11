#################################################################################
# Utility functions for data process and evaluation on 1-D sequences.
#################################################################################

from collections import OrderedDict
from typing import List
import numpy as np
from sqlalchemy import null
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import os

from Bio.PDB.ResidueDepth import residue_depth
from Bio.PDB.ResidueDepth import ca_depth
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.PDBParser import PDBParser

from numpy import pi
from Bio.PDB.vectors import rotaxis2m
from Bio.PDB.vectors import Vector

from Bio.PDB import *

import torch
from torch import nn

### CONSTANTS

matrix = matlist.blosum62

PFAM_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("A", 4),
    ("C", 5),
    ("D", 6),
    ("E", 7),
    ("F", 8),
    ("G", 9),
    ("H", 10),
    ("I", 11),
    ("K", 12),
    ("L", 13),
    ("M", 14),
    ("N", 15),
    ("O", 16),
    ("P", 17),
    ("Q", 18),
    ("R", 19),
    ("S", 20),
    ("T", 21),
    ("U", 22),
    ("V", 23),
    ("W", 24),
    ("Y", 25),
    ("B", 26),
    ("Z", 27)])

dict_PSAIA_table_file_name_to_col_idx = {'chain id': 0, 'ch total ASA': 1, 'ch b-bone ASA': 2, 'ch s-chain ASA': 3, 'ch polar ASA': 4, 
'ch n-polar ASA': 5, 'res id': 6, 'res name': 7, 'total ASA': 8, 'b-bone ASA': 9, 's-chain ASA': 10, 'polar ASA': 11, 
'n-polar ASA': 12, 'total RASA': 13, 'b-bone RASA': 14, 's-chain RASA': 15, 'polar RASA': 16, 'n-polar RASA': 17, 
'average DPX': 18, 's_avg DPX': 19, 's-ch avg DPX': 20, 's-ch s_avg DPX': 21, 'max DPX': 22, 'min DPX': 23, 
'average CX': 24, 's_avg CX': 25, 's-ch avg CX': 26, 's-ch s_avg CX': 27, 'max CX': 28, 'min CX': 29, 'Hydrophobicity': 30}

set_bb_atom_name = set(["N", "C", "CA", "O"])

eps = 1e-8

BLOSUM62_AA_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

#################################################################################
# 1-D Sequences Process
#################################################################################

def OneHot_encoding(seq, char_keys):  # by SZ
    """Transform a sequence in to an one-hot encoding sequences.

    Args:
        seq (str): The target sequence.
        char_keys (str): An oredered string for all the elementary tokens of the sequences.

    Returns:  
        numpy.ndarray: The one-hot encoding matrix.
    """
    result = np.zeros((len(string),len(char_keys)))
    for i, char in enumerate(string):
        result[i,char_keys.index(char)] = 1
    return result

#################################################################################
# 1-D Sequences Evaluation
#################################################################################

def sequence_indentity(seq_1, seq_2, version = 'BLAST'): # by SZ
    """Calculate the identity between two sequences.

    Args:
        seq_1, seq_2 (str): The protein sequence.
        version (str): The version of the sequence identity.

    Returns:  
        float: The sequence identity.
    """
    l_x = len(seq_1)
    l_y = len(seq_2)
    X = seq_1.upper()
    Y = seq_2.upper()

    if version == 'BLAST':
        alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
        max_iden = 0
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
    else:
        print('Error! No sequence identity version named %s!'%version)
    return identity

def msi_cal(seq_output, seq_input): # by SZ
    """Calculate the maximum sequence identity of the target sequences.

    Args:
        seq_output (list): The list of the target protein sequences.
        seq_input (list): The list of the reference sequences.

    Returns:  
        list: The list of the maximum sequence identities.
    """
    id_list = []
    for sout in seq_output:
        max_id=0.
        for sin in seq_input:
            idd = Identity(sin, sout)
            max_id = max(max_id, idd)
        id_list.append(max_id)
    return id_list

def calculate_biophysical_prop(myfile):  # by SZ
    """Calculate the biological properties including stability, aromativity, GRAVY, and AAF.

    Args:
        myfile (str): The path of the sequence file

    Returns:  
        tuple: The biological properties of the sequences.
    """
    ssf = []
    aromaticity = []
    gravy = []
    aaf = {}
    count=0.
    with open(myfile) as f:
         for line in f:
             line = line.strip()
             if "No Successful Sequence." in line:
                return -1
             x = ProteinAnalysis(line)
             ssf.append(x.secondary_structure_fraction())
             aromaticity.append(x.aromaticity())
             gravy.append(x.gravy())
             aaf_temp = x.get_amino_acids_percent()
             aaf = { k: aaf_temp.get(k, 0) + aaf.get(k, 0) for k in set(aaf_temp) | set(aaf) }
             count += 1.

    aaf = { k: round(aaf.get(k, 0)/count,3) for k in set(aaf) }
    ssf = np.asarray(ssf)
    ssf_mean = np.mean(ssf,axis=0)
    aromaticity = np.asarray(aromaticity)
    aromaticity_mean = np.mean(aromaticity)
    gravy = np.asarray(gravy)
    gravy_mean = np.mean(gravy)
    return ssf,aromaticity,gravy,aaf


class SequenceTokenizersClass():
    """Tokenizer class for protein sequences. 
        Can use different vocabs depending on the model and applicable to other sequence datasets, e.g. Uniref

        Author: Yuanfei Sun
    """

    def __init__(self, vocab: str = 'pfam'):
        if vocab == 'pfam':
            self.vocab = PFAM_VOCAB
        else:
            raise Exception("vocab not known!")
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab
        self.blosum62Vec = self.convert_blosumMat_to_blosumVec(matrix)
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    @property
    def blosum62Vec(self) -> dict:
        return self.blosum62Vec

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)
    
    def convert_token_to_blosum62Vec(self, token: str) -> np.ndarray:
        return self.blosum62Vec[token]    
    
    @classmethod
    def convert_blosumMat_to_blosumVec(blosumMat: dict) -> dict:
        blosumVec = {}
        blosum_aa_idx = {}
        len_aa = len(BLOSUM62_AA_LIST)
        for aa_i in range(len_aa):
            blosum_aa_idx[BLOSUM62_AA_LIST[aa_i]] = aa_i
            blosumVec[BLOSUM62_AA_LIST[aa_i]] = np.zeros(len_aa)
        for key, value in blosumMat.items():
            aa1,aa2 = key
            blosumVec[aa1][blosum_aa_idx[aa2]] = value
            blosumVec[aa2][blosum_aa_idx[aa1]] = value
        return blosumVec


def pssm_from_msa_a3m(msa_file: str):
    """Calculate positional specific substitution matrix PSSM from a MSA input (a3m format)
    
    Author: Yuanfei Sun

    Args:
        msa_file: MSA file path
    
    Returns:
        numpy.ndarray: PSSM matrix of size L*20
    """
    return null

def pssm_from_msa_stockholm(msa_file: str):
    """Calculate positional specific substitution matrix PSSM from a MSA input (stockholm format)
    
    Author:Yuanfei Sun

    Args:
        msa_file: MSA file path
    
    Returns:
        numpy.ndarray: PSSM matrix of size L*20
    """
    return null

def seq_predicted_ss():
    """Predicted Secondary Structure from sequence alone
    Need to find sota methods
    
    """
    return null

def seq_predicted_rsa():
    """Predicted Relative Solvent Accessibility from sequence alone
    Need to find sota methods
    
    """
    return null


### The function for imputing the Cb position for Gly ###
def getGlyCbPosBynccaCoord(list_atomCoords):
	""" Calculate the C Beta position for the Glycine

	    Args:
	        1. list_atomCoords (list): A list of tuples of xyz coordinates for N, C, CA atoms. ([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)])

	    Returns:
	        1. cb (list): The position coordinates vector for C Beta

	"""

	# check if n/c/ca coordinate info exists. (If no, return None)
	try:
		# get atom coordinates as vectors
		n = Vector(list_atomCoords[0])
		c = Vector(list_atomCoords[1])
		ca = Vector(list_atomCoords[2])
	except Exception:
		return None

	# center at origin
	n = n - ca
	c = c - ca
	# find rotation matrix that rotates n -120 degrees along the ca-c vector
	rot = rotaxis2m(-pi * 120.0 / 180.0, c)
	# apply rotation to ca-n vector
	cb_at_origin = n.left_multiply(rot)
	# put on top of ca atom
	cb = cb_at_origin + ca
	cb = list(cb)

	return cb

def parse_PSAIA_table_file(path_to_txt_file):
    """ parse the output file from PSAIA to get the vertex features {"rASA", "PI", "Hydrophobicity"}

    Args:
        path_to_txt_file (str): the path to the file to be parsed

    Returns:
        dict_out (dict): a python dictionary containing the information about the ATOM residues including 
            1) chain id: the chain they are on 
            2) lists of floats/floats for {"rASA", "PI", "Hydrophobicity"} 
            3) res name
            4) res id
        dict_res_ind_ls (dict): a python dictionary containing the lists of res ids for all chain ids ({chain id: lists of res ids})

    """
    # a list of column header names that are useful
    ls_col_name = ['chain id', 'res id', 'res name', 'total RASA', 'average CX', 's_avg CX', 's-ch avg CX', 's-ch s_avg CX', 'max CX', 'min CX', 'Hydrophobicity']

    dict_out = {}
    dict_res_ind_ls = {}

    start_line_idx = 8

    idx = 0
    with open(path_to_txt_file, "r") as rf:
        # skip the headers
        for line in rf:
            if idx < start_line_idx:
                idx += 1
                continue
            else:
                idx += 1
                splt_line = line.strip().split()

                chain_id = splt_line[dict_PSAIA_table_file_name_to_col_idx['chain id']]
                res_id = splt_line[dict_PSAIA_table_file_name_to_col_idx['res id']]
                res_name = splt_line[dict_PSAIA_table_file_name_to_col_idx['res name']]
                total_RASA = float(splt_line[dict_PSAIA_table_file_name_to_col_idx['total RASA']])
                CX_np = [float(splt_line[dict_PSAIA_table_file_name_to_col_idx[ls_col_name[i]]]) for i in range(4,10)]
                Hydrophobicity = float(splt_line[dict_PSAIA_table_file_name_to_col_idx['Hydrophobicity']])

                # whether there is a chain in the both dicts
                if chain_id not in dict_out:
                    dict_out[chain_id] = {}
                if chain_id not in dict_res_ind_ls:
                    dict_res_ind_ls[chain_id] = []
                
                # complete both the dict for The residue
                dict_out[chain_id][res_id] = {}
                dict_out[chain_id][res_id]['res name'] = res_name
                dict_out[chain_id][res_id]['total RASA'] = total_RASA
                dict_out[chain_id][res_id]['CX'] = CX_np
                dict_out[chain_id][res_id]['Hydrophobicity'] = Hydrophobicity

                dict_res_ind_ls[chain_id].append(res_id)

    return dict_out, dict_res_ind_ls

def NormalizeData(data, axis = 0):
    """ Normalize data along some axis 
    """
    return (data - np.min(data, axis=axis)) / (np.max(data, axis=axis) - np.min(data, axis=axis))


def parse_pdb_list_file(path_to_pdb_list):
    """ parse the text file for a bunch of pdb list

        Args:
            path_to_pdb_list (str): the path to the text file which contains all the pdb file names 
                (one file name for one line)
        
        Returns:
            abs_pdb_list (list(str)): list of absolute paths to the pdb files
            rel_pdb_list (list(str)): list of relative paths to the pdb files (against the inner most folder)
            ls_name (list(str)): list of names of proteins (the relative path w/o ".pdb")

    """

    abs_pdb_list = []
    rel_pdb_list = []
    ls_name = []
    with open(path_to_pdb_list, "r") as rf:
        for line in rf:
            abs_path = line.strip()
            ls_name.append(line.strip().split('/')[-1][:-4])
            rel_pdb_list.append(line.strip().split('/')[-1])
            abs_pdb_list.append(abs_path)
    
    return abs_pdb_list, rel_pdb_list, ls_name




####TODO: Pass in a chain order list argument and make the function read out the features in this order

### rASA, Normalized Protrusion Index, Normalized Hydrophobicity calculation function ###
def calc_PSAIA_features(path_to_pdb_list, path_to_exe = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/bin/linux/psa/psa", \
path_to_config = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/psaia_config_file_input.txt", \
path_to_output_dir = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/OUTPUT",\
rASA = True, PI = True, Hydrophobicity = True, \
preferred_chain_order_file = None):
    """ rASA, Normalized Protrusion Index, Normalized Hydrophobicity calculation function

        Args:
            path_to_pdb_list (str): the path to the text file which contains all the pdb file names 
                (one file name for one line)
            path_to_exe (str): the path to the installed the PSAIA executables 
                (default value: "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/bin/linux/psa/psa")
            path_to_config (str): the path to the configuration file for the PSAIA software
                (default value: "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/psaia_config_file_input.txt")
            path_to_output_dir (str): the path to the output directory for the PSAIA software which should correspond to the specifications in the config file
                (default value: "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/OUTPUT")
            rASA (bool): whether rASA is part of the outputs
            PI (bool): whether PI is part of the outputs
            Hydrophobicity (bool): whether Hydrophobicity is part of the outputs
            preferred_chain_order_file (str): a text file contains chain order strings with one line for one protein each (for one line e.g. AB)
                (default value: None)
    
        Returns:
            Out_dict (dict): a python dictionary containing keys {the pdb file names w/o the suffix ".pdb"} 
                whose corresponding value is a python dict with 
                (1) keys some/all of {"rASA", "PI", "Hydrophobicity"} and values {a numpy array with shape (n_res, n_features)} pairs 
                (2) chain_id: the list of res idx
                (3) "chain order": the order according to which the chains are concatenated

        Notes:
            "Normalized" here means normalizing the features on a per protein basis

    """

    # Initialize the output python dict
    Out_dict = {}

    #
    # read the text files specified in the path_to_pdb_list
    abs_pdb_list, rel_pdb_list, ls_name = parse_pdb_list_file(path_to_pdb_list)

    # set of names
    set_name = set(ls_name)


    #
    # execute the executables
    cmd_str = path_to_exe + " " + path_to_config + " " + path_to_pdb_list
    os.system(cmd_str)
    
    #
    # read the output from the output directotry containing all the output table files

    # check whether path_to_output_dir ends with "/" and append one if not
    if path_to_output_dir[-1] != "/":
        path_to_output_dir += "/"

    ls_output_file_names = os.listdir(path_to_output_dir)

    for file_name in ls_output_file_names:

        name = file_name[:-25]

        if name not in set_name:
            continue

        abs_fname = path_to_output_dir + file_name

        dict_out, dict_res_ind_ls = parse_PSAIA_table_file(abs_fname)

        # rASA (shape: (n_res, ))
        # Normalized Protrusion Index (shape: (n_res, 6))
        # Hydrophobicity (shape: (n_res, ))

        pro_rASA = []
        pro_PI = []
        pro_H = []
        chain_order = []

        for chain_id in dict_res_ind_ls:
            ls_idx = dict_res_ind_ls[chain_id]

            # figure out the chain order
            chain_order.append(chain_id)
            
            for idx in ls_idx:
                total_RASA = dict_out[chain_id][idx]['total RASA']
                CX_np = dict_out[chain_id][idx]['CX']
                Hydrophobicity = dict_out[chain_id][idx]['Hydrophobicity']

                pro_rASA.append(total_RASA)
                pro_PI.append(CX_np)
                pro_H.append(Hydrophobicity)
        
        # convert to numpy array
        pro_rASA = np.array(pro_rASA, dtype=np.float)
        pro_PI = np.array(pro_PI, dtype=np.float)
        pro_H = np.array(pro_H, dtype=np.float)

        # #####
        # print(pro_rASA.shape)
        # print(pro_PI.shape)
        # print(pro_H.shape)

        # normalize PI and Hydrophobicity on a per protein basis
        normed_pro_PI = NormalizeData(pro_PI)
        normed_pro_H = NormalizeData(pro_H)

        #
        # complete the Out_dict
        Out_dict[name] = {}

        # (1) keys some/all of {"rASA", "PI", "Hydrophobicity"} and values {a numpy array with shape (n_res, n_features)} pairs
        Out_dict[name]["rASA"] = pro_rASA
        Out_dict[name]["PI"] = normed_pro_PI
        Out_dict[name]["Hydrophobicity"] = normed_pro_H

        # (2) chain_id: the list of res idx
        Out_dict[name].update(dict_res_ind_ls)

        # (3) "chain order": the order according to which the chains are concatenated
        Out_dict[name]["chain order"] = chain_order

        #####
        print(name)

    return Out_dict

#
# test function calc_PSAIA_features(path_to_pdb_list, path_to_exe = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/bin/linux/psa/psa", \
# path_to_config = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/psaia_config_file_input.txt", \
# path_to_output_dir = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/OUTPUT",\
# rASA = True, PI = True, Hydrophobicity = True, \
# preferred_chain_order_file = None)
# path = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/pdb_list.fls"
# print(calc_PSAIA_features(path))


### residue depth calculation function ###
def cal_residue_depth(path_to_pdb_list):
    """ Calculate residue depth using biopython for average distance and C-alpha distance

        Args:
            path_to_pdb_list (str): the path to the text file which contains all the pdb file names 
                (one file name for one line)
            
        Returns:
            Out_dict (dict): a python dictionary containing keys {the pdb file names w/o the suffix ".pdb"} 
                whose corresponding value is a numpy array with shape (n_res, 2)
                (0th feature is for the average distance and 1th for the C-alpha distance)
        Notes:
            First please install the MSMS software tool and correctly set the env var PATH

    """
    
    # output
    Out_dict = {}


    #
    # read the text files specified in the path_to_pdb_list

    # get: (1) the name of the pdb files (the one wo ".pdb"); 
    #      (2) the abs path for the pdb files 

    # the list of abs path for the pdb files
    # the list of the name of the pdb files
    abs_pdb_list, rel_pdb_list, ls_name = parse_pdb_list_file(path_to_pdb_list)

    #
    # get all the residue depths
    for path, name in zip(abs_pdb_list, ls_name):
        parser = PDBParser()
        structure = parser.get_structure(name, path)
        model = structure[0]

        # get the surface
        surface = get_surface(model)

        # get all the residue in the model and compute two kinds of residue depths
        ls_all_rd = []
        for res in model.get_residues():
            ls_rd = []
            # ls_rd.append(float(residue_depth(res, surface)))
            # ls_rd.append(float(ca_depth(res, surface)))
            # print(type(residue_depth(res, surface)))
            ls_rd.append(residue_depth(res, surface))
            ls_rd.append(ca_depth(res, surface))
            # ls_rd = np.array(ls_rd)
            ls_all_rd.append(ls_rd)
    
        #
        # complete the dictionary
        Out_dict[name] =  np.array(ls_all_rd, dtype=float)

        ######
        print(name)
    
    return Out_dict

# test the residue depth calculation
# path = "/scratch/user/rujieyin/Uniprot-ID-Filter/yes_pdb_file_path_list.txt"
# cal_residue_depth(path)

### Half Sphere Amino Acid Composition calculation function ###
def cal_HSAAC(path_to_pdb_list, cut_off = 8.0):
    """ calculate Half Sphere Amino Acid Composition

        Args:
            path_to_pdb_list (str): the path to the text file which contains all the pdb file names 
                (one file name for one line) 
            cut_off (float): the cut off distance used for neighborhood definition 
                (default value: 8.0) 

        Returns:
            Out_dict (dict): a python dictionary containing keys {the pdb file names w/o the suffix ".pdb"} 
                whose corresponding value is a numpy array with shape (n_res, 40)
                (0-19th percentage features are for the up sphere and 20-39th percentage features are for the down sphere)

    """
    # initialize the output Out_dict
    Out_dict = {}

    #
    # read the text files specified in the path_to_pdb_list
    abs_pdb_list, rel_pdb_list, ls_name = parse_pdb_list_file(path_to_pdb_list)

    
    for abs_path, rel_path, name in zip(abs_pdb_list, rel_pdb_list, ls_name):

        #
        # parse the pdb files with biopython PDBparser
        parser = PDBParser()

        structure = parser.get_structure(name, abs_path)
        model = structure[0]

        #
        # the side chain vector (the average of the unit vectors of the side chain vectors)
        #                       (if the interested residue is GLY, the side chain vector is the CA-virtual_CB vector)

        dict_res_to_sc_vec = {}
        ls_res = []

        for r in model.get_residues():
            #
            # average side chain vector
            ave_sc_vec = None
            # list of res objects
            ls_res.append(r)
            
            if r.get_resname() == 'GLY':
                # For GLY
                try:
                    list_atomCoords = [tuple(r['N'].get_vector()), tuple(r['C'].get_vector()), tuple(r['CA'].get_vector())]
                    

                    # below two objects are python list
                    CB_coord = getGlyCbPosBynccaCoord(list_atomCoords)

                    CA_coord = list(r['CA'].get_vector())
        #             The following line will throw the error message: "TypeError: unsupported operand type(s) for -: 'list' and 'list'""
        #             print(CA_coord-CB_coord)
                    CA_CB_vec = Vector(CB_coord)-Vector(CA_coord)
        #             print(CA_CB_vec)

                    CA_CB_vec = CA_CB_vec/CA_CB_vec.norm()

                    # use CA-CB vector as the average side chain vector for GLY
                    ave_sc_vec = CA_CB_vec
                except:
                    # TODO:
                    print("the GLY doesn't have all three bb atoms coordinates!!!")
                    print(r.get_resname())
                    # for atom in r:
                    #     print(atom.get_name())
                    pass
            else:
                try:
                    sum_norm_vec = Vector(0,0,0)
                    ct_vec = 0

                    for atom in r.get_atoms():
                        if atom.get_name() in set_bb_atom_name:
                            continue
                        else:
                            diff_vec = atom.get_vector()-r['CA'].get_vector()
                            sum_norm_vec += diff_vec/diff_vec.norm()
                            ct_vec += 1

                    ave_sc_vec = sum_norm_vec / (ct_vec * 1.0)
                except:
                    print("the non GLY doesn't have some atoms coordinates!!!")
            
            # complete the dict_res_to_sc_vec
            dict_res_to_sc_vec[r] = ave_sc_vec

        #
        # complete the Output numpy array

        atom_list = Selection.unfold_entities(model, 'A') # A for atoms
        # print(atom_list)
        ns = NeighborSearch(atom_list)
        neighbors = ns.search_all(cut_off, 'R') # 8.0 for distance in angstrom
        neighbors = set(neighbors)

        # dict from res to idx
        dict_res_to_idx = dict([(res, i) for i, res in enumerate(ls_res)])

        # !!!!!!!!
        n_res = len(ls_res)

        # Output (dtype: np.array(n_res, 40); (0-19) for the positively correlated residues; (20-39)  for the negatively correlated residues)
        Output = np.zeros((n_res, 40))

        for pair in neighbors:
            if pair[0] == pair[1]:
        #         print("!!!!!!!!")
                continue
            for i in range(2):
                
                try:
                    tar_res, nb_res = pair[(0+i)%2], pair[(1+i)%2]
        #             print(tar_res, nb_res)

                    if not is_aa(nb_res, standard=True):
                        continue
                    # side chain vector
                    sc_vec = dict_res_to_sc_vec[tar_res]
                    # displacement vector (CA_nb - CA_tar)
                    dp_vec = nb_res['CA'].get_vector() - tar_res['CA'].get_vector()

                    #
                    # determine whether the dp_vec is on the up/down sphere
                    resname = nb_res.get_resname()
                    if resname not in dict_AA_to_idx:
                        continue

                    if sc_vec*dp_vec >= 0:
                        Output[dict_res_to_idx[tar_res], dict_AA_to_idx[resname]] += 1
                    else:
                        Output[dict_res_to_idx[tar_res], dict_AA_to_idx[resname]+20] += 1
                except:
                    print("Some key error in the line: for pair in neighbors:")
                    pass
        
        # percentagize Output
        Output[:, :20] /= (np.sum(Output[:, :20], axis = 1).reshape(-1, 1) + eps)
        Output[:, 20:] /= (np.sum(Output[:, 20:], axis = 1).reshape(-1, 1) + eps)


        #
        # store the value into the Out_dict

        Out_dict[name] = Output

        #####
        print(name)
    
    return Out_dict


""" PSSM
Reference: https://support.cyrusbio.com/workflows/create-a-pssm/
1. Donwload alignment database: wget ftp://ftp.ncbi.nih.gov/blast/db/FASTA/nr.gz
2. Build database: PATH\ncbi-blast-2.7.1+\bin\makeblastdb -in PATH\nr -input_type fasta -title nonR -dbtype prot
3. Prepare your fasta file for proteins, e.g. 5ORB.fasta
4. PSSM: perl PATH\blast+\bin\legacy_blast.pl blastpgp -d PATH\nr -j 4 -b 1 -a 2 -Q PATH\5ORB.pssm -i PATH\5ORB.fasta –path PATH\blast+\bin
"""

class AAEmbedding:
    """residue level features: hydropathy; volume; charge; polarity; a hydrogen bond donor or receptor
    """

    def __init__(self):
        super(self).__init__()
        self.hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
        self.volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
        self.charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
        self.embedding = torch.tensor([
            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa],
            self.polarity[aa], self.acceptor[aa], self.donor[aa]]
            for aa in ALPHABET
        ]).cuda()

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = D_mu.view(1,1,-1)  # [1, 1, K]
        D_expand = torch.unsqueeze(D, -1)  # [B, N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, :, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, :, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, :, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, :, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        B, N = x.size(0), x.size(1)
        aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs

    def soft_forward(self, x):
        B, N = x.size(0), x.size(1)
        aa_vecs = torch.matmul(x.reshape(B * N, -1), self.embedding).view(B, N, -1)
        rbf_vecs = self.transform(aa_vecs)
        return rbf_vecs
