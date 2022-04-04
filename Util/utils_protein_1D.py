#################################################################################
# Utility functions for data process and evaluation on 1-D sequences.
#################################################################################

import numpy as np
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist 

matrix = matlist.blosum62

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
