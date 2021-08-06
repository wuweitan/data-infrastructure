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
    '''Calculate the TM-scores between two protein structures

    :param pdb_1, pdb_2: path of the pdb files
    :type pdb_1, pdb_2: str
    :rtype: float
    ''' 
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

class Structure():
    """
    Structure-based evaluation
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



def pdb_info_load(pdb_file, chains = None):
    '''
    Extract the residue information inside a pdb file. Ignore the missing residues.
    '''
    AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I', 'HSE':'H',
               'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

    protein_dict = {}
    index_dict = {} # The indexes of the residues.

    with open(pdb_file,'r') as p_file:
        lines = p_file.readlines()
        for line in lines:
            if line[0:4] == 'ATOM':
                ### residue-wise info ###
                if line[16] == ' ' or line[16] == 'A' or line[16] == '1':
                    atom = remove(line[12:16],' ')
                else:
                    atom = remove(line[12:16],' ') + '_' + line[16]
                resi = line[17:20]
                chain = line[21]

                if chains is None or chain in chains:
                    index_all = remove(line[22:27],' ')
                    ### atom-wise info ###
                    x = float(remove(line[30:38],' '))
                    y = float(remove(line[38:46],' '))
                    z = float(remove(line[46:54],' '))
        ############ Judge whether a new chain begins. ########################
                    if not chain in protein_dict.keys():
                        protein_dict[chain] = {'coor':{}, 'seq':''}
                        index_dict[chain] = []
        ############ Save the sequence infomation. ######################## 
                    if not index_all in protein_dict[chain]['coor'].keys():
                        protein_dict[chain]['coor'][index_all] = {'resi':resi}
                        index_dict[chain].append(index_all)
                        protein_dict[chain]['seq'] += AA_dict[resi]
                    elif resi != protein_dict[chain]['coor'][index_all]['resi']:
                        print('PDB read error! The residue kind of resi %s is not consistent for %s!'%(index_all,pdb_file))
                        return 0
        ############ atom coordinates. ########################
                    if not atom in protein_dict[chain]['coor'][index_all].keys():
                        protein_dict[chain]['coor'][index_all][atom] = np.array([x,y,z])
    print(pdb_file)
    print('%d chains processed.'%len(index_dict.keys()))
    for c in index_dict.keys():
        print('Chain %s: %d residues'%(c, len(index_dict[c])))
    return protein_dict, index_dict

    

def fnat(target_path, native_path, recepter_chains, ligand_chains, cutoff = 5.0):
    """
    target_path: path pf the target pdb file
    native_path: path of the groundtruth pdb file 
    recepter_chains: list of receptor chains
    ligand_chains: list of ligand chains
    cufoff: distance cutoff
    """
    ### load the info
    target_rec, target_rec_index = pdb_info_load(target_path, recepter_chains)
    target_lig, target_lig_index = pdb_info_load(target_path, ligand_chains)

    native_rec, native_rec_index = pdb_info_load(native_path, recepter_chains)
    native_lig, native_lig_index = pdb_info_load(native_path, ligand_chains)

    ### residue map
    map_dict = {} # from target to native
    # receptor
    print('Receptor alignment:')
    for chain in recepter_chains:
        if chain in target_rec.keys() and chain in native_rec.keys():
            map_dict[chain] = {}
            seq_tar = target_rec[chain]['seq']
            seq_nat = native_rec[chain]['seq']
            alignment = pairwise2.align.globaldd(seq_tar,seq_nat, matrix,-11,-1,-11,-1)[0] 
            print('Chain %s:'%chain)
            print(alignment[0])
            print(alignment[1])

            idx_t = 0
            idx_n = 0
            for i in range(alignment[-1]):
                if alignment[0][i] != '-':
                    if alignment[0][i] == alignment[1][i]:
                        map_dict[chain][target_rec_index[chain][idx_t]] = native_rec_index[chain][idx_n]
                    idx_t += 1
                if alignment[1][i] != '-':
                    idx_n += 1
    # ligand
    print('Ligand alignment:')
    for chain in ligand_chains:
        if chain in target_lig.keys() and chain in native_lig.keys():
            map_dict[chain] = {}
            seq_tar = target_lig[chain]['seq']
            seq_nat = native_lig[chain]['seq']
            alignment = pairwise2.align.globaldd(seq_tar,seq_nat, matrix,-11,-1,-11,-1)[0]
            print('Chain %s:'%chain)
            print(alignment[0])
            print(alignment[1])

            idx_t = 0
            idx_n = 0
            for i in range(alignment[-1]):
                if alignment[0][i] != '-':
                    if alignment[0][i] == alignment[1][i]:
                        map_dict[chain][target_lig_index[chain][idx_t]] = native_lig_index[chain][idx_n]
                    idx_t += 1
                if alignment[1][i] != '-':
                    idx_n += 1
    ### target contact pairs
    print('Counting target contacts...')
    contact_dict = {}
    concat_num = 0
    # chain
    for chain_r in target_rec.keys():
        for chain_l in target_lig.keys():
            print((chain_r,chain_l))
            contact_dict[(chain_r,chain_l)] = []
            # residue 
            for resi_r in target_rec[chain_r]['coor'].keys():
                for resi_l in target_lig[chain_l]['coor'].keys():
                    # atom
                    for atom_r in target_rec[chain_r]['coor'][resi_r].keys():
                        flag = False
                        if atom_r != 'resi': #and atom_r[0] != 'H':
                            for atom_l in target_lig[chain_l]['coor'][resi_l].keys():
                                if atom_l != 'resi': #and atom_l[0] != 'H':
                                    dist = np.linalg.norm(target_rec[chain_r]['coor'][resi_r][atom_r] - target_lig[chain_l]['coor'][resi_l][atom_l])
                                    # judge
                                    if dist <= cutoff: 
                                        contact_dict[(chain_r,chain_l)].append((resi_r, resi_l))
                                        concat_num += 1
                                        flag = True
                                        break
                        if flag:
                            break
    print('%d contacts in the target complex.'%concat_num) 
    ### native contacts
    nat_num = 0
    for pair in contact_dict.keys():
        chain_r = pair[0]
        chain_l = pair[1]
        if chain_r in native_rec.keys() and chain_l in native_lig.keys():
            for resi_pair in contact_dict[pair]:
                resi_r = resi_pair[0]
                resi_l = resi_pair[1]
                if resi_r in map_dict[chain_r].keys() and resi_l in map_dict[chain_l].keys():
                    nat_resi_r = map_dict[chain_r][resi_r]
                    nat_resi_l = map_dict[chain_l][resi_l]
 
                    for atom_r in native_rec[chain_r]['coor'][nat_resi_r].keys():
                         flag = False
                         if atom_r != 'resi': #and atom_r[0] != 'H':
                             for atom_l in native_lig[chain_l]['coor'][nat_resi_l].keys():
                                 if atom_l != 'resi': #and atom_l[0] != 'H':
                                     dist = np.linalg.norm(native_rec[chain_r]['coor'][nat_resi_r][atom_r] - native_lig[chain_l]['coor'][nat_resi_l][atom_l])
                                     # judge
                                     if dist <= cutoff:
                                         nat_num += 1
                                         flag = True
                                         break
                         if flag:
                             break
    return float(nat_num) / concat_num



