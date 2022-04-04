#################################################################################
# Contain functions and classes to help to extract secondary structure elements
# and generate adjacent matrix and edge type. 
#################################################################################

import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import random
import networkx as nx
import requests
import json

#################################################################################
# Auxiliary Funtions 
#################################################################################

def remove(string, char): # by SZ
    """
    Remove the certain characters in a string.
    """
    string_char = string.split(char)
    return ''.join(string_char)

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
# Acquire Information from the Sturctures
#################################################################################

def read_pdb(pdb_file):  # by SZ
    """Extract the residue information from a pdb file into dictionaries.

    Args:
        pdb_file (str): The path of the PDB file.

    Returns: 
        dict: A hierarchical dictionary containing the information of each atom in the PDB file.
        dict: A dictionary containing the ordered residue indices of each chain.
    """

    protein_dict = {}
    index_dict = {} # The indexes of the residues. Two level: the first is for the number, the second is for the letter.
                    # e.g. '100', '100A', '100B','101','102'...then the dictionary is {...,100:[' ','A','B'],101:[' '],...}
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
                index_all = remove(line[22:27],' ')
                index = int(remove(line[22:26],' '))
                insertion_code = line[26]
                ### atom-wise info ###
                x = float(remove(line[30:38],' '))
                y = float(remove(line[38:46],' '))
                z = float(remove(line[46:54],' '))
                occupancy = float(remove(line[54:60],' '))
                temp_factor = float(remove(line[60:66],' '))
        ############ Judge whether a new chain begins. ########################      
                if not chain in protein_dict.keys():
                    protein_dict[chain] = {}
                    index_dict[chain] = {}
        ############ Save the sequence infomation. ######################## 
                if not index_all in protein_dict[chain].keys():
                    protein_dict[chain][index_all] = {'index':index,'insertion_code':insertion_code,'resi':resi}
                    if not index in index_dict[chain]:
                        index_dict[chain][index] = [insertion_code]
                    else:
                        index_dict[chain][index].append(insertion_code)
                elif resi != protein_dict[chain][index_all]['resi']:
                    print('PDB read error! The residue kind of resi %s is not consistent for %s!'%(index_all,pdb_file))
                    return 0
                if not atom in protein_dict[chain][index_all].keys():
                    protein_dict[chain][index_all][atom] = [x,y,z,occupancy,occupancy,temp_factor]
    return protein_dict, index_dict


class PDB_information(object): # by SZ
    """Extract the information from the *.pdb file with the DSSP program.

    Args:
        pdb_file (str): path of the target PDB file. 
        ss_kind (int): version of the secondary structure, 3 or 8. Default: ``3``

    """
    def __init__(self,pdb_file,ss_kind = 3):

        BREAK = False

        ss_map_8_3 = {'H':'H','G':'H','I':'H','E':'E','B':'E','S':'C','T':'C','-':'C'} # 8-classes to 3-classes

        if ss_kind == 3:
            ss_map_dict = ss_map_8_3 # 8-classes to 3-classes 
        elif ss_kind == 8:
            ss_map_dict = {'H':'H','G':'G','I':'I','E':'E','B':'B','S':'S','T':'T','-':'-'} # 8-classes to 8-classes
        else:
            print('Error! Secondary structure can only be 3-classes or 8-classes, not %s-classes!'%str(ss_kind))
            BREAK = True

        if not BREAK:
            AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
                       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
                       'ASX':'B','GLX':'Z'}

            self.protein_dict, self.index_dict = read_pdb(pdb_file)
            dssp_dict = dssp_dict_from_pdb_file(pdb_file)[0]

            self.Seq_dict = {}
            self.SS_dict = {}
            self.SS_dict_3 = {}

            for chain in self.protein_dict.keys():
                Complete_Seq = ''
                Complete_SS = ''
                Complete_SS_3 = ''

                index_list = self.index_dict[chain].keys()
                index_min = min(index_list)
                index_max = max(index_list)

                for index_n in range(index_min, index_max + 1):
                    if index_n in index_list:
                        for index_l in self.index_dict[chain][index_n]:

                            index = remove(str(index_n) + index_l,' ')

                            if self.protein_dict[chain][index]['resi'] in AA_dict.keys():
                                Resi_AA = AA_dict[self.protein_dict[chain][index]['resi']]
                            else:
                                print('Unknown amino acid kind "%s" at position %s in %s'%(self.protein_dict[chain][index]['resi'],index,pdb_file))
                                Resi_AA = 'X'

                            if (chain, (' ', index_n, index_l)) in dssp_dict.keys():

                                SecStru = dssp_dict[chain, (' ', index_n, index_l)][1]
                                DSSP_AA = dssp_dict[chain, (' ', index_n, index_l)][0]

                                ASA = dssp_dict[chain, (' ', index_n, index_l)][2]

                                SS_3 = ss_map_8_3[SecStru]
                                if SS_3 == 'C':
                                    if ASA <= 25:
                                        ASA_level = 'core'
                                    elif ASA < 40:
                                        ASA_level = 'boundary'
                                    else:
                                        ASA_level = 'surface'
                                else:
                                    if ASA <= 15:
                                        ASA_level = 'core'
                                    elif ASA < 60:
                                        ASA_level = 'boundary'
                                    else:
                                        ASA_level = 'surface'

                                SecStru = ss_map_dict[SecStru]

                                Phi = dssp_dict[chain, (' ', index_n, index_l)][3]
                                Psi = dssp_dict[chain, (' ', index_n, index_l)][4]
                                DSSP_idx = dssp_dict[chain, (' ', index_n, index_l)][5]
                                NH_O1_relix = dssp_dict[chain, (' ', index_n, index_l)][6]
                                NH_O1_energy = dssp_dict[chain, (' ', index_n, index_l)][7]
                                O_NH1_relix = dssp_dict[chain, (' ', index_n, index_l)][8]
                                O_NH1_energy = dssp_dict[chain, (' ', index_n, index_l)][9]
                                NH_O2_relix = dssp_dict[chain, (' ', index_n, index_l)][10]
                                NH_O2_energy = dssp_dict[chain, (' ', index_n, index_l)][11]
                                O_NH2_relix = dssp_dict[chain, (' ', index_n, index_l)][12]
                                O_NH2_energy = dssp_dict[chain, (' ', index_n, index_l)][13]

                                if Resi_AA != DSSP_AA and DSSP_AA != 'X':
                                    print('Residue Error! %s and %s do not match! %s'%(Resi_AA,DSSP_AA,pdb_file))

                            else:
                                ASA = np.nan
                                ASA_level = None
                                Phi = np.nan
                                Psi = np.nan
                                DSSP_idx = np.nan
                                NH_O1_relix = np.nan
                                NH_O1_energy = np.nan
                                O_NH1_relix = np.nan
                                O_NH1_energy = np.nan
                                NH_O2_relix = np.nan
                                NH_O2_energy = np.nan
                                O_NH2_relix = np.nan
                                O_NH2_energy = np.nan
                                SecStru = 'M'
                                SS_3 = 'M'

                            self.protein_dict[chain][index]['AminoAci'] = Resi_AA
                            self.protein_dict[chain][index]['SeconStru'] = SecStru
                            self.protein_dict[chain][index]['ASA'] = ASA
                            self.protein_dict[chain][index]['ASA_level'] = ASA_level
                            self.protein_dict[chain][index]['Phi'] = Phi
                            self.protein_dict[chain][index]['Psi'] = Psi
                            self.protein_dict[chain][index]['DSSP_idx'] = DSSP_idx
                            self.protein_dict[chain][index]['NH_O1_relix'] = NH_O1_relix
                            self.protein_dict[chain][index]['NH_O1_energy'] = NH_O1_energy
                            self.protein_dict[chain][index]['O_NH1_relix'] = O_NH1_relix
                            self.protein_dict[chain][index]['O_NH1_energy'] = O_NH1_energy
                            self.protein_dict[chain][index]['NH_O2_relix'] = NH_O2_relix
                            self.protein_dict[chain][index]['NH_O2_energy'] = NH_O2_energy
                            self.protein_dict[chain][index]['O_NH2_relix'] = O_NH2_relix
                            self.protein_dict[chain][index]['O_NH2_energy'] = O_NH2_energy

                            Complete_Seq += Resi_AA
                            Complete_SS += SecStru
                            Complete_SS_3 += SS_3
                    else:
                        Complete_Seq += '*'
                        Complete_SS += '*'
                        Complete_SS_3 += '*'
                self.Seq_dict[chain] = Complete_Seq
                self.SS_dict[chain] = Complete_SS
                self.SS_dict_3[chain] = Complete_SS_3

#**************************** Acquire Secondary Structures *********************************

def read_pdb_ss(header_file, chain_ref): # by SZ
    """Extract the secondary structure region for the certain chain from a PDB header.

    Args:
        header_file (str): The path of the PDB file (with the header) or the PDB Header file.
        chain_ref (str): The chain to be queied.

    Returns: 
        dict: A dictionary tell the region of the helices and the strands of the target chain (the other parts are coils).
    """

    ss_index_dict = {'SHEET':[], 'HELIX':[]}
    with open(header_file, 'r') as rf:
        lines = rf.readlines()
    flag = True
    for line in lines:
        if line.startswith('HELIX'):
            chain = line[19]
            if chain == chain_ref:
                if flag and chain != line[31]:
                    print('Error! Chains do not match for %s!'%header_file)
                    flag = False
                start = remove(line[20:25], ' ')
                end = remove(line[32:37], ' ')
                ss_index_dict['HELIX'].append((start, end))
        elif line.startswith('SHEET'):
            chain = line[21]
            if chain == chain_ref:
                if flag and chain != line[32]:
                    print('Error! Chains do not match for %s!'%header_file)
                    flag = False
                start = remove(line[22:26], ' ')
                end = remove(line[33:37], ' ')
                ss_index_dict['SHEET'].append((start, end))
    return ss_index_dict


def rcsb_api_ss(pdb, chain, PDB_chain = True): # by SZ
    """Extract the secondary structure region with the RCSB API.

    Args:
        pdb (str): The PDB ID.
        chain_ref (str): The chain to be queied.

    Returns: 
        dict: A dictionary tell the region of the helices and the strands of the target chain (the other parts are coils).
    """

    rcsbBase_url = "https://data.rcsb.org/graphql"
    ### map the PDB (author) chain ID to the RCSB chain ID
    if PDB_chain:
        # query entity ids
        entityIds_query = '''
        {{entries(entry_ids: ["{}"]) {{
            rcsb_entry_container_identifiers {{
                polymer_entity_ids}}
                }}
          }}  
          '''.format(pdb)
        res_entityIds = requests.post(rcsbBase_url,json={'query':entityIds_query})
        if res_entityIds.status_code != 200:
            print('Query fail for %s!'%pdb)
            return None, None
        # query asym_ids, auth_asym_ids
        entityIds_list = res_entityIds.json()
        if len(entityIds_list['data']['entries']) == 0:
            print('Empty query for %s!'%pdb)
            return None, None
        entityIds_list = entityIds_list['data']['entries'][0]['rcsb_entry_container_identifiers']['polymer_entity_ids']
        if len(entityIds_list) > 0:
            flag = True
            for ent_id in entityIds_list:
                asymIds_query = '''
                {{polymer_entities(entity_ids:["{}_{}"]) {{
                    rcsb_polymer_entity_container_identifiers {{
                      asym_ids
                      auth_asym_ids}}
                    entity_poly {{
                      pdbx_strand_id}}
                  }}
                }}
                '''.format(pdb, ent_id)
                res_asymIds = requests.post(rcsbBase_url,json={'query':asymIds_query})
                if res_asymIds.status_code != 200:
                    print('Query fail for %s_%s!'%(pdb, ent_id))
                    return None, None
                else:
                    rec_asymIds_json = res_asymIds.json()
                    asymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['asym_ids']
                    pdbx_strandId_list = rec_asymIds_json['data']['polymer_entities'][0]['entity_poly']['pdbx_strand_id'].split(',')
                    if chain in pdbx_strandId_list:
                        chain = asymIds_list[pdbx_strandId_list.index(chain)]
                        flag = False
                        break
            if flag:
                print('Chain %s is not found for %s!'%(chain, pdb))
        else:
            print('No entity for %s!'%pdb)
            return None, None
    ### API query format
    pdb_instance = '{}.{}'.format(pdb,chain)
    query = '''
    {{polymer_entity_instances(instance_ids: ["{pdb_info}"]) {{
        rcsb_id
        rcsb_polymer_instance_feature {{
          type
          feature_positions {{
            beg_seq_id
            end_seq_id
          }}
        }}
        rcsb_polymer_entity_instance_container_identifiers {{
        auth_to_entity_poly_seq_mapping
        }}
      }}
    }}
    '''.format(pdb_info = pdb_instance)
    ### API query
    info_dict = requests.post(rcsbBase_url,json={'query':query})
    if info_dict.status_code != 200:
        print('Query fail for %s!'%pdb_instance)
        return None, None
    info_dict = info_dict.json()
    if len(info_dict['data']['polymer_entity_instances']) == 0:
        print('Empty query for %s!'%pdb)
        return None, None
    ### extract the SS index info
    ss_info_dict = {'SHEET':[], 'HELIX':[]}
    if info_dict['data']['polymer_entity_instances'][0]['rcsb_polymer_instance_feature'] is not None:
        for info in info_dict['data']['polymer_entity_instances'][0]['rcsb_polymer_instance_feature']:
            if 'SHEET' in info['type']:
                for ss_range in info['feature_positions']:
                    start = ss_range['beg_seq_id']
                    end = ss_range['end_seq_id']
                    ss_info_dict['SHEET'].append((start, end))
            elif 'HELIX' in info["type"]:
                for ss_range in info['feature_positions']:
                    start = ss_range['beg_seq_id']
                    end = ss_range['end_seq_id']
                    ss_info_dict['HELIX'].append((start, end))
        ### transform the sequence index to pdb index
    ss_index_dict = {'SHEET':[], 'HELIX':[]}
    pdb_index = info_dict['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_to_entity_poly_seq_mapping']
    for key in ss_info_dict.keys():
        for seg in ss_info_dict[key]:
            ss_index_dict[key].append((pdb_index[seg[0] - 1], pdb_index[seg[1] - 1]))
    return ss_index_dict

#################################################################################
# 2-D Features: Element-wise Graphs (by SZ)
#################################################################################

#******************** Auxiliary Funtions for Graphs *****************************

def seqtial_mat(node_num, direct = False):
    result = np.zeros((node_num, node_num))
    result[1:,:-1] = np.eye(node_num - 1)
    if not direct:
        result[:-1,1:] += np.eye(node_num - 1)
    return result

def ele_distance(ele_1, ele_2, kind = 'PROTEINS', atom = 'CB'):
    """
    Distance between two elements 
    kind: PROTEINS, TOPS, closest (based on "atom")
    """
    resi_info_1 = ele_1['resi']
    resi_info_2 = ele_2['resi']

    if kind == 'PROTEINS':
        center_1 = PROTEIN_ele_center(resi_info_1)
        center_2 = PROTEIN_ele_center(resi_info_2)
        return np.linalg.norm(center_1 - center_2)
    elif kind == 'TOPS':
        center_1 = TOPS_ele_center(resi_info_1)
        center_2 = TOPS_ele_center(resi_info_2)
        return np.linalg.norm(center_1 - center_2)
    elif kind == 'closest':
        dist_min = np.inf
        for resi_1 in resi_info_1:
            if atom in resi_1['coor'].keys():
                coor_1 = np.array(resi_1['coor'][atom][:3])
                ### 2nd resi
                for resi_2 in resi_info_2:
                    if atom in resi_2['coor'].keys():
                        coor_2 = np.array(resi_2['coor'][atom][:3])
                        dist = np.linalg.norm(coor_1 - coor_2)
                        if dist < dist_min:
                            dist_min = dist
        return dist_min

### For TOPS ###

def TOPS_ele_center(resi_info):
    atom_list = ['N','CA', 'C']
    coor_list = []
    for resi in resi_info:
        for atom in atom_list:
            if atom in resi['coor'].keys():
                coor_list.append(np.array(resi['coor'][atom][:3]))
    return np.mean(coor_list, axis=0)

def TOPS_best_fit(resi_info):
    atom_list = ['N', 'CA', 'C']
    coor_list = []
    for resi in resi_info:
        for atom in atom_list:
            if atom in resi['coor'].keys():
                coor_list.append(np.array(resi['coor'][atom][:3]))
    return best_fit(coor_list)

def best_fit(coors):  # best-fit vector
    if len(coors) < 2:
        print('Error! Only one cooridinate!')
    center = np.mean(coors,0)
    uu, dd, vv = np.linalg.svd(coors - center)
    dire = vv[0]
    return dire, center

def resi_hbond_judge(resi_1, resi_2):
    if not ('DSSP_idx' in resi_1.keys() and 'DSSP_idx' in resi_2.keys()\
    and 'hbond_pair' in resi_1.keys() and 'hbond_pair' in resi_2.keys()):
        return False
    dssp_idx_1 = resi_1['DSSP_idx']
    dssp_idx_2 = resi_2['DSSP_idx']
    if (dssp_idx_1 in resi_2['hbond_pair']) or (dssp_idx_2 in resi_1['hbond_pair']):
        return True
    else:
        return False

def ele_hbond_judge(ele_1, ele_2):
    for resi_1 in ele_1['resi']:
        for resi_2 in ele_2['resi']:
            if resi_hbond_judge(resi_1, resi_2):
                return True
    return False

def atomic_contact(resi_1, resi_2, atom_threshold = 4.5): # updated 01/23/2021
    """
    atomic contact: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-292
    """
    # atom of 1st atom
    for atom_1 in resi_1['coor'].keys():
        if atom_1[0] != 'H':
            coor_1 = np.array(resi_1['coor'][atom_1][:3])
            # atom of 2nd residue
            for atom_2 in resi_2['coor'].keys():
                if atom_2[0] != 'H':
                    coor_2 = np.array(resi_2['coor'][atom_2][:3])
                    # judge
                    dist = np.linalg.norm(coor_1 - coor_2)
                    if dist <= atom_threshold:
                        return True
    return False

def Alpha_contact(ele_1, ele_2, contact_num = 3):
    # when atoms of at least three residues of each helix form atomic contacts between the two helices
    resi_info_1 = ele_1['resi']
    resi_info_2 = ele_2['resi']

    num_1 = 0  # for the 1st ele
    num_2 = 0  # for the 2nd ele
    flag = False
    sele_list = []
    for i,resi_1 in enumerate(resi_info_1):      # 1st resi
        for j,resi_2 in enumerate(resi_info_2):  # 2nd resi
            if atomic_contact(resi_1, resi_2):
                num_1 += 1
                if not j in sele_list:
                    num_2 += 1
                    sele_list.append(j)
            # judge 
            if num_1 >= contact_num and num_2 >= contact_num:
                flag = True
                break
        if num_1 >= contact_num and num_2 >= contact_num:
            flag = True
            break
    return flag

### For PROTEINS ###

def PROTEIN_ele_center(resi_info):
    flag = False
    if len(resi_info) == 1:
        if 'CA' in resi_info[0]['coor'].keys():
            return np.array(resi_info[0]['coor']['CA'][:3])
        else:
            flag = True
    else:
        if 'CA' in resi_info[0]['coor'].keys() and 'CA' in resi_info[-1]['coor'].keys():
            c_1 = np.array(resi_info[0]['coor']['CA'][:3])
            c_2 = np.array(resi_info[-1]['coor']['CA'][:3])
            return 0.5 * (c_1 + c_2)
        else:
            flag = True
    if flag:
        return TOPS_ele_center(resi_info)

#************************* Class of the Graphs **********************************

class Protein_Graph(object):
    """Generate graph elements according to the protein information..

    Args:
        protein_dict (dict): the dictionary containing the protein information.
        index_dict (dict): the dictionary containing the indices information. 
        ss_seq (str): reference of the secondary structure sequence
        sequence_ref (str): reference of the amino acid sequence
        ss_kind (int): version of the secondary structure, 3 or 8. Default: ``3``

    """
    def __init__(self, protein_dict, index_dict, ss_seq=None, sequence_ref=None, ss_kind=3, treat_as_missing=True, ss_padding = True):
        '''
        Warning: if the input information is based on 3-classes and ss_kind = 8, the graph construction will still be based on 3-classes and treat 'C' as '-'.
        ss_kind: 3 or 8
        treat_as_missing: whether treat the residue without known coordinates as a missing residue
        '''
        ################ Parameters ################

        self.ss_kind = ss_kind
        self.treat_as_missing = treat_as_missing
        self.ss_map_dict_8_3 = {'H':'H','G':'H','I':'H','E':'E','B':'E','S':'C','T':'C','-':'C','C':'C','M':'M','*':'*'} # 8-classes to 3-classes and 3 to 3

        if ss_kind == 3:
            self.ss_keys = ['H','E','C']
            self.ss_map_dict = self.ss_map_dict_8_3 # 8-classes to 3-classes and 3 to 3
        elif ss_kind == 8:
            self.ss_keys = ['H','G','I','E','B','S','T','-']
            self.ss_map_dict = {'H':'H','G':'G','I':'I','E':'E','B':'B','S':'S','T':'T','-':'-','C':'S','M':'M','*':'*'} # 8-classes to 8-classes
        else:
            print('Error! Secondary structure can only be 3-classes or 8-classes, not %s-classes!'%str(ss_kind))
            return None
         
        self.non_coor_term = ['index','insertion_code','resi',  # keys of the dictionary except the coordinates
                              'AminoAci','SeconStru','ASA','ASA_level','Phi','Psi','DSSP_idx','NH_O1_relix','NH_O1_energy',
                              'O_NH1_relix','O_NH1_energy','NH_O2_relix','NH_O2_energy','O_NH2_relix','O_NH2_energy']

        # For residue 'B', 'Z', 'X', randomly change them into a residue from the related list
        B_list = ['D','N']
        Z_list = ['Q','E']
        X_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

        ################ Protein information ################

        ### SS sequence
        index_list = index_dict.keys()
        index_min = min(index_list)
        index_max = max(index_list)
 
        ss_ref_seq = ''
        if ss_seq:
            if 'C' in ss_seq and ss_kind == 8:
                print("Warning! Converting 3-classes SS to 8-classes SS, using 'S' for 'C'!")
            for i in range(len(ss_seq)):
                ss_ref_seq += self.ss_map_dict[ss_seq[i]]
        else:
            warning_flag = True
            for i in range(index_min, index_max + 1):  # For different indexes
                if i in index_list:   # Not a missing residue.
                    for ins_code in index_dict[i]:   # For different insertion codes.
                        index = remove(str(i) + ins_code, ' ')      
                        resi_dict = protein_dict[index]  
                        if warning_flag and (resi_dict['SeconStru'] == 'C' and ss_kind == 8):
                            print("Warning! Converting 3-classes SS to 8-classes SS, using 'S' for 'C'!")
                            warning_flag = False
                        ss_ref_seq += self.ss_map_dict[resi_dict['SeconStru']]
                else:
                    ss_ref_seq += '*'

        if sequence_ref:
            sequence_ref = sequence_ref.upper()
        if sequence_ref and not '*' in sequence_ref:
            self.seq_emb_keys = 'ARNDCQEGHILKMFPSTWYV'
        else:
            self.seq_emb_keys = 'ARNDCQEGHILKMFPSTWYV*'

        #################### Compare the sequences #######################################

        seq_processed = ''
        for idx in range(index_min, index_max + 1):  # For different indexes
            if not (idx in index_list): # A missing residue.
                seq_processed += '*'
            else:
                for ins_code in index_dict[idx]:   # For different insertion codes.
                    index = remove(str(idx) + ins_code, ' ')
                    seq_processed += protein_dict[index]['AminoAci'] 

        if sequence_ref and sequence_ref != seq_processed:
            align_flag, seq_padding, sequence_ref_new = Sequence_helper.seq_truncate(seq_processed,sequence_ref)
            if not align_flag:
                print('Error! The sequence and the reference sequence do not match!')
                print(seq_processed)
                print(sequence_ref)
                return None
            else:
                sequence_ref = sequence_ref_new
        else:
            seq_padding = seq_processed
        
        ### make up the missing residues in the pdb file

        make_up_idx = []
        if seq_padding != seq_processed:
            padding_num = 0
            for i in range(len(seq_processed)):
                i_padding = i + padding_num
                while seq_padding[i_padding] != seq_processed[i]:
                    make_up_idx.append(i_padding)
                    padding_num += 1
                    i_padding = i + padding_num

        ### Modify SS sequence accordingly

        if ss_padding:
            for m_idx in make_up_idx:
                ss_ref_seq = ss_ref_seq[:m_idx] + '*' + ss_ref_seq[m_idx:]

        self.SS_origin = ss_ref_seq

        ################### Extra Residue Information ####################################

        self.residues_dict = {}
        self.ordered_index = []
        self.missing_index = []
        self.seq_out = ''

        resi_order = 0

        for idx in range(index_min, index_max + 1):  # For different indexes
            if not (idx in index_list): 
                # A missing residue.
                index = str(idx)
                self.ordered_index.append(index)
                self.missing_index.append(index)
                self.residues_dict[index] = {'status':'missing_resi'}

                ss = ss_ref_seq[resi_order]
                if ss != '*' and ss_padding:
                    print('Missing residue index error! Residue %d should be missing!'%idx)
                    return None
                self.residues_dict[index]['ss'] = ss

                if sequence_ref:
                    if sequence_ref[resi_order] == 'B':
                        self.residues_dict[index]['aa'] = random.choice(B_list) 
                    elif sequence_ref[resi_order] == 'Z':
                        self.residues_dict[index]['aa'] = random.choice(Z_list)
                    elif sequence_ref[resi_order] == 'X':
                        self.residues_dict[index]['aa'] = random.choice(X_list)
                    else:
                        self.residues_dict[index]['aa'] = sequence_ref[resi_order]
                else:
                    self.residues_dict[index]['aa'] = '*'
                self.seq_out += self.residues_dict[index]['aa']

                resi_order += 1

                ### make up residue
                while resi_order in make_up_idx:
                    index_make_up = '%d_make_up'%resi_order
                    self.ordered_index.append(index_make_up)
                    self.missing_index.append(index_make_up)
                    self.residues_dict[index_make_up] = {'status':'missing_resi','ss':'*'}

                    resi_amino = sequence_ref[resi_order]
                    if resi_amino  == 'B':
                        self.residues_dict[index_make_up]['aa'] = random.choice(B_list)
                    elif resi_amino  == 'Z':
                        self.residues_dict[index_make_up]['aa'] = random.choice(Z_list)
                    elif resi_amino  == 'X':
                        self.residues_dict[index_make_up]['aa'] = random.choice(X_list)
                    else:
                        self.residues_dict[index_make_up]['aa'] = resi_amino
                    self.seq_out += self.residues_dict[index_make_up]['aa']

                    resi_order += 1

            ##############################################################################

            else:   # Not a missing residue.
                for ins_code in index_dict[idx]:   # For different insertion codes.
                    index = remove(str(idx) + ins_code, ' ')
                    self.ordered_index.append(index)
                    self.residues_dict[index] = {}  # needed information of each residue           
 
                    ############### Residue Information ##########################

                    resi_dict = protein_dict[index]
                    
                    ### for feature information
                    ss = ss_ref_seq[resi_order]   # Secondary Structure
                    resi_aa = resi_dict['AminoAci'] # Amino Acid
                    self.residues_dict[index]['ss'] = ss
                    self.residues_dict[index]['ASA_level'] = resi_dict['ASA_level']
                    ### for sequence 
                    if resi_aa == 'B':
                        self.residues_dict[index]['aa'] = random.choice(B_list)
                    elif resi_aa == 'Z':
                        self.residues_dict[index]['aa'] = random.choice(Z_list)
                    elif resi_aa == 'X':
                        self.residues_dict[index]['aa'] = random.choice(X_list)
                    else:
                        self.residues_dict[index]['aa'] = resi_aa
                    self.seq_out += self.residues_dict[index]['aa']
                    ### check output of DSSP
                    if ss == 'M':
                        self.residues_dict[index]['status'] = 'missing_SS'
                        if self.residues_dict[index]['ASA_level'] != None:
                            print('Missing SS error! Residue %d have the information from DSSP'%idx)
                            return None
                    else:
                        self.residues_dict[index]['status'] = 'normal'
                        ### for hbond                    
                        self.residues_dict[index]['DSSP_idx'] = resi_dict['DSSP_idx']
                        self.residues_dict[index]['hbond_pair'] = []
                        if resi_dict['NH_O1_energy'] < -0.5:
                            self.residues_dict[index]['hbond_pair'].append(resi_dict['DSSP_idx'] + resi_dict['NH_O1_relix'])
                        if resi_dict['O_NH1_energy'] < -0.5:
                            self.residues_dict[index]['hbond_pair'].append(resi_dict['DSSP_idx'] + resi_dict['O_NH1_relix'])
                        if resi_dict['NH_O2_energy'] < -0.5:
                            self.residues_dict[index]['hbond_pair'].append(resi_dict['DSSP_idx'] + resi_dict['NH_O2_relix'])
                        if resi_dict['O_NH2_energy'] < -0.5:
                            self.residues_dict[index]['hbond_pair'].append(resi_dict['DSSP_idx'] + resi_dict['O_NH2_relix'])

                    ### prepare the coordinates of each residue

                    self.residues_dict[index]['coor'] = {}

                    for k in resi_dict.keys():
                        if not k in self.non_coor_term:
                            self.residues_dict[index]['coor'][k] = resi_dict[k]
 
                    if len(self.residues_dict[index]['coor'].keys()) == 0 and treat_as_missing: 
                        self.missing_index.append(index)
                        self.residues_dict[index]['status'] = 'missing_resi'
                        self.residues_dict[index]['ss']= '*'
                        print('Warning! Resi %s %s: no atom! Take as a missing residue.'%(index, resi_aa, atom))

                    resi_order += 1

                    ### make up residue
                    while resi_order in make_up_idx:
                        index_make_up = '%d_make_up'
                        self.ordered_index.append(index_make_up)
                        self.missing_index.append(index_make_up)
                        self.residues_dict[index_make_up] = {'status':'missing_resi','ss':'*'}

                        resi_amino = sequence_ref[resi_order]
                        if resi_amino  == 'B':
                            self.residues_dict[index_make_up]['aa'] = random.choice(B_list)
                        elif resi_amino  == 'Z':
                            self.residues_dict[index_make_up]['aa'] = random.choice(Z_list)
                        elif resi_amino  == 'X':
                            self.residues_dict[index_make_up]['aa'] = random.choice(X_list)
                        else:
                            self.residues_dict[index_make_up]['aa'] = resi_amino 
                        self.seq_out += self.residues_dict[index_make_up]['aa']

                        resi_order += 1

        ################ Construct the graph elements ##########################

        self.elements = []

        status_pre = None
        ss_pre = None
        resi_order = 0
        single_ele = {'seq':'','resi':[],'ASA_level':[]}  # initialize a new element

        for idx in self.ordered_index: 
            resi_dict = self.residues_dict[idx]
            if ss_padding:
                status = resi_dict['status']
            else:
                status = 'normal'
            ss = resi_dict['ss']
            ### whether a new element
            if status != status_pre or ss != ss_pre:
                new_ele_flag = True
            else:
                new_ele_flag = False

            if resi_order != 0 and new_ele_flag:  # Record the element
                single_ele['status'] = status_pre
                single_ele['SS'] = ss_pre

                single_ele = self.element_arrange(single_ele, ss_padding) # get the arranged element 
                self.elements.append(single_ele) 

                single_ele = {'seq':'','resi':[],'ASA_level':[]} # initialize a new element
  
            single_ele['seq'] += resi_dict['aa']
            if resi_dict['status'] != 'missing_resi':
                if resi_dict['status'] != 'missing_SS':
                    single_ele['ASA_level'].append(resi_dict['ASA_level'])
                    resi_dict_temp = {'DSSP_idx':resi_dict['DSSP_idx'], 'coor':resi_dict['coor'], 'hbond_pair':resi_dict['hbond_pair']}
                else:
                    single_ele['ASA_level'].append(resi_dict['ASA_level'])
                    resi_dict_temp = {'coor':resi_dict['coor']}
                single_ele['resi'].append(resi_dict_temp)
           
            status_pre = status
            ss_pre = ss
            resi_order += 1  
        ### for the last element
        single_ele['status'] = status_pre
        single_ele['SS'] = ss_pre
        single_ele = self.element_arrange(single_ele, ss_padding) # get the arranged element 
        self.elements.append(single_ele)


    def element_arrange(self, element, ss_padding = True):
        '''
        Arrange the element information and generate a complete element     
        '''
        element['seq_embed'] = OneHot_encoding(element['seq'],self.seq_emb_keys)
        element['length'] = len(element['seq'])

        if element['status'] == 'normal':
            if element['length'] == 1 and len(element['ASA_level']) > 0: #?
                element['ASA_level'] = element['ASA_level'][0]
            else:
                surface_amount = element['ASA_level'].count('surface')
                boundary_amount = element['ASA_level'].count('boundary')
                core_amount = element['ASA_level'].count('core')
                max_amount = max(surface_amount,boundary_amount,core_amount)
                if surface_amount == max_amount:
                    element['ASA_level'] = 'surface'
                elif boundary_amount == max_amount:
                    element['ASA_level'] = 'boundary'
                else:
                    element['ASA_level'] = 'core'
        else:
            element['ASA_level'] = None

        feature = np.zeros(2 + self.ss_kind + 3)
        if element['status'] == 'normal':
            feature[2:2+self.ss_kind] = OneHot_encoding(element['SS'],self.ss_keys).squeeze()
            if element['ASA_level'] == 'surface':
                feature[2+self.ss_kind] = 1
            elif element['ASA_level'] == 'boundary':
                feature[3+self.ss_kind] = 1
            elif element['ASA_level'] == 'core':
                feature[4+self.ss_kind] = 1
            elif ss_padding:
                print('Error! No ASA level named %s!'%element['ASA_level'])
            else:
                feature[1] = 1
        elif element['status'] == 'missing_resi':
            feature[0] = 1
        else:
            feature[1] = 1
        element['feature'] = feature

        return element
             
    ####################### Graph Construction ##########################

    ###### TOPS Graph Construction ######

    def TOPS_graph(self, coil = True, beta = 'hbond', contact_threshold = 8, center_threshold = 12, alpha_contact_num = 3, direct = True):
        '''
        Construct the protein graphs based on TOPS topology.
        Versions:
             coil: whether consider coil elements (including missing_resi and missing_ss)
             beta: beta connection ('hbond' or 'distance', 'distance' is based on center_threshold)
        contact_threshold: for the fifth channel
        '''

        elements_TOPS = []
        Node_features = []
        Element_sequences = []
        Sequence_Embeddings = []
        Adjacency_tensor = []

        for ele in self.elements:
            if coil or (ele['status'] == 'normal' and ele['SS'] in ['H','E']): 
                elements_TOPS.append(ele)
                Node_features.append(ele['feature'])
                Element_sequences.append(ele['seq'])
                Sequence_Embeddings.append(ele['seq_embed'])
    
        ### adjacency tensor ###

        ele_num = len(elements_TOPS)
        connected_list = [] # for the 5th channel to not cover the others

        ### 1st channel: sequential

        seq_mat = seqtial_mat(ele_num, direct)
        Adjacency_tensor.append(seq_mat)
        if coil:
            connected_list = [(i,i+1) for i in range(ele_num - 1)] 

        ### 2nd and 3rd channels: beta-strand 

        beta_mat_2 = np.zeros((ele_num,ele_num))
        beta_mat_3 = np.zeros((ele_num,ele_num))
        for i in range(ele_num):
            ele_1 = elements_TOPS[i]
            if ele_1['status'] == 'normal' and ele_1['SS'] == 'E':
                dire_1, center_1 = TOPS_best_fit(ele_1['resi'])
                ### second element
                for j in range(i+1, ele_num):
                    ele_2 = elements_TOPS[j]
                    if ele_2['status'] == 'normal' and ele_2['SS'] == 'E':
                        connect_flag = False
                        if beta == 'hbond':
                            if ele_hbond_judge(ele_1, ele_2):
                                connect_flag = True
                        elif beta == 'distance':
                            dist = ele_distance(ele_1, ele_2, kind = 'TOPS')
                            if dist < center_threshold:
                                connect_flag = True
                        else:
                            print('Error! No beta connection version named %s!'%beta)
                            return None

                        if connect_flag:
                            connected_list.append((i,j))
                            dire_2, center_2 = TOPS_best_fit(ele_2['resi'])
                            if np.dot(dire_1, dire_2) < 0:
                                beta_mat_3[i,j] = 1
                                beta_mat_3[j,i] = 1
                            else:
                                beta_mat_2[i,j] = 1
                                beta_mat_2[j,i] = 1
        Adjacency_tensor.append(beta_mat_2)
        Adjacency_tensor.append(beta_mat_3)

        ### 4th channel: alpha packing

        alpha_mat = np.zeros((ele_num,ele_num))
        for i in range(ele_num):
            ele_1 = elements_TOPS[i]
            if ele_1['status'] == 'normal' and ele_1['SS'] == 'H':
                ### second element
                for j in range(i+1, ele_num):
                    ele_2 = elements_TOPS[j]
                    if ele_2['status'] == 'normal' and ele_2['SS'] == 'H':
                        if Alpha_contact(ele_1, ele_2, contact_num = alpha_contact_num):
                            connected_list.append((i,j))
                            alpha_mat[i,j] = 1
                            alpha_mat[j,i] = 1
        Adjacency_tensor.append(alpha_mat)

        ### 5th channel: other spatial connection

        spatial_mat = np.zeros((ele_num,ele_num))
        for i in range(ele_num):
            ele_1 = elements_TOPS[i]
            if ele_1['status'] != 'missing_resi':
                ### second element
                for j in range(i+1, ele_num):
                    ele_2 = elements_TOPS[j]
                    if ele_2['status'] != 'missing_resi' and (not (i,j) in connected_list) and (not (j,i) in connected_list):
                        dist = ele_distance(ele_1, ele_2, kind = 'closest', atom = 'CB') 
                        if dist < contact_threshold:
                            spatial_mat[i,j] = 1
                            spatial_mat[j,i] = 1
        Adjacency_tensor.append(spatial_mat)

        return Element_sequences, Sequence_Embeddings, np.array(Node_features), np.array(Adjacency_tensor)
      
    
    ###### PROTEINS Graph Construction ######

    def PROTEIN_graph(self, direct = False, hetero = True):
        '''
        Construct the protein graphs according to the PROTEIN dataset.
        Get the element seqeunces, feature matrices and adjacency matrices.
        '''
        ele_num = len(self.elements) 
        seq_mat = seqtial_mat(ele_num, direct)

        Node_features = []
        Element_sequences = []
        Sequence_Embeddings = []

        dist_mat = np.zeros([ele_num, ele_num])
        for i in range(ele_num):
            ele = self.elements[i]
            ### Node features ###
            Node_features.append(ele['feature'])
            ### Sequences ###
            Element_sequences.append(ele['seq'])
            Sequence_Embeddings.append(ele['seq_embed'])
            ### distance matrix ### 
            if ele['status'] != 'missing_resi':
                for j in range(i+1, ele_num): 
                    if self.elements[j]['status'] != 'missing_resi':
                       dist_mat[i,j] = ele_distance(ele, self.elements[j], kind = 'PROTEINS') 
                       dist_mat[j,i] = dist_mat[i,j]
            else:
                dist_mat[i] = np.inf
                dist_mat[:,i] = np.inf
        ### asymmetric adjacency matrix ###
        adja_mat = np.zeros([ele_num, ele_num])
        if ele_num <= 4:
            adja_mat[dist_mat < np.inf] = 1
            adja_mat[seq_mat == 1] = 0
        else:
            for i in range(ele_num):
                if self.elements[i]['status'] != 'missing_resi':
                    dist_vec = dist_mat[i]
                    min_list = sorted([(dist_vec[j],j) for j in range(ele_num) if dist_vec[j] < np.inf and j != i])
                    if len(min_list) > 3:
                        min_list = min_list[:3]
                    sele_list = [x[1] for x in min_list]
                    adja_mat[i][sele_list] = 1
            adja_mat[seqtial_mat(ele_num, False) == 1] = 0

        if hetero:
            Adjacency_tensor = np.array([seq_mat, adja_mat])
        else:
            adja_mat[seq_mat == 1] = 1
            Adjacency_tensor = np.array([adja_mat]) 

        return Element_sequences, Sequence_Embeddings, np.array(Node_features), Adjacency_tensor

#********************** Graph Formats Transformation ****************************

def to_nxGragh(A, X=None, Y=None):
    """Transformer the graph into networkx format.

    Args:
        A (torch.Tensor or numpy.ndarray): The adjacency tensor.
        X (torch.Tensor or numpy.ndarray): The node feature matrix.
        Y (torch.Tensor or numpy.ndarray): The label.

    Returns:  
        networkx.graph: The graph in networkx format.
    """
    ## for 1-channel graph
    G = nx.Graph()
    if Y:
        G.graph['label'] = Y
    node_num = A.shape[0]
    if X.any():
        for i in range(node_num):
            G.add_node(i,feat=X[i])
    for i in range(node_num):
        for j in range(i+1,node_num):
            #print(i,j,node_num)
            if A[i,j] != 0:
                G.add_edge(i, j, weight= A[i,j])
                G.add_edge(j, i, weight= A[j,i])
    return G

def Adjacency_to_edge_index(A_matrix):
    """
    Transform the adjacency matrices into edge index format.
    """
    result = from_scipy_sparse_matrix(csr_matrix(A_matrix))[0]
    return result[0].numpy(), result[1].numpy()

def to_dglGraph(A, version = 'Homogeneous', edge_type = None):
    """Transformer the graph into the DGL format.

    Args:
        A (torch.Tensor or numpy.ndarray): The adjacency tensor..
        version (str): The version of the graphs, "Homogeneous" or "Heterogenous". Default: "Homogeneous"
        edge_type (list): List of the edge types.

    Returns:  
        DGL.graph: The graph in DGL format.   
    """
    node_num = A.shape[-1]
    if version == 'Homogeneous':
        edge_index = Adjacency_to_edge_index(A)
        graph = dgl.graph(data = edge_index, num_nodes = node_num)
    elif version == 'Heterogeneous':
        edges_heter = {}
        for i,etype in enumerate(edge_type):
            edges_heter[etype] = Adjacency_to_edge_index(A[i])
        graph = dgl.heterograph(data = edges_heter, num_nodes = node_num)
    return graph

#################################################################################
# 3-D Features
#################################################################################


#################################################################################
# Evaluations
#################################################################################

def TM_score(pdb_1, pdb_2):
    '''Calculate the TM-scores between two protein structures with pdb_1 as the target and pdb_2 as the reference.

    Args:
        pdb_1 (str): The path of the target pdb file. 
        pdb_2 (str): The path of the reference pdb file.

    Returns:  
        float: The TM-score.
    '''
    command_1 = './TMalign ' + pdb_1 + ' ' + pdb_2 + ' -a'
    output_1 = os.popen(command_1)
    out_1 = output_1.read()
    if "(if normalized by length of Chain_2" in out_1:
        k_1 = out_1.index("(if normalized by length of Chain_2")
        tms_1 = float(out_1[k_1-8:k_1-1])
        return tms_1
    else:
        return None

def TMscore_sym(pdb_1, pdb_2):
    '''Calculate the symmetric TM-scores between two protein structures

    Args:
        pdb_1, pdb_2 (str): The path of the pdb files.

    Returns:  
        float: The symmetric TM-score.
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

