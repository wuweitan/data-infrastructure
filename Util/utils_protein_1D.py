#################################################################################
# Utility functions for data process and evaluation on 1-D sequences.
#################################################################################

from typing import List
import numpy as np
from sqlalchemy import null
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import os

matrix = matlist.blosum62


### CONSTANTS

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

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


def blosum62_from_msa_a3m(msa_file: str):
    """Calculate substitution matrix BLOSUM62 from a MSA input (a3m format)
    
    Args:
        msa_file: MSA file path
    
    Returns:
        numpy.ndarray: BLOSUM62 matrix of size 20*20
    """
    return null

def blosum62_from_msa_stockholm(msa_file: str):
    """Calculate substitution matrix BLOSUM62 from a MSA input (stockholm format)
    
    Args:
        msa_file: MSA file path
    
    Returns:
        numpy.ndarray: BLOSUM62 matrix of size 20*20
    """
    return null


def pssm_from_msa_a3m(msa_file: str):
    """Calculate positional specific substitution matrix PSSM from a MSA input (a3m format)
    
    Args:
        msa_file: MSA file path
    
    Returns:
        numpy.ndarray: PSSM matrix of size L*20
    """
    return null

def pssm_from_msa_stockholm(msa_file: str):
    """Calculate positional specific substitution matrix PSSM from a MSA input (stockholm format)
    
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

### Following is a helper function to parse the PSAIA table file ###
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
    for open(path_to_txt_file, "r") as rf:
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


#TODO: Pass in a chain order list argument and make the function read out the features in this order
### calculate rASA, Normalized Protrusion Index, Normalized Hydrophobicity ###
def calc_PSAIA_features(path_to_pdb_list, path_to_exe = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/bin/linux/psa/psa", \
path_to_config = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/psaia_config_file_input.txt", \
path_to_output_dir = "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/OUTPUT",\
rASA = True, PI = True, Hydrophobicity = True):
    """ rASA, Normalized Protrusion Index, Normalized Hydrophobicity calculation function

    Args:
        path_to_pdb_list (str): the path to the text file which contains all the pdb file names 
            (one file name for one line)
        path_to_exe (str): the path to the installed the PSAIA executables 
            (default value: "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/bin/linux/psa/psa")
        path_to_config (str): the path to the configuration file for the PSAIA software
            (default value: "/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/INPUT/psaia_config_file_input.txt")
        path_to_output_dir (str): the path to the output directory for the PSAIA software which should correspond to the specifications in the config file
            (default value:"/scratch/user/rujieyin/seq_process/PSAIA/Programs/PSAIA_1.0_source/OUTPUT")
        rASA (bool): whether rASA is part of the outputs
        PI (bool): whether PI is part of the outputs
        Hydrophobicity (bool): whether Hydrophobicity is part of the outputs
    
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

    # relative path (against the inner most directory)
    rel_pdb_list = []
    with open(path_to_pdb_list, "r") as rf:
        for line in rf:
            rel_path = line.strip().split("/")[-1]
            abs_pdb_list.append(rel_path)

    # the list of name strings w/o ".pdb"
    ls_name = []
    for name in rel_pdb_list:
        ls_name.append(name[:-4])


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
        pro_rASA = np.array(pro_rASA)
        pro_PI = np.array(pro_PI)
        pro_H = np.array(pro_H)

        # normalize PI and Hydrophobicity on a per protein basis
        normed_pro_PI = NormalizeData(pro_PI)
        normed_pro_H = NormalizeData(pro_H)

        #
        # complete the Out_dict
        name = file_name[:-25]

        # (1) keys some/all of {"rASA", "PI", "Hydrophobicity"} and values {a numpy array with shape (n_res, n_features)} pairs
        Out_dict[name]["rASA"] = pro_rASA
        Out_dict[name]["PI"] = normed_pro_PI
        Out_dict[name]["Hydrophobicity"] = normed_pro_H

        # (2) chain_id: the list of res idx
        Out_dict[name].update(dict_res_ind_ls)

        # (3) "chain order": the order according to which the chains are concatenated
        Out_dict[name]["chain order"] = chain_order

    return Out_dict


``` PSSM
Reference: https://support.cyrusbio.com/workflows/create-a-pssm/
1. Donwload alignment database: wget ftp://ftp.ncbi.nih.gov/blast/db/FASTA/nr.gz
2. Build database: PATH\ncbi-blast-2.7.1+\bin\makeblastdb -in PATH\nr -input_type fasta -title nonR -dbtype prot
3. Prepare your fasta file for proteins, e.g. 5ORB.fasta
4. PSSM: perl PATH\blast+\bin\legacy_blast.pl blastpgp -d PATH\nr -j 4 -b 1 -a 2 -Q PATH\5ORB.pssm -i PATH\5ORB.fasta –path PATH\blast+\bin
```
