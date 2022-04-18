#################################################################################
# Utility functions for data process and evaluation on 1-D sequences.
#################################################################################

from typing import List
import numpy as np
from sqlalchemy import null
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist 

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