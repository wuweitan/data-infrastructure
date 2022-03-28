#################################################################################
# Data processing utility for 1D protein sequences
#
# Authors: Rujie Yin; Shaowen Zhu; Yuanfei Sun; Yuning You
# Date: 03/2022
# Version: 0.1
#################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

import random, os, re, sys, time, json, lmdb
import pickle as pkl
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections
from prody import MSAFile
from Bio import pairwise2, SeqIO
from Bio.SubsMat import MatrixInfo as matlist


#### Constants

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

def raw_msa2dict(data_path: str = None,
                 input_format: str = None,
                 fileNm: str = None,
                 famAcc: str = None,
                 clanAcc: str = None,
                 aligned: bool = True,
                 weight_in_header: bool = True,
                 keep_X: bool = True):
  """extract sequences from MSA files downloaded from Pfam dataset,
     bundle other informaton and output a dictionary list
    
  
  Args: 
    - data_path (str): the path of folder containing MSA file
    - input_format (str): MSA file format
    - fileNm (str): name of MSA file
    - famAcc (str): identifier of family
    - clanAcc (str): identifier of clan
    - aligned (bool): if True, sequences in MSA file are aligned
    - weight_in_header (bool): if True, sequence reweighting score is provided and in header of sequence
    - keep_X (bool): if True, keep seuqences with 'X' letter

  Returns: 
    a list of dictionaries with each dictionary corresponds to one protein sequence/segment.
    
    dictionary keys:
      - primary: raw protein sequence in aa symbols(upper case)
      - protein_length: length of raw sequence
      - unpIden: uniprot_id
      - range: start_idx-end_idx
      - family: family name
      - clan: clan name
      - seq_reweight: redundancy weight
  """
  seq_dict_list = []
  print('loading sequences from %s' % (fileNm), flush=True)
  # convert pfam,clan to id(int)
  famId = int(famAcc[2:7])
  clanId = int(clanAcc[2:]) if clanAcc is not None else -1
  # initilize msa object
  msa = MSAFile('{}/{}'.format(data_path, fileNm), format=input_format, aligned=aligned)
  family_weight = 0.
  for seq in msa:
    # get unaligned sequence
    raw_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
    if 'X' in raw_seq: 
      if keep_X:
        pass ## keep seqs containing 'X'
      else:
        continue ## jump over seqs containing 'X'
      
    if not weight_in_header:
      res_idx = seq.getResnums()
      start_idx = res_idx[0]
      end_idx = res_idx[-1]
      label = seq.getLabel()
      one_protein = {
        'primary': raw_seq,
        'msa_seq': str(seq),
        'protein_length': len(raw_seq),
        'family': famId,
        'clan': clanId,
        'unpIden': label,
        'range': '{}-{}'.format(start_idx,end_idx)
        }   
    else: ## weight in header: patten follow Yue's definition
      whole_id = seq.getLabel()
      label = whole_id.split(';')[1].split('/')[0]
      start_idx = whole_id.split(';')[1].split('/')[1].split(':')[0].split('-')[0]
      end_idx = whole_id.split(';')[1].split('/')[1].split(':')[0].split('-')[1]
      weight_value = whole_id.split(';')[1].split('/')[1].split(':')[1]
      family_weight += float(weight_value)
      one_protein = {
        'primary': raw_seq,
        'msa_seq': str(seq),
        'protein_length': len(raw_seq),
        'family': famId,
        'clan': clanId,
        'unpIden': label,
        'range': '{}-{}'.format(start_idx,end_idx),
        'seq_reweight': float(weight_value)
        }
    '''
    # old protein dict
    one_protein = {
        'sequence': raw_seq,
        'aligned_seq': str(seq),
        'seq_length': len(raw_seq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family': famAcc,
        'clan': clanAcc
        }
    ''' 
    seq_dict_list.append(one_protein)

  return seq_dict_list, family_weight

def create_family_seq_dataSet(
    working_path: str = None,
    seq_file_name: str = None,
    seq_format: str = 'stockholm',
    family_list: List = None,
    familyList_file: str = None,
    output_dir: str = None,
    output_format: str = 'lmdb',
    iden_cutoff: float = 0.8,
    align_pairwise: bool = False,
    reweight_bool: bool = True,
    use_diamond: bool = False,
    diamond_path: str = None,
    use_mmseqs2: bool = True,
    mmseqs2_path: str = None,
    len_hist: bool = False,
    weight_in_header: bool = False):

  """prepare sequence dataset (lmdb format) from Pfam MSA files
  
  Args
    - working_path (str): path of folder containing input sequence files
    - seq_file_name (str): name of input sequence file
    - seq_format (str): format of input sequence file, default = 'stockholm'
    - family_list (List): list of family ids 
    - familyList_file (str): if family_list is None, give the name of a file containing family ids, one family at each line
    - output_dir (str): output folder path
    - output_format (str): output file format, default = 'lmdb'
    - iden_cutoff (float): sequence identity cutoff for use in reweighting scores, default = 0.8
    - align_pairwise (bool): if True, use pairwise alignment when calculating reweighting scores ,default = False
    - reweight_bool (bool): if True, calcualte reweighting scores, default = True
    - use_diamond (bool): if True, use diamond, default = False
    - diamond_path (str):
    - use_mmseqs2 (bool): if True, use mmseq2, default = True
    - mmseqs2_path (str): 
    - len_hist (bool): count sequence length distribution, default = False
    - weight_in_header (bool): if True, reweighting score in the header of sequence, default = False

  Returns:
    processed data in lmdb format
  """

  if family_list is not None:
    family_list = np.asarray(family_list,dtype='str').reshape((-1,))
  ## load family list to process
  if familyList_file is not None:
    family_list = np.loadtxt('{}/{}'.format(working_path,familyList_file), dtype='str', delimiter=',')

  # loop through family list
  for l in range(family_list.shape[0]):
  #for l in range(1):
    famAcc = family_list[l]
    print('>>> Processing %s' % (famAcc), flush=True)

    ## get clan 
    #clanAcc = get_clanAcc(famAcc,famClanMap)

    # transform raw fasta sequence to list of distionarys
    print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    ## remove family version number
    #if re.search(r'\.\d+',famAcc) is not None:
    #  famAcc = re.split(r'\.', famAcc)[0]
   
    seq_dict_list,family_reweight = raw_msa2dict(
                                      data_path=f'{working_path}/{seq_file_name}',
                                      input_format=seq_format,
                                      fileNm=f'{famAcc}.{seq_format}',  #famAcc,
                                      famAcc=famAcc,
                                      clanAcc=None, #clanAcc,
                                      aligned=True,
                                      weight_in_header=weight_in_header,
                                      keep_X=True)
    
    print('>>> Num of seqs: {}'.format(len(seq_dict_list)))
    if reweight_bool:
      if use_diamond:
        print('>>Using Diamond')
        seqReweight_dict = {}
        ## stockholm to fasta
        os.system('esl-reformat -u -o {}/{}/{}.fa fasta {}/{}/{}'.format(diamond_path,famAcc,famAcc,working_path,seq_file_name,famAcc))
        ## build diamond database
        os.system('diamond makedb --in {}/{}/{}.fa -d {}/{}/{}'.format(diamond_path,famAcc,famAcc,diamond_path,famAcc,famAcc))
        ## diamond search
        # get split fasta names
        fasta_splitList = os.popen("ls -v {}/{}/splits".format(diamond_path,famAcc)).readlines() #contain '\n'
        # loop fasta splits
        for fa_split in fasta_splitList:
          fa_split = fa_split.replace('\n','')
          # search
          #dmSearch_out = os.popen(f'diamond blastp -q {diamond_path}/splits/{fa_split} -d {diamond_path}/{famAcc}/{famAcc} -o {diamond_path}/{famAcc}/{famAcc}.out.tsv -v -k0 --compress 1 -f 6 qseqid sseqid pident length mismatch').readlines()
          os.system(f'diamond blastp -q {diamond_path}/splits/{fa_split} -d {diamond_path}/{famAcc}/{famAcc} -o {diamond_path}/{famAcc}/{famAcc}.out.tsv -v -k0 --compress 1 -f 6 qseqid sseqid pident length mismatchi')
          # loop each seq
          with open(f'{diamond_path}/splits/{fa_split}') as handle:
            for record in SeqIO.parse(handle, "fasta"):
              seqId = record.id
              # extract lines for this seq
              iden_tar = os.popen(f"grep '^{seqId}' {diamond_path}/{famAcc}/{famAcc}.out.tsv | cut -d$'\t' -f3").readlines()
              iden_tar = np.array([float(iden.strip('\n')) for iden in iden_tar])
              num_neighbors = np.sum(iden_tar >= iden_cutoff) - 1 
              seqReweight_dict[seqId] == 1. / num_neighbors
              ##TODO unfinished
      elif use_mmseqs2: ## use mmSeqs2 to cluster seqs, then use reweight = 1/cluster_size for seqs in one cluster
        print('>>Using MMseqs2')
        ## stockholm to fasta
        if not os.path.isdir(f'{mmseqs2_path}/{famAcc}'):
          os.mkdir(f'{mmseqs2_path}/{famAcc}')
        os.system(f'esl-reformat -u -o {mmseqs2_path}/{famAcc}/{famAcc}.fa fasta {working_path}/{seq_file_name}/{famAcc}.{seq_format}')
        ## run mmseqs2 (BFD -c 0.9 --cov-mode 1)
        os.system(f'mmseqs easy-linclust {mmseqs2_path}/{famAcc}/{famAcc}.fa {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2 {mmseqs2_path}/tmpDir -c 0.9 --cov-mode 1 --cluster-mode 2 --alignment-mode 3 --min-seq-id {iden_cutoff}')
        ## loop representative seqs
        seqReweight_dict = {}
        family_reweight = 0.
        with open(f'{mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_rep_seq.fasta') as handle:
          for record in SeqIO.parse(handle,"fasta"):
            seqId = record.id
            ## get neighbor seqIds and nums
            num_neighbors = int(os.popen(f"grep -c '{seqId}' {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_cluster.tsv").read().strip('\n'))-1# remove count of itself
            assert num_neighbors >= 0
            if num_neighbors == 0:
              seqReweight_dict[seqId] = 1.0
              family_reweight += 1.0
            else:
              seq_neighbors = os.popen(f"grep '{seqId}' {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_cluster.tsv").read().strip('\n').split('\n')
              for seq_pair in seq_neighbors:
                seqNei_id = seq_pair.split('\t')[-1]
                seqReweight_dict[seqNei_id] = 1./num_neighbors
                family_reweight += 1./num_neighbors
        ## check num of seqs in seqReweight_dict
        ## CAUTION: seqs containing 'X'
        #assert len(seq_dict_list) == len(seqReweight_dict)
        ## assign seq reweight score
        for seq_dict in seq_dict_list:
          seqId = '{}/{}'.format(seq_dict['unpIden'],seq_dict['range'])
          seq_dict['seq_reweight'] = seqReweight_dict[seqId]
      else: ## calculate pairwise %identity mamually
        print('>>manual pairwise iden')
        start_time = time.time()
        print("*** Calculating sequence reweighting scores ***", flush=True)
        ## calculate reweighting score for each sequence and whole family
        family_reweight = 0.
        for seq_dict_query in seq_dict_list:
          idenScore_list = []
          for seq_dict in seq_dict_list:
            if align_pairwise:
              iden_score = seq_align_identity(seq_dict_query['primary'],seq_dict['primary'],matrix=blosum_matrix)
            else:
              iden_score = seq_identity(seq_dict_query['msa_seq'],seq_dict['msa_seq'])
            idenScore_list.append(iden_score)
          idenScore_list = np.array(idenScore_list).astype(float)
          ## exclude compare to itself(-1), avoid devided by 0
          num_similar_neighbors = np.sum(idenScore_list >= iden_cutoff) - 1.
          seq_reweight = min(1., 1. / (num_similar_neighbors + 1e-6))
          seq_dict_query['seq_reweight'] = seq_reweight
          family_reweight += seq_reweight
        end_time = time.time()
        print('>>> Takes {}s'.format(end_time - start_time))
    
    ## save family reweighting score, seq_length, uniq_char
    seqLen_list = []
    uniq_chars = []
    if weight_in_header:
      assert family_reweight > 0.
    for seq_dict in seq_dict_list:
      #rand_num = rng.random()
      if reweight_bool or weight_in_header:
        seq_dict['family_reweight'] = family_reweight
      seqLen_list.append(seq_dict['protein_length'])
      seq = seq_dict['primary']
      uniq_chars = list(set(uniq_chars + list(seq)))
    print("*** Save data and draw figures ***", flush=True)
    if len_hist:
      ## seq length histogram figure
      fig = plt.figure()
      plt.hist(seqLen_list, density=False, bins=50)  # density=False would make counts
      plt.ylabel('Count')
      plt.xlabel('Length')
      plt.savefig('{}/{}/seqLenDist_{}.png'.format(working_path,output_dir,famAcc))
      plt.close()
      ## save
      np.savetxt('{}/{}/seqLenList_{}'.format(working_path,output_dir,famAcc),seqLen_list,fmt='%s',delimiter=',')
      np.savetxt('{}/{}/uniqCharList_{}'.format(working_path,output_dir,famAcc),uniq_chars,fmt='%s',delimiter=',')
    
    if output_format == 'json':
      with open('{}/{}/{}.json'.format(working_path,output_dir,famAcc),'w') as fl2wt:
        json.dump(seq_dict_list,fl2wt)
    elif output_format == 'lmdb':
      wrtEnv = lmdb.open('{}/{}/{}.lmdb'.format(working_path,output_dir,famAcc), map_size=(1024 * 20)*(2 ** 20))
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(seq_dict_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i + 1))
      wrtEnv.close()
    else:
      Exception('invalid output format: {}'.format(output_format))
    print('*** In total, write {} instances ***'.format(len(seq_dict_list)), flush=True)
  return None



class SequenceTokenizersClass():
  """Tokenizer class for pfam sequences. 
      Can use different vocabs depending on the model and extened to other sequence datasets, e.g. Uniref
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