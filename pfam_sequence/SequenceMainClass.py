from absl import app
from absl import flags
import random, os, re, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import collections
from prody import MSAFile


## define global variables (TODO: convert to class vars)
FLAGS = flags.FLAGS

#/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/
flags.DEFINE_string('PROJ_DIR', '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/', 
  "Project root dir")

flags.DEFINE_string('DATA_PATH', 'data_process/pfam_32.0', 
  "Path for data files")

flags.DEFINE_string('STAT_PATH', 'data_process/pfam_32.0/stat_rp15',
  "Path for statistic files")

flags.DEFINE_string('OUTPUT_PATH', 'data_process/pfam_32.0/seq_json_rp15',
  "Folder name to save output files")

flags.DEFINE_string('FAM_CLAN_FILE', 'data_process/pfam_32.0/Pfam-A.clans.tsv',
  "family, clan and domain description tsv file")

flags.DEFINE_string('FLNM_PFAM_FILE', 'data_process/pfam_32.0/pfam_rp15_seqs_files_famAcc', 
  'data file name and pfam number, clan number corresponds')

flags.DEFINE_integer('RANDOM_SEED', 25, 'random seed')

flags.DEFINE_boolean('SPLIT', True, 
  'If True, split seqs in every fasta file into subsets')

flags.DEFINE_integer('NUM_SPLIT', 50,
  'number of subsets to split seqs into tfReocrd files')

flags.DEFINE_integer('VOCAB_SIZE', 29,
  'number of vocabularies/tokens')

# 'max_seq_length' takes the value of 'sgl_domain_len_up' + 2 (start and end tokens)
flags.DEFINE_list('MASK_PARAMS', [202, 0.15, 30],
  'params for mask: max_seq_length, mask_prob, max_mask_per_seq')

flags.DEFINE_boolean('SIG_DOM', True,
  'flag for only saving single domain sequences')

flags.DEFINE_integer('sgl_domain_len_low', 18,
  'the lower bound length of single domain')

flags.DEFINE_integer('sgl_domain_len_up', 200,
  'the upper bound length of single domain')

flags.DEFINE_string('IN_FORMAT', 'stockholm', 'format for input seq files')

flags.DEFINE_boolean('MASK_FLAG', True, 'Whether to mask seqs')


class SequenceMainClass():
  """The class for pfam sequence generation
     * extract sequence from pfam MSA file
     * split data into train, validation and test set
     * filter sequence by length
  """

def raw_msa2dict(data_path, input_format, fileNm, famAcc, clanAcc):
  '''
  input: 
    -data_path: data dir
    -input_format: msa file format
    -fileNm: path of file
    -famAcc: identifier of family
    -clanAcc: identifier of clan
  output: 
    -a list of dictionaries with each dictionary corresponds to one protein sequence/segment.
    keys of dictionary:
    -sequence: raw protein sequence in aa symbols(upper case)
    -aligned_seq: aligned sequence in msa file
    -seq_length: length of raw sequence
    -uni_iden: uniprot_id/start_idx-end_idx
    -family: family id
    -clan: clan id
  Count statistics in create_instance function
  '''
  seq_dict_list = []
  print('loading sequences from %s' % (fileNm), flush=True)
  # convert pfam,clan to id(int)
  famId = int(famAcc[2:])
  clanId = int(clanAcc[2:]) if len(clanAcc) > 0 else -1
  # initilize msa object
  msa = MSAFile('{}/{}'.format(data_path, fileNm), format=input_format, aligned=True)
  for seq in msa:
    res_idx = seq.getResnums()
    start_idx = res_idx[0]
    end_idx = res_idx[-1]
    label = seq.getLabel()
    # get unaligned sequence
    raw_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
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
    # new protein dict
    if 'X' in raw_seq:
      #print('X in raw_seq:{}'.format(raw_seq))
      continue
    else:
      one_protein = {
          'primary': raw_seq,
          'protein_length': len(raw_seq),
          'family': famId,
          'clan': clanId,
          'unpIden': label,
          'range': '{}-{}'.format(start_idx,end_idx)
          }

      seq_dict_list.append(one_protein)

  return seq_dict_list

def create_json_dataSet(argv):
  '''
  Output:
  *train, validation, test set: list, each element a dist for a seq
  '''
  ## copy common global params
  PROJ_DIR = FLAGS.PROJ_DIR
  DATA_PATH = PROJ_DIR+FLAGS.DATA_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  FAM_CLAN_FILE = PROJ_DIR+FLAGS.FAM_CLAN_FILE
  FLNM_PFAM_FILE = PROJ_DIR+FLAGS.FLNM_PFAM_FILE
  RANDOM_SEED = FLAGS.RANDOM_SEED
  SPLIT = FLAGS.SPLIT
  NUM_SPLIT = FLAGS.NUM_SPLIT
  [MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ] = FLAGS.MASK_PARAMS
  SIG_DOM = FLAGS.SIG_DOM
  IN_FORMAT = FLAGS.IN_FORMAT
  MASK_FLAG = FLAGS.MASK_FLAG

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  rng = random.Random(RANDOM_SEED)
  
  # holdout set: mutation set
  holdOut_pfamDir = '{}data_process/pfam_32.0/holdOut_sets/muta_pfam_small_set.txt'.format(PROJ_DIR)
  holdOut_clanDir = '{}data_process/pfam_32.0/holdOut_sets/muta_clan_small_set.txt'.format(PROJ_DIR)
  holdOut_pfams = np.loadtxt(holdOut_pfamDir, dtype='str')
  holdOut_clans = np.loadtxt(holdOut_clanDir, dtype='str')

  # load fileNm-PfamAcc pairs for msa files
  flNm_Pfam = np.loadtxt(FLNM_PFAM_FILE, dtype='str', delimiter=' ')
  
  # train - val rate: 9:1, 
  train_set = []
  val_set = []
  holdOut_set = []

  # counters for file writting
  writer_idx = 0 
  num_written = [0, 0, 0] #seq counting [train_written, val_written, test_written]


  # loop through file names for each family
  for l in range(flNm_Pfam.shape[0]):
  #for l in range(1):
    [fileNm, famAcc, clanAcc] = flNm_Pfam[l]
    print('>>> Processing file %s from family %s' % (fileNm, famAcc), flush=True)

    # transform raw fasta sequence to list of distionarys
    print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    famAcc_no_version = re.split(r'\.', famAcc)[0] 
    seq_dict_list = raw_msa2dict(DATA_PATH, IN_FORMAT, fileNm, famAcc_no_version, clanAcc)
    if famAcc in holdOut_pfams or clanAcc in holdOut_clans:
      # add 'id' pair
      tmp_id = 0
      for seq_dict in seq_dict_list:
        seq_dict["id"] = str(tmp_id)
        tmp_id += 1
      # save a individual file
      with open('{}/holdout_indiSet/pfam_holdout_{}.json'.format(OUTPUT_PATH,famAcc_no_version),'w') as fl2wt:
        json.dump(seq_dict_list,fl2wt)
      
      # append to whole holdout set
      for seq_dict in seq_dict_list:
        seq_dict['id'] = str(num_written[2])
        num_written[2] += 1
      holdOut_set.extend(seq_dict_list)
    else:
      for seq_dict in seq_dict_list:
        rand_num =  rng.random()
        if rand_num < 0.1:
          seq_dict["id"] = str(num_written[1])
          num_written[1] += 1
          val_set.append(seq_dict)
        else:
          seq_dict["id"] = str(num_written[0])
          num_written[0] += 1
          train_set.append(seq_dict)
  # save three sets
  with open('{}/pfam_train.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(train_set,fl2wt)
  with open('{}/pfam_valid.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(val_set,fl2wt)
  with open('{}/pfam_holdout.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(holdOut_set,fl2wt)


  print('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1], num_written[2]), flush=True)

  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/seq_json_num".format(sum(num_written), num_written[0], num_written[1],num_written[2], STAT_PATH))

def filterByLen(argv):
  """
  filter out sequences with length <= 500
  """
  PROJ_DIR = FLAGS.PROJ_DIR 
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH

  # load json data
  with open('{}/pfam_train.json'.format(OUTPUT_PATH),'r') as fl:
    train_json = json.load(fl)
  with open('{}/pfam_valid.json'.format(OUTPUT_PATH),'r') as fl:
    val_json = json.load(fl)
  with open('{}/pfam_holdout.json'.format(OUTPUT_PATH),'r') as fl:
    test_json = json.load(fl)

  len_cutoff = 500
  # loop through json and filter seq by length
  train_lenCut = []
  val_lenCut = []
  test_lenCut = []

  for train_one in train_json:
    if int(train_one['protein_length']) <= len_cutoff:
      train_lenCut.append(train_one)

  for val_one in val_json:
    if int(val_one['protein_length']) <= len_cutoff:
      val_lenCut.append(val_one)
  
  for test_one in test_json:
    if int(test_one['protein_length']) <= len_cutoff:
      test_lenCut.append(test_one)

  # save to json files
  with open('{}/pfam_train_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(train_lenCut,fl2wt)
  with open('{}/pfam_valid_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(val_lenCut,fl2wt)
  with open('{}/pfam_holdout_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(test_lenCut,fl2wt)
  
  num_written = [len(train_lenCut), len(val_lenCut), len(test_lenCut)]
  print('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1], num_written[2]), flush=True)

  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/seq_json_num_lenCut".format(sum(num_written), num_written[0], num_written[1],num_written[2], STAT_PATH))

