import numpy as np
import re, collections, os
import pandas as pd
import json, lmdb
import pickle as pkl
import random
from collections import OrderedDict
import Bio.PDB
from pdbmap_process import asym_mapping, queryApi_pdbInfo, get_unp_pdb_seqIdx_mapping, unmodel_pdb_idx, check_valid_pos, kth_diag_indices, NumpyArrayEncoder


class MutationStatisticsClass():
  """Statistical count class for mutation data
     
  """

  def __init__(self, **kwargs):
    self.arg1 = kwargs.pop('arg1',None)

  def label_statistic(working_dir, out_name='label_count'):
    """
     Each mutagenesis dataset may provide multiple fitness scores(e.g. growth rate under different dose concentration)
     this function is to count number of measured cases for each fitness score and save to a json file

     Input:
      * working_dir: str; directory to save output json file; default: current dir(./)
      * set_path: str; path of file which stores file name of each mutagenesis set 
      * out_name: str; name of output json file; default: 'label_count'
    """

    file_list = '{}/DeepSequenceMutaSet_flList'.format(working_dir)
    mut_stat = {}
    with open(file_list,'r') as fl:
      for one_fl in fl:
        print('processing {}'.format(one_fl[:-1]))
        mut_stat[one_fl[:-1]] = []
        #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
        mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),index_col=0)
        for col_name, col_count in mut_labels.count().iteritems():
          mut_stat[one_fl[:-1]].append({col_name:col_count})
    
    with open('{}/label_count.json'.format(working_dir), 'w', encoding='utf-8') as f:
      json.dump(mut_stat, f, ensure_ascii=False, indent=2)


  def mutant_label_distribution(working_dir):
    '''
    check label distribution of wt, single and multi-site mutation
    '''
    label_target = {}
    with open('{}/target_label_set.csv'.format(working_dir),'r') as fl:
      for line in fl:
        line_split = re.split(',',line[:-1])
        label_target[line_split[0]] = line_split[1]

    # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
    for set_nm, label_nm in label_target.items():
      mutant_count_dict = OrderedDict()
      print('>processing {}'.format(set_nm))
      os.system("echo '>processing {}' >> {}/mutant_label_dist.txt".format(set_nm,working_dir))
      #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
      mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
      target_df = mut_labels[mut_labels[label_nm].notnull()]
      for idx, row in target_df.iterrows():
        target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
        target_label = row[label_nm]
        
        if target_muts in ['wt', 'WT']:
          if 0 not in mutant_count_dict.keys():
            mutant_count_dict[0]=[]
            mutant_count_dict[0].append(target_label)
          else:
            mutant_count_dict[0].append(target_label)
        elif len(target_muts) == 1:
          one_mut = target_muts[0]
          wt_aa = one_mut[0]
          mut_aa = one_mut[-1]
          if wt_aa == mut_aa:
            if 0 not in mutant_count_dict.keys():
              mutant_count_dict[0]=[]
              mutant_count_dict[0].append(target_label)
            else:
              mutant_count_dict[0].append(target_label)
          else:
            if 1 not in mutant_count_dict.keys():
              mutant_count_dict[1]=[]
              mutant_count_dict[1].append(target_label)
            else:
              mutant_count_dict[1].append(target_label)
        elif len(target_muts) > 1:
          mut_num = len(target_muts)
          if mut_num not in mutant_count_dict.keys():
            mutant_count_dict[mut_num]=[]
            mutant_count_dict[mut_num].append(target_label)
          else:
            mutant_count_dict[mut_num].append(target_label)
      print('>>label value dist:')      
      os.system("echo '>>label value dist:' >> {}/mutant_label_dist.txt".format(working_dir))
      for k, v in mutant_count_dict.items():
        print('>**>{}-site mut: min {}; max {}; mean {}; sd {}'.format(k,np.amin(v),np.amax(v),np.mean(v),np.std(v)))
        os.system("echo '>**>{}-site mut: min {}; max {}; mean {}; sd {}' >> {}/mutant_label_dist.txt".format(k,np.amin(v),np.amax(v),np.mean(v),np.std(v),working_dir))


  def mutant_count(working_dir):
    '''
    count wt-change, single-site, multi-site mutants
    '''
    label_target = {}
    with open('{}/target_label_set.csv'.format(working_dir),'r') as fl:
      for line in fl:
        line_split = re.split(',',line[:-1])
        label_target[line_split[0]] = line_split[1]

    # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
    for set_nm, label_nm in label_target.items():
      mutant_count_dict = OrderedDict({0:0,
                                       1:0})
      print('>processing {}'.format(set_nm))
      os.system("echo '>processing {}' >> {}/mutant_count.txt".format(set_nm,working_dir))
      #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
      mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
      target_df = mut_labels[mut_labels[label_nm].notnull()]
      for idx, row in target_df.iterrows():
        target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
        if target_muts in ['wt', 'WT']:
          mutant_count_dict[0] += 1
        elif len(target_muts) == 1:
          one_mut = target_muts[0]
          wt_aa = one_mut[0]
          mut_aa = one_mut[-1]
          if wt_aa == mut_aa:
            mutant_count_dict[0] += 1
          else:
            mutant_count_dict[1] += 1
        elif len(target_muts) > 1:
          mut_num = len(target_muts)
          if mut_num not in mutant_count_dict.keys():
            mutant_count_dict[mut_num] = 1
          else:
            mutant_count_dict[mut_num] += 1
      print('>>total rows: {}'.format(target_df.shape[0]))
      print('>>detailed counts:')
      os.system("echo '>>total rows: {}' >> {}/mutant_count.txt".format(target_df.shape[0], working_dir))
      os.system("echo '>>detailed counts:' >> {}/mutant_count.txt".format(working_dir))
      sum_total = 0
      for k, v in mutant_count_dict.items():
        print('>>{}:{}'.format(k,v))
        os.system("echo '>>{}:{}' >> {}/mutant_count.txt".format(k,v,working_dir))
        sum_total += v
      assert sum_total == target_df.shape[0]


