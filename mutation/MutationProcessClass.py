import numpy as np
import re, collections, os
import pandas as pd
import json, lmdb
import pickle as pkl
import random
from collections import OrderedDict
import Bio.PDB
import matplotlib.pyplot as plt
import seaborn as sns
from pdbmap_process import asym_mapping, queryApi_pdbInfo, get_unp_pdb_seqIdx_mapping, unmodel_pdb_idx, check_valid_pos, kth_diag_indices, NumpyArrayEncoder

def proc_ind():
  file_list = '{}/DeepSequenceMutaSet_flList'.format(working_dir)
  with open(file_list,'r') as fl:
    for one_fl in fl:
      print('processing {}'.format(one_fl[:-1]))
      mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
      #print(mut_dt.shape)
      prim_seq_idx = {} #idx:wt_aa
      mut_list = mut_dt[:,1]
      
      for muta_one in mut_list:
        multi_muta = re.split(r':',muta_one)
        for muta in multi_muta:
          if len(muta) > 2:
            wt_aa = muta[0]
            mut_aa = muta[-1]
            mut_idx = int(muta[1:-1])
            if mut_idx not in prim_seq_idx.keys():
              prim_seq_idx[mut_idx] = [wt_aa]
            else:
              if wt_aa not in prim_seq_idx[mut_idx]:
                print('>>>More than 1 aa at one pos!!!')
              else:
                pass
      sorted_primSeqIdx = collections.OrderedDict(sorted(prim_seq_idx.items()))
      idx_list = list(sorted_primSeqIdx.keys())
      range_len = idx_list[-1]-idx_list[0]+1
      print('>>>start:{},end:{},range:{}'.format(idx_list[0],idx_list[-1],len(idx_list)))
      if range_len != len(idx_list):
        print('>>>Some aa not covered!!!')
        sorted_primSeqIdx_list = []
        for manul_idx in range(idx_list[0],idx_list[-1]+1):
          if manul_idx in idx_list:
            sorted_primSeqIdx_list.append([manul_idx,sorted_primSeqIdx[manul_idx][0]])
          else:
            sorted_primSeqIdx_list.append([manul_idx,'*'])
        sorted_primSeqIdx_list = np.array(sorted_primSeqIdx_list)
        #np.savetxt('{}/DeepSequenceMutaSet_priSeq/{}_idxSeqMiss.csv'.format(working_dir,one_fl[:-5]),sorted_primSeqIdx_list,fmt='%s',delimiter=',')
        #with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,one_fl[:-5]),'w') as fl:
        #  fl.write(''.join(sorted_primSeqIdx_list[:,1]))
      else:
        # generate primary seq
        sorted_primSeqIdx_list = []
        for idx,aa in sorted_primSeqIdx.items():
          sorted_primSeqIdx_list.append([idx,aa[0]])
        sorted_primSeqIdx_list = np.array(sorted_primSeqIdx_list)
        #np.savetxt('{}/DeepSequenceMutaSet_priSeq/{}_idxSeq.csv'.format(working_dir,one_fl[:-5]),sorted_primSeqIdx_list,fmt='%s',delimiter=',')
        #with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,one_fl[:-5]),'w') as fl:
        #  fl.write(''.join(sorted_primSeqIdx_list[:,1]))


def prepare_seq_label(working_dir):
  ''' 
  prepare mutant sequence and label sets 
  '''
  label_target = {}
  with open('{}/target_label_set.csv'.format(working_dir),'r') as fl:
    for line in fl:
      line_split = re.split(',',line[:-1])
      label_target[line_split[0]] = line_split[1]
  
  priSeq_range = np.loadtxt('{}/DeepSequenceMutaSet_priSeq_idxRange'.format(working_dir),dtype='str',delimiter=',')
  priSeq_range_dict = {}
  for i in range(priSeq_range.shape[0]):
    priSeq_range_dict[priSeq_range[i][0]]=priSeq_range[i,1:]

  # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
  all_num = 0
  for set_nm, label_nm in label_target.items():
    if not os.path.isdir('{}/processed_data/set_data/{}'.format(working_dir,set_nm)):
      os.mkdir('{}/processed_data/set_data/{}'.format(working_dir,set_nm))

    set_dt_list = [] # list of jsons
    set_wt_list = [] # list to hold wt cases
    print('>processing {}'.format(set_nm))
    i_num = 0
    #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
    mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
    
    with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,set_nm),'r') as fl:
      wt_seq = fl.read().replace('\n','')
    mut_seq_list = list(wt_seq)
    target_df = mut_labels[mut_labels[label_nm].notnull()]
    for idx, row in target_df.iterrows():
      mut_seq_list = list(wt_seq) 
      target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
      # replace aa to get mutant seq
      for one_mut in target_muts:
        if one_mut not in ['wt', 'WT']:
          wt_aa = one_mut[0]
          mut_aa = one_mut[-1]
          if wt_aa not in ['_','X'] and mut_aa not in ['_', 'X']:
            idx_aa = int(one_mut[1:-1])
            priSeq_startIdx = int(priSeq_range_dict[set_nm][0])
            priSeq_endIdx = int(priSeq_range_dict[set_nm][1])
            #print('idx_aa:{}; priSeq_startIdx:{}'.format(idx_aa, priSeq_startIdx))
            assert wt_seq[idx_aa - priSeq_startIdx] == wt_aa
            mut_seq_list[idx_aa - priSeq_startIdx] = mut_aa
      if ''.join(mut_seq_list) != wt_seq:
        one_json = {"set_nm": set_nm,
                  "wt_seq": wt_seq,
                  "seq_len": len(mut_seq_list),
                  "mutants": target_muts,
                  "mut_seq": ''.join(mut_seq_list),
                  "fitness": row[label_nm],
                  "mutation_effect_prediction_vae_ensemble": row["mutation_effect_prediction_vae_ensemble"],
                  "mutation_effect_prediction_vae_1": row["mutation_effect_prediction_vae_1"],
                  "mutation_effect_prediction_vae_2": row["mutation_effect_prediction_vae_2"],
                  "mutation_effect_prediction_vae_3": row["mutation_effect_prediction_vae_3"],
                  "mutation_effect_prediction_vae_4": row["mutation_effect_prediction_vae_4"],
                  "mutation_effect_prediction_vae_5": row["mutation_effect_prediction_vae_5"],
                  "mutation_effect_prediction_pairwise": row["mutation_effect_prediction_pairwise"],
                  "mutation_effect_prediction_independent": row["mutation_effect_prediction_independent"]}
        set_dt_list.append(one_json)
        i_num += 1
        all_num += 1
      else:
        wt_one_json = {"set_nm": set_nm,
                  "wt_seq": wt_seq,
                  "seq_len": len(wt_seq),
                  "mutants": target_muts,
                  "fitness": row[label_nm],
                  "mutation_effect_prediction_vae_ensemble": row["mutation_effect_prediction_vae_ensemble"],
                  "mutation_effect_prediction_vae_1": row["mutation_effect_prediction_vae_1"],
                  "mutation_effect_prediction_vae_2": row["mutation_effect_prediction_vae_2"],
                  "mutation_effect_prediction_vae_3": row["mutation_effect_prediction_vae_3"],
                  "mutation_effect_prediction_vae_4": row["mutation_effect_prediction_vae_4"],
                  "mutation_effect_prediction_vae_5": row["mutation_effect_prediction_vae_5"],
                  "mutation_effect_prediction_pairwise": row["mutation_effect_prediction_pairwise"],
                  "mutation_effect_prediction_independent": row["mutation_effect_prediction_independent"]}
        set_wt_list.append(wt_one_json)
        #print('wt mut:',target_muts)
    print('>->- {} examples for this set (exclude wt cases)'.format(i_num))
    # save data
    sample_dt = random.sample(set_dt_list, 5)
    with open('{}/processed_data/set_data/{}/{}_mut_all.json'.format(working_dir, set_nm, set_nm),'w') as fl:
      json.dump(set_dt_list,fl)
    if len(set_wt_list) > 0:
      with open('{}/processed_data/set_data/{}/{}_wt_all.json'.format(working_dir, set_nm, set_nm),'w') as fl:
        json.dump(set_wt_list,fl)
    with open('{}/processed_data/set_data/{}/{}_mut_samples.json'.format(working_dir,set_nm,set_nm),'w') as fl:
      json.dump(sample_dt,fl)
  
    map_size = (1024 * 15) * (2 ** 20) # 15G
    wrtEnv = lmdb.open('{}/processed_data/set_data/{}/{}_mut_all.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(set_dt_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    wrtEnv.close()
    
    if len(set_wt_list) > 0:
      wrtEnv = lmdb.open('{}/processed_data/set_data/{}/{}_wt_all.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(set_wt_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()

    sample_dt = random.sample(set_dt_list, min(len(set_dt_list), 500))
    wrtEnv = lmdb.open('{}/processed_data/set_data/{}/{}_mut_samples.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(sample_dt):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    wrtEnv.close()

    print('>saving data done for {}'.format(set_nm))
  print('>In total, {} mut cases'.format(all_num))

def split_dataset(working_dir):
  '''
  * split-1 (single/mul-site mixed together)
    80% of each set gathered together as training set
    evaluate on remaining 20% for each set
  '''
  train_mut_list = []
  val_mut_list = []
  test_mut_list = []
  map_size = (1024 * 15) * (2 ** 20) # 15G

  set_list = np.loadtxt('{}/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  # loop through sets
  for set_nm in set_list:
    print('>process set: {}'.format(set_nm))
    test_mut_list_indi = []
    with open('{}/processed_data/set_data/{}/{}_mut_all.json'.format(working_dir,set_nm,set_nm)) as fl:
      mut_dt_all = json.load(fl)
    # loop through examples
    for one_dt in mut_dt_all:
      prob = random.random()
      if prob < 0.8:
        prob /= 0.8
        if prob < 0.1:
          val_mut_list.append(one_dt)
        else:
          train_mut_list.append(one_dt)
      else:
        test_mut_list.append(one_dt)
        test_mut_list_indi.append(one_dt)

    # save test_individual
    wrtEnv = lmdb.open('{}/processed_data/set_data/{}/{}_mut_holdout.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(test_mut_list_indi):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    wrtEnv.close()
    print('>*>* set individual test num: {}'.format(len(test_mut_list_indi)))

  # save train, val, test set
  wrtEnv = lmdb.open('{}/processed_data/mut_train.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(train_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  
  wrtEnv = lmdb.open('{}/processed_data/mut_valid.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(val_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()

  wrtEnv = lmdb.open('{}/processed_data/mut_holdout.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(test_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  print('>In total, train num: {}, validation num: {}, test num: {}'.format(len(train_mut_list),len(val_mut_list),len(test_mut_list)))


def generate_contact_map(working_dir):
  '''
  generate comtact map from pdb structures
  '''
  
  cutoff = 8.0 # distance cutoff for contact
  # strcture coverage in uniprot indices
  wt_seq_struct_range = np.loadtxt('{}/wt_seq_structure/wt_structure.csv'.format(working_dir),dtype='str',delimiter=',',skiprows=1)
  noAsymId_pdb, noRsp_pdb, noAlign_pdb, noExpAtom_pdb, noValidReg_pdb, wtNotUnp = [], [], [], [], [], []
  packed_data = []
  bio_pdbList = Bio.PDB.PDBList()
  bio_pdbParser = Bio.PDB.PDBParser()
  bio_mmcifParser = Bio.PDB.FastMMCIFParser(QUIET=True)
  for l in range(wt_seq_struct_range.shape[0]):
    set_nm = wt_seq_struct_range[l,0]
    wtSeq_unp_start,wtSeq_unp_end = re.split('-',wt_seq_struct_range[l,1])
    wtSeq_unp_start,wtSeq_unp_end = int(wtSeq_unp_start),int(wtSeq_unp_end)
    if len(wt_seq_struct_range[l,2]) > 0:
      pdbId,pdbChain = re.split('-',wt_seq_struct_range[l,2])
      pdbId = pdbId.upper()
    else:
      pdbId,pdbChain = None,None
    if len(wt_seq_struct_range[l,3]) > 0:
      struc_unp_start,struc_unp_end = re.split('-',wt_seq_struct_range[l,3])
      struc_unp_start,struc_unp_end = int(struc_unp_start), int(struc_unp_end)
    else:
      struc_unp_start,struc_unp_end = None,None
    unpAcc = wt_seq_struct_range[l,4]
    print('{},{}-{},{}-{},{}-{}'.format(set_nm,wtSeq_unp_start,wtSeq_unp_end,pdbId,pdbChain,struc_unp_start,struc_unp_end))
    
    # load wt seq
    with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,set_nm),'r') as fl:
      wt_seq = fl.read().replace('\n','')
    
    if pdbId is not None:
      # mapping {auth_asym_id:asym_id}
      asym_mapping_dict = asym_mapping(pdbId)
      if asym_mapping_dict is None:
        noAsymId_pdb.append([set_nm,pdbId,pdbChain])
        # make a dumb asym_mapping_dict
        asym_mapping_dict = {pdbChain:pdbChain}
      # query pdb info
      res_flag,auth_pdbSeq_mapping,unp_seq,pdb_seq,aligned_regions,unobserved_residues,unobserved_atoms=queryApi_pdbInfo(working_dir,pdbId,asym_mapping_dict[pdbChain],unpAcc)
      if not res_flag: # no response
        noRsp_pdb.append([set_nm,pdbId,pdbChain])
        continue
      else:
        # fetch pdb file and generate pdb object
        pdb_flNm = bio_pdbList.retrieve_pdb_file(pdbId,pdir='tmp_download/{}_pdb'.format(set_nm),file_format='mmCif',overwrite=True)
        #pdb_struc = bio_pdbParser.get_structure(pdbId,'{}'.format(pdb_flNm))
        pdb_struc = bio_mmcifParser.get_structure(pdbId,'{}'.format(pdb_flNm))
        pdb_model = pdb_struc[0]
        #os.remove(pdb_flNm)

        # build residue id dict
        resiId_dict={}
        for resi_obj in pdb_model[pdbChain]:
          resiId_tuple = resi_obj.get_id()
          if resiId_tuple[2] != ' ':
            resiId_dict['{}{}'.format(resiId_tuple[1],resiId_tuple[2])] = resiId_tuple
          else:
            resiId_dict['{}'.format(resiId_tuple[1])] = resiId_tuple

        if aligned_regions is None: # this unpAcc not covered by this pdb
          # get unp_seq, do seq alignment
          noAlign_pdb.append([set_nm,pdbId,pdbChain])
          #unp_seq = query_unpSeq(unpAcc)
          #aligned_range = pdb_unp_align(pdb_seq,unp_seq)
          continue
        else:
          unp_pdb_seqIdx_mapping = get_unp_pdb_seqIdx_mapping(aligned_regions)
          unmodelResi_pdbIdxs,unmodelAtom_pdbIdxs = unmodel_pdb_idx(unobserved_residues,unobserved_atoms)
          valid_unpIdx_list = check_valid_pos(wtSeq_unp_start,wtSeq_unp_end,aligned_regions,unmodelResi_pdbIdxs,unp_pdb_seqIdx_mapping)
          if len(valid_unpIdx_list) > 0:
            # loop over multiple valid regions of the pdb
            print('>>val_region:{}-{},len:{}'.format(valid_unpIdx_list[0],valid_unpIdx_list[-1],len(valid_unpIdx_list)))
            # build json
            tmp_data_dict = {} 
            tmp_data_dict['unpAcc'] = unpAcc
            tmp_data_dict['pfamAcc'] = 'PF'
            # uniprot seq is used as target seq
            seq_tar = ""
            unpSeq_len = len(unp_seq)
            unp_pfam_range_maxUnp = range(wtSeq_unp_start, min(wtSeq_unp_end,unpSeq_len)+1) # index should not exceed unp seq length
            unp_pfam_range = range(wtSeq_unp_start, wtSeq_unp_end+1) # index should not exceed unp seq length
            unp_pfam_range_len = len(unp_pfam_range)
            assert len(wt_seq) == unp_pfam_range_len
            for unpIdx_i in unp_pfam_range_maxUnp:
              seq_tar += unp_seq[unpIdx_i-1]
            # wt seq verification
            # UBC9_HUMAN_Roth2017 (extra Y159 - as unmodeled residue)
            if not wt_seq == seq_tar and set_nm != 'UBC9_HUMAN_Roth2017':
              wtNotUnp.append(set_nm)
              continue
            tmp_data_dict['set_nm'] = set_nm
            tmp_data_dict['unp_pfam_range'] = '{}-{}'.format(unp_pfam_range[0],unp_pfam_range[-1])
            tmp_data_dict['target_seq'] = wt_seq
            tmp_data_dict['targetSeq_len'] = unp_pfam_range_len
            tmp_data_dict['best_pdb'] = pdbId
            tmp_data_dict['chain_id'] = pdbChain
            tmp_data_dict['valid_unpIdxs_len'] = len(valid_unpIdx_list)
            tmp_data_dict['valid_unpIdxs'] = valid_unpIdx_list
            
            valid_mask = [False]*unp_pfam_range_len
            valid_pos = [val_i - unp_pfam_range[0] for val_i in valid_unpIdx_list]
            for i in valid_pos:
              valid_mask[i] = True
            tmp_data_dict['valid_mask'] = valid_mask 
            
            # generate contact-map (with self-self, self-neighbor as 1)
            contact_mat = np.zeros((unp_pfam_range_len,unp_pfam_range_len))
            diagIdx = kth_diag_indices(contact_mat,[-1,0,1])
            contact_mat[diagIdx] = 1
            # loop rows and cols
            for valReg_row in range(unp_pfam_range_len):
              for valReg_col in range(unp_pfam_range_len):
                unp_idx_row = unp_pfam_range[valReg_row]
                unp_idx_col = unp_pfam_range[valReg_col]
                if unp_idx_row in valid_unpIdx_list and unp_idx_col in valid_unpIdx_list:
                  # get pdb natural seq idx(1-idxed), then author-defined index
                  resIdx_pdbSeq_row = unp_pdb_seqIdx_mapping[unp_idx_row]
                  resIdx_pdbSeq_col = unp_pdb_seqIdx_mapping[unp_idx_col]
                  resIdx_pdbAuth_row = str(auth_pdbSeq_mapping[resIdx_pdbSeq_row-1])
                  resIdx_pdbAuth_col = str(auth_pdbSeq_mapping[resIdx_pdbSeq_col-1])
                  if 'CB' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CB'
                  elif 'CA' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CA'
                  elif 'CB1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CB1'
                  elif 'CA1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CA1'
                  else:
                    '''
                    # debug
                    print(resiId_dict[resIdx_pdbAuth_row])
                    for atm in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                      print(atm.get_name)
                    '''
                    #raise Exception("No atoms CB,CA,CB1,CA1")
                    noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_row]])
                    # set as no-contact if CB,CA,CB1,CA1 not exist
                    continue

                  if 'CB' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CB'
                  elif 'CA' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CA'
                  elif 'CB1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CB1'
                  elif 'CA1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CA1'
                  else:
                    '''
                    # debug
                    print(resiId_dict[resIdx_pdbAuth_col])
                    for atm in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                      print(atm.get_name)
                    '''
                    #raise Exception("No atoms CB,CA,CB1,CA1")
                    noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_col]])
                    # set as no-contact if CB,CA,CB1,CA1 not exist
                    continue

                  #print('{},{}'.format(resiId_dict[resIdx_pdbAuth_row],resiId_dict[resIdx_pdbAuth_col]))
                  if pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]][atomNm_row]-pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]][atomNm_col] <= cutoff:
                    contact_mat[valReg_row][valReg_col] = 1
                  else:
                    continue
            #build json
            tmp_data_dict['contact-map'] = contact_mat
            packed_data.append(tmp_data_dict)
            print('>*ADDED*<')
          else:
            noValidReg_pdb.append([set_nm,pdbId,pdbChain])
    else: # use predicted structure
      # build json
      tmp_data_dict = {} 
      tmp_data_dict['pfamAcc'] = 'PF'
      tmp_data_dict['unpAcc'] = unpAcc
      unp_pfam_range = range(wtSeq_unp_start, wtSeq_unp_end+1) # index should not exceed unp seq length
      unp_pfam_range_len = len(unp_pfam_range)
      valid_unpIdx_list = unp_pfam_range
      assert len(wt_seq) == unp_pfam_range_len
      tmp_data_dict['set_nm'] = set_nm
      tmp_data_dict['unp_pfam_range'] = '{}-{}'.format(unp_pfam_range[0],unp_pfam_range[-1])
      tmp_data_dict['target_seq'] = wt_seq
      tmp_data_dict['targetSeq_len'] = unp_pfam_range_len
      tmp_data_dict['best_pdb'] = pdbId
      tmp_data_dict['chain_id'] = pdbChain
      tmp_data_dict['valid_unpIdxs_len'] = len(valid_unpIdx_list)
      tmp_data_dict['valid_unpIdxs'] = list(valid_unpIdx_list)
      
      valid_mask = [True]*unp_pfam_range_len
      tmp_data_dict['valid_mask'] = valid_mask 

      # generate contact-map (with self-self, self-neighbor as 1)
      contact_mat = np.zeros((unp_pfam_range_len,unp_pfam_range_len))
      diagIdx = kth_diag_indices(contact_mat,[-1,0,1])
      contact_mat[diagIdx] = 1
      chain_id = 'A'
      pdb_struc = bio_pdbParser.get_structure('model1','{}/wt_seq_structure/{}_trR_results/model1.pdb'.format(working_dir,set_nm))
      pdb_model = pdb_struc[0]
      # build residue id dict
      resiId_dict={}
      for resi_obj in pdb_model[chain_id]:
        resiId_tuple = resi_obj.get_id()
        if resiId_tuple[2] != ' ':
          resiId_dict['{}{}'.format(resiId_tuple[1],resiId_tuple[2])] = resiId_tuple
        else:
          resiId_dict['{}'.format(resiId_tuple[1])] = resiId_tuple

      # loop rows and cols
      for valReg_row in range(unp_pfam_range_len):
        for valReg_col in range(unp_pfam_range_len):
          resIdx_pdbAuth_row = str(valReg_row+1)
          resIdx_pdbAuth_col = str(valReg_col+1)
          ifct[resIdx_pdbAuth_col]]:
            atomNm_col = 'CA'
          elif 'CB1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CB1'
          elif 'CA1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CA1'
          else:
            '''
            # debug
            print(resiId_dict[resIdx_pdbAuth_col])
            for atm in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
              print(atm.get_name)
            '''
            #raise Exception("No atoms CB,CA,CB1,CA1")
            noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_row]])
            # set as no-contact if CB,CA,CB1,CA1 not exist
            continue

          #print('{},{}'.format(resiId_dict[resIdx_pdbAuth_row],resiId_dict[resIdx_pdbAuth_col]))
          if pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]][atomNm_row]-pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]][atomNm_col] <= cutoff:
            contact_mat[valReg_row][valReg_col] = 1
          else:
            continue
      #build json
      tmp_data_dict['contact-map'] = contact_mat
      packed_data.append(tmp_data_dict)
      print('>*ADDED*<')
  print('>>Exception occurs: noRsp_pdb-{},noAlign_pdb-{},noValidReg_pdb-{},noAsymId_pdb-{},wtNotUnp-{}'.format(len(noRsp_pdb),len(noAlign_pdb),len(noValidReg_pdb),len(noAsymId_pdb),len(wtNotUnp)))
  # save all data
  print('>>total {} seqs'.format(len(packed_data)))
  with open('{}/wt_seq_structure/allData_lenCut_l8h500_wt.json'.format(working_dir),'w') as fl:
    json.dump(packed_data,fl,cls=NumpyArrayEncoder)
  
  wrtDir = '{}/wt_seq_structure/allData_lenCut_l8h500_wt.lmdb'.format(working_dir)
  map_size = (1024 * 10) * (2 ** 20) # 10G
  wrtEnv = lmdb.open(wrtDir, map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(packed_data):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  wrtEnv.close() 
  
  np.savetxt('{}/wt_seq_structure/contact_NoRsps.csv'.format(working_dir),noRsp_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoAligns.csv'.format(working_dir),noAlign_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoValids.csv'.format(working_dir),noValidReg_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoAsymIds.csv'.format(working_dir),noAsymId_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoExpAtoms.csv'.format(working_dir),noExpAtom_pdb,fmt='%s',delimiter=';')
  np.savetxt('{}/wt_seq_structure/contact_wtNotUnp.csv'.format(working_dir),wtNotUnp,fmt='%s',delimiter=',')


