import numpy as np
from prody import *
import re, sys

class SequenceStatisticsClass():
  """statistical count class for Pfam sequence
     Provides functions to process pfam sequence data. e.g. MSA in 'selex', 'stockholm' and 'fasta' format
     Currently only support processing 'stockholm' format, others will be added later.

     Functions:
  """
  def __init__(self, data_dir, **kwargs):
    self.data_dir
    #self.arg1 = kwargs.pop('arg1', None)

  def gather_stat(rund):
    '''count the following statistics:
        * number of sequences in each Pfam entry
        * number of Pfam entries in each type of six (Coiled-coil;Disordered;Domain;Family;Motif;Repeat)
        * total number of Pfam entries
        * total number of sequencs
       and save these informatoin to file

       input: 
        - file_name: Pfam MSA file(STOCKHOLM format)
        - out_stat: output file to save statistics
    '''
    for i in rund:
      print('Working on Pfam-A.rp{}...'.format(i))
      records = []
      with open(data_dir+'/Pfam-A.rp{}'.format(i), 'r', encoding='latin-1') as fl:
        for line in fl:
          ID = re.findall("^#=GF ID (.+)\n", line)
          AC = re.findall("^#=GF AC (.+)\n", line)
          DE = re.findall("^#=GF DE (.+)\n", line)
          TP = re.findall("^#=GF TP (.+)\n", line)
          SQ = re.findall("^#=GF SQ (.+)\n", line)
          if len(ID) > 0:
            print('Pfam - {}'.format(ID))
            rec = [ID[0]]
          elif len(AC) > 0:
            rec.append(AC[0])
          elif len(DE) > 0:
            rec.append(DE[0])
          elif len(TP) > 0:
            rec.append(TP[0])
          elif len(SQ) > 0:
            rec.append(SQ[0])
            #print('one record: {}'.format(rec))
            records.append(rec)

      # stat count
      records = np.array(records)
      total_num = records.shape[0]
      fields = records[:,3]
      field_uniq, fields_count = np.unique(fields, return_counts=True)
      seq_num = records[:,4].astype(np.int)
      seq_num_sum = sum(seq_num)
      # count of each type

      # comment line
      field_str = ''
      type_seq_num = []
      for j in range(len(field_uniq)):
        idx_list = np.where(records[:,3]==field_uniq[j])[0]
        seq_num = records[idx_list,4].astype(np.int)
        type_seq_num.append(np.sum(seq_num))
        field_str += '{}:{}/{}, '.format(field_uniq[j],fields_count[j],np.sum(seq_num))

      type_seq_num_all = np.sum(type_seq_num)
      #assert(type_seq_num_all == seq_num_sum)
      comm = '{}total_fam_num:{}, total_seq_num:{} =? type_seq_num_all:{}'.format(field_str,total_num,seq_num_sum,type_seq_num_all)
      #print(comm)
      
      # save to file
      
      np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}.stat'.format(i,i), records, fmt='%s', delimiter='\t',header=comm)

      # order by seq number, Fam iden
      rec_by_seqNum = records[np.argsort(records[:,-1])]
      rec_by_famIden = records[np.argsort(records[:,1])]
      np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}_sortBySeqsize.stat'.format(i,i),rec_by_seqNum,fmt='%s', delimiter='\t',header=comm)
      np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}_sortByFacc.stat'.format(i,i),rec_by_famIden,fmt='%s', delimiter='\t',header=comm)

  def collect(rund):
    '''
    * count length of aligned sequence and unaligned sequence
    * find unique letters occured in sequences
    * record sequences containing ambiguous letters
    '''
    for i in rund:
      # define recorders
      rawSeqLen_type = {'Family':[],'Domain':[],'Motif':[],'Repeat':[],'Coiled-coil':[],'Disordered':[]}
      msaSeqLen_type = {'Family':[],'Domain':[],'Motif':[],'Repeat':[],'Coiled-coil':[],'Disordered':[]}
      rawSeqLen_pfam = {}
      rawSeqLen_clan = {'Orphan':[]}
      uniq_chars = [] #list to hold unique characters occuring in msa
      ambig_chars = ['B','Z','X','O','U','b','z','x','u']
      seqs_ambigChar = [] # list to store seqs containing ambiguous tokens
      
      
      # load family acc
      #pfam_info_list = np.loadtxt(data_dir+'/Pfam-A.rp{}.stat'.format(i), dtype='str', delimiter='\t')
      #new_label_list = [] # to store all labels
      #seq_list = [] # to store all seqs
      
      # load msa file list
      msa_files = np.loadtxt(data_dir+'/pfam_rp{}_seqs_files_famAcc'.format(i),dtype='str',delimiter=' ')
      #print(msa_files.shape)
      

      for idx in range(len(msa_files)):
      #for idx in range(1,2):
        #pfam_acc = pfam_info_list[idx,1]
        #pfam_dscp = pfam_info_list[idx,0]
        #print('download seqs of pfam: {}, rp {}...'.format(pfam_acc, i), flush=True)
        
        # download msa in fasta format for this fam
        #fetchPfamMSA(pfam_acc, alignment='rp{}'.format(i), 
        #             format='fasta', order='tree', inserts='upper',
        #             gaps=None, outname='tmp_msa')
        # modify comment line of msa file
        #msa = MSAFile('tmp_msa_rp{}.fasta'.format(i), aligned=False)
        msa_nm = msa_files[idx][0]
        pfam_nm = msa_files[idx][1]
        clan_nm = msa_files[idx][2]
        print('>>>{},{},{}'.format(msa_nm,pfam_nm, clan_nm))
        # initialize list to store seq len
        if pfam_nm not in rawSeqLen_pfam.keys():
          rawSeqLen_pfam[pfam_nm] = []
        if len(clan_nm) > 0 and clan_nm not in rawSeqLen_clan.keys():
          rawSeqLen_clan[clan_nm] = []
        # load msa file as a string
        msa_fl = open('{}/{}'.format(data_dir, msa_nm), 'r', encoding='latin-1')
        msa_content = msa_fl.readlines()
        # extract info from meta-header
        #ID, AC = '', ''
        for line in msa_content:
          '''
          ID_tmp = re.findall("^#=GF ID (.+)\n", line)
          AC_tmp = re.findall("^#=GF AC (.+)\n", line)
          if len(ID_tmp) > 0:
            ID = ID_tmp[0]
          if len(AC_tmp) > 0:
            AC = AC_tmp[0]
          if len(ID) > 0 and len(AC) > 0:
            break
          '''
          TP_tmp = re.findall("^#=GF TP (.+)\n", line)
          if len(TP_tmp) > 0:
            TP = TP_tmp[0]
            break
        print('type: {}'.format(TP))
        # initilize msa from stockholm file
        msa = MSAFile('{}/{}'.format(data_dir, msa_nm), format='Stockholm')
        for seq in msa:
          #extract from header
          res_idx = seq.getResnums()
          start_idx = res_idx[0]
          end_idx = res_idx[-1]
          label = seq.getLabel()
          uniP_id = label
          #new_label_list.append('{}/{}-{} {} {};{};'.format(label,start_idx,end_idx, uniP_id, AC, ID))
          pro_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
          #print(new_label_list[-1])
          #print(pro_seq)
          #seq_list.append(pro_seq)
          
          # store length
          msaSeq_len = len(str(seq))
          rawSeq_len = len(pro_seq)
          rawSeqLen_type[TP].append(rawSeq_len)
          msaSeqLen_type[TP].append(msaSeq_len)
          rawSeqLen_pfam[pfam_nm].append(rawSeq_len)
          if len(clan_nm) > 0:
            rawSeqLen_clan[clan_nm].append(rawSeq_len)
          else:
            rawSeqLen_clan['Orphan'].append(rawSeq_len)
          # collect uniq characters
          chars = list(set(str(seq)))
          for c in chars:
            if c not in uniq_chars:
              uniq_chars.append(c)
            if c in ambig_chars:
              seqs_ambigChar.append(['{}/{}/{}/{}-{}/{}'.format(pfam_nm,clan_nm,label,start_idx,end_idx,len(pro_seq)),pro_seq])
      
      # create new msa object
      #new_msa = MSA(seq_list, title='new_fasta', labels=new_label_list, aligned=False)
      # save to fasta file
      #writeMSA('Pfam-A.rp{}.fasta'.format(i), new_msa, aligned=False)
      '''
      print('writing to fasta file...') 
      with open(data_dir+'/Pfam-A.rp{}.fasta'.format(i), 'w') as fl:
        for seq_i in range(len(new_label_list)):
          fl.write('>{}\n'.format(new_label_list[seq_i]))
          fl.write(seq_list[seq_i]+'\n')
      '''

      # save counts and draw figures
      '''
      with open('{}/stat_rp{}/rawSeqLen_type.json'.format(data_dir,rund[0]), 'w') as fl:
        json.dump(rawSeqLen_type, fl)
      with open('{}/stat_rp{}/msaSeqLen_type.json'.format(data_dir,rund[0]), 'w') as fl:
        json.dump(msaSeqLen_type, fl)
      np.savetxt('{}/stat_rp{}/seqs_ambigChar'.format(data_dir,rund[0]),seqs_ambigChar,fmt='%s',delimiter=' ')
      uniq_chars.sort()
      np.savetxt('{}/stat_rp{}/uniq_chars'.format(data_dir,rund[0]),uniq_chars,fmt='%s')
      '''
      with open('{}/stat_rp{}/rawSeqLen_pfam.json'.format(data_dir,rund[0]), 'w') as fl:
        json.dump(rawSeqLen_pfam, fl)
      with open('{}/stat_rp{}/rawSeqLen_clan.json'.format(data_dir,rund[0]), 'w') as fl:
        json.dump(rawSeqLen_clan, fl) 
 
  def filter_stat(rund,len_thres):
    '''
    '''
    for i in rund:
      rp_stat = np.loadtxt(data_dir+'/stat_rp{}/Pfam-A.rp{}.stat'.format(i,i),dtype='str',delimiter='\t')
      pfam_clan = np.loadtxt('{}/Pfam-A.clans.tsv'.format(data_dir),dtype='str',delimiter='\t')
      clan_list = np.loadtxt('{}/Pfam-A.clan_list'.format(data_dir),dtype='str')
      clan_count = []
      
      for clan in clan_list:
        print(clan)
        tar_idx = np.where(pfam_clan[:,1]==clan)[0]
        tar_pfams = [pfam_clan[p,0] for p in tar_idx]
        # remove version
        rp_pfams = np.array([re.split(r'\.',pf)[0] for pf in rp_stat[:,1]])
        tar_idx = []
        for pf in tar_pfams:
          idx = np.where(rp_pfams==pf)[0]
          if len(idx) > 0:
            tar_idx.append(idx[0])
          else:
            pass
        if len(tar_idx) > 0:
          clan_count.append([clan,np.sum(rp_stat[tar_idx,-1].astype(int))])
      clan_count = np.array(clan_count)
      
      np.savetxt('{}/stat_rp{}/family_counts.tsv'.format(data_dir,i),rp_stat[:,[1,4]],fmt='%s',delimiter='\t',header='total pfam seq:{},total pfam num:{}'.format(np.sum(rp_stat[:,-1].astype(int)),rp_stat.shape[0]))
      np.savetxt('{}/stat_rp{}/clan_counts.tsv'.format(data_dir,i),clan_count,fmt='%s',delimiter='\t',header='total clan seq:{},total clan num:{}'.format(np.sum(clan_count[:,-1].astype(int)),clan_count.shape[0]))

      # filter by length
      if len_thres > 0:
        with open('{}/stat_rp{}/rawSeqLen_type.json'.format(data_dir,i),'r') as fl:
          rawSeqLen_type = json.load(fl)
        seqNum_byLen = []
        for key,value in rawSeqLen_type.items():
          value = np.array(value).astype(int)
          seqNum = len(np.where(value <= len_thres)[0])
          seqNum_byLen.append([key,seqNum])
        seqNum_byLen = np.array(seqNum_byLen).astype(str)
        np.savetxt('{}/stat_rp{}/seqNumType_filterBy{}.tsv'.format(data_dir,i,len_thres),seqNum_byLen,fmt='%s',delimiter='\t',header='total num:{}'.format(np.sum(seqNum_byLen[:,1].astype(int))))

  def testSet_stat(rund,level,fl_dir):
    rp_rund = rund[0]
    fl_path = '{}/holdOut_sets/{}'.format(data_dir,fl_dir)
    holdOut_list = np.loadtxt(fl_path,dtype='str')
    seqNum_count = []
    if level=='pfam':
      stat_dt = np.loadtxt('{}/stat_rp{}/family_counts.tsv'.format(data_dir,rp_rund),dtype='str',delimiter='\t')
    elif level=='clan':
      stat_dt = np.loadtxt('{}/stat_rp{}/clan_counts.tsv'.format(data_dir,rp_rund),dtype='str',delimiter='\t')
    for iden in holdOut_list:
      tar_idx = np.where(np.array([re.split(r'\.',i)[0] for i in stat_dt[:,0]])==iden)[0]
      if len(tar_idx) > 0:
        seqNum_count.append([iden,int(stat_dt[tar_idx[0],1])])
      else:
        seqNum_count.append([iden,0])
    seqNum_count = np.array(seqNum_count)
    np.savetxt('{}/stat_rp{}/{}_stat.tsv'.format(data_dir,rp_rund,fl_dir),seqNum_count,fmt='%s',delimiter='\t',header='total num:{}'.format(np.sum(seqNum_count[:,1].astype(int))))

  def clan_anno(rp):
    clan_set = {}
    clan_set_seqNum = 0
    noClan_pfam_set = []
    # load whole pfam-clan pairs
    pfam_clan = np.loadtxt('{}/Pfam-A.clans.tsv'.format(data_dir),dtype='str',delimiter='\t')
    # load this rp's stat
    pfam_seqNum_count = np.loadtxt('{}/stat_rp{}/Pfam-A.rp{}.stat'.format(data_dir,rp,rp),dtype='str',delimiter='\t')
    #print(pfam_seqNum_count)
    for i in range(len(pfam_seqNum_count)):
      line = pfam_seqNum_count[i]
      #print(line)
      pfam_nm = line[1]
      pfam_nm_noVer = re.split(r'\.',pfam_nm)[0]
      tar_idx = np.where(pfam_clan[:,0]==pfam_nm_noVer)[0][0]
      clan_nm = pfam_clan[tar_idx,1]
      #pfam_nm = re.split(r'\.',pfam_nm_ver)[0]
      if len(clan_nm) == 0:
        tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
        noClan_pfam_set.append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
      else:
        if clan_nm not in clan_set.keys():
          clan_set[clan_nm] = []
          tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
          clan_set[clan_nm].append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
          clan_set_seqNum += int(pfam_seqNum_count[tar_idx,4])
        else:
          tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
          clan_set[clan_nm].append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
          clan_set_seqNum += int(pfam_seqNum_count[tar_idx,4])

    clan_entryNum = []
    for key, value in clan_set.items():
      value = np.array(value)
      num_entry = value.shape[0]
      uniq_types = np.unique(value[:,1])
      clan_entryNum.append([key,num_entry,';'.join(uniq_types)])
    clan_entryNum = np.array(clan_entryNum)
    clan_entryNum_all = np.sum(clan_entryNum[:,1].astype(np.int))
    
    noClan_pfam_set = np.array(noClan_pfam_set)
    noClan_SeqNum = np.sum(noClan_pfam_set[:,2].astype(np.int))
    noClan_pfamNum = len(noClan_pfam_set)
    noClan_pfam_set_sorted = noClan_pfam_set[noClan_pfam_set[:,2].astype(np.int).argsort()]
    np.savetxt('{}/stat_rp{}/pfam_noClan.csv'.format(data_dir,rp),noClan_pfam_set_sorted,fmt='%s',delimiter=',',header='seqNum_clan:{}, seqNum_noClan:{}, pfamNum_clan:{}, pfamNum_noClan:{}'.format(clan_set_seqNum, noClan_SeqNum,clan_entryNum_all,noClan_pfamNum))

    clan_entryNum_sorted = clan_entryNum[clan_entryNum[:,1].astype(np.int).argsort()]
    np.savetxt('{}/stat_rp{}/clan_entryNum.csv'.format(data_dir,rp),clan_entryNum_sorted,fmt='%s',delimiter=',',header='seqNum_clan:{}, seqNum_noClan:{}, pfamNum_clan:{}, pfamNum_noClan:{}'.format(clan_set_seqNum,noClan_SeqNum,clan_entryNum_all,noClan_pfamNum))

    #print(clan_set)
    with open('{}/stat_rp{}/clan_set.json'.format(data_dir,rp),'w') as fl:
      json.dump(clan_set, fl)
   
