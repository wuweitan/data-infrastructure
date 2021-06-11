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


def mut_fig():
  # params
  working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_name = 'mut_processed_data'
  config_set = {'con': ['_2_0'],
                'nonCon': ['_2_0'],
                'ce': ['_2_0'],
                'pretrain': ['']}
  init_epoch = '20'
  rp_set = 'rp15_all'
  set_list = np.loadtxt('{}/data_process/mutagenesis/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  set_order=['POL_HV1N5-CA_Ndungu2014','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Ostermeier2014','P84126_THETH_b0','BLAT_ECOLX_Palzkill2012','RL401_YEAST_Bolon2013','RASH_HUMAN_Kuriyan','B3VI55_LIPSTSTABLE','HG_FLU_Bloom2016','BG_STRSQ_hmmerbit','TIM_SULSO_b0','AMIE_PSEAE_Whitehead','BG505_env_Bloom2018','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','TIM_THEMA_b0','KKA2_KLEPN_Mikkelsen2014','BF520_env_Bloom2018','YAP1_HUMAN_Fields2012-singles','MK01_HUMAN_Johannessen','UBC9_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','RL401_YEAST_Bolon2014','BLAT_ECOLX_Tenaillon2013','HSP82_YEAST_Bolon2016','RL401_YEAST_Fraser2016','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','MTH3_HAEAESTABILIZED_Tawfik2015','IF1_ECOLI_Kishony','SUMO1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','PA_FLU_Sun2015','BRCA1_HUMAN_BRCT','UBE4B_MOUSE_Klevit2013','HIS7_YEAST_Kondrashov2017','BRCA1_HUMAN_RING','B3VI55_LIPST_Whitehead2015','TPK1_HUMAN_Roth2017','parEparD_Laub2015_all','CALM1_HUMAN_Roth2017','POLG_HCVJF_Sun2014']
  df_list = []
  # load score in other modes
  for mode in ['con','nonCon', 'ce', 'pretrain']:
    for new_i in config_set[mode]:
      for test_set in ['holdout']:
        print('loading set: {}{}'.format(mode,new_i))
        log_fl = 'mutation_{}_{}_torch_eval_{}{}.{}.0.out'.format(rp_set,init_epoch,mode,new_i,test_set)
        print('>log file: {}'.format(log_fl))
        os.system("grep 'loading weights file' job_logs/{} | cut -d'/' -f12 > tmp_rec".format(log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system("rm tmp_rec")
        print('>model dir:',tar_dir)
        print('>json file: results_metrics_{}_{}.json'.format(data_name,test_set))
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name,test_set),'r') as f:
          metric_json = json.load(f)
        for set_nm in set_list:
          data_num = metric_json[set_nm+'_num']
          mse = metric_json[set_nm+'_mse']
          spearmanr = metric_json[set_nm+'_spearmanr']
          df_list.append([mode,rp_set,init_epoch,new_i,test_set,set_nm,data_num,mse,np.abs(spearmanr)])
  df = pd.DataFrame(df_list,columns=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr'])

  # draw point plot
  filter_df = df.loc[(df["test_set"]=='holdout') & (df["rp_nm"]=='rp15_all') & (df["init_epoch"]=='20')]
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0, 'figure.figsize':(120,80)}, font_scale=6)
  gax = sns.pointplot(x="set_nm",y="spearmanr",hue="mode",data=filter_df,join=False,scale=8,
                      ci=None,dodge=False,order=set_order,hue_order=['pretrain','ce','nonCon','con'])
  gax.set_xticklabels(gax.get_xticklabels(), rotation=270)
  tar_fig_dir = 'mut'
  if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
    os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
  plt.savefig('{}/results_to_keep/figures/{}/spearmanr_all_sets.png'.format(working_dir,tar_fig_dir))
  plt.clf()

def mut_precision_fitness_fig():
  '''
  * delta precision vs delta spearmanr
  * delta precision vs delta mse
  '''

  # params
  working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_name = 'mut_processed_data'
  data_name_wt = 'wt_seq_structure_wt'
  config_set = {'con': ['_2_0'],
                'nonCon': ['_2_0'],
                'ce': ['_2_0'],
                'pretrain': ['_0']}
  init_epoch = '20'
  rp_set = 'rp15_all'
  set_list = np.loadtxt('{}/data_process/mutagenesis/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  set_order=['POL_HV1N5-CA_Ndungu2014','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Ostermeier2014','P84126_THETH_b0','BLAT_ECOLX_Palzkill2012','RL401_YEAST_Bolon2013','RASH_HUMAN_Kuriyan','B3VI55_LIPSTSTABLE','HG_FLU_Bloom2016','BG_STRSQ_hmmerbit','TIM_SULSO_b0','AMIE_PSEAE_Whitehead','BG505_env_Bloom2018','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','TIM_THEMA_b0','KKA2_KLEPN_Mikkelsen2014','BF520_env_Bloom2018','YAP1_HUMAN_Fields2012-singles','MK01_HUMAN_Johannessen','UBC9_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','RL401_YEAST_Bolon2014','BLAT_ECOLX_Tenaillon2013','HSP82_YEAST_Bolon2016','RL401_YEAST_Fraser2016','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','MTH3_HAEAESTABILIZED_Tawfik2015','IF1_ECOLI_Kishony','SUMO1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','PA_FLU_Sun2015','BRCA1_HUMAN_BRCT','UBE4B_MOUSE_Klevit2013-singles','HIS7_YEAST_Kondrashov2017','BRCA1_HUMAN_RING','B3VI55_LIPST_Whitehead2015','TPK1_HUMAN_Roth2017','parEparD_Laub2015_all','CALM1_HUMAN_Roth2017','POLG_HCVJF_Sun2014']
  df_list = []
  # load scores
  # * apc precision@all,short,medium,long; L1,2,5
  # * spearmanr
  # * mse
  for mode in ['con','nonCon', 'ce', 'pretrain']:
    for new_i in config_set[mode]:
      for test_set in ['holdout']:
        print('**loading set: {}{}>>'.format(mode,new_i))
        log_fl = 'mutation_{}_{}_torch_eval_{}{}.{}.0.out'.format(rp_set,init_epoch,mode,new_i,test_set)
        print('>log file: {}'.format(log_fl))
        os.system("grep 'loading weights file' job_logs/{} | cut -d'/' -f12 > tmp_rec".format(log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system("rm tmp_rec")
        print('>model dir:',tar_dir)
        print('>json file: results_metrics_{}_{}.json'.format(data_name,test_set))
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name,test_set),'r') as f:
          metric_json = json.load(f)
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name_wt),'r') as f:
          metric_wt_json = json.load(f)
        for set_nm in set_list:
          data_num = metric_json[set_nm+'_num']
          mse = metric_json[set_nm+'_mse']
          spearmanr = metric_json[set_nm+'_spearmanr']
          for topK in ['1','2','5']:
            for ran in ['all','short','medium','long']:
              prec_arr = np.array(metric_wt_json['{}_apc_precision_{}_{}_wt'.format(set_nm,ran,topK)])
              for lay in range(prec_arr.shape[0]):
                for hea in range(prec_arr.shape[1]):
                  head_idx = hea + 1
                  layer_idx = lay + 1
                  prec_val = prec_arr[lay][hea]
                  df_list.append([mode,rp_set,init_epoch,new_i,test_set,set_nm,data_num,mse,np.abs(spearmanr),'L/'+topK,ran,layer_idx,head_idx,prec_val])
  df = pd.DataFrame(df_list,columns=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr','topK','range','layer_idx','head_idx','precision'])

  # plot delta-prec vs delta-spearmanr
  for mode_i in ['ce','con','nonCon']:
    for conf_i in config_set[mode_i]:
      for topK_i in ['1','2','5']:
        precS2plot = []
        precM2plot = []
        precL2plot = []
        spr2plot = []
        mse2plot = []
        for set_i in set_order:
          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr'])
          df_filter = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]==mode_i) &
                                  (df_uniq["config_set"]==conf_i) &
                                  (df_uniq["set_nm"]==set_i)]
          #print(df_filter["spearmanr"].values[0])
          
          ## pretrain
          df_filter_pretrain = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]=='pretrain') &
                                  (df_uniq["set_nm"]==set_i)]

          
          try:
            assert((len(df_filter.index) == 1) & (len(df_filter_pretrain.index)==1))
          except:
            Exception('{}\n{}'.format(df_filter,df_filter_pretrain))

          spr2plot.append(df_filter["spearmanr"].values[0] - df_filter_pretrain["spearmanr"].values[0])
          mse2plot.append(df_filter["mse"].values[0] - df_filter_pretrain["mse"].values[0])


          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','topK','range','layer_idx','head_idx','precision'])
          df_filter = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]==mode_i) &
                                  (df_uniq["config_set"]==conf_i) &
                                  (df_uniq["set_nm"]==set_i) &
                                  (df_uniq["layer_idx"]==4) &
                                  (df_uniq["topK"]=='L/'+topK_i)]
          
          df_filter_pretrain = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]=='pretrain') &
                                  (df_uniq["set_nm"]==set_i) &
                                  (df_uniq["layer_idx"]==4) &
                                  (df_uniq["topK"]=='L/'+topK_i)]


          #print(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan))
          #precS2plot.append(df_filter.loc[(df_filter["range"]=='short')]["precision"].replace(0.0,np.nan).mean())
          #precM2plot.append(df_filter.loc[(df_filter["range"]=='medium')]["precision"].replace(0.0,np.nan).mean())
          #precL2plot.append(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan).mean())

          try:
            assert(len(df_filter.loc[(df_filter["range"]=='short')].index) == 8)
            assert(len(df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')].index) == 8)
          except:
            Exception('check this spot')

          precS2plot.append(df_filter.loc[(df_filter["range"]=='short')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')]["precision"].mean())
          precM2plot.append(df_filter.loc[(df_filter["range"]=='medium')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='medium')]["precision"].mean())
          precL2plot.append(df_filter.loc[(df_filter["range"]=='long')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='long')]["precision"].mean())

        #print('spr2plot',len(spr2plot),spr2plot)
        #print('mse2plot',len(mse2plot),mse2plot)
        #print('precS2plot',len(precS2plot),precS2plot)
        #print('precM2plot',len(precM2plot),precM2plot)
        #print('precL2plot',len(precL2plot),precL2plot)
        
        ''' 
        ## fig: x-set_nm
        fig, host = plt.subplots(figsize=(30,15))
        par1 = host.twinx()
        par2 = host.twinx()
        #host.set_ylim(0, 2)
        #par1.set_ylim(0, 4)
        #par2.set_ylim(1, 65)
        host.set_xlabel("setName")
        host.set_ylabel("deltaPrecision")
        par1.set_ylabel("deltaSpearmanR")
        par2.set_ylabel("deltaMSE")
        x_tick = np.arange(1,2+1*(len(set_order)-1),1)
        p1, = host.plot(x_tick,precS2plot, marker='1', markersize=10, color='r', linestyle = 'None',label="short")
        p1_1, = host.plot(x_tick,precM2plot, marker='+', markersize=10, color=p1.get_color(), linestyle = 'None', label="medium")
        p1_2, = host.plot(x_tick,precL2plot, marker='^', markersize=10, color=p1.get_color(), linestyle = 'None', label="long")
        p2, = par1.plot(x_tick,spr2plot, color='g', marker='.', markersize=10, linestyle = 'None', label="spearmanR")
        p3, = par2.plot(x_tick,mse2plot, color='b', marker='.', markersize=10, linestyle = 'None', label="mse")
        host.axhline(y=0, color='r', linestyle='--')
        par1.axhline(y=0, color='g', linestyle='--')
        par2.axhline(y=0, color='b', linestyle='--')
        lns = [p1,p1_1,p1_2,p2,p3]
        host.legend(handles=lns, loc='best')
        # right, left, top, bottom
        par2.spines['right'].set_position(('outward', 60))
        
        # no x-ticks                 
        par2.xaxis.set_ticks([])
        
        # Sometimes handy, same for xaxis
        #par2.yaxis.set_ticks_position('right')
        
        # Move "Velocity"-axis to the left
        # par2.spines['left'].set_position(('outward', 60))
        # par2.spines['left'].set_visible(True)
        # par2.yaxis.set_label_position('left')
        # par2.yaxis.set_ticks_position('left')
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(x_tick)
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        # Adjust spacings w.r.t. figsize
        fig.tight_layout()
        # Alternatively: bbox_inches='tight' within the plt.savefig function 
        #                (overwrites figsize)
        
        host.grid(which='major', axis='x', linestyle='--')
        
        # Best for professional typesetting, e.g. LaTeX
        plt.savefig('{}/results_to_keep/figures/mut/delta_prec_R_mse_{}{}_L{}.png'.format(working_dir,mode_i,conf_i,topK_i))
        plt.clf()
        # For raster graphics use the dpi argument. E.g. '[...].png",dpi=200)'
        '''

        ## x, y both delta-score
        prec_ran_list = [precS2plot,precM2plot,precL2plot]
        ran_list = ['short','medium','long']
        for ran_i in range(3):
          ## R
          fig, host = plt.subplots()
          ax.scatter(prec_ran_list[ran_i], spr2plot)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            ax.annotate(txt, (prec_ran_list[ran_i][i], spr2plot[i]))
          plt.savefig('{}/results_to_keep/figures/mut/xyDelta_prec_R_{}{}_L{}_{}.png'.format(working_dir,mode_i,conf_i,topK_i,ran_list[ran_i]))
          plt.clf()
          ## mse
          fig, host = plt.subplots()
          ax.scatter(prec_ran_list[ran_i], mse2plot)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            ax.annotate(txt, (prec_ran_list[ran_i][i], mse2plot[i]))
          plt.savefig('{}/results_to_keep/figures/mut/xyDelta_mse_R_{}{}_L{}_{}.png'.format(working_dir,mode_i,conf_i,topK_i,ran_list[ran_i]))
          plt.clf()
