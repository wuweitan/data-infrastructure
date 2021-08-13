###########################################################################################
# prepare the gene-gene interaction matrix
###########################################################################################

import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import urllib.request

parser = argparse.ArgumentParser()

parser.add_argument('--gene_list_file', type=str, default='../other_DB/NIHMS1036707-supplement-Supp_TableS3.xlsx')
parser.add_argument('--seq_path', type=str, default='../other_DB/Ref_seq/', help='path of the processed data')
parser.add_argument('--info_path', type=str, default='../other_DB/Gene_info/', help='path of the images')

args = parser.parse_args()

seq_path = args.seq_path
info_path = args.info_path

if not seq_path.endswith('/'):
    seq_path += '/'
if not info_path.endswith('/'):
    info_path += '/'

web_path = info_path + 'WebPage/'
if not os.path.exists(web_path):
    os.mkdir(web_path)

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

####################### load the gene list #####################################

print('Load gene list...')

gene_list_path = info_path + 'gene_ID.list'    
gene_list_file = args.gene_list_file

gene_xlsx = pandas.read_excel(gene_list_file, engine='openpyxl')
gene_list = [char for char in list(gene_xlsx['Gene']) if type(char) == str]
chr_list = [char for char in list(gene_xlsx['Chr band']) if type(char) == str]

_save = dict_save(gene_list, gene_list_path)

gene_id_num = len(gene_list)
print('%d genes of the ID pannel.'%gene_id_num)

####################### download gene information #####################################

print('Download the gene information...')
for gene in gene_list:
    if gene == 'NLGN4':
        gene = 'NLGN4X'
    if not os.path.exists(web_path + gene):
         _ = urllib.request.urlretrieve('https://www.ncbi.nlm.nih.gov/gene/?term=%s'%gene, web_path + gene)
   
####################### download gene information #####################################

print('Read the gene information and download the sequences...')

gene_info_dict = {}

with open(info_path + 'Gene_Info.txt', 'w') as wf:
    for i,gene in enumerate(gene_list):
        if gene == 'NLGN4':
            gene = 'NLGN4X'

        try:
            chrom = chr_list[i]
            if 'p' in chrom:
                chrom = chrom.split('p')[0]
            elif 'q' in chrom:
                chrom = chrom.split('q')[0]
            else:
                print('Error! The Chr band of %s is not correct!'%gene)
                continue
    
            gene_info_dict[gene] = {'chr':chrom}
    
            temp_path = web_path + gene
      
            with open(temp_path, 'r') as tf:
                lines = tf.readlines()
    
            flag = False
            for line in lines:
                if 'click_feat_download' in line:  # for sequences and id
                    url = line.split('"')[1]
                    gene_id = url.split('id=')[-1].split(',')
                    if 'RefSeq transcripts' in line:
                        _ = urllib.request.urlretrieve(url, seq_path + '%s_transcripts.fasta'%gene)
                        gene_info_dict[gene]['transcripts_id'] = gene_id
                    elif 'RefSeqGene' in line:
                        _ = urllib.request.urlretrieve(url, seq_path + '%s_GeneSeq.fasta'%gene)
                        gene_info_dict[gene]['GeneSeq_id'] = gene_id
                        gene_seq = ''
                        with open(seq_path + '%s_GeneSeq.fasta'%gene, 'r') as rf:
                            lines = rf.readlines()
                        for line in lines:
                            if not line.startswith('>'):
                                gene_seq += line.strip('\n')
                        gene_info_dict[gene]['GeneSeq'] = gene_seq
                        gene_seq_len = len(gene_seq)
                        gene_info_dict[gene]['Gene_Len'] = gene_seq_len
    
                elif 'Gene ID:' in line:  # for gene id
                    g_id = line.split('Gene ID: ')[-1].split('<')[0]
                    gene_info_dict[gene]['gene_id'] = g_id 
                    flag = True
    
                elif 'complement' in line and flag:  # for position
                    line = line.split('ID: ')
                    for seg in line:
                        if seg.startswith(g_id):
                            start = seg.split('..')[0].split('(')[-1]
                            end = seg.split('..')[-1].split(')')[0]
                            if ',' in end:
                                end = end.split(',')[0]
                            gene_info_dict[gene]['posi'] = [int(start), int(end)]
                            if gene_seq_len == int(end) - int(start):
                                label = 'PASS'
                            else:
                                label = 'FAIL'
                            break
                    flag = False

            wf.write('%s\t%s\t%d\t%d\t%d\t%s\n'%(gene, gene_info_dict[gene]['gene_id'], gene_info_dict[gene]['posi'][0], gene_info_dict[gene]['posi'][1],
                                                 len(gene_seq), label))
        except Exception as e:
            print(gene, e)
 
_save = dict_save(gene_info_dict, info_path + 'Gene_Info_dict.pickle')

print('Done. %d genes processed.'%len(gene_info_dict.keys()))
