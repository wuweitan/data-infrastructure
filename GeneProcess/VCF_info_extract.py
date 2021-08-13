###########################################################################################
# Transform the VCF format into position converter format
###########################################################################################

import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

parser = argparse.ArgumentParser()

parser.add_argument('--VCF_path', type=str, default='../idpanel/VCF_CAGI5_renamed/')
parser.add_argument('--output_path', type=str, default='../Processed_data/', help='path of the processed data')
parser.add_argument('--gene_info_dict', type=str, default='../other_DB/Gene_info/Gene_Info_dict.pickle')

args = parser.parse_args()

output_path = args.output_path
VCF_path = args.VCF_path
gene_info_dict = dict_load(args.gene_info_dict)

if not output_path.endswith('/'):
    output_path += '/'
if not VCF_path.endswith('/'):
    VCF_path += '/'

title = VCF_path.split('/')[-2]
record_path = output_path + '%s_record.txt'%title
output_path += '%s_info.pickle'%title

### Read the VCF files

vcf_list = [f for f in os.listdir(VCF_path) if f.endswith('.vcf')]

VCF_info_dict = {}

with open(record_path, 'w') as wf:
    wf.write('Sample_ID\tmut_num\tgene_num\tuncover_num\tmax_mut_range\n')
    for f in vcf_list:
        temp_path = VCF_path + f
        name = '.'.join(f.split('.')[:-1])
        VCF_info_dict[name] = {'uncovered':[], 'gene_mut':{}, 'gene_posi':{}, 'mut_range':{}}
       
        mut_num = 0

        with open(temp_path, 'r') as rf:
            lines = rf.readlines()

        for line in lines:
            if line.startswith('chr'):
                mut_num += 1

                line = line.split('\t')
                chrom = line[0].split('chr')[-1]
                posi = int(line[1])

                ref = line[3]
                mut = line[4]
                command = '%s:g.%s%s>%s\n'%(chrom, posi, ref, mut)

                ### compare with the genes
                flag = True
                for gene in gene_info_dict.keys():
                    if chrom == gene_info_dict[gene]['chr']:
                         if posi >= gene_info_dict[gene]['posi'][0] and posi <= gene_info_dict[gene]['posi'][1]:
                             flag = False
                             if not gene in VCF_info_dict[name]['gene_mut'].keys():
                                 VCF_info_dict[name]['gene_mut'][gene] = [command]
                                 VCF_info_dict[name]['gene_posi'][gene] = [posi]
                             else:
                                 VCF_info_dict[name]['gene_mut'][gene].append(command)
                                 VCF_info_dict[name]['gene_posi'][gene].append(posi)

                max_mut_range = 0
                for gene in VCF_info_dict[name]['gene_posi'].keys():
                    VCF_info_dict[name]['gene_posi'][gene] = sorted(VCF_info_dict[name]['gene_posi'][gene])
                    VCF_info_dict[name]['mut_range'][gene] = VCF_info_dict[name]['gene_posi'][gene][-1] - VCF_info_dict[name]['gene_posi'][gene][0] + 1
                    if VCF_info_dict[name]['mut_range'][gene] > max_mut_range:
                        max_mut_range = VCF_info_dict[name]['mut_range'][gene]

                ### uncoverd mutations
                if flag:
                    VCF_info_dict[name]['uncovered'].append(command)

        wf.write('%s\t%d\t%d\t%d\t%d\n'%(name, mut_num, len(VCF_info_dict[name]['gene_mut'].keys()), len(VCF_info_dict[name]['uncovered']), max_mut_range))

_save = dict_save(VCF_info_dict, output_path)






