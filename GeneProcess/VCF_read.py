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
parser.add_argument('--index_path', type=str, default='../Processed_data/', help='path of the processed data')

args = parser.parse_args()

output_path = args.output_path
VCF_path = args.VCF_path
index_path = args.index_path

if not output_path.endswith('/'):
    output_path += '/'
if not VCF_path.endswith('/'):
    VCF_path += '/'
if not index_path.endswith('/'):
    index_path += '/'

title = VCF_path.split('/')[-2]
submit_path = output_path + 'Submit_Posi_%s.txt'%title
output_path += 'Posi_%s.txt'%title
index_path += 'Posi_%s_index.pickle'%title

### Read the VCF files

vcf_list = [f for f in os.listdir(VCF_path) if f.endswith('.vcf')]

index_dict = {}
num = 0
with open(output_path, 'w') as wf, open(submit_path, 'w') as sf:
    for f in vcf_list:
        temp_path = VCF_path + f
        name = '.'.join(f.split('.')[:-1])
        temp_num = 0
        index_dict[name] = []

        with open(temp_path, 'r') as rf:
            lines = rf.readlines()

        for line in lines:
            if line.startswith('chr'):
                num += 1
                temp_num += 1
                index_dict[name].append(num)
                line = line.split('\t')
                chrom = line[0]
                posi = line[1]

                ref = line[3]
                mut = line[4]

                command = '%s:g.%s%s>%s\n'%(chrom, posi, ref, mut)
                wf.write(command)
                command_2 = '%s:g.%s%s>%s\n'%(chrom, posi, ref[0], mut[0])
                sf.write(command_2)

        print('%s: %d mutations'%(name, temp_num))

_save = dict_save(index_dict, index_path)






