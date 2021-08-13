###########################################################################################
# prepare the gene-gene interaction matrix
###########################################################################################

import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas

parser = argparse.ArgumentParser()

parser.add_argument('--gene_list_file', type=str, default='../idpanel/other_DB/NIHMS1036707-supplement-Supp_TableS3.xlsx')
parser.add_argument('--gene_embed_file', type=str, default='../Gene2vec/pre_trained_emb/gene2vec_dim_200_iter_9.txt')
parser.add_argument('--output_path', type=str, default='../Processed_data/', help='path of the processed data')
parser.add_argument('--image_path', type=str, default='../Processed_data/Images/', help='path of the images')

parser.add_argument('--threshold', type=float, default=None)

args = parser.parse_args()

output_path = args.output_path
image_path = args.image_path

if not output_path.endswith('/'):
    output_path += '/'
if not image_path.endswith('/'):
    image_path += '/'

threshold = args.threshold

def dict_save(dictionary,path):
    with open(path,'wb') as handle:
        pickle.dump(dictionary, handle)
    return 0

def dict_load(path):
    with open(path,'rb') as handle:
        result = pickle.load(handle)
    return result

### load the gene embedding dictionary

print('Load gene embedding dictionary...')

gene_embed_dict_path = output + 'gene_embed_dict.pickle' 
if os.path.exists(gene_embed_dict_path):
    gene_embed_dict = dict_load(gene_embed_dict_path)
else:
    gene_embed_file = args.gene_embed_file
    emb_dim = 200
    gene_embed_dict = {}
    with open(gene_embed_file, 'r') as rf:
        lines = rf.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        name = line[0]
        vec = np.array([float(char) for char in line[1].split(' ') if char != ''])
        if vec.shape[0] != emb_dim:
            print('Error! The embedding dimension of Gene %s is not %d but %d!'%(name, emb_dim, vec.shape[0]))
            continue
        gene_embed_dict[name] = vec

    _save = dict_save(gene_embed_dict, gene_embed_dict_path)

print('%d genes in all.'%len(gene_embed_dict.keys()))

### load the gene list

print('Load gene list...')

gen_list_path = output + 'gene_ID.list' 
if os.path.exists(gene_list_path):
    gene_list = dict_load(gene_list_path)
else:
    gene_list_file = args.gene_list_file
    gene_xlsx = pandas.read_excel(gene_list_file, engine='openpyxl')
    gene_list = [char for char in list(gene_xlsx['Gene']) is type(char) == str]
    _save = dict_save(gene_list, gene_list_path)

gene_id_num = len(gene_list)
print('%d genes of the ID pannel.'%gene_id_num)

print('Check the genes...')
for gene in gene_list:
    if not gene in gene_embed_dict.keys():
        print('Error! The gene %s is not in the embedding dictionary!'%gene)
        quit()

### distance-map cal

print('Prepare distance map...')

dist_map = np.zeros([gene_id_num, gene_id_num])

for i in range(gene_id_num):
   gene_emb_1 = gene_embed_dict[gene_list[i]]
   for j in range(i + 1, gene_id_num):
       gene_emb_2 = gene_embed_dict[gene_list[j]]
       dist_map[i,j] = np.linalg.norm(gene_emb_1 - gene_emb_2)
       dist_map[j,i] = dist_map[i,j]

np.savetxt(output_path + 'ID_GG_dict.txt', dist_map)
print('Heatmap...')
plt.fi

### GGI graph

if threshold is not None:
    print('Prepare gene-gene interaction map...')
    ggi_map = np.zeros([gene_id_num, gene_id_num])
    ggi_map[dist_map <= threshold] = 1
    ggi_map -= np.eye(gene_id_num)
print('Done.')


