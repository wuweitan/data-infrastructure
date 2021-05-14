import numpy as np
import os

chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
resolution = 1e5
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM'])

chr_size = np.asarray(chr_length // resolution + 1, dtype = int) 

hic_list = eval(open('hic_list.txt', 'r').read())

path = 'if_intermediate_' + str(int(resolution)) + '/'

def if_txt_to_npy(hic_name):
	for chr_i in range(len(list_chr)):
		for chr_j in range(len(list_chr)):
			if(chr_i <= chr_j):
				output = np.zeros((chr_size[chr_i], chr_size[chr_j]))
				if(os.path.isfile(path + hic_name + list_chr[chr_i] + '_' + list_chr[chr_j] + '.txt')):
					input = open(path + hic_name + list_chr[chr_i] + '_' + list_chr[chr_j] + '.txt').readlines()
					coordinate_1 = []
					coordinate_2 = []
					counts = []
					for i in range(len(input)):
						if_float = float(input[i].split('\t')[2].split('\n')[0])
						if(not np.isnan(if_float)):
							coordinate_1.append(int(input[i].split('\t')[0]) // resolution)
							coordinate_2.append(int(input[i].split('\t')[1]) // resolution)
							counts.append(if_float)
					coordinate_1 = np.asarray(coordinate_1)
					coordinate_2 = np.asarray(coordinate_2)
					counts = np.asarray(counts)
					if(len(counts) == 0):
						print(list_chr[chr_i] + '_' + list_chr[chr_j])
					elif(np.min(counts) == 0):
						print(list_chr[chr_i] + '_' + list_chr[chr_j])
					for i in range(len(counts)):
						output[int(coordinate_1[i]), int(coordinate_2[i])] = counts[i]
						if(chr_i == chr_j):
							output[int(coordinate_2[i]), int(coordinate_1[i])] = counts[i]
				np.save(path + hic_name + list_chr[chr_i] + '_' + list_chr[chr_j] + '.npy', output)
					

for key_index_i in range(len(list(hic_list.keys()))):
	hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
	for hic_i in range(len(hic_list_temp)):
		if_txt_to_npy(hic_list_temp[hic_i] + '_novc_')
		

