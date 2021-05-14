import numpy as np

resolution = 1e5
seq_path = 'hg19_downloaded/'
node_list = open('ENCFF013TGD_' + str(int(resolution)) + '_npy_index.txt').readlines()
output_path = 'seq_' + str(int(resolution)) + '/'

dict = {}

chr_temp = node_list[0].split('\t')[0]
chr_seq = np.load(seq_path + chr_temp + '.npy')
for node_i in range(len(node_list)):
	if(node_list[node_i].split('\t')[0] != chr_temp):
		chr_temp = node_list[node_i].split('\t')[0]
		chr_seq = np.load(seq_path + chr_temp + '.npy')
	start_index = int(node_list[node_i].split('\t')[1])
	end_index = int(node_list[node_i].split('\t')[2])
	dict[str(int(node_i))] = chr_seq[start_index:end_index]
	np.save(output_path + 'node_' + str(int(node_i)) + '_seq.npy', chr_seq[start_index:end_index])

np.save(output_path + 'seq_dictionary.npy', dict)



