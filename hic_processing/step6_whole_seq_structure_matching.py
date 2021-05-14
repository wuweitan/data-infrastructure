import numpy as np

resolution = 5e4
input_path = ''
output_path = 'end_to_end_concatenation_input_' + str(int(resolution)) + '/'
input_all = open(input_path + 'allTFs.pos.bed').readlines()
input_all_chr = []
input_all_start = []
for i in range(2204000):
	input_all_temp = input_all[i].split('\t')
	input_all_chr.append(input_all_temp[0])
	input_all_start.append(input_all_temp[1])

for i in np.arange(2309366, 2536878):
	input_all_temp = input_all[i].split('\t')
	input_all_chr.append(input_all_temp[0])
	input_all_start.append(input_all_temp[1])

input_all_start = np.asarray(np.asarray(input_all_start, dtype = float) // resolution * resolution, dtype = int)
input_all_chr = np.asarray(input_all_chr)

print('seq position loading finished')

input_structure = open(input_path + 'if_matrix_' + str(int(resolution)) + '/ENCFF013TGD_' + str(int(resolution)) + '_npy_index.txt').readlines()
input_structure_chr = []
input_structure_start = []
for i in range(len(input_structure)):
	input_structure_temp = input_structure[i].split('\t')
	input_structure_chr.append(input_structure_temp[0])
	input_structure_start.append(input_structure_temp[1])

print('structure position loading finished')

input_structure_chr = np.asarray(input_structure_chr)
input_structure_start = np.asarray(input_structure_start, dtype = int)

index_matching = np.zeros(len(input_all_chr))
for i in range(len(input_all_chr)):
	index_matching[i] = np.where((input_all_chr[i] == input_structure_chr) & (input_all_start[i] == input_structure_start))[0][0]

#np.save(path + 'whole_genome_matching_index_to_structure.npy', index_matching)

list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

for chr_i in range(len(list_chr)):
	np.save(output_path + list_chr[chr_i] + '_matching_index_to_structure.npy', index_matching[input_all_chr == list_chr[chr_i]])


