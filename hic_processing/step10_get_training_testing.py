import numpy as np

#list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

resolution = 5e4
path = 'end_to_end_concatenation_input_' + str(int(resolution)) + '/'
section_size = 10000
section_chr = []
section_index = []
for chr_i in range(len(list_chr)):
	matching_index = np.asarray(np.load(path + list_chr[chr_i] + '_matching_index_to_structure.npy'), dtype = int)
	for section_i in range(int(len(matching_index) // section_size + 1)):
		section_chr.append(list_chr[chr_i])
		section_index.append(section_i)
matching_index = np.asarray(np.load(path + 'chr7_matching_index_to_structure.npy'), dtype = int)
matching_index = matching_index[0:int(len(matching_index)-4000)]
for section_i in range(int(len(matching_index) // section_size + 1)):
	section_chr.append('chr7')
	section_index.append(section_i)
np.save(path + 'training_section_chr.npy', np.asarray(section_chr))
np.save(path + 'training_section_index.npy', np.asarray(section_index))
list_chr = ['chr8', 'chr9']
section_chr = []
section_index = []
for chr_i in range(len(list_chr)):
	matching_index = np.asarray(np.load(path + list_chr[chr_i] + '_matching_index_to_structure.npy'), dtype = int)
	for section_i in range(int(len(matching_index) // section_size + 1)):
		section_chr.append(list_chr[chr_i])
		section_index.append(section_i)

np.save(path + 'testing_section_chr.npy', np.asarray(section_chr))
np.save(path + 'testing_section_index.npy', np.asarray(section_index))


