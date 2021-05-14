import numpy as np

#list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

resolution = 5e4
path = 'end_to_end_concatenation_input_' + str(int(resolution)) + '/'
section_size = 10000

for chr_i in range(len(list_chr)):
	matching_index = np.asarray(np.load(path + list_chr[chr_i] + '_matching_index_to_structure.npy'), dtype = int)
	for section_i in range(int(len(matching_index) // section_size + 1)):
		if(section_i < (len(matching_index) // section_size)):
			output = np.zeros(section_size)
			for index_i in range(section_size):
				output[index_i] = matching_index[int(section_i * section_size + index_i)]
		else:
			output = np.zeros(len(matching_index) - int(len(matching_index) // section_size * section_size))
			for index_i in range(output.shape[0]):
				output[index_i] = matching_index[int(len(matching_index) // section_size * section_size + index_i)]
		np.save(path + list_chr[chr_i] + '_section' + str(int(section_i)) + '_matching_index.npy', output)
matching_index = np.asarray(np.load(path + 'chr7_matching_index_to_structure.npy'), dtype = int)
matching_index_valid = matching_index[int(len(matching_index) - 4000):len(matching_index)]
matching_index = matching_index[0:int(len(matching_index) - 4000)]
for section_i in range(int(len(matching_index) // section_size + 1)):
	if(section_i < (len(matching_index) // section_size)):
		output = np.zeros(section_size)
		for index_i in range(section_size):
			output[index_i] = matching_index[int(section_i * section_size + index_i)]
	else:
		output = np.zeros(len(matching_index) - int(len(matching_index) // section_size * section_size))
		for index_i in range(output.shape[0]):
			output[index_i] = matching_index[int(len(matching_index) // section_size * section_size + index_i)]
	np.save(path + 'chr7_section' + str(int(section_i)) + '_matching_index.npy', output)
output = np.zeros(len(matching_index_valid))
for index_i in range(len(matching_index_valid)):
	output[index_i] = matching_index_valid[index_i]
np.save(path + 'validation_matching_index.npy', output)

