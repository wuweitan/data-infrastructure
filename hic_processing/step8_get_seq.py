import numpy as np

#list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

path = '../../3dstructure_sanitycheck_data/deepsea_data/'
output_path = 'end_to_end_concatenation_input_seq_label/'
section_size = 10000
for chr_i in range(len(list_chr)):
	seq_input = np.load(path + list_chr[chr_i] + '_seq.npy')
	for section_i in range(int(seq_input.shape[0] // section_size + 1)):
		if(section_i < (seq_input.shape[0] // section_size)):
			output = np.zeros((section_size, seq_input.shape[1], seq_input.shape[2]))
			for index_i in range(section_size):
				output[index_i, :, :] = seq_input[int(section_i * section_size + index_i), :, :]
		else:
			output = np.zeros((seq_input.shape[0] - int(seq_input.shape[0] // section_size * section_size), seq_input.shape[1], seq_input.shape[2]))
			for index_i in range(output.shape[0]):
				output[index_i, :] = seq_input[int(seq_input.shape[0] // section_size * section_size + index_i), :]
		np.save(output_path + list_chr[chr_i] + '_seq_section' + str(int(section_i)) + '.npy', output)
seq_input = np.load(path + 'chr7_seq.npy')
for section_i in range(int((seq_input.shape[0] - 4000)// section_size + 1)):
                if(section_i < ((seq_input.shape[0] - 4000)// section_size)):
                        output = np.zeros((section_size, seq_input.shape[1], seq_input.shape[2]))
                        for index_i in range(section_size):
                                output[index_i, :, :] = seq_input[int(section_i * section_size + index_i), :, :]
                else:
                        output = np.zeros(((seq_input.shape[0] - 4000) - int((seq_input.shape[0] - 4000) // section_size * section_size), seq_input.shape[1], seq_input.shape[2]))
                        for index_i in range(output.shape[0]):
                                output[index_i, :, :] = seq_input[int(seq_input.shape[0] // section_size * section_size + index_i), :]
                np.save(output_path + 'chr7_seq_section' + str(int(section_i)) + '.npy', output)
output = np.zeros((4000, seq_input.shape[1], seq_input.shape[2]))
for index_i in range(4000):
	output[index_i, :, :] = seq_input[index_i + int(seq_input.shape[0] - 4000), ]
np.save(output_path + 'seq_validation.npy', output)

