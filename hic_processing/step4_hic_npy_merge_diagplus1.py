import numpy as np

chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
resolution = 1e6
list_chr = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY', 'chrM'])
chr_size = np.asarray(chr_length // resolution + 1, dtype = int)
chr_start = np.zeros(len(chr_length))
for i in np.arange(1, len(chr_length)):
	chr_start[i] = np.sum(chr_size[0:i])

total_length = int(np.sum(chr_size))
output = np.zeros((total_length, total_length))

path = 'if_intermediate_' + str(int(resolution)) + '/'

hic_list = eval(open('hic_list.txt', 'r').read())

for key_index_i in range(len(list(hic_list.keys()))):
	hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
	for hic_i in range(len(hic_list_temp)):
		hic_name_i = hic_list_temp[hic_i] + '_'
		for chr_i in range(len(list_chr)):
			for chr_j in  range(len(list_chr)):
				if(chr_i <= chr_j):
					input = np.load(path + hic_name_i + 'novc_' + list_chr[chr_i] + '_' + list_chr[chr_j] + '.npy')
				else:
					input = np.load(path + hic_name_i + 'novc_' + list_chr[chr_j] + '_' + list_chr[chr_i] + '.npy').T
				output[int(chr_start[chr_i]):int(chr_start[chr_i]+chr_size[chr_i]), int(chr_start[chr_j]):int(chr_start[chr_j]+chr_size[chr_j])] = input
		np.save(path + hic_name_i + 'novc_whole.npy', output)
		for diag_i in range(output.shape[0]):
			output[diag_i, diag_i] = output[diag_i, diag_i] + 1
		output_sum = np.sum(output)
		col_sum = np.mean(output, axis = 0)
		for i in range(output.shape[0]):
			if(col_sum[i] > 0):
				output[i, :] = output[i, :] / col_sum[i]
				output[:, i] = output[:, i] / col_sum[i]
		output = output / np.sum(output) * output_sum
		#np.save(path + hic_name_i + 'replicate_vc_whole.npy', output)
		np.save(path + hic_name_i + 'vc_whole_diagplus1.npy', output)
		#index_log = open(path + hic_name_i + str(int(resolution)) + '_npy_index.txt', 'w')
		#for i in range(len(list_chr)):
		#	for j in range(chr_size[i]-1):
		#		index_log.write(list_chr[i] + '\t' + str(int(j * resolution)) + '\t' + str(int((j+1) * resolution)) + '\n')
		#	j = j + 1
		#	index_log.write(list_chr[i] + '\t' + str(int(j * resolution)) + '\t' + str(int(chr_length[i])) + '\n')
		#index_log.close()


