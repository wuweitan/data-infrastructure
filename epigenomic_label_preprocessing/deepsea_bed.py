import numpy as np

chr_index = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']

chr_length = np.array([249250621, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 243199373, 63025520, 48129895, 51304566, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 155270560, 59373566])

with open('whole.bed', 'w+') as f:
	for i in range(len(chr_length)):
		chr_length[i] = (chr_length[i] // 200)
		for index in range(chr_length[i]):
			f.write(str(chr_index[i] + '\t' + str(index*200) + '\t' + str(index*200+200) + '\n'))
		


