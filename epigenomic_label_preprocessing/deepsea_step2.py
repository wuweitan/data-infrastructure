import numpy as np
import math

chr_length = np.array([249250621, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 243199373, 63025520, 48129895, 51304566, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 155270560, 59373566])

chr_start_coordinates = [0]

chr_label = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']
chr_label = np.asarray(chr_label)

for i in range(len(chr_length)):
	chr_start_coordinates.append(chr_start_coordinates[i] + (chr_length[i] // 200))

path = '/scratch/user/wuwei_tan/noncoding_project/deepsea_peak_data/'

#origin_bed_list = open(path + 'list_all_ChIP_origin.txt').readlines()

origin_bed_list = open(path + 'wholelist_chipoptimal_replaced_chiponly.txt').readlines()

avail_files = []
for i in range(len(origin_bed_list)):
	avail_files.append(origin_bed_list[i].split('\n')[0])

for file_index in range(len(avail_files)):
	output = np.zeros([chr_start_coordinates[len(chr_start_coordinates) - 1]])
	input = open(path + 'deepsea_chip_replaced_data/' + avail_files[file_index]).readlines()
	#input = open(path + 'deepsea_origin_data/' + avail_files[file_index]).readlines()
	input_chr_index = []
	input_coordinates = []
	for i in range(len(input)):
		bed_input_row = input[i].split('\t')
		if bed_input_row[0] in chr_label:
			input_coordinates.append( math.floor(int(bed_input_row[1]) / 200.0) )
			input_coordinates.append( math.ceil(int(bed_input_row[2]) / 200.0) )
			input_chr_index.append( bed_input_row[0] )
	for i in range(len(input_coordinates)//2-1):
		if(input_coordinates[2*i]!=input_coordinates[2*i+1]):
			temp_coordinates = input_coordinates[2*i]
			while(temp_coordinates < input_coordinates[2*i+1]):
				output[chr_start_coordinates[int(np.asarray(np.where(chr_label == input_chr_index[i])))] + temp_coordinates] = 1
				temp_coordinates = temp_coordinates + 1
	i = i + 1
	temp_coordinates = input_coordinates[2*i]
	while(temp_coordinates < input_coordinates[2*i+1]):
		output[chr_start_coordinates[int(np.asarray(np.where(chr_label == input_chr_index[i])))] + temp_coordinates] = 1
		temp_coordinates = temp_coordinates + 1
	np.save('boolean_' + avail_files[file_index], np.array(output, dtype = 'bool'))


