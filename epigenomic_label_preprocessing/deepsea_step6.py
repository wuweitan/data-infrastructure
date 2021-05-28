import numpy as np

path = '/scratch/user/wuwei_tan/noncoding_project/deepsea_peak_data/'

#origin_bed_list = open(path + 'list_all.txt').readlines()
origin_bed_list = open(path + 'wholelist_chipoptimal_replaced.txt').readlines()

avail_files = []
for i in range(len(origin_bed_list)):
	avail_files.append(origin_bed_list[i].split('\n')[0])

input_temp = np.load('filtered_' + avail_files[0] + '.npy')

output = np.array(np.zeros([len(input_temp), len(avail_files)]), dtype = 'bool')

for file_index in range(len(avail_files)):
	input = np.load('filtered_' + avail_files[file_index] + '.npy')
	output[:, file_index] = input	

np.save('filtered_whole', np.array(output, dtype = 'bool'))

