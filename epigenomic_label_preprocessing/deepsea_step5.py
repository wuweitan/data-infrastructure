import numpy as np

path = '/scratch/user/wuwei_tan/noncoding_project/deepsea_peak_data/'

#origin_bed_list = open(path + 'list_all.txt').readlines()
#origin_bed_list = open(path + 'DeepSEA_origin_list.txt').readlines()
origin_bed_list = open(path + 'wholelist_chipoptimal_replaced.txt').readlines()
avail_files = []
for i in range(len(origin_bed_list)):
	avail_files.append(origin_bed_list[i].split('\n')[0])

boolean = np.load('chip_boolean.npy')
#boolean = np.load('chip_boolean_origin.npy')

for file_index in range(len(avail_files)):
	input = np.load('boolean_' + avail_files[file_index] + '.npy')
	output = np.delete(input, np.where(boolean == False), 0)	
	np.save('filtered_' + avail_files[file_index], np.array(output, dtype = 'bool'))

