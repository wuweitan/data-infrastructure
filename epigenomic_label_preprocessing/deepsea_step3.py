import numpy as np

path = '/scratch/user/wuwei_tan/noncoding_project/deepsea_peak_data/'

origin_bed_list = open(path + 'wholelist_chipoptimal_replaced_chiponly.txt').readlines()
avail_files = []
for i in range(len(origin_bed_list)):
	avail_files.append(origin_bed_list[i].split('\n')[0])

chip_boolean = np.load('boolean_' + avail_files[0] + '.npy')

for file_index in np.arange(1, len(avail_files)):
	input = np.load('boolean_' + avail_files[file_index] + '.npy')
	chip_boolean[np.where(input == True )] = True
np.save('chip_boolean', np.array(chip_boolean, dtype = 'bool'))

