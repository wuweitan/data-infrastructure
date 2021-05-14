import numpy as np
import os

hic_list = eval(open('hic_list.txt', 'r').read())

#for resolution in ['50000', '100000', '500000', '1000000']:
for resolution in ['1000000']:
	os.system('mkdir if_intermediate_' + resolution)
	for key_index_i in range(len(list(hic_list.keys()))):
		hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
		for hic_i in range(len(hic_list_temp)):
			hic_name_i = hic_list_temp[hic_i]
			for chr_i in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
				chr_i_name = chr_i
				for chr_j in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
					chr_j_name = chr_j
					#os.system('java -jar juicer_tools_1.22.01.jar dump observed VC hic/' + hic_name_i + '.hic ' + chr_i_name + ' ' + chr_j_name + ' BP 1000000 ' + 'if_intermediate_' + resolution + '/' + hic_name_i + '_chr' + chr_i_name + '_chr' + chr_j_name + '.txt')
					os.system('java -jar juicer_tools_1.22.01.jar dump observed None hic/' + hic_name_i + '.hic ' + chr_i_name + ' ' + chr_j_name + ' BP 1000000 ' + 'if_intermediate_' + resolution + '/' + hic_name_i + '_novc_chr' + chr_i_name + '_chr' + chr_j_name + '.txt')


