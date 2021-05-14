import numpy as np


hic_list = eval(open('hic_list.txt', 'r').read())

for key_index_i in range(len(list(hic_list.keys()))):
	hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
	for hic_i in range(len(hic_list_temp)):
		hic_name_i = hic_list_temp[hic_i] + '_'
		for resolution_i in [1e5]:
			resolution = str(int(resolution_i))
			if_matrix_input = np.load('if_matrix_' + resolution + '/' + hic_name_i + 'replicate_vc_whole.npy')
			if_matrix_square = np.matmul(if_matrix_input, if_matrix_input)
			np.save('if_matrix_square_' + resolution + '/' + hic_name_i + 'replicate_vc_whole.npy', if_matrix_square)


