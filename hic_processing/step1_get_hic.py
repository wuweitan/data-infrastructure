import wget

hic_list = eval(open('hic_list.txt', 'r').read())

for key_index_i in range(len(list(hic_list.keys()))):
	hic_list_temp = hic_list[list(hic_list.keys())[key_index_i]]
	for hic_i in range(len(hic_list_temp)):
		hic_name_i = hic_list_temp[hic_i]
		wget.download('https://www.encodeproject.org/files/' + hic_name_i + '/@@download/' + hic_name_i + '.hic', out = 'hic/')


