import numpy as np

bed = open('whole.bed').readlines()

boolean = np.load('chip_boolean.npy')

output = []
for i in range(len(boolean)):
	if(boolean[i] == True):
		output.append(bed[i])

output_file = open('filtered.bed', 'w+')
for i in range(len(output)):
	output_file.writelines(output[i])
