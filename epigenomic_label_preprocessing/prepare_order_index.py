import numpy as np

origin = np.asarray(open('allTFs.bed').readlines())
whole = np.asarray(open('whole.bed').readlines())

index = np.zeros(len(origin))

origin_index = 0

for i in range(len(whole)):
	if(origin_index < len(origin)):
		if(whole[i] == origin[origin_index]):
			index[origin_index] = i
			origin_index = origin_index + 1

output = np.zeros(len(whole))

for i in range(len(index)):
	output[int(index[i])] = 1

np.save('chip_boolean_origin.npy', output)

