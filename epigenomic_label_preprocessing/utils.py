import numpy as np

"""
Downloading, preprocessing the genome sequence and epigenomic events paired data.
This is based on hg19, which can be easily changed to hg38 or other reference version.
"""

chr_index = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']
chr_length = np.array([249250621, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 243199373, 63025520, 48129895, 51304566, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 155270560, 59373566])

def prepare_whole_genome_bed_file(chr_index, chr_length, region_length, output_path):
	"""
	Generate a bed files contains all candidate regions.
	'chr_index, chr_length' are the informations for the given genome reference, e.g. hg19.
	'region_length' is the length of the each candidate region. Here we use 200bp.
	'output_path' is where the generated bed file will be saved at.
	"""
	with open(output_path + 'whole.bed', 'w+') as f:
		for i in range(len(chr_length)):
			chr_length[i] = (chr_length[i] // region_length)
			for index in range(chr_length[i]):
				f.write(str(chr_index[i] + '\t' + str(index*region_length) + '\t' + str(index * region_length + region_length) + '\n'))

def convert_peak_to_binary(input_path, output_path, chr_length, bed_file_list_name, region_length):
	"""
	Read the a peaking calling result, .narrowPeak, convert the peak regions into the binary label for each candidate regions. 
	For a given candidate region, if more than 50% of is covered by a peak region, then we assign a positive label to that candidate region. Otherwise, we assign a negative label.
	'input_path' is where the peak callling files, .narrowPeak, are saved.
	'output_path' is where the intermediate files will be saved.
	'chr_length' is the length of each chromosome for the given genome reference.
	'bed_file_list_name' is the list of the peak calling files we are analyzing.
	'region_length' is the length of the each candidate region.
	"""
	chr_start_coordinates = [0]
	for i in range(len(chr_length)):
		chr_start_coordinates.append(chr_start_coordinates[i] + (chr_length[i] // 200))
	bed_file_list = open(input_path + bed_file_list_name).readlines()
	avail_files = []
	for i in range(len(origin_bed_list)):
		avail_files.append(bed_file_list[i].split('\n')[0])
	region_length_half = region_length // 2
	for file_index in range(len(avail_files)):
		output = np.zeros([chr_start_coordinates[len(chr_start_coordinates) - 1]])
		input = open(input_path + avail_files[file_index]).readlines()
		input_chr_index = []
		input_coordinates = []
		for i in range(len(input)):
			bed_input_row = input[i].split('\t')
			if bed_input_row[0] in chr_label:
				left = int(bed_input_row[1])
				right = int(bed_input_row[2])
				diff = (right / region_length) - (right // region_length)
				if(i < (len(input) - 1)):
					if((input[i].split('\t')[0] == input[i+1].split('\t')[0]) & (diff > 0) & (diff < 0.5) & (int(input[i+1].split('\t')[1]) - right < 100)):
						right = right + region_length_half
				if( (right - left) >= region_length_half ):
					if( ((left // region_length) != (left / region_length)) & ((left // region_length_half) == (left / region_length_half)) ):
						left = left - region_length_half			
					if( ((right // region_length) != (right / region_length)) & ((right // region_length_half) == (right / region_length_half)) ):
						right = right - region_length_half
					input_coordinates.append( round( left / region_length ) )
					input_coordinates.append( round( right / region_length ) )
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
		np.save(output_path + 'boolean_' + avail_files[file_index], np.array(output, dtype = 'bool'))

def merge_binary_label(input_path, output_path, chip_bed_file_list_name):
	"""
	After all peak calling results are converted into binary labels. 
	This function will read all binary labels and get all selected regions(the regions with at least one positive labels, note as M regions)
	'input_path' is where the preprocessed files, the binary label for all candidate regions, are saved.
	'output_path' is where the intermediate files will be saved.
	'chip_bed_file_list_name' is the list of all ChIP-seq, which we used to determine the interested regions from all candidate regions.
	"""
	bed_list = open(input_path + chip_bed_file_list_name).readlines()
	avail_files = []
	for i in range(len(bed_list)):
		avail_files.append(bed_list[i].split('\n')[0])
	chip_boolean = np.load(input_path + 'boolean_' + avail_files[0] + '.npy')
	for file_index in np.arange(1, len(avail_files)):
		input = np.load('boolean_' + avail_files[file_index] + '.npy')
		chip_boolean[np.where(input == True )] = True
	np.save(output_path + 'chip_boolean.npy', np.array(chip_boolean, dtype = 'bool'))

def write_selected_region_to_bed(input_path, output_path, output_bed_name):
	"""
	After the interested regions are selected(from merge_binary_label), 
	this function will write a .bed file, showing the detailed genome coordinates for all selected regions.
	'input_path' is where the 'chip_boolean.npy' file is saved, which is an array of boolean variable showing which candidate regions are selected.
	'output_path' is where the output bed files, showing all interested regions, will be saved.
	'output_bed_name' is the name of the output bed file.
	"""
	bed = open(input_path + 'whole.bed').readlines()
	boolean = np.load(input_path + 'chip_boolean.npy')
	output = []
	for i in range(len(boolean)):
		if(boolean[i] == True):
			output.append(bed[i])
	output_file = open(output_path + output_bed_name, 'w+')
	for i in range(len(output)):
		output_file.writelines(output[i])

def select_regions_on_all_peak_fils(input_path, output_path, selected_region, bed_file_list):
	"""
	Based on the TF features, the interested regions have been selected.
	This function will filter the candidate regions for other epigenomic events, only the TF selected regions are left.
	'input_path' is where the preprocessed binary label for all candidate regions for each epigenomic events are saved.
	'output_path' is where the filtered, the binary label only for interested regions will be saved.
	'selected_region' is the array of boolean variable indicating which regions have been selected as 'interested regions'.
	'bed_file_list' is the list of all the epigenomic events we are analyzing.
	"""
	origin_bed_list = open(input_path + bed_file_list).readlines()
	avail_files = []
	for i in range(len(origin_bed_list)):
		avail_files.append(origin_bed_list[i].split('\n')[0])
	selected_region_boolean = np.load(input_path + selected_region)
	for file_index in range(len(avail_files)):
		input = np.load(input_path + 'boolean_' + avail_files[file_index] + '.npy')
		output = np.delete(input, np.where(boolean == False), 0)	
		np.save(output_path + 'filtered_' + avail_files[file_index], np.array(output, dtype = 'bool'))

def perpare_final_binary_label_matrix(input_path, output_path, bed_file_list, output_name):
	"""
	This function will prepare a N*M matrix, where N is the number of the interested epigenomic events. M is the selected regions.
	1 in this matrix means there is a peak(covering more than 50%). 0 means less than 50% of the given region is covered by the given epigenomic event peak.
	'input_path' is where the processed binary label are saved.
	'output_path' is where the final output, the N*M matrix will be saved.
	'bed_file_list' is the list of all the epigenomic events we are analyzing.
	'output_name' is the name of the final output, that N*M matrix.
	"""
	origin_bed_list = open(input_path + bed_file_list).readlines()
	avail_files = []
	for i in range(len(origin_bed_list)):
		avail_files.append(origin_bed_list[i].split('\n')[0])
	input_temp = np.load(input_path + 'filtered_' + avail_files[0] + '.npy')
	output = np.array(np.zeros([len(input_temp), len(avail_files)]), dtype = 'bool')
	for file_index in range(len(avail_files)):
		input = np.load('filtered_' + avail_files[file_index] + '.npy')
		output[:, file_index] = input	
	np.save(output_path + output_name, np.array(output, dtype = 'bool'))

