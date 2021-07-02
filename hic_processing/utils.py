import numpy as np
import os

"""
This will download the chromatin structure data, .hic, process the interaction frequency.
This code require the juicer tools.
This is based on hg19, which can be easily changed to hg38 or other reference version.
"""

def download_hic(path, hic_name):
    """
    This function download the chromatin structure data, .hic file.
    'path' is the folder to save the downloaded experiment data.
    'hic_name' is the ENCODE project name for the wanted hic experiment.
    """
    wget.download('https://www.encodeproject.org/files/' + hic_name + '/@@download/' + hic_name_i + '.hic', out = path)

def extract_if_from_hic(path, juicer_path, hic_name, resolution, normalize_method):
    """
    By calling juicer tools, extract the interaction frequency from the .hic file. multiple interaction frequency resolution, normalizion methods can be selected.
    You may get more information about juicer usage by checking the github https://github.com/aidenlab/juicer/wiki/Juicer-Tools-Quick-Start
    'path' is the working directory, used to save and process the .hic file.
    'juicer_path' is the path to the juicer tools.
    'hic_name' is the ENCODE project experiment name for the analyzed .hic data.
    'resolution' is the chromatin structure resolution. It can be determined by the .hic file. Usually choose 100000 or 1000000.
    'normalized_method' is the normalization method we want to use, VC, NONE, etc.
    """
    output_path = path + 'if_matrix_' + resolution + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for chr_i in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
        chr_i_name = chr_i
            for chr_j in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
                chr_j_name = chr_j
                os.system('java -jar ' + juicer_path + 'dump observed ' + normalize_method + ' ' + path + hic_name + '.hic ' + chr_i_name + ' ' + chr_j_name + ' BP ' + resolution + ' ' + output_path + hic_name + '_' + chr_i_name + '_' + chr_j_name + '.txt')

def if_txt_to_npy(path, hic_name, resolution):
    """
    This function convert the juicer tool output text file into numpy array. The output is a chromatin-chromatin pairwise result.
    'path' is the processed chromatin structure and the output will be saved.
    'hic_name' is the encode project experiment name for the given .hic.
    'resolution' is the selected chromatin structure resolution.
    """
    list_chr = list(np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))))
    for chr_i in range(len(list_chr)):
        for chr_j in range(len(list_chr)):
            if(chr_i <= chr_j):
                output = np.zeros((chr_size[chr_i], chr_size[chr_j]))
                if(os.path.isfile(path + hic_name + list_chr[chr_i] + '_' + list_chr[chr_j] + '.txt')):
                    input = open(path + hic_name + '_' + list_chr[chr_i] + '_' + list_chr[chr_j] + '.txt').readlines()
                    coordinate_1 = []
                    coordinate_2 = []
                    counts = []
                    for i in range(len(input)):
                        if_float = float(input[i].split('\t')[2].split('\n')[0])
                        if(not np.isnan(if_float)):
                            coordinate_1.append(int(input[i].split('\t')[0]) // resolution)
                            coordinate_2.append(int(input[i].split('\t')[1]) // resolution)
                            counts.append(if_float)
                    coordinate_1 = np.asarray(coordinate_1)
                    coordinate_2 = np.asarray(coordinate_2)
                    counts = np.asarray(counts)
                    if(len(counts) == 0):
                        print(list_chr[chr_i] + '_' + list_chr[chr_j])
                    elif(np.min(counts) == 0):
                        print(list_chr[chr_i] + '_' + list_chr[chr_j])
                    for i in range(len(counts)):
                        output[int(coordinate_1[i]), int(coordinate_2[i])] = counts[i]
                        if(chr_i == chr_j):
                            output[int(coordinate_2[i]), int(coordinate_1[i])] = counts[i]
                np.save(path + hic_name + list_chr[chr_i] + '_' + list_chr[chr_j] + '.npy', output)

def merge_if_npy(path, hic_name, resolution):
    """
    This function merge the chromatin-chromatin pairwise result into a whole genome numpy array.
    'path' is the processed chromatin structure and the output will be saved.
    'hic_name' is the encode project experiment name for the given .hic.
    'resolution' is the selected chromatin structure resolution.
    """
    chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
    chr_length_hg38 = np.array([248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468, 156040895, 57227415, 16569])
    list_chr = list(np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))))
    chr_size = np.asarray(chr_length // resolution + 1, dtype = int)
    chr_start = np.zeros(len(chr_length))
    for i in np.arange(1, len(chr_length)):
    chr_start[i] = np.sum(chr_size[0:i])
    total_length = int(np.sum(chr_size))
    output = np.zeros((total_length, total_length))
    for chr_i in range(len(list_chr)):
        for chr_j in  range(len(list_chr)):
            if(chr_i <= chr_j):
                input = np.load(path + hic_name_i + 'novc_' + list_chr[chr_i] + '_' + list_chr[chr_j] + '.npy')
            else:
                input = np.load(path + hic_name_i + 'novc_' + list_chr[chr_j] + '_' + list_chr[chr_i] + '.npy').T
            output[int(chr_start[chr_i]):int(chr_start[chr_i]+chr_size[chr_i]), int(chr_start[chr_j]):int(chr_start[chr_j]+chr_size[chr_j])] = input
    np.save(path + hic_name_i + 'novc_whole.npy', output)