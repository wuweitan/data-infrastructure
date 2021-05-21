import numpy as np
import os

def download_hic(path, hic_name):
    wget.download('https://www.encodeproject.org/files/' + hic_name + '/@@download/' + hic_name_i + '.hic', out = path)

def extract_if_from_hic(path, juicer_path, hic_name, resolution, normalize_method):
    output_path = path + 'if_matrix_' + resolution + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for chr_i in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
        chr_i_name = chr_i
            for chr_j in np.concatenate((np.arange(22) + 1, np.array(['X', 'Y', 'M']))):
                chr_j_name = chr_j
                os.system('java -jar ' + juicer_path + 'dump observed ' + normalize_method + ' ' + path + hic_name + '.hic ' + chr_i_name + ' ' + chr_j_name + ' BP ' + resolution + ' ' + output_path + hic_name + '_' + chr_i_name + '_' + chr_j_name + '.txt')

def if_txt_to_npy(path, hic_name, resolution):
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
    chr_length = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,  63025520, 48129895, 51304566, 155270560, 59373566, 16571])
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
