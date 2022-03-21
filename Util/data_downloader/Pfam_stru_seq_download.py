##################################################################################
# Download the original struture and sequence data from Protein Data Bank
# according to the non-redundant protein structure files.
# Input: the pdb index list file
#        download objective ('pdb' or 'seq')
#        download path
##################################################################################

import os
import sys
import download_helper

LIST_FILE = sys.argv[1]
DOWN_OB = sys.argv[2]
DOWN_PATH = sys.argv[3]
fail_file = sys.argv[4]

if not DOWN_PATH.endswith('/'):
    DOWN_PATH += '/'

if os.path.exists(fail_file):
    os.system('rm %s'%fail_file)

########################## Read the list file ####################################

PDB_list = []

with open(LIST_FILE,'r') as p_list_file:
    lines = p_list_file.readlines()

    for line in lines:
        line = line.strip('\n').split('\t')
        pdb_id = line[0].split('_')[0]
        info = line[1]
        PDB_list.append((pdb_id, info))

print('%d PDB files in the list.'%len(PDB_list))

######################### Download the data #######################################

num = 0
fail_num = 0

if DOWN_OB == 'pdb':
    for index in PDB_list:
        print(index)
        PDB_index = index[0]
        info = index[1]
        
        try:
            chain = info.split(' ')[1]
            resi = info.split(' ')[-1].strip(')')
            file_name = DOWN_PATH + PDB_index + '_' + chain + '_' + resi + '.pdb'
            print(file_name)
            feedback = download_helper.pdb_download_with_info(PDB_index,info,file_name)
            if feedback == 0:
                num += 1
            else:
                fail_num += 1
        except Exception as e:
            fail_num += 1
            with open(fail_file, 'a') as f:
                f.write(index + '\n' + e + '\n\n')
              
    print('%d pdb files have been downloaded. %d jobs failed.'%(num, fail_num))

elif DOWN_OB == 'seq':
    for index in PDB_list:
        print(index)
        PDB_index = index[0]
        info = index[1]

        try:
            chain = info.split(' ')[1]
            file_name = DOWN_PATH + PDB_index + '_' + chain + '.fasta'
            print(file_name)
            feedback = download_helper.seq_download_with_chain(PDB_index,chain,file_name)
            if feedback == 0:
                num += 1
            else:
                fail_num += 1
        except Exception as e:
            fail_num += 1
            with open(fail_file, 'a') as f:
                f.write(index + '\n' + e + '\n\n')
    print('%d sequence files have been downloaded. %d jobs failed.'%(num, fail_num))

else:
    print('Error! Cannot download %s!'%DOWN_OB)
 



