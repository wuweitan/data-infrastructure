##################################################################################
# Download the original struture and sequence data from Protein Data Bank
# according to the non-redundant protein structure files.
# Input: the pdb index list file
#        download objective ('pdb' or 'seq')
#        download path
##################################################################################

import sys
import download_helper

LIST_FILE = sys.argv[1]
DOWN_OB = sys.argv[2]
DOWN_PATH = sys.argv[3]

if not DOWN_PATH.endswith('/'):
    DOWN_PATH += '/'

########################## Read the list file ####################################

PDB_list = []

with open(LIST_FILE,'r') as p_list_file:
    lines = p_list_file.readlines()

    for line in lines:
        line = [p for p in line.strip('\n').split(' ') if p != '']
        for p in line:
            PDB_list.append(p.split('_'))

print '%d PDB files in the list.'%len(PDB_list)

######################### Download the data #######################################

num = 0

if DOWN_OB == 'pdb':
    for index in PDB_list:
        print index
        PDB_index = index[0]
        info = '(chain %s)'%index[1]
        file_name = DOWN_PATH + PDB_index + '_' + index[1] + '.pdb'
        feedback = download_helper.pdb_download_with_info(PDB_index,info,file_name)
        if feedback == 0:
            num += 1
    print '%d pdb files have been downloaded.'%num

elif DOWN_OB == 'seq':
    for index in PDB_list:
        print index
        PDB_index = index[0]
        chain = index[1]
        file_name = DOWN_PATH + PDB_index + '_' + index[1] + '.fasta'
        feedback = download_helper.seq_download_with_chain(PDB_index,chain,file_name)
        if feedback == 0:
            num += 1
    print '%d sequence files have been downloaded.'%num

else:
    print 'Error! Cannot download %s!'%DOWN_OB
 



