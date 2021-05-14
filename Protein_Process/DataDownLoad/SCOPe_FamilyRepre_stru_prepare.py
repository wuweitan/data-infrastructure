#######################################################################################
# Check the representative proteins and the opriginal proteins. Download the missing ones.
# Input: the original SCOPe file
#        the representative SCOPe file
#        structure path
#        downloading log path
#######################################################################################

import sys
import download_helper

SCOPe_FILE = sys.argv[1]
REPRE_FILE = sys.argv[2]
STRU_PATH = sys.argv[3]
LOG = sys.argv[4]

if not STRU_PATH.endswith('/'):
    STRU_PATH += '/'

########################## Read the list file ####################################

miss_stru_num = 0
contradict_num = 0
same_num = 0

origi_dict = {}

with open(SCOPe_FILE,'r') as scope_file:
    lines = scope_file.readlines()
    for line in lines:
        if line[0] == '>':
            prot = line.split(' ')[0][1:]
            origi_dict[prot] = line

with open(REPRE_FILE,'r') as repre_file, open(LOG,'w') as log:
    lines = repre_file.readlines()
    for line in lines:
        if line[0] == '>':
            prot = line.split(' ')[0][1:]
            if not prot in origi_dict.keys():
                title = line[1:].strip('\n')
                pdb_id,fold,info = download_helper.read_SCOPe_title(title)
                pdb_file_name = STRU_PATH + line[1:8] + '.pdb'
                dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
                log.write(prot + '\nnot in the origin file.\n\n')
                miss_stru_num += 1
            elif origi_dict[prot] != line:
                title = line[1:].strip('\n')
                pdb_id,fold,info = download_helper.read_SCOPe_title(title)
                pdb_file_name = STRU_PATH + line[1:8] + '.pdb'
                dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
                log.write(prot + '\nContradictary Information.\n')
                log.write('Original file: ' + origi_dict[prot])
                log.write('Represent file: ' + line + '\n')
                contradict_num += 1
            else:
                same_num += 1

print('%d structures have been downloaded. %d not in the orginal file and %d with contradictary title.'%(miss_stru_num + contradict_num, miss_stru_num, contradict_num))
print('%d structures are in the original file.'%same_num)
