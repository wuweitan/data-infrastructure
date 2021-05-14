##################################################################################
# Download the original struture data and extract the sequence data according to 
# the SCOPe sequence file.
# Input: the SCOPe seq file
#        structure path
#        sequence path
##################################################################################

import sys
import download_helper

SCOPe_FILE = sys.argv[1]
STRU_PATH = sys.argv[2]
SEQ_PATH = sys.argv[3]

if not STRU_PATH.endswith('/'):
    STRU_PATH += '/'
if not SEQ_PATH.endswith('/'):
    SEQ_PATH += '/'

########################## Read the list file ####################################

stru_num = 0
seq_num = 0

with open(SCOPe_FILE,'r') as scope_file:
    lines = scope_file.readlines()
    length = len(lines)
    for i in range(length):
        line = lines[i]
        if line[0] == '>':

            seq_num += 1
            
            if i != 0:
                if pdb_id != 1:
                    dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
                    download_helper.fasta_write(title,seq,seq_file_name,record_type = 'w',line_length = 60)
                    if dl_result == 0:
                        stru_num += 1
            title = line[1:].strip('\n')
            pdb_id,fold,info = download_helper.read_SCOPe_title(title)            

            pdb_file_name = STRU_PATH + line[1:8] + '.pdb'
            seq_file_name = SEQ_PATH + line[1:8] + '.fasta'
            seq = ''
        else:
            seq += line.strip('\n') 
    if pdb_id != 1:
        dl_result = download_helper.pdb_download_with_info(pdb_id,info,pdb_file_name)
        download_helper.fasta_write(title,seq,seq_file_name,record_type = 'w',line_length = 60)
        if dl_result == 0:
            stru_num += 1

print('%d sequences in the list.'%seq_num)
print('%d structures have been downloaded'%stru_num)

