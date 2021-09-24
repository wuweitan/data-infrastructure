##################################################################################
# Download the original struture data and extract the sequence data according to 
# the SCOPe sequence file.
# Input: the SCOPe seq file
#        structure path
#        sequence path
##################################################################################

import sys
import download_helper
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str, default='../Datasets/CATH/train_domain.txt', help='path of the input data')
parser.add_argument('--stru_path', type=str, default='../Datasets/CATH/sequences/', help='path of the pdb')
parser.add_argument('--seq_path', type=str, default='../Datasets/CATH/sequences/', help='path of the sequences')

args = parser.parse_args()

input_file = args.input_file
STRU_PATH = args.stru_path
SEQ_PATH = args.seq_path

if not STRU_PATH.endswith('/'):
    STRU_PATH += '/'
if not SEQ_PATH.endswith('/'):
    SEQ_PATH += '/'

########################## Read the list file ####################################

stru_num = 0
seq_num = 0

with open(input_file,'r') as in_file:
    lines = in_file.readlines()

for i,line in enumerate(lines):
    line = line.strip('\n')
    pdb_id = line[:4].upper()
    chain = line[4]
    if '/' in line:
        resi = line.split('/')[-1]
    else:
        resi = '-'.join(line.split('-')[-2:])
           
    name = '%s-%s%s'%(pdb_id, chain, resi)
    info = '(chain %s and resi %s)'%(chain, resi)
    
    pdb_file_name = STRU_PATH + name + '.pdb'
    seq_file_name = SEQ_PATH + name + '.fasta'

    dl_result = download_helper.pdb_download_with_info(pdb_id, info, pdb_file_name)
    if dl_result == 0:
        stru_num += 1

    feedback = download_helper.seq_download_with_chain(pdb_id, chain, seq_file_name)
    if feedback == 0:
        seq_num += 1


print('%d pdbs have been downloaded'%stru_num)
print('%d sequences have been downloaded'%seq_num)
