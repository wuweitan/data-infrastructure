import numpy as np
import sys
import difflib

#***********************************************************************************************************

def remove(string,char):
    '''
    Remove the character in a string.
    '''
    #string_char = [i for i in string.split(char) if i != '']
    #return string_char[0]
    string_char = string.split(char)
    return ''.join(string_char)

def seq_extraction(pdb,absolute_position=True):
    AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I', 'HSE':'H',
               'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    with open(pdb,'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ''
    flag = True
    for line in lines:
        if line[0:4] == 'ATOM':
            if absolute_position:
                try:
                    if line[21] == ' ':
                        line = line[1:]

                    if line[17:20] in AA_dict.keys():
                        AA = AA_dict[line[17:20]]
                    else:
                        AA = 'X'
                    chain = line[21]

                    index = int(remove(line[22:26],' '))
                    index_all = str(index) + line[26]
 
                    if flag:
                        seq += AA
                        start = index_all
                        flag = False
                        missing = []
                        if index_all[-1] == ' ':
                            inser_num = 0
                        else:
                            inser_num = 1 
                    elif chain != chain_pre:
                        seqs.append({'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
                        seq = AA
                        start = index_all
                        missing = []
                        if index_all[-1] == ' ':
                            inser_num = 0
                        else:
                            inser_num = 1
                    elif index_all != index_all_pre:
                        gap = 1
                        while(index > int(index_all_pre[:-1]) + gap):
                            seq += 'x'
                            missing.append(int(index_all_pre[:-1]) + gap)
                            gap += 1
                        seq += AA
                        if index_all[-1] != ' ':
                            inser_num += 1
                    
                except:
                    print('PDB read error!')
                    print('################################################')
                    print(pdb)
                    print(line)
                    print('################################################')

            else:
                line = [char for char in line.strip('\n').split(' ') if char != '']
                if line[3] in AA_dict.keys():
                    AA = AA_dict[line[3]]
                else:
                    AA = 'X'
                chain = line[4]

                index_all = line[5]
                if index_all[-1] in '1234567890':
                    index = int(index_all)
                    index_all += ' '
                else:
                    index = int(index_all[:-1])

                if flag:
                    seq += AA
                    start = index_all
                    flag = False
                    missing = []
                    if index_all[-1] == ' ':
                        inser_num = 0
                    else:
                        inser_num = 1
                elif chain != chain_pre:
                    seqs.append({'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
                    seq = AA
                    start = index_all
                    missing = []
                    if index_all[-1] == ' ':
                        inser_num = 0
                    else:
                        inser_num = 1
                elif index_all != index_all_pre:
                    gap = 1
                    while(index > int(index_all_pre[:-1]) + gap):
                        seq += 'x'
                        missing.append(int(index_all_pre[:-1]) + gap)
                        gap += 1
                    seq += AA
                    if index_all[-1] != ' ':
                        inser_num += 1

            chain_pre = chain
            index_all_pre =  index_all

        elif not flag and line[0:3] == 'TER':
            seqs.append({'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
            seq = ''
            flag = True

    if seq != '':
        seqs.append({'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})

    return seqs


def chain_dict(pdb,absolute_position=True):
    result = {}
    seqs = seq_extraction(pdb,absolute_position)
    ordered_keys = []
    for s in seqs:
        chain = s['chain']
        ordered_keys.append(chain)
        if not chain in result.keys():
            result[chain] = {'seq': s['seq'], 'length': s['length'], 'resi_range':s['resi_range'], 
                             'miss_resi': s['miss_resi'], 'insertion_num':s['insertion_num']}
        else:
            index = 2
            chain_new = chain + str(index)
            while (chain_new in result.keys()):
                index += 1
                chain_new = chain + str(index)
            result[chain_new] = {'seq': s['seq'], 'length': s['length'], 'resi_range':s['resi_range'], 
                                 'miss_resi': s['miss_resi'], 'insertion_num':s['insertion_num']}
    return result, ordered_keys
    
def seq_similarity(seq_1,seq_2):
    return difflib.SequenceMatcher(None,seq_1,seq_2).quick_ratio()

#######################################################################################################

if __name__ == "__main__":
    
    for seq in seq_extraction(sys.argv[1]):
        for k in seq.keys():
            print('%s:'%k, seq[k])
        print()
