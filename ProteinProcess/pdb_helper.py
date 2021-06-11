import numpy as np
import sys
import difflib

from auxiliary_funtions import remove

#************************************* general pdb process *************************************************

def read_pdb(pdb_file):
    '''
    Extract the residue information inside a pdb file.
    '''
    protein_dict = {}
    index_dict = {} # The indexes of the residues. Two level: the first is for the number, the second is for the letter.
                    # e.g. '100', '100A', '100B','101','102'...then the dictionary is {...,100:[' ','A','B'],101:[' '],...}
    with open(pdb_file,'r') as p_file:
        lines = p_file.readlines()
        for line in lines:
            if line[0:4] == 'ATOM':
                ### residue-wise info ###
                if line[16] == ' ' or line[16] == 'A' or line[16] == '1':
                    atom = remove(line[12:16],' ')
                else:
                    atom = remove(line[12:16],' ') + '_' + line[16]
                resi = line[17:20]
                chain = line[21]
                index_all = remove(line[22:27],' ')
                index = int(remove(line[22:26],' '))
                insertion_code = line[26]
                ### atom-wise info ###
                x = float(remove(line[30:38],' '))
                y = float(remove(line[38:46],' '))
                z = float(remove(line[46:54],' '))
                occupancy = float(remove(line[54:60],' '))
                temp_factor = float(remove(line[60:66],' '))
        ############ Judge whether a new chain begins. ########################
                if not chain in protein_dict.keys():
                    protein_dict[chain] = {}
                    index_dict[chain] = {}
        ############ Save the sequence infomation. ######################## 
                if not index_all in protein_dict[chain].keys():
                    protein_dict[chain][index_all] = {'index':index,'insertion_code':insertion_code,'resi':resi}
                    if not index in index_dict[chain]:
                        index_dict[chain][index] = [insertion_code]
                    else:
                        index_dict[chain][index].append(insertion_code)
                elif resi != protein_dict[chain][index_all]['resi']:
                    print('PDB read error! The residue kind of resi %s is not consistent for %s!'%(index_all,pdb_file))
                    return 0
                if not atom in protein_dict[chain][index_all].keys():
                    protein_dict[chain][index_all][atom] = [x,y,z,occupancy,temp_factor]
    return protein_dict, index_dict


def pdb_truncate(pdb_file,chain,start,end,relative_start=False,fix_length=False):
    '''
    Select a certain range of the pdb file.    
    '''
    output_line = ''
    with open(pdb_file,'r') as pdb_f:
        lines = pdb_f.readlines()
        if len(lines) < 10:
            print(pdb_file)
        index_start = start
        index_end = end
        idx_start_flag = True
        resi_num = 0
        idx_pre = None
        length = end - start + 1

        for line in lines:
            if line[0:4] == 'ATOM':
                index = int(line[22:26])
                if chain == line[21]:
                    if relative_start and idx_start_flag:
                        index_start = index + start - 1
                        index_end = index_start + (end - start)
                        idx_start_flag = False

                    if fix_length and (resi_num >= length):
                        break
                    elif (not fix_length) and (index > index_end):
                        break
                    elif index >= index_start:
                        output_line += line
                        if index != idx_pre:
                            resi_num += 1
                        idx_pre = index
    return output_line


def unify_chain(temp_dict, model_dict, input_file, output_file):
    if len(temp_dict.keys()) != len(temp_dict.keys()):
        print('Chains amounts do not match! Cannot be unified!')
        return 1
    select_chain = []
    map_dict = {}
    for chain in model_dict.keys():
        if chain in temp_dict.keys():
            map_dict[chain] = chain
        else:
            best_score = 0
            for chain_2 in temp_dict.keys():
               if not (chain_2 in model_dict.keys() or chain_2 in select_chain):
                   score = seq_similarity(model_dict[chain]['seq'],temp_dict[chain_2]['seq'])
                   if score > best_score:
                       best_score = score
                       best_chain = chain_2
            map_dict[chain] = best_chain
            select_chain.append(best_chain)
    with open(input_file,'r') as in_f, open(output_file,'w') as out_f:
        lines = in_f.readlines()
        for line in lines:
            if len(line) > 4 and line[0:4] == 'ATOM':
                chain = line[21]
                line_new = line[:21] + map_dict[chain] + line[22:]
                out_f.write(line_new)
            elif len(line) > 21 and line[0:3] == 'TER' and line[21] == chain:
                line_new = line[:21] + map_dict[chain] + line[22:]
                out_f.write(line_new)
            else:
                out_f.write(line)
    return 0


def assembly_extract(pdb):

    return None

def pdb_indexing(pdb, version = 'Kabat'):
    return None

#************************************* sequence process *************************************************

def seq_extraction(pdb,absolute_position=True):
    '''
    Extract the sequence from the pdb files.
    Output a list of chain informations. (The same chain index may refer to different chains in different models.)
    '''
    AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I', 'HSE':'H',
               'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    with open(pdb,'r') as f:
        lines = f.readlines()
    seqs = []
    seq = ''
    flag = True
    model = None
    for line in lines:
        if line[0:5] == 'MODEL':
            model = line.strip('\n').split(' ')[-1]
        elif line[0:4] == 'ATOM':
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
                        seqs.append({'model':model, 'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
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
                    seqs.append({'model':model, 'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
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
            seqs.append({'model':model, 'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})
            seq = ''
            flag = True

    if seq != '':
        seqs.append({'model':model, 'chain': chain_pre, 'seq': seq, 'length': len(seq), 'resi_range':[start, index_all_pre], 'miss_resi':missing, 'insertion_num':inser_num})

    return seqs


def seq_dict(pdb,absolute_position=True):
    result = {}
    seqs = seq_extraction(pdb,absolute_position)
    ordered_keys = []
    for s in seqs:
        model = s['model']
        chain = s['chain']
        ordered_keys.append(chain)
        if not model in result.keys():
            result[model] = {}
        if not chain in result[model].keys():
            result[model][chain] = {'seq': s['seq'], 'length': s['length'], 'resi_range':s['resi_range'],
                                    'miss_resi': s['miss_resi'], 'insertion_num':s['insertion_num']}
            else:
                index = 2
                chain_new = chain + str(index)
                while (chain_new in result.keys()):
                    index += 1
                    chain_new = chain + str(index)
                result[model][chain_new] = {'seq': s['seq'], 'length': s['length'], 'resi_range':s['resi_range'],
                                            'miss_resi': s['miss_resi'], 'insertion_num':s['insertion_num']}
    return result, ordered_keys


def seq_similarity(seq_1,seq_2):
    '''
    Fastly compare two sequences.
    '''
    return difflib.SequenceMatcher(None,seq_1,seq_2).quick_ratio()

#************************************* PPI process ******************************************************

def binding_site_filter(protein_dict, chain_info, threshold = 5):
    binding_dict = {'heavy_chain':{}, 'light_chain':{}, 'antigen':{}}
    for chain_1 in chain_info['antigen']:
        binding_dict['antigen'][chain_1] = [] 
        A_info = protein_dict[chain_1]
        ### heavy chain
        for chain_2 in chain_info['heavy_chain']:
            binding_dict['heavy_chain'][chain_2] = []
            H_info = protein_dict[chain_2]
            ### residue-wise
            for idx_1 in A_info.keys():
                for idx_2 in H_info.keys():
                    contact_flag = False
                    ### atom-wise 
                    for atom_1 in A_info[idx_1].keys():
                        if atom_1[0] != 'H' and not atom_1 in ['index', 'insertion_code', 'resi']:
                            for atom_2 in H_info[idx_2].keys():
                                if atom_2[0] != 'H' and not atom_2 in ['index', 'insertion_code', 'resi']:
                                    d = np.linalg.norm(np.array(A_info[idx_1][atom_1][:3]) - np.array(H_info[idx_2][atom_2][:3]))
                                    if d < threshold:
                                        contact_flag = True
                                        break
                        if contact_flag:
                            break
                    if contact_flag:
                        if not idx_1 in binding_dict['antigen'][chain_1]:
                            binding_dict['antigen'][chain_1].append(idx_1)
                        if not idx_2 in binding_dict['heavy_chain'][chain_2]:
                            binding_dict['heavy_chain'][chain_2].append(idx_2)
        ### light chain
        for chain_2 in chain_info['light_chain']:
            binding_dict['light_chain'][chain_2] = []
            L_info = protein_dict[chain_2]
            ### residue-wise
            for idx_1 in A_info.keys():
                for idx_2 in L_info.keys():
                    contact_flag = False
                    ### atom-wise 
                    for atom_1 in A_info[idx_1].keys():
                        if atom_1[0] != 'H' and not atom_1 in ['index', 'insertion_code', 'resi']:
                            for atom_2 in L_info[idx_2].keys():
                                if atom_2[0] != 'H' and not atom_2 in ['index', 'insertion_code', 'resi']:
                                    d = np.linalg.norm(np.array(A_info[idx_1][atom_1][:3]) - np.array(L_info[idx_2][atom_2][:3]))
                                    if d < threshold:
                                        contact_flag = True
                                        break
                        if contact_flag:
                            break
                    if contact_flag:
                        if not idx_1 in binding_dict['antigen'][chain_1]: 
                            binding_dict['antigen'][chain_1].append(idx_1)
                        if not idx_2 in binding_dict['light_chain'][chain_2]:
                            binding_dict['light_chain'][chain_2].append(idx_2)
    return binding_dict

def binding_sele_dict(protein_dict, binding_dict, index_dict):
    sele_dict = {}
    for mole in ['heavy_chain', 'light_chain', 'antigen']:
        for chain in binding_dict[mole].keys():
            sele_dict[chain] = {}

            index_all = []
            for idx in sorted(index_dict[chain].keys()):
                for code in index_dict[chain][idx]:
                    if code == ' ':
                        index_all.append(str(idx))
                    else:
                        index_all.append(str(idx) + code)
            # search start and end
            index_idx_list = [index_all.index(i) for i in binding_dict[mole][chain]]
            idx_min = min(index_idx_list)
            idx_max = max(index_idx_list)
            for i in range(idx_min, idx_max + 1):
                idx_temp = index_all[i]
                sele_dict[chain][idx_temp] = protein_dict[chain][idx_temp]
    return sele_dict

#######################################################################################################

if __name__ == "__main__":
    
    for seq in seq_extraction(sys.argv[1]):
        for k in seq.keys():
            print('%s:'%k, seq[k])
        print()
