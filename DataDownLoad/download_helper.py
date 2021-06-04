######################################################################################
# Functions that help to download structures and sequences from the website.
######################################################################################

import os
import numpy as np	
import requests

def link_exist(file_name):
    '''
    Check whether the downloaded file is from a valid link.
    '''
    with open(file_name,'r') as f:
        lines = f.readlines()
        result = True
        for line in lines:
            line = line.strip('\n')
            if line.startswith('<!DOCTYPE html PUBLIC'):
                result = False
                break
            elif line.endswith('PDBID|CHAIN|SEQUENCE') or line.startswith('HEADER'):
                break
    return result

def pdb_download(PDB_index,file_name):
    '''
    Download pdb file of the input index.
    '''
    link = 'https://files.rcsb.org/download/%s.pdb'%PDB_index.upper()
    os.system("curl -o %s %s"%(file_name,link))
    if link_exist(file_name):
        return True
    else:
        print('The PDB %s is not found!'%PDB_index)
        os.system('rm ' + file_name)
        return False

def pdb_download_with_info(PDB_index,info,file_name):
    '''
    Download pdb file of the input index and info (chain or resi).
    '''
    temp_index = 0
    while(os.path.exists('temp_%d'%temp_index)):
        temp_index += 1
    valid = pdb_download(PDB_index,'temp_%d'%temp_index)
    if valid:
        pymol_index = 0
        while(os.path.exists('load_pdb_%d.pml'%pymol_index)):
            pymol_index += 1
        pymol=[]
        pymol.append('load temp_%d'%temp_index)
        pymol.append('select ' + info)
        pymol.append('save ' + file_name + ', sele')
        np.savetxt('load_pdb_%d.pml'%pymol_index, pymol, fmt='%s')
        os.system('pymol -cq load_pdb_%d.pml'%pymol_index)
        ### remove the temporary files ###
        os.system('rm temp_%d'%temp_index)
        os.system('rm load_pdb_%d.pml'%pymol_index)
        return 0 
    else:
        return 1 

def web_download(url,file_name):
    '''
    Download the materials on a website through the link.
    '''
    seq_req = requests.get(url,stream=True)
    with open(file_name,'w') as f_w:
        for chunk in seq_req.iter_content(chunk_size=1024):
            f_w.write(chunk)
    return 0

def seq_download(PDB_index,file_name):
    '''
    Download fasta file of the input index.
    '''
    link = 'https://www.rcsb.org/pdb/download/downloadFastaFiles.do?structureIdList=%s&compressionType=uncompressed'%PDB_index.upper()
    web_download(link,file_name)
    if link_exist(file_name):
        return True
    else:
        print('The PDB %s is not found!'%PDB_index)
        os.system('rm ' + file_name)
        return False

def seq_read(file_name):
    '''
    Return a dictionary that the sequences of different chains belong to different keys.
    '''
    result_dict = {}
    with open(file_name,'r') as f_r:
        lines = f_r.readlines()
        l = len(lines)
        for i in range(l):
            line = lines[i]
            if line[0] == '>':
                if i != 0:
                    if not chain in result_dict.keys():
                        result_dict[chain] = [seq]
                    else:
                        result_dict[chain].append(seq)
                chain = line[6]
                seq = ''
            else:
                seq += line.strip('\n')
        if not chain in result_dict.keys():
            result_dict[chain] = [seq]
        else:
            result_dict[chain].append(seq)
        return result_dict

def fasta_write(title,seq,file_name,record_type = 'a',line_length = 80):
    '''
    Record a sequence as fasta format.
    ''' 
    seq = seq.upper()
    with open(file_name,record_type) as f_w:
        f_w.write('>' + title + '\n')
        while len(seq) > line_length:
            f_w.write(seq[0:line_length] + '\n')
            seq = seq[line_length:]
        f_w.write(seq + '\n')
    return 0

def seq_download_with_chain(PDB_index,chain,file_name):
    '''
    Download pdb file of the input index and info (chain or resi).
    '''
    temp_index = 0
    while(os.path.exists('seq_temp_%d'%temp_index)):
        temp_index += 1
    valid = seq_download(PDB_index,'seq_temp_%d'%temp_index)
    if valid:
        seq_dict = seq_read('seq_temp_%d'%temp_index)
        if not chain in seq_dict:
            print('The chain %s is not found for pdb %s'%(chain,PDB_index))
            os.system('rm seq_temp_%d'%temp_index)
            return 1
        else:
            seqs = seq_dict[chain]
            f_w = open(file_name,'w')
            f_w.close()
            for i in range(len(seqs)):
                fasta_write(PDB_index + '_' + chain + '_' + str(i+1),seqs[i],file_name,record_type = 'a',line_length = 80)
            os.system('rm seq_temp_%d'%temp_index)
            return 0
    else:
        return 1

def read_SCOPe_title(title):
    title = title.strip('>')
    pdb_id = title[1:5].upper()
    fold = '.'.join(title.split(' ')[1].split('.')[0:2])
    chain_resi = title.split(' ')[2]

    if chain_resi[0] == '(' and chain_resi[-1] == ')':
        chain_resi = chain_resi.strip('(').strip(')').split(',')
        info = ''
        for in_s in chain_resi:
            info += '(chain %s'%in_s[0]
            if len(in_s) >= 3:
                info += ' and resi %s'%in_s[2:]
            info += ') or '
        info = info.strip(' or ')
        return pdb_id, fold, info
    else:
        print('Abnormal title!')
        print(title)
        return 1,1,1


