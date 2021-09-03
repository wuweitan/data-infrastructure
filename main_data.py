from getGlyCbPosBynccaCoord import *
from utils import *
import numpy as np
from Bio.PDB import *
from os import path

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import atomium
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='test')
parser.add_argument('--startingIdx', type=int, default=0)

args = parser.parse_args()


pth = args.path
st = args.startingIdx
print(args)

with open(pth+'/'+pth+'_wChain_n', 'r') as f:
    ct = 0
    for line in f:
        if ct < st:
            ct += 1
            continue
        ct += 1
        instance_u_r, instance_u_l, instance_b_r, instance_b_l = [], [], [], []
        
        '''
        AAAAAAAAAA       FOR extra
        '''

        '''

        # step 1
        ll = line.strip().split()
        # bound pdb id
        instance_b_r.append(ll[0].split(':')[0][0:4])
        instance_b_l.append(ll[0].split(':')[0][0:4])
        b_r = ll[0].split(':')[0][5:7]
        b_l = ll[0].split(':')[1]
        instance_b_r.append(b_r)
        instance_b_l.append(b_l)
        
        # unbound pdb id
        instance_u_r.append(ll[1].split('_')[0])
        instance_u_l.append(ll[2].split('_')[0])
        u_r = ll[1].split('_')[1]
        u_l = ll[2].split('_')[1]
        instance_u_r.append(u_r)
       	instance_u_l.append(u_l)

        for i1 in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            instance_b_r[1] = b_r[i1[0]]
            instance_b_l[1] = b_l[i1[1]]
            instance_u_r[1] = u_r[i1[0]]
            instance_u_l[1] = u_l[i1[1]]
        '''

        '''
        BBBBBBBBBB
        1. see which unbound protein has two different chains
        2. extract the coresponding bound and unbound chain ids
        '''
        '''
        # step 1
        ll = line.strip().split()
        
        flag = -1
        
        if ll[1][-1] != ')' and len(ll[1].split('_')[1]) == 2:
            print(ll[1].split('_')[0], ll[1].split('_')[1])
            pdb_id = ll[1].split('_')[0]
#             print(pdb_id)
            if atomium.fetch(pdb_id).model.chain(ll[1].split('_')[1][0]).sequence != atomium.fetch(pdb_id).model.chain(ll[1].split('_')[1][1]).sequence:
                flag = 1
                instance_u_r.append(pdb_id)
                instance_u_r.append(ll[1].split('_')[1][1])
                # bound
                instance_b_r.append(ll[0].split(':')[0][0:4])
                instance_b_l.append(ll[0].split(':')[0][0:4])
                instance_b_r.append(ll[0].split(':')[0][6])
                instance_b_l.append(ll[0].split(':')[1][0])
                # unbound
                if ll[2][-1] == ')':
                    instance_u_l.append(ll[2][0:4]+'('+ll[2].split('(')[1])
                    instance_u_l.append(ll[2].split('_')[1][0])
                else:
                    instance_u_l.append(ll[2][0:4])
                    instance_u_l.append(ll[2].split('_')[1][0])
        
        if ll[2][-1] != ')' and len(ll[2].split('_')[1]) == 2:
            print(ll[2].split('_')[0], ll[2].split('_')[1])
            pdb_id = ll[2].split('_')[0]
            if atomium.fetch(pdb_id).model.chain(ll[2].split('_')[1][0]).sequence != atomium.fetch(pdb_id).model.chain(ll[2].split('_')[1][1]).sequence:
                flag = 1
                instance_u_l.append(pdb_id)
                instance_u_l.append(ll[2].split('_')[1][1])
                # bound
                instance_b_r.append(ll[0].split(':')[0][0:4])
                instance_b_l.append(ll[0].split(':')[0][0:4])
                instance_b_r.append(ll[0].split(':')[0][5])
                instance_b_l.append(ll[0].split(':')[1][1])
                # unbound
                if ll[1][-1] == ')':
                    instance_u_r.append(ll[1][0:4]+'('+ll[1].split('(')[1])
                    instance_u_r.append(ll[1].split('_')[1][0])
                else:
                    instance_u_r.append(ll[1][0:4])
                    instance_u_r.append(ll[1].split('_')[1][0])
        
        if flag == -1:
            continue
        '''

        '''
        CCCCCCCCCCCC
        ORIGINAL
        '''

        
        ll = line.strip().split()
        # bound
        instance_b_r.append(ll[0].split(':')[0][0:4])
        instance_b_l.append(ll[0].split(':')[0][0:4])
        instance_b_r.append(ll[0].split(':')[0][5])
        instance_b_l.append(ll[0].split(':')[1][0])
        # unbound
        # there might be an exact model num
        if ll[1][-1] == ')':
            instance_u_r.append(ll[1][0:4]+'('+ll[1].split('(')[1])
            instance_u_r.append(ll[1].split('_')[1][0])
        else:
            instance_u_r.append(ll[1][0:4])
            instance_u_r.append(ll[1].split('_')[1][0])
        
        if ll[2][-1] == ')':
            instance_u_l.append(ll[2][0:4]+'('+ll[2].split('(')[1])
            instance_u_l.append(ll[2].split('_')[1][0])
        else:
            instance_u_l.append(ll[2][0:4])
            instance_u_l.append(ll[2].split('_')[1][0])
        
        print(instance_u_r, instance_u_l, instance_b_r, instance_b_l)
        #if path.exists(pth+'/'+'distances/dist_cb_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+ '_' + instance_u_l[0][0:4] +  '_' + instance_u_l[1] +'.npy'):
        #    continue

        seq_u_r, seq_u_l, dict_ABS_u_to_b_r, dict_ABS_u_to_b_l, \
        list_residx_mapping_u_r, list_residx_mapping_u_l, list_residx_mapping_b_r, list_residx_mapping_b_l = data_prepare(instance_u_r, instance_u_l, instance_b_r, instance_b_l)
        
        # Inter (bound)
        dist_ca = -1.0 * np.ones((len(seq_u_r), len(seq_u_l)))
        dist_cb = -1.0 * np.ones((len(seq_u_r), len(seq_u_l)))
        ct_closest = np.zeros((len(seq_u_r), len(seq_u_l)))

        cut_off = 5

        pdb_b = atomium.fetch(instance_b_r[0][0:4])
        mdl = None
        if instance_b_r[0][-1] == ')':
            mdl = pdb_b.models[int(instance_b_r[0][5:-1])-1]
        else:
            mdl = pdb_b.model

        for i in dict_ABS_u_to_b_r:
            idx_r = list_residx_mapping_b_r[dict_ABS_u_to_b_r[i]]
            res1 = mdl.residue(instance_b_r[1]+'.'+idx_r)

            if not res1: 
                continue
            for j in dict_ABS_u_to_b_l:
                idx_l = list_residx_mapping_b_l[dict_ABS_u_to_b_l[j]]
                res2 = mdl.residue(instance_b_l[1]+'.'+idx_l)

                if not res2:
                    continue


                # 1. Construct two lists of atom coordinate NUMPY ARRAYS for two residues respectively (l1, l2)
                # 2. Construct two dictionaries of {desired atom name : coordinate NUMPY ARRAY} for two residues respectively (dict_atom1, dict_atom2)
                # With C betas imputed (if imputable)

                ll1 = []
                ll2 = []

                for k in res1.atoms():
                    # {atom_object}.location returns a tuple
                    ll1.append(np.array(k.location))

                for k in res2.atoms():
                    ll2.append(np.array(k.location))


                min_dis = float('inf')

                for k in ll1:
                    for l in ll2:

                        diff = k - l

                        d = np.sqrt(np.sum(diff * diff))
        #                                 print(d)
                        if d < min_dis:
                            min_dis = d

                if min_dis <= cut_off:
                    ct_closest[i][j] = 1

                CA_coords1 = []
                CA_coords2 = []

                if res1.atoms(name = 'CA'):
                    for k in res1.atoms(name = 'CA'):
                        CA_coords1.append(k.location)

                if res2.atoms(name = 'CA'):
                    for k in res2.atoms(name = 'CA'):
                        CA_coords2.append(k.location)

                CB_coords1 = []
                CB_coords2 = []

                if res1.atoms(name = 'CB'):
                    for k in res1.atoms(name = 'CB'):
                        CB_coords1.append(k.location)

                if res2.atoms(name = 'CB'):
                    for k in res2.atoms(name = 'CB'):
                        CB_coords2.append(k.location)

                if (not res1.atoms(name = 'CB')) and res1.atoms(name = 'N') and res1.atoms(name = 'C') and res1.atoms(name = 'CA'):
                    # res1.atoms(name = 'CB') return a set

                    list_atomCoords = []

                    for k in res1.atoms(name = 'N'):
                        list_atomCoords.append(k.location)
                    for k in res1.atoms(name = 'C'):
                        list_atomCoords.append(k.location)
                    for k in res1.atoms(name = 'CA'):
                        list_atomCoords.append(k.location)

                    CB_coords1.append(getGlyCbPosBynccaCoord(list_atomCoords))

                if (not res2.atoms(name = 'CB')) and res2.atoms(name = 'N') and res2.atoms(name = 'C') and res2.atoms(name = 'CA'):
                    # res2.atoms(name = 'CB') return a set

                    list_atomCoords = []

                    for k in res2.atoms(name = 'N'):
                        list_atomCoords.append(k.location)
                    for k in res2.atoms(name = 'C'):
                        list_atomCoords.append(k.location)
                    for k in res2.atoms(name = 'CA'):
                        list_atomCoords.append(k.location)

                    CB_coords2.append(getGlyCbPosBynccaCoord(list_atomCoords))

                if len(CB_coords1) != 0 and len(CB_coords2) != 0:
                    diff = np.array(CB_coords1[0]) - np.array(CB_coords2[0])
                    dist_cb[i][j] = np.sqrt(np.sum(diff * diff))

                if len(CA_coords1) != 0 and len(CA_coords2) != 0:
                    diff = np.array(CA_coords1[0]) - np.array(CA_coords2[0])
                    dist_ca[i][j] = np.sqrt(np.sum(diff * diff))
        with open(pth+'/'+'ctmaps/ctmap_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+ '_' + instance_u_l[0][0:4] +  '_' + instance_u_l[1]+ \
'_' + instance_b_r[0] + '_'+ instance_b_r[1]+ '_' + instance_b_l[1] + '.npy', 'wb') as f:
            np.save(f, ct_closest)


        with open(pth+'/'+'distances/dist_ca_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+ '_' + instance_u_l[0][0:4] +  '_' + instance_u_l[1] + \
'_' + instance_b_r[0] + '_'+ instance_b_r[1]+ '_' + instance_b_l[1] +'.npy', 'wb') as f:
            np.save(f, dist_ca)

        with open(pth+'/'+'distances/dist_cb_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+ '_' + instance_u_l[0][0:4] +  '_' + instance_u_l[1] + \
'_' + instance_b_r[0] + '_'+ instance_b_r[1]+ '_' + instance_b_l[1] + '.npy', 'wb') as f:
            np.save(f, dist_cb)
        
        if not path.exists('ctmaps_u/ctmap_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+'.npy'):
            calc(seq_u_r, instance_u_r, list_residx_mapping_u_r)
        with open('seq/'+instance_u_r[0][0:4] +  '_' + instance_u_r[1], 'w') as f:
            f.write(seq_u_r)
        
        if not path.exists('ctmaps_u/ctmap_' + instance_u_l[0][0:4] +  '_' + instance_u_l[1]+'.npy'):
            calc(seq_u_l, instance_u_l, list_residx_mapping_u_l)
        with open('seq/' + instance_u_l[0][0:4] +  '_' + instance_u_l[1], 'w') as f:
            f.write(seq_u_l)
        #break #################
