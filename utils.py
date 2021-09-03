from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from getGlyCbPosBynccaCoord import *
import Bio.PDB
import numpy as np
import re,os,requests,sys,random
import json
from json import JSONEncoder
import matplotlib.pyplot as plt
import atomium


def data_prepare(instance_u_r, instance_u_l, instance_b_r, instance_b_l):
    '''
    Input:
    ## The chain_id's below are all author-defined chain ids ##
    1. [pdb_id_u_r, chain_id]   
    2. [pdb_id_u_l, chain_id]
    3. [pdb_id_b_r, chain_id]
    4. [pdb_id_b_l, chain_id]



    Procedures:
    1. use 'pdb1 = atomium.fetch('1FS1')' and 'pdb1.model.chain('B').sequence' to get the unbound/bound seqs (4 in total)
    2. use align_u_to_b(seq_u, seq_b) to find out the alignment between seq_u and seq_b (ABS indexing and starting from 0)
    3. use asym_mapping(best_pdb) to find out asym_mapping_dict: {auth_asym_id:asym_id} for 4 chains and get the
       asym_id for each chain
    4. use residx_mapping(pdb_id, asym_chain_id) to find out the 4 list_residx_mapping's.



    Output:
    1. seq_u_r
    2. seq_u_l
    3. dict_ABS_u_to_b_r (starting from index 0)
    4. dict_ABS_u_to_b_l (starting from index 0)
    ## The entries in the following lists are all strings!!!!!!!! ##
    5. list_residx_mapping_u_r 
    6. list_residx_mapping_u_l
    7. list_residx_mapping_b_r
    8. list_residx_mapping_b_l
    '''
    
    # step 1
    pdb_u_r = atomium.fetch(instance_u_r[0][0:4])
    pdb_u_l = atomium.fetch(instance_u_l[0][0:4])
    pdb_b_r = atomium.fetch(instance_b_r[0][0:4])
    pdb_b_l = atomium.fetch(instance_b_l[0][0:4])
    
    seq_u_r = con_atomium_mdl(pdb_u_r, instance_u_r[0]).chain(instance_u_r[1]).sequence
    seq_u_l = con_atomium_mdl(pdb_u_l, instance_u_l[0]).chain(instance_u_l[1]).sequence
    seq_b_r = con_atomium_mdl(pdb_b_r, instance_b_r[0]).chain(instance_b_r[1]).sequence
    seq_b_l = con_atomium_mdl(pdb_b_l, instance_b_l[0]).chain(instance_b_l[1]).sequence
    
    # step 2
    dict_ABS_u_to_b_r = align_u_to_b(seq_u_r, seq_b_r)
    dict_ABS_u_to_b_l = align_u_to_b(seq_u_l, seq_b_l)
    
    # step 3
    asym_mapping_b = asym_mapping(instance_b_r[0][0:4])
    asym_mapping_u_r = asym_mapping(instance_u_r[0][0:4])
    asym_mapping_u_l = asym_mapping(instance_u_l[0][0:4])
    
    # step 4
    list_residx_mapping_u_r = residx_mapping(instance_u_r[0][0:4], asym_mapping_u_r[instance_u_r[1]])
    list_residx_mapping_u_l = residx_mapping(instance_u_l[0][0:4], asym_mapping_u_l[instance_u_l[1]])
    list_residx_mapping_b_r = residx_mapping(instance_b_r[0][0:4], asym_mapping_b[instance_b_r[1]])
    list_residx_mapping_b_l = residx_mapping(instance_b_l[0][0:4], asym_mapping_b[instance_b_l[1]])
    
    return seq_u_r, seq_u_l, dict_ABS_u_to_b_r, dict_ABS_u_to_b_l, \
        list_residx_mapping_u_r, list_residx_mapping_u_l, list_residx_mapping_b_r, list_residx_mapping_b_l
    
    
def con_atomium_mdl(pdb_o, instance_idx0):
    mdl = None
    if instance_idx0[-1] == ')':
        mdl = pdb_o.models[int(instance_idx0[5:-1])-1]
    else:
        mdl = pdb_o.model
    return mdl




def align_u_to_b(seq_u, seq_b):
    '''
    Input:
    1. seq_u
    2. seq_b
    
    Output:
    1. a dictionary for align_u_to_b
    '''
    
    matrix = matlist.blosum62
    alignment = pairwise2.align.globaldd(seq_b,seq_u, matrix,-11,-1,-11,-1)[0]
    idx_t = 0
    idx_n = 0
    map_u_to_b = {}
    for i in range(alignment[-1]):
        if alignment[0][i] != '-':
            if alignment[0][i] == alignment[1][i]:
                map_u_to_b[idx_n] = idx_t
            idx_t += 1
        if alignment[1][i] != '-':
            idx_n += 1
    return map_u_to_b



def asym_mapping(best_pdb):
    """
    generate mapping between natural chain id and author defined chain id (Might be some missing chains)

    *Input:
    - best_pdb id
    *Output:
    - asym_mapping_dict: {auth_asym_id:asym_id}
    """
    asym_mapping_dict = {}
    rcsbBase_url = "https://data.rcsb.org/graphql"
    # query entity ids
    entityIds_query = '''
        {{entries(entry_ids: ["{}"]) {{
            rcsb_entry_container_identifiers {{
              polymer_entity_ids}}
          }}
        }}
    '''.format(best_pdb)
    res_entityIds = requests.post(rcsbBase_url,json={'query':entityIds_query})
    if res_entityIds.status_code != 200:
        return None
    else:
        try:
#             print(res_entityIds.json())
            entityIds_list = res_entityIds.json()['data']['entries'][0]['rcsb_entry_container_identifiers']['polymer_entity_ids']
            for ent_id in entityIds_list:
                # query asym_ids, auth_asym_ids
                asymIds_query = '''
                  {{polymer_entities(entity_ids:["{}_{}"])
                      {{rcsb_polymer_entity_container_identifiers {{
                          asym_ids
                          auth_asym_ids}}
                        entity_poly {{
                          pdbx_strand_id}}
                      }}
                  }}
                '''.format(best_pdb,ent_id)
                res_asymIds = requests.post(rcsbBase_url,json={'query':asymIds_query})
                if res_asymIds.status_code != 200:
                    return None
                else:
                    rec_asymIds_json = res_asymIds.json()
                    asymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['asym_ids']
                    #authAsymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] # auth_asym_ids may not have the right order of ids
                    pdbx_strandId_list = re.split(r',',rec_asymIds_json['data']['polymer_entities'][0]['entity_poly']['pdbx_strand_id'])
                    assert len(asymIds_list) == len(pdbx_strandId_list), "asym_ids length not same with auth_asym_ids"
                    # upate mapping dict
#                     print(rec_asymIds_json)
                    for asym_i in range(len(asymIds_list)):
                        asym_mapping_dict[pdbx_strandId_list[asym_i]] = asymIds_list[asym_i]
            return asym_mapping_dict
        except:
            return None
        



def residx_mapping(pdb_id, chain_id):
    '''
    Get a list of author-defined INDICES for {pdb_id}.{chain_id}

    Input:
    1. pdb_id (eg. '1FS1')
    2. chain_id (eg. 'B')  (asym_id, not author defined)!!!!!!

    Output:
    1. a list of author-defined INDICES
    '''
    rcsbBase_url = "https://data.rcsb.org/graphql"
    rcsb1d_url = "https://1d-coordinates.rcsb.org/graphql"

    pdb_instance = '{}.{}'.format(pdb_id,chain_id)

    query_idxMap = '''
      {{polymer_entity_instances(instance_ids: ["{pdb_ins}"]) {{
        rcsb_id
        rcsb_polymer_entity_instance_container_identifiers {{
          auth_asym_id
          entity_id
          auth_to_entity_poly_seq_mapping}}
        }}
      }}
      '''.format(pdb_ins=pdb_instance)

    res_idxMap = requests.post(rcsbBase_url,json={'query':query_idxMap})

    res_idxMap_json = res_idxMap.json()
    auth_pdbSeq_mapping=res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_to_entity_poly_seq_mapping']
    auth_asym_id=res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_asym_id']

    return auth_pdbSeq_mapping
#     auth_asym_id

# Intra (unbound)
def calc(seq_u_r, instance_u_r, list_residx_mapping_u_r):
    dist_ca = -1.0 * np.ones((len(seq_u_r), len(seq_u_r)))
    dist_cb = -1.0 * np.ones((len(seq_u_r), len(seq_u_r)))
    ct_closest = np.zeros((len(seq_u_r), len(seq_u_r)))

    cut_off = 5

    pdb_b = atomium.fetch(instance_u_r[0][0:4])
    
    mdl = None
    if instance_u_r[0][-1] == ')':
        mdl = pdb_b.models[int(instance_u_r[0][5:-1])-1]
    else:
        mdl = pdb_b.model

    for i in range(len(list_residx_mapping_u_r)):
        idx_r = list_residx_mapping_u_r[i]
        res1 = mdl.residue(instance_u_r[1]+'.'+idx_r)

        if not res1: 
            continue
        for j in range(len(list_residx_mapping_u_r)):
            idx_l = list_residx_mapping_u_r[j]
            res2 = mdl.residue(instance_u_r[1]+'.'+idx_l)

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
    with open('ctmaps_u/ctmap_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+'.npy', 'wb') as f:
        np.save(f, ct_closest)


    with open('distances_u/dist_ca_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+'.npy', 'wb') as f:
        np.save(f, dist_ca)

    with open('distances_u/dist_cb_' + instance_u_r[0][0:4] +  '_' + instance_u_r[1]+'.npy', 'wb') as f:
        np.save(f, dist_cb)


# For unbound structures (two chains)
def calc_two_chains(seq_u_r1, instance_u_r1, list_residx_mapping_u_r1, seq_u_r2, instance_u_r2, list_residx_mapping_u_r2):
    dist_ca = -1.0 * np.ones((len(seq_u_r1), len(seq_u_r2)))
    dist_cb = -1.0 * np.ones((len(seq_u_r1), len(seq_u_r2)))
    ct_closest = np.zeros((len(seq_u_r1), len(seq_u_r2)))

    cut_off = 5

    pdb_b = atomium.fetch(instance_u_r1[0][0:4])
    
    mdl = None
    if instance_u_r1[0][-1] == ')':
        mdl = pdb_b.models[int(instance_u_r1[0][5:-1])-1]
    else:
        mdl = pdb_b.model

    for i in range(len(list_residx_mapping_u_r1)):
        idx_r = list_residx_mapping_u_r1[i]
        res1 = mdl.residue(instance_u_r1[1]+'.'+idx_r)

        if not res1: 
            continue
        for j in range(len(list_residx_mapping_u_r2)):
            idx_l = list_residx_mapping_u_r2[j]
            res2 = mdl.residue(instance_u_r2[1]+'.'+idx_l)

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
    with open('ctmaps_u/ctmap_' + instance_u_r1[0][0:4] +  '_' + instance_u_r1[1] + '_' + instance_u_r2[1]+'.npy', 'wb') as f:
        np.save(f, ct_closest)


    with open('distances_u/dist_ca_' + instance_u_r1[0][0:4] +  '_' + instance_u_r1[1] + '_' + instance_u_r2[1]+'.npy', 'wb') as f:
        np.save(f, dist_ca)

    with open('distances_u/dist_cb_' + instance_u_r1[0][0:4] +  '_' + instance_u_r1[1] + '_' + instance_u_r2[1]+'.npy', 'wb') as f:
        np.save(f, dist_cb)
