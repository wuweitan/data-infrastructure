##########################################################################################################
########################## Generate Cb-Cb distance/phi/psi/theta angle 2D maps ############################
##########################################################################################################
""" Three Major Stand-alone Functions 

1. preprocessing(pdb_ids, chains_b_r, chains_b_l, chains_u_r, chains_u_l, path_u_r, path_u_l, 
   path_b_r, path_b_l)
   --> read in four pdb files and calculate Cb-Cb distance/phi/psi/theta angle 2D maps

2. xyz_to_c6d_modified(xyz, mask_seq)
   --> convert cartesian coordinates into 2d distance and orientation maps

3. pdb_info_load(pdb_file, chains = None)
   --> read in a pdb file and get parsed structure info

"""
import numpy as np
# from pdb_helper import *
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
# import atomium
import pickle
import torch

matrix = matlist.blosum62



# No 'HSE'
dict_AA_to_atom = {'ALA': ['N', 'CA', 'C', 'O', 'CB'], 'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'], \
'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],\
'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'], 'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],\
'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'], 'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],\
'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'], 'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG'],\
'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], 'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],\
'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'], 'GLY': ['N', 'CA', 'C', 'O'],\
'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'], 'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],\
'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'], 'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],\
'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],\
'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],\
'A': ['N', 'CA', 'C', 'O', 'CB'], 'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'], \
'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],\
'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'], 'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],\
'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'], 'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],\
'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'], 'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],\
'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'], 'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],\
'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'], 'G': ['N', 'CA', 'C', 'O'],\
'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'], 'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],\
'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'], 'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],\
'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],\
'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2']}


# ============================================================
def get_pair_dist(a, b):  #########
    """calculate pair distances between two sets of points
    
    Args:
        a (torch.Tensor): pytorch tensor of shape [batch,nres,3]
            store Cartesian coordinates of a set of atoms
        b (torch.Tensor): pytorch tensor of shape [batch,nres,3]
            store Cartesian coordinates of a set of atoms

    Returns:
        dist: pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c, eps=1e-8):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i]) from Cartesian coordinates of three sets of atoms a,b,c 
    
    Note:
        If the angle does not exist, then we expect the calculation will give a mean value (pi/2 here). 
        (This is the case when we add the epsilon value to gain numerical stability)

    Args:
        a (torch.Tensor): pytorch tensor of shape [batch,nres,3]
            store Cartesian coordinates of a set of atoms
        b (torch.Tensor): pytorch tensor of shape [batch,nres,3]
            store Cartesian coordinates of a set of atoms
        a (torch.Tensor): pytorch tensor of shape [batch,nres,3]
            store Cartesian coordinates of a set of atoms
	    
    Returns:
        ang (torch.Tensor): pytorch tensor of shape [batch,nres]
            stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= (torch.norm(v, dim=-1, keepdim=True)+eps)  ################# +eps
    w /= (torch.norm(w, dim=-1, keepdim=True)+eps)  ################# +eps
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw) # [0, pi]

# ============================================================
def get_dih(a, b, c, d, eps=1e-8):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i]) given Cartesian coordinates of four sets of atoms a,b,c,d
    
    Note:
        If the angle does not exist, then we expect the calculation will give a mean value (0 here). 
        (This is the case when we add the epsilon value to gain numerical stability)

    Args:
        a (torch.Tensor): pytorch tensor of shape [batch,nres,3]
              store Cartesian coordinates of a set of atoms
        b (torch.Tensor): pytorch tensor of shape [batch,nres,3]
              store Cartesian coordinates of a set of atoms
	c (torch.Tensor): pytorch tensor of shape [batch,nres,3]
              store Cartesian coordinates of a set of atoms
	d (torch.Tensor): pytorch tensor of shape [batch,nres,3]
              store Cartesian coordinates of a set of atoms
    Returns:
        dih (torch.Tensor): pytorch tensor of shape [batch,nres]
            stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    # print("a", torch.sum(a))
    # print("b", torch.sum(b))
    # print("c", torch.sum(c))
    # print("b0", torch.sum(b0))
    # print("b1", torch.sum(b1))
    # print("b2", torch.sum(b2))

    # print("b1", b1)
    # print("norm!!!!!!!!!", torch.sum(torch.norm(b1, dim=-1, keepdim=True)==0))
    b1 /= (torch.norm(b1, dim=-1, keepdim=True)+eps)  ################# +eps
    

    # print("b1", torch.sum(torch.isnan(b1)))

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1
    # print("v", torch.sum(torch.isnan(v)))
    # print("w", torch.sum(torch.isnan(w)))

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    # print("x", torch.sum(torch.isnan(x)))
    # print("y", torch.sum(torch.isnan(y)))

    # print("x and y", torch.sum(torch.logical_or(torch.isnan(x), torch.isnan(y))))


    return torch.atan2(y, x) # [-pi, pi]

#### MODIFYING ####
def xyz_to_c6d_modified(xyz, mask_seq):
    """convert cartesian coordinates into 2d distance and orientation maps
    
    Args:
        xyz (torch.Tensor): pytorch tensor of shape [batch,nres,3,3]
            stores Cartesian coordinates of backbone N,Ca,C atoms
        mask_seq (torch.Tensor): pytorch tensor of shape [batch,nres]

    Returns:
        c6d (torch.Tensor): pytorch tensor of shape [batch,nres,nres,4]
            stores stacked dist,omega,theta,phi 2D maps
        mask_pair (torch.Tensor): pytorch tensor of shape [batch, nres, nres]
            stores 2D maps where the distance is below 20 angstroms
    """

    ### 1. There is nan for any unseen coordinates, the distances assigned to such related pairs are very large.
    ###    Also the self-distance is very large.
    ###    Also the ditance is larger than 20 angstroms then fixing them to 999.99 angstroms.
    ### 2. The other features exist only when the ditance is below 20 angstroms.
    ###    Otherwise 0.
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # print("N", torch.sum(N[0][mask_seq[0]]==0))
    # print("N", N[0][mask_seq[0]])
    # print("Ca", Ca[0][mask_seq[0]])
    # print("C", C[0][mask_seq[0]])
    # print("Ca", torch.sum(torch.isnan(Ca[0][mask_seq[0]])))
    # print("C", torch.sum(torch.isnan(C[0][mask_seq[0]])))

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # print("Cb", torch.sum(Cb[0][mask_seq[0]]==0))

    # mask for pair features
    mask_pair = torch.zeros((batch, nres,nres), device=xyz.device)
    for b_idx in range(batch):
        mask_pair[b_idx, mask_seq[b_idx], :] += 1
        mask_pair[b_idx, :, mask_seq[b_idx]] += 1
    mask_pair = (mask_pair > 1)
    
    # print("mask_pair", torch.sum(mask_pair))


    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb) # (B, L, L)
    # print("dist", torch.sum(torch.isnan(dist[mask_pair])))  
    # dist[torch.isnan(dist)] = 999.9 
    c6d[...,0] = dist #+ 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(mask_pair==True) # DMAX = 20

    


    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j]) # torch.full_like(input, fill_value, \*, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
    # print("c6d", torch.sum(torch.isnan(c6d[b,i,j,torch.full_like(b,1)])))
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # torch.tensor([1.0,2.0,5.0])
    # torch.tensor([2.0,3.0,4.0])

    # print("The results when special cases", get_dih(torch.tensor([2.0,3.0,4.0]), torch.tensor([1.0,2.0,5.0]), torch.tensor([1.0,2.0,5.0]), torch.tensor([2.0,3.0,4.0])))
    # print("The results when special cases", get_ang(torch.tensor([2.0,3.0,4.0]), torch.tensor([1.0,2.0,5.0]), torch.tensor([1.0,2.0,5.0])))
    print("c6d", torch.sum(torch.isnan(c6d[:,:,:,3][mask_pair])))
    print("c6d", torch.sum(torch.isnan(c6d[:,:,:,2][mask_pair])))
    print("c6d", torch.sum(torch.isnan(c6d[:,:,:,1][mask_pair])))

    # fix long-range distances
    # c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    # mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    # mask[b,i,j] = 1.0
    return c6d, mask_pair

def conv_to_aatype(seq_str):

    restypes = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
        'S', 'T', 'W', 'Y', 'V'
    ]

    restype_order = {restype: i for i, restype in enumerate(restypes)}

    out = []

    for i in range(len(seq_str)):
        if seq_str[i] in restype_order:
            out.append(restype_order[seq_str[i]])
        else:
            out.append(20)
    
    return out


def pdb_info_load(pdb_file, chains = None):
    """Extract the residue information inside a pdb file. Ignore the missing residues.
    """
    AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I', 'HSE':'H',
               'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

    protein_dict = {}
    index_dict = {} # The indexes of the residues.

    with open(pdb_file,'r') as p_file:
        lines = p_file.readlines()
        for line in lines:
            if line[0:4] == 'ATOM':
                ### residue-wise info ###  TODO: ask Shaowen about this
                if line[16] == ' ' or line[16] == 'A' or line[16] == '1':
                    atom = line[12:16].replace(' ', '')
                else:
                    atom = line[12:16].replace(' ', '') + '_' + line[16]
                resi = line[17:20]
                chain = line[21]
                if chain == ' ':
                    chain = chains[0]
                # print(chain)

                # some residues are not in the AA_dict
                if resi not in AA_dict:
                    continue

                if chains is None or chain in chains:
                    index_all = line[22:27].replace(' ', '')
                    ### atom-wise info ###
                    x = float(line[30:38].replace(' ', ''))
                    y = float(line[38:46].replace(' ', ''))
                    z = float(line[46:54].replace(' ', ''))
        ############ Judge whether a new chain begins. ########################
                    if not chain in protein_dict.keys():
                        protein_dict[chain] = {'coor':{}, 'seq':''}
                        index_dict[chain] = []
        ############ Save the sequence infomation. ######################## 
                    if not index_all in protein_dict[chain]['coor'].keys():
                        protein_dict[chain]['coor'][index_all] = {'resi':resi}
                        index_dict[chain].append(index_all)
                        # if resi == 'ACE':
                        #     print("resi !!!!!", index_all)
                        protein_dict[chain]['seq'] += AA_dict[resi]
                    elif resi != protein_dict[chain]['coor'][index_all]['resi']:
                        print('PDB read error! The residue kind of resi %s is not consistent for %s!'%(index_all,pdb_file))
                        return 0
        ############ atom coordinates. ########################
                    if not atom in protein_dict[chain]['coor'][index_all].keys():
                        protein_dict[chain]['coor'][index_all][atom] = np.array([x,y,z])
    print(pdb_file)
    print('%d chains processed.'%len(index_dict.keys()))
    for c in index_dict.keys():
        print('Chain %s: %d residues'%(c, len(index_dict[c])))


    '''
    protein_dict = {chain : {'coor':{index_all : {'resi': 'ALA', atom_name: np.array(x, y, z)}}, 'seq':''}}
    index_dict = {chain: [index_all]}

    # [index_all] is a list of characters

    E.g.
    protein_dict = {'A': {'coor': {'1': {'resi': 'CYS', 'N': array([39.722,  6.322, -2.689]), 'CA': array([38.916,  6.004, -1.494])}}, 
    'seq': 'CGVPAIQPVLIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTTSDVVVAGEFDQGSSSEKIQKLKIAKVFKNSKYNSLTINNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYNTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGASGVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWVQQTLAAN'}}
    index_dict = {'A': ['1', '2', '3', '4', '5', '6', '7', '8']}

    '''
    return protein_dict, index_dict

def nat_idx_conv_conc_to_inchain(conc_idx, chains, index_dict):
    """Convert some natural residue idx for the concatenated seq to the natural res idx for some chain and find out the chain id.
    
       Args:
           conc_idx (int): the natural index for the residue in the concatenated sequence 
	       with the order of the chains specified in the chains argument
           chains (list(str)): a list of chain names 
           index_dict (dict): a dict object as one of the output items of the pdb_info_loader function
	   
       Returns:
           c (str): chain id for the chain where the residue resides on
           difference (int): natural res idx for the chain c
    """
    cur_len = 0
    for i, c in enumerate(chains):
        if cur_len <= conc_idx and conc_idx < cur_len + len(index_dict[c]):
	    difference = conc_idx - cur_len
            return c, difference
        else:
            cur_len += len(index_dict[c])
    print("Cannot find the residue's position! Please check the input!")


def preprocessing(pdb_ids, chains_b_r, chains_b_l, chains_u_r, chains_u_l, path_u_r, path_u_l, 
	path_b_r, path_b_l):
	
	"""Process the docking data (include the unbound and bound structure) to get a dictionary containing a set of features/masks/labels

	Args:
	    pdb_ids: a list of pdb id triplets in the format of items in DB5. 
		   (dtype: list, eg. ["1AHW_AB:C" "1FGN_LH" "1TFH_A"])

	    chains_b_r (list(str)): a list of chain ids for receptor in the bound state 
	    chains_b_l (list(str)): a list of chain ids for ligand in the bound state 
	    chains_u_r (list(str)): a list of chain ids for receptor in the unbound state 
	    chains_u_l (list(str)): a list of chain ids for ligand in the unbound state 

	    path_u_r (str): path to the pdb files for receptor in the bound state 
	    path_u_l (str): path to the pdb files for ligand in the bound state 
	    path_b_r (str): path to the pdb files for receptor in the unbound state 
	    path_b_l (str): path to the pdb files for ligand in the unbound state 

	Notes:
	    A pickled python dictionary file named with the "labels/{complex_code}.pkl" with the following keys:
	    
	    1. out_dict["complex_code"] (dtype: str)
	    2. out_dict["conc_seq"]["rec"], out_dict["conc_seq"]["lig"] (dtype: str)
	    3. out_dict["conc_bb_coord"]["rec"]["u"], out_dict["conc_bb_coord"]["rec"]["b"], out_dict["conc_bb_coord"]["lig"]["u"], out_dict["conc_bb_coord"]["lig"]["b"] (dtype: np.array (n_res, 3, 3))
	    4. out_dict["mask"] (dtype: np.array (n_res_r+n_res_l))
	    5. out_dict["labels"] (dtype: np.array (n_res_r+n_res_l, n_res_r+n_res_l, 4))
	    6. out_dict["mask_pair"] (dtype: np.array (n_res_r+n_res_l, n_res_r+n_res_l))
	"""

    pdb_b = pdb_ids[0][0:4]

    # TODO:
    # with model number for the multiple generated decoys:
    # preprocessing(pdb_ids, chains_b_r, chains_b_l, chains_u_r, chains_u_l, path_u_r, path_u_l, 
	# path_b_r, path_b_l, model_num=0, cutoff = 5.0):

    # path_u_r = "benchmark5_cleaned/structures/"+pdb_b+"_r_u"+".pdb"
    # path_u_l = "benchmark5_cleaned/structures/"+pdb_b+"_l_u"+".pdb"
    # path_b_r = "benchmark5_cleaned/structures/"+pdb_b+"_r_b"+".pdb"
    # path_b_l = "benchmark5_cleaned/structures/"+pdb_b+"_l_b"+".pdb"

    # path_u_r = path_to_the_scratch_folder+'/PPI/Data/zdock/2c/{}_r_u.pdb.ms'.format(pdb_ids[0][0:4])

    # path_u_l = path_to_the_scratch_folder+'/PPI/Data/zdock/2c/{}_l_u_{}.pdb'.format(pdb_ids[0][0:4], model_num)
    

    # path_b_r = path_to_the_scratch_folder+'/PPI/Data/zdock/benchmark/{}_r_b.pdb'.format(pdb_ids[0][0:4])
    # path_b_l = path_to_the_scratch_folder+'/PPI/Data/zdock/benchmark/{}_l_b.pdb'.format(pdb_ids[0][0:4])

    ### load the info
    target_rec, target_rec_index = pdb_info_load(path_u_r, chains_u_r)
    target_lig, target_lig_index = pdb_info_load(path_u_l, chains_u_l)

    native_rec, native_rec_index = pdb_info_load(path_b_r, chains_b_r)
    native_lig, native_lig_index = pdb_info_load(path_b_l, chains_b_l)


    ############################################### 0 for rec, 1 for lig #########################################################
    # dict seq_conc for u_r and u_l; dict_seq_conc_u={0: "ADB", 1: "VW"}
    dict_seq_conc_u = {}

    # dict bb_coord for b_r, u_r, b_l, u_l; dict_bb_coord = {0:{"b": np.array(...), "u": np.array(...)}, 1:{"b": np.array(...), "u": np.array(...)}}
    dict_bb_coord = {}

    # dict mask for r, l mask = {0: np.array(...), 1: np.array(...)}
    mask = {}

    



    # rec and lig

    # chains_b --> chains_b_r/l
    # chains_u --> chains_u_r/l

    # native --> native_rec
    # target --> target_rec

    # native_index --> native_rec_index
    # target_index --> target_rec_index

    list_chains_b = [chains_b_r, chains_b_l]
    list_chains_u = [chains_u_r, chains_u_l]

    list_native = [native_rec, native_lig]
    list_target = [target_rec, target_lig]

    list_native_index = [native_rec_index, native_lig_index]
    list_target_index = [target_rec_index, target_lig_index]

    map_idx_u_to_b = {} # in rec/lig

    # 0 for rec, 1 for lig
    for j in range(2):
        dict_bb_coord[j] = {}


        chains_b = list_chains_b[j]
        chains_u = list_chains_u[j]

        native = list_native[j]
        target = list_target[j]

        native_index = list_native_index[j]
        target_index = list_target_index[j]

        # alignment
        seq_conc_b = ""
        seq_conc_u = ""

        for chain_b in chains_b:
            seq_conc_b += native[chain_b]['seq']
        
        for chain_u in chains_u:
            seq_conc_u += target[chain_u]['seq']
        
        dict_seq_conc_u[j] = seq_conc_u
        

        map_idx_u_to_b[j] = {} # ub and b are all in conc forms
        alignment = pairwise2.align.globaldd(seq_conc_u, seq_conc_b, matrix,-11,-1,-11,-1)[0]

        idx_u = 0
        idx_b = 0

        for i in range(alignment[-1]):
            if alignment[0][i] != '-':
                if alignment[0][i] == alignment[1][i]:
                    map_idx_u_to_b[j][idx_u] = idx_b
                    # map_r[chain_u][str(idx_u)] = native_index[chain_b][idx_b]
                idx_u += 1
            if alignment[1][i] != '-':
                idx_b += 1
        
        bb_coord_u = np.full((len(seq_conc_u), 3, 3), float('nan'))
        bb_coord_b = np.full((len(seq_conc_u), 3, 3), float('nan'))
        mask_ = np.zeros((len(seq_conc_u)), dtype=bool)

        for nat_idx in range(len(seq_conc_u)):
            if nat_idx in map_idx_u_to_b[j]:
                # chain id and chain idx
                # print(nat_idx, chains_u)
                chain_u, chain_nat_idx_u = nat_idx_conv_conc_to_inchain(nat_idx, chains_u, target_index)
                chain_b, chain_nat_idx_b = nat_idx_conv_conc_to_inchain(map_idx_u_to_b[j][nat_idx], chains_b, native_index)
                # unbound
                if ('N' in target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]) and \
                ('CA' in target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]) and \
                ('C' in target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]):
                # {'A': {'coor': {'1': {'resi': 'CYS', 'N': array([39.722,  6.322, -2.689]), 'CA':
                # bound
                    if ('N' in native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]) and \
                    ('CA' in native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]) and \
                    ('C' in native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]):


                        mask_[nat_idx] = True
                        bb_coord_b[nat_idx, 0] = native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]['N']
                        bb_coord_b[nat_idx, 1] = native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]['CA']
                        bb_coord_b[nat_idx, 2] = native[chain_b]['coor'][native_index[chain_b][chain_nat_idx_b]]['C']

                        bb_coord_u[nat_idx, 0] = target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]['N']
                        bb_coord_u[nat_idx, 1] = target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]['CA']
                        bb_coord_u[nat_idx, 2] = target[chain_u]['coor'][target_index[chain_u][chain_nat_idx_u]]['C']
        
        mask[j] = mask_
        dict_bb_coord[j]["b"] = bb_coord_b
        dict_bb_coord[j]["u"] = bb_coord_u
    
    # c6d for conc(b_r, b_l)
    conc_coord_bb_b = np.concatenate((dict_bb_coord[0]["b"], dict_bb_coord[1]["b"]))
    conc_mask = np.concatenate((mask[0], mask[1]))
    c6d, mask_pair = xyz_to_c6d_modified(torch.from_numpy(conc_coord_bb_b)[None,...], torch.from_numpy(conc_mask)[None,...])

    c6d_numpy = np.array(c6d[0,...])
    mask_pair_numpy = np.array(mask_pair[0,...])

    out_dict = {}

    out_dict["complex_code"] = pdb_b
    out_dict["conc_seq"] = {"rec": dict_seq_conc_u[0], "lig": dict_seq_conc_u[1]}
    # out_dict["Ls"] = Ls
    out_dict["conc_bb_coord"] = {"rec": dict_bb_coord[0], "lig": dict_bb_coord[1]}
    # out_dict["conc_bb_b"] = conc_bb_b
    out_dict["mask"] = conc_mask

    out_dict["labels"] = c6d_numpy
    out_dict["mask_pair"] = mask_pair_numpy

    # print(out_dict["complex_code"], len(out_dict["conc_seq"]["lig"]), out_dict["conc_bb_coord"]["rec"]["b"].shape, out_dict["mask"].shape, out_dict["labels"].shape
    # , out_dict["mask_pair"])#.shape)
    # )


    with open("labels/"+"{}.pickle".format(pdb_b), "wb") as f:
        pickle.dump(out_dict, f)
    
