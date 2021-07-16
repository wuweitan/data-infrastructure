import atomium
from uniChainMapping import *
from chainAuthIndexMapping import *
from getGlyCbPos import *
from chainUniRangeMapping import *
import numpy as np
from Bio.PDB import *
from os import path
from numpy import pi
from Bio.PDB.vectors import Vector
from Bio.PDB.vectors import calc_dihedral


class ProtPairs:
	def __init__(self, Input = None, re_path = 'data'): # for every row, [uniprot1, uniprot2, pdbid]
		self.re_path = re_path
		self.dataList = None
		self.PDBid = None
		self.uni1 = None
		self.uni2 = None
		self.uniToChain_dict = None
		self.closet_chains = [None, None]

		if Input:
			dt = Input
			self.dataList = dt
			self.PDBid = dt[2].lower()
			self.uni1 = dt[0]
			self.uni2 = dt[1]
			self.uniToChain_dict = None

# May 28
# TODO1: which domains are interacting with each other (write the class template)
# TODO2: The atom availability statistics of the pdb
# TODO3: Complete the complex class

# June 4
# TODO1: write the helper functions to find the matching relationship between uniprot ids and chain ids (using APIs)
# TODO2: find the mapping from chain ids to the seq

# Jnue 11
# TODO1: write a function for finding the author index for a given chain
# TODO2: complete class ChainStatistics

# June 18
# TODO1: fixed the part about loading the preprocessed data
# TODO2: position C-beta for Glycine
# TODO3: complete the Dist class

# June 25
# TODO1: add a ComPDB class for constructing new PDB files with missing C-Betas added
# TODO2: complete the Ang class

# July 2
# TODO1: make the comments for every class more complete
# TODO2: complete the domain class

# July 9
# TODO1: start the documentation
# TODO2: find out the mapping between the pfam data and the uniprot data

# July 16
# TODO1: explore the IntAct mutations influencing interactions dataset
# TODO2: complete the mapping from the uniprot id to the pfam id(s) 





class ChainStatistics(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getChainStatForEachUniprot(self): 
		"""
		Find out the statistics of the unmodeled residue

		Output
		1. dict_resStat ({chain_assym_id : ratio_of_unmodeled_to_total_residue})

		"""

		if not Input:
			return np.load(re_path+"/dict_resStat.npy")


		dict_resStat = {}
		pdbl = PDBList()
		pdbl.retrieve_pdb_file(self.PDBid)

		parser = MMCIFParser()
		path_to_cif = self.PDBid[1]+self.PDBid[2]+'/'+self.PDBid+'.cif'

		structure = parser.get_structure(self.PDBid, path_to_cif)

		uniprot_list = [self.uni1, self.uni2]

		if not self.uniToChain_dict:
			
			self.uniToChain_dict = uniChainMapping(self.PDBid, uniprot_list)

		# TODO: give the statistics of the chain according to the user input

		for uni in uniprot_list:

			for chainid in self.uniToChain_dict[uni]:

				# find out the list of author defined indices to the chain sequence
				authIndex_list = chainAuthIndexMapping(self.PDBid, chainid)

				ct_missing = 0

				for In in authIndex_list:
					try:
						res = structure[0][chainid][In]

					except Exception:
						ct_missing += 1

				dict_resStat[chainid] = ct_missing/len(authIndex_list)

		return dict_resStat





            	

		









class Complex(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getComplexInfoForPDB(self):

		"""
		Output:
		1. 'Ass' if there is biological assembly info for the input 
		   'AsymmetricUnit' if there is only assymetric unit information for the input
		   'NoEvidence' if there is only NMR info for the input or there is no such info for the input


		"""

		if not Input:
			return np.load(re_path+"/ComplexInfo.npy")

		pdb3 = atomium.fetch(self.PDBid)
		if len(pdb3.assemblies) == 0:
			return 'AsymmetricUnit'
		else:
			for n in range(len(pdb3.assemblies)):
				if len(pdb3.generate_assembly(n+1).chains()) >= 2:
					return 'Ass'
			return 'NoEvidence'




class Seq(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getSeqInfoForEachUniprot(self): 
		"""
		Output dictionary (chainid : sequences)
	
		"""

		if not Input:
			return np.load(re_path+"/SeqInfo.npy")

		dict_chainSeq = {}
		uniprot_list = [self.uni1, self.uni2]
		mapp = uniChainMapping(self.PDBid, uniprot_list)

		# record the mapping to the self.uniToChain_dict
		self.uniToChain_dict = mapp

		pdb_id = self.PDBid


		pdbl = PDBList()

		pdbl.retrieve_pdb_file(pdb_id)

		parser = MMCIFParser()

		path_to_cif = pdb_id[1]+pdb_id[2]+'/'+pdb_id+'.cif'

		structure = parser.get_structure(pdb_id, path_to_cif)

		if mapp == None:
			return None
		else:
			for chain_list in mapp.values():
				for chain in chain_list:
					dict_chainSeq[chain] = structure[0][chain_id].get_residues()

		return dict_chainSeq




class Domain(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getInteractingDomainInfoForPDB(self): # return # 'interacting doamin pairs' # 'NoEvidence'

		if not Input:
			return np.load(re_path+"/interDomainPairs.npy")

		dict_uniprotToDomain = np.load(re_path+"/dict_uniprotToDomain.npy")

		if (uni1, uni2) in dict_uniprotToDomain:
			return dict_uniprotToDomain[(uni1, uni2)]

		if (uni2, uni1) in dict_uniprotToDomain:
			return [dict_uniprotToDomain[(uni1, uni2)][1], dict_uniprotToDomain[(uni1, uni2)][0]]


class ComPDB(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getNewPDBwCB(self): # return # pdb_id_wCB.cif

		if not Input:
			return re_path+"/PDBwCB"

		pdb_id = self.PDBid

		pdbl = PDBList()
		pdbl.retrieve_pdb_file(pdb_id)

		parser = MMCIFParser()
		path_to_cif = pdb_id[1]+pdb_id[2]+'/'+pdb_id+'.cif'

		structure = parser.get_structure(pdb_id, path_to_cif)

		# read in the files
		with open(path_to_cif, 'r') as f:
		    in_file = f.readlines()

		# build the output
		flag1 = 0 # if 1, enter the matching area (the value part of the atom_site category)
		out_file = []
		last_auth_seq_id = None

		dict_res_atom = {}

		for i, line in enumerate(in_file):
			out_file.append(line)

			if line[0] == '#':
				flag1 = 0

			if flag1 == 1:
				# Find out whether the Cbeta insertion should be done
				# (All the atom info collected (check the ordinal) && Missing CB and existing CA, N, C)

				# Include the atom id in the dictionary
				dict_res_atom[line.split()[-2]] = 1
				
				# The ending of the residue
				if in_file[i+1][0] == '#' or line.split()[-5] != in_file[i+1].split()[-5]:

					# Determine whether CB is to be added
					if ('CB' not in dict_res_atom) and ('CA' in dict_res_atom) and ('N' in dict_res_atom) and ('C' in dict_res_atom):
						
						res = structure[0][line.split()[-3]][int(line.split()[-5])]
						CB_posit = getGlyCbPos(res)

						# print(CB_posit)

						# Construct the xyz coordinates with the target style
						posit = [None, None, None] # initialized as None's
						for q in range(3):
							posit[q] = str(round(CB_posit[q], 3)) + ' '*(list_posit[11+q] - list_posit[10+q] - len(str(round(CB_posit[q], 3))))

						out_str = line[list_posit[0]:list_posit[2]] + 'C' + ' '*(list_posit[3] - list_posit[2] - 1) + 'CB' + ' '*(list_posit[4] - list_posit[3] - 2) + line[list_posit[4]:list_posit[10]] + posit[0] + posit[1] + posit[2] + line[list_posit[13]:list_posit[19]] + 'CB' + ' '*(list_posit[20] - list_posit[19] - 2) + line[list_posit[20]:]
						
						#print(out_str) #######

						out_file.append(out_str)

					# Reset the dictionary
					dict_res_atom = {}


			if line[0: 21] == '_atom_site.pdbx_PDB_m':

				flag1 = 1
				ll = list(in_file[i+1])

				# from the position of the first char of this item to the position of the first char of next item
				cc = 1
				st = 0

				list_posit = [0]

				while cc <= 20:

					while ll[st] != ' ':
						st += 1
					while ll[st] == ' ':
						st += 1

					list_posit.append(st)

					cc += 1


		# Revise the _atom_site.id


		stt = 0
		flag1 = 0

		for i, line in enumerate(out_file):

			if line[0] == '#':
				flag1 = 0

			if flag1 == 1:
				stt += 1
				out_file[i] = line[:list_posit[1]] + str(stt) + ' '*(list_posit[2]-list_posit[1]-len(str(stt))) + line[list_posit[2]:]


			if line[0: 21] == '_atom_site.pdbx_PDB_m':
				flag1 = 1

		print(stt)





		with open(pdb_id[1]+pdb_id[2]+'/'+pdb_id+'_wCB.cif', 'w') as of:
			of.writelines(out_file)






class Ang(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getAngInfoForResiduePairs(self): 

		if not Input:
			return re_path+"/Angles"

		
		# Find out the pair of chains which have the most contacts (cutoff = 8 anstrong)

		if not self.closet_chains[0]:
			# Find out the corresponding chains
			uniprot_list = [self.uni1, self.uni2]

			if not self.uniToChain_dict:
				self.uniToChain_dict = uniChainMapping(self.PDBid, uniprot_list)


			# Find out the pdb Author Index list for each chain
			dict_chainPDBIndex = {}

			for chain_id in self.uniToChain_dict[self.uni1]:
				dict_chainPDBIndex[chain_id] = chainAuthIndexMapping(self.PDBid, chain_id)
			for chain_id in self.uniToChain_dict[self.uni2]:
				dict_chainPDBIndex[chain_id] = chainAuthIndexMapping(self.PDBid, chain_id)

			# Find out the Uniprot index range for each chain
			# {chain_id : [uniprot_seq_beg, uniprot_seq_end]}

			dict_chainUniRange = chainUniRangeMapping(self.PDBid, uniprot_list) 


			# Find out the pair of chains which have the most contacts (cutoff = 8 anstrong)

			pdb_id = self.PDBid
			pdbl = PDBList()
		    pdbl.retrieve_pdb_file(pdb_id)

		    parser = MMCIFParser()
		    path_to_cif = pdb_id[1]+pdb_id[2]+'/'+pdb_id+'.cif'

		    structure = parser.get_structure(pdb_id, path_to_cif)

		    dis_best = None
		    inter_pairs = None
		    num_contact_most = 0


			for chain_id1 in self.uniToChain_dict[self.uni1]:
				for chain_id2 in self.uniToChain_dict[self.uni2]:

			
				    chain_ids = [chain_id1, chain_id2]

				    list_PDBIndex1 = dict_chainPDBIndex[chain_id1]
				    list_PDBIndex2 = dict_chainPDBIndex[chain_id2]

				    uniprot_seq_begs = [dict_chainUniRange[chain_id1][0],dict_chainUniRange[chain_id2][0]]
				    uniprot_seq_ends = [dict_chainUniRange[chain_id1][1],dict_chainUniRange[chain_id2][1]]
				    
				    uniprot_ids = [self.uni1, self.uni2]
				    uniprot_lens = [dict_chainUniRange[chain_id1][1]-dict_chainUniRange[chain_id1][0]+1, dict_chainUniRange[chain_id2][1]-dict_chainUniRange[chain_id2][0]+1] #######

				    chain_lens = [uniprot_seq_ends[0] - uniprot_seq_begs[0]+1, uniprot_seq_ends[1] - uniprot_seq_begs[1]+1]
				    conc_len = sum(chain_lens)


				    # distances
				    distances = -1.0 * np.ones((conc_len, conc_len))

				    for i in range(conc_len):
				        if i < chain_lens[0]:
				            chain_id = chain_ids[0]
				            res_ind_in_chain = list_PDBIndex1[i]
				        else:
				            chain_id = chain_ids[1]
				            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

				        try:
				            res1 = structure[0][chain_id][res_ind_in_chain]
				            if 'CA' in res1 or 'CB' in res1:
				                distances[i][i] = .0
				                for j in range(conc_len):
				                    if j < i:
				                        distances[i][j] = distances[j][i]
				                    elif j > i:

				                        if j < chain_lens[0]:
				                            chain_id2 = chain_ids[0]
				                            res_ind_in_chain2 = list_PDBIndex2[j]
				                        else:
				                            chain_id2 = chain_ids[1]
				                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

				                        try:
				                            res2 = structure[1][chain_id2][res_ind_in_chain2]

				                            if 'CA' in res2 or 'CB' in res2:
				                                min_dis = float('inf')
				                                for k in res1.get_atoms():
				                                    for l in res2.get_atoms():
				                                        diff = k.coord - l.coord
				                                        d = np.sqrt(np.sum(diff * diff))
				                                        if d < min_dis:
				                                            min_dis = d
				                                distances[i][j] = min_dis

				                            else:
				                                pass

				                        except Exception:
				                            pass

				            else:
				                pass



				        except Exception:
				            pass
				    
				    # complete distances matrix
				    comp_len = sum(uniprot_lens)
				    dists = -1.0 * np.ones((comp_len, comp_len))
				    
				    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
				    
				    # Monomer is an exception
				    for c1 in range(chain_lens[0]):
				        for c2 in range(chain_lens[0]):
				            dists[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[c1][c2]
				            
				    for c1 in range(chain_lens[1]):
				        for c2 in range(chain_lens[1]):
				            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[chain_lens[0]+c1][chain_lens[0]+c2]
				    
				    for c1 in range(chain_lens[1]):
				        for c2 in range(chain_lens[0]):
				            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[chain_lens[0]+ c1][c2]
				    
				    for c1 in range(chain_lens[0]):
				        for c2 in range(chain_lens[1]):
				            dists[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[c1][chain_lens[0]+c2]


				    # contacts
				    cut_off = 8

				    ct_map = np.zeros((comp_len, comp_len))

				    # if the distance is -1.0, then the pair is not considered (denoted as non-contacts)
				    # 1's for contacts and 0's for non-contacts

				    for i in range(comp_len):
				        for j in range(comp_len):
				            if dists[i][j] != -1.0 and dists[i][j] <= cut_off :
				                ct_map[i][j] = 1
				    if sum(sum(ct_map)) >= num_contact_most:
				    	num_contact_most = sum(sum(ct_map))
				    	dis_best = dist
				    	inter_pairs = [chain_id1, chain_id2]

			if num_contact_most == 0:
				return None
			else:
				self.closet_chains[0] = chain_id1
				self.closet_chains[1] = chain_id2
				print('The interaction is between chain {} and chain {}.'.format(self.closet_chains[0], self.closet_chains[1]))

		# Compute angles

		# Omega (CA1-CB1-CB2-CA2)
		Omegas = []

		chain_id1 = self.closet_chains[0]
		chain_id2 = self.closet_chains[1]

		chain_ids = [chain_id1, chain_id2]
	    list_PDBIndex1 = dict_chainPDBIndex[chain_id1]
	    list_PDBIndex2 = dict_chainPDBIndex[chain_id2]

	    uniprot_seq_begs = [dict_chainUniRange[chain_id1][0],dict_chainUniRange[chain_id2][0]]
	    uniprot_seq_ends = [dict_chainUniRange[chain_id1][1],dict_chainUniRange[chain_id2][1]]
	    
	    uniprot_ids = [self.uni1, self.uni2]
	    uniprot_lens = [dict_chainUniRange[chain_id1][1]-dict_chainUniRange[chain_id1][0]+1, dict_chainUniRange[chain_id2][1]-dict_chainUniRange[chain_id2][0]+1] #######

	    chain_lens = [uniprot_seq_ends[0] - uniprot_seq_begs[0]+1, uniprot_seq_ends[1] - uniprot_seq_begs[1]+1]
	    conc_len = sum(chain_lens)


	    Omegas = -1.0 * np.ones((conc_len, conc_len))

	    for i in range(conc_len):
	        if i < chain_lens[0]:
	            chain_id = chain_ids[0]
	            res_ind_in_chain = list_PDBIndex1[i]
	        else:
	            chain_id = chain_ids[1]
	            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

	        try:
	            res1 = structure[0][chain_id][res_ind_in_chain]
	            if 'CA' in res1 and 'CB' in res1:
	                Omegas[i][i] = -1.0
	                for j in range(conc_len):
	                    if j < i:
	                        Omegas[i][j] = Omegas[j][i]
	                    elif j > i:

	                        if j < chain_lens[0]:
	                            chain_id2 = chain_ids[0]
	                            res_ind_in_chain2 = list_PDBIndex2[j]
	                        else:
	                            chain_id2 = chain_ids[1]
	                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

	                        try:
	                            res2 = structure[1][chain_id2][res_ind_in_chain2]

	                            if 'CA' in res2 and 'CB' in res2:
	                            	Omegas[i][j] = calc_dihedral(Vector(tuple(res1['CA'].coord)), Vector(tuple(res1['CB'].coord)), Vector(tuple(res2['CB'].coord)), Vector(tuple(res2['CA'].coord)))

	                            else:
	                                pass

	                        except Exception:
	                            pass

	            else:
	                pass



	        except Exception:
	            pass
	    
	    # complete the matrix
	    comp_len = sum(uniprot_lens)
	    omegas = -1.0 * np.ones((comp_len, comp_len))
	    
	    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
	    
	    # Monomer is an exception
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[0]):
	            omegas[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Omegas[c1][c2]
	            
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[1]):
	            omegas[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Omegas[chain_lens[0]+c1][chain_lens[0]+c2]
	    
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[0]):
	            omegas[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Omegas[chain_lens[0] + c1][c2]
	    
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[1]):
	            omegas[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Omegas[c1][chain_lens[0] + c2]


	    # Theta (N1-CA1-CB1-CB2)

	    Thetas = -1.0 * np.ones((conc_len, conc_len))

	    for i in range(conc_len):
	        if i < chain_lens[0]:
	            chain_id = chain_ids[0]
	            res_ind_in_chain = list_PDBIndex1[i]
	        else:
	            chain_id = chain_ids[1]
	            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

	        try:
	            res1 = structure[0][chain_id][res_ind_in_chain]
	            if 'N' in res1 and 'CB' in res1 and 'CA' in res1:
	                Thetas[i][i] = -1.0
	                for j in range(conc_len):
                        if j < chain_lens[0]:
                            chain_id2 = chain_ids[0]
                            res_ind_in_chain2 = list_PDBIndex2[j]
                        else:
                            chain_id2 = chain_ids[1]
                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

                        try:
                            res2 = structure[1][chain_id2][res_ind_in_chain2]

                            if 'CB' in res2:
                            	Thetas[i][j] = calc_dihedral(Vector(tuple(res1['N'].coord)), Vector(tuple(res1['CA'].coord)), Vector(tuple(res1['CB'].coord)), Vector(tuple(res2['CB'].coord)))

                            else:
                                pass

                        except Exception:
                            pass

	            else:
	                pass



	        except Exception:
	            pass
	    
	    # complete the matrix
	    comp_len = sum(uniprot_lens)
	    thetas = -1.0 * np.ones((comp_len, comp_len))
	    
	    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
	    
	    # Monomer is an exception
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[0]):
	            thetas[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Thetas[c1][c2]
	            
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[1]):
	            thetas[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Thetas[chain_lens[0]+c1][chain_lens[0]+c2]
	    
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[0]):
	            thetas[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Thetas[chain_lens[0] + c1][c2]
	    
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[1]):
	            thetas[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Thetas[c1][chain_lens[0] + c2]

	    # Phi (CA1-CB1-CB2)

	    Phis = -1.0 * np.ones((conc_len, conc_len))

	    for i in range(conc_len):
	        if i < chain_lens[0]:
	            chain_id = chain_ids[0]
	            res_ind_in_chain = list_PDBIndex1[i]
	        else:
	            chain_id = chain_ids[1]
	            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

	        try:
	            res1 = structure[0][chain_id][res_ind_in_chain]
	            if 'CB' in res1 and 'CA' in res1:
	                Phis[i][i] = -1.0
	                for j in range(conc_len):
                        if j < chain_lens[0]:
                            chain_id2 = chain_ids[0]
                            res_ind_in_chain2 = list_PDBIndex2[j]
                        else:
                            chain_id2 = chain_ids[1]
                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

                        try:
                            res2 = structure[1][chain_id2][res_ind_in_chain2]

                            if 'CB' in res2:
                            	Phis[i][j] = calc_angle(Vector(tuple(res1['CA'].coord)), Vector(tuple(res1['CB'].coord)), Vector(tuple(res2['CB'].coord)))

                            else:
                                pass

                        except Exception:
                            pass

	            else:
	                pass



	        except Exception:
	            pass
	    
	    # complete the matrix
	    comp_len = sum(uniprot_lens)
	    phis = -1.0 * np.ones((comp_len, comp_len))
	    
	    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
	    
	    # Monomer is an exception
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[0]):
	            phis[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Phis[c1][c2]
	            
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[1]):
	            phis[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Phis[chain_lens[0]+c1][chain_lens[0]+c2]
	    
	    for c1 in range(chain_lens[1]):
	        for c2 in range(chain_lens[0]):
	            phis[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = Phis[chain_lens[0] + c1][c2]
	    
	    for c1 in range(chain_lens[0]):
	        for c2 in range(chain_lens[1]):
	            phis[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = Phis[c1][chain_lens[0] + c2]



	    ##################################################################################################
	    # return the list [omegas, thetas, phis]
	    return [omegas, thetas, phis]




	    








class Dist(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getDistInfoForResiduePairs(self, Type = 'closest'): # (Type = 'closest' # 'alpha' # 'beta')

		if not Input:
			return np.load(re_path+"/dist_"+Type+".npy")

		# Find out the corresponding chains
		uniprot_list = [self.uni1, self.uni2]

		if not self.uniToChain_dict:
			self.uniToChain_dict = uniChainMapping(self.PDBid, uniprot_list)


		# Find out the pdb Author Index list for each chain
		dict_chainPDBIndex = {}

		for chain_id in self.uniToChain_dict[self.uni1]:
			dict_chainPDBIndex[chain_id] = chainAuthIndexMapping(self.PDBid, chain_id)
		for chain_id in self.uniToChain_dict[self.uni2]:
			dict_chainPDBIndex[chain_id] = chainAuthIndexMapping(self.PDBid, chain_id)

		# Find out the Uniprot index range for each chain
		# {chain_id : [uniprot_seq_beg, uniprot_seq_end]}
		dict_chainUniRange = chainUniRangeMapping(self.PDBid, uniprot_list) 


		# Find out the pair of chains which have the most contacts (cutoff = 8 anstrong)

		pdb_id = self.PDBid
		pdbl = PDBList()
	    pdbl.retrieve_pdb_file(pdb_id)

	    parser = MMCIFParser()
	    path_to_cif = pdb_id[1]+pdb_id[2]+'/'+pdb_id+'.cif'

	    structure = parser.get_structure(pdb_id, path_to_cif)

	    dis_best = None
	    inter_pairs = None
	    num_contact_most = 0

		for chain_id1 in self.uniToChain_dict[self.uni1]:
			for chain_id2 in self.uniToChain_dict[self.uni2]:

		
			    chain_ids = [chain_id1, chain_id2]

			    list_PDBIndex1 = dict_chainPDBIndex[chain_id1]
			    list_PDBIndex2 = dict_chainPDBIndex[chain_id2]

			    uniprot_seq_begs = [dict_chainUniRange[chain_id1][0],dict_chainUniRange[chain_id2][0]]
			    uniprot_seq_ends = [dict_chainUniRange[chain_id1][1],dict_chainUniRange[chain_id2][1]]
			    
			    uniprot_ids = [self.uni1, self.uni2]
			    uniprot_lens = [dict_chainUniRange[chain_id1][1]-dict_chainUniRange[chain_id1][0]+1, dict_chainUniRange[chain_id2][1]-dict_chainUniRange[chain_id2][0]+1] #######

			    chain_lens = [uniprot_seq_ends[0] - uniprot_seq_begs[0]+1, uniprot_seq_ends[1] - uniprot_seq_begs[1]+1]
			    conc_len = sum(chain_lens)


			    # distances
			    distances = -1.0 * np.ones((conc_len, conc_len))

			    for i in range(conc_len):
			        if i < chain_lens[0]:
			            chain_id = chain_ids[0]
			            res_ind_in_chain = list_PDBIndex1[i]
			        else:
			            chain_id = chain_ids[1]
			            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

			        try:
			            res1 = structure[0][chain_id][res_ind_in_chain]
			            if 'CA' in res1 or 'CB' in res1:
			                distances[i][i] = .0
			                for j in range(conc_len):
			                    if j < i:
			                        distances[i][j] = distances[j][i]
			                    elif j > i:

			                        if j < chain_lens[0]:
			                            chain_id2 = chain_ids[0]
			                            res_ind_in_chain2 = list_PDBIndex2[j]
			                        else:
			                            chain_id2 = chain_ids[1]
			                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

			                        try:
			                            res2 = structure[1][chain_id2][res_ind_in_chain2]

			                            if 'CA' in res2 or 'CB' in res2:
			                                min_dis = float('inf')
			                                for k in res1.get_atoms():
			                                    for l in res2.get_atoms():
			                                        diff = k.coord - l.coord
			                                        d = np.sqrt(np.sum(diff * diff))
			                                        if d < min_dis:
			                                            min_dis = d
			                                distances[i][j] = min_dis

			                            else:
			                                pass

			                        except Exception:
			                            pass

			            else:
			                pass



			        except Exception:
			            pass
			    
			    # complete distances matrix
			    comp_len = sum(uniprot_lens)
			    dists = -1.0 * np.ones((comp_len, comp_len))
			    
			    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
			    
			    # Monomer is an exception
			    for c1 in range(chain_lens[0]):
			        for c2 in range(chain_lens[0]):
			            dists[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[c1][c2]
			            
			    for c1 in range(chain_lens[1]):
			        for c2 in range(chain_lens[1]):
			            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[chain_lens[0]+c1][chain_lens[0]+c2]
			    
			    for c1 in range(chain_lens[1]):
			        for c2 in range(chain_lens[0]):
			            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[chain_lens[0]+ c1][c2]
			    
			    for c1 in range(chain_lens[0]):
			        for c2 in range(chain_lens[1]):
			            dists[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[c1][chain_lens[0]+c2]


			    # contacts
			    cut_off = 8

			    ct_map = np.zeros((comp_len, comp_len))

			    # if the distance is -1.0, then the pair is not considered (denoted as non-contacts)
			    # 1's for contacts and 0's for non-contacts

			    for i in range(comp_len):
			        for j in range(comp_len):
			            if dists[i][j] != -1.0 and dists[i][j] <= cut_off :
			                ct_map[i][j] = 1
			    if sum(sum(ct_map)) >= num_contact_most:
			    	num_contact_most = sum(sum(ct_map))
			    	dis_best = dist
			    	inter_pairs = [chain_id1, chain_id2]

		if num_contact_most == 0:
			return None
		else:
			self.closet_chains[0] = chain_id1
			self.closet_chains[1] = chain_id2

		if Type == 'closest':
			print('The distance type is the distance between the two closest atoms in the specified residues respectively.')
			print('The interaction is between chain {} and chain {}.'.format(inter_pairs[0], inter_pairs[1]))
			return dis_best

		if Type == 'alpha':

			chain_id1 = inter_pairs[0]
			chain_id2 = inter_pairs[1]

			chain_ids = [chain_id1, chain_id2]
		    list_PDBIndex1 = dict_chainPDBIndex[chain_id1]
		    list_PDBIndex2 = dict_chainPDBIndex[chain_id2]

		    uniprot_seq_begs = [dict_chainUniRange[chain_id1][0],dict_chainUniRange[chain_id2][0]]
		    uniprot_seq_ends = [dict_chainUniRange[chain_id1][1],dict_chainUniRange[chain_id2][1]]
		    
		    uniprot_ids = [self.uni1, self.uni2]
		    uniprot_lens = [dict_chainUniRange[chain_id1][1]-dict_chainUniRange[chain_id1][0]+1, dict_chainUniRange[chain_id2][1]-dict_chainUniRange[chain_id2][0]+1] #######

		    chain_lens = [uniprot_seq_ends[0] - uniprot_seq_begs[0]+1, uniprot_seq_ends[1] - uniprot_seq_begs[1]+1]
		    conc_len = sum(chain_lens)


		    # distances
		    distances = -1.0 * np.ones((conc_len, conc_len))

		    for i in range(conc_len):
		        if i < chain_lens[0]:
		            chain_id = chain_ids[0]
		            res_ind_in_chain = list_PDBIndex1[i]
		        else:
		            chain_id = chain_ids[1]
		            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

		        try:
		            res1 = structure[0][chain_id][res_ind_in_chain]
		            if 'CA' in res1:
		                distances[i][i] = .0
		                for j in range(conc_len):
		                    if j < i:
		                        distances[i][j] = distances[j][i]
		                    elif j > i:

		                        if j < chain_lens[0]:
		                            chain_id2 = chain_ids[0]
		                            res_ind_in_chain2 = list_PDBIndex2[j]
		                        else:
		                            chain_id2 = chain_ids[1]
		                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

		                        try:
		                            res2 = structure[1][chain_id2][res_ind_in_chain2]

		                            if 'CA' in res2:
                                        diff = res1['CA'].coord - res2['CA'].coord
                                        d = np.sqrt(np.sum(diff * diff))

		                                distances[i][j] = d

		                            else:
		                                pass

		                        except Exception:
		                            pass

		            else:
		                pass



		        except Exception:
		            pass
		    
		    # complete distances matrix
		    comp_len = sum(uniprot_lens)
		    dists = -1.0 * np.ones((comp_len, comp_len))
		    
		    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
		    
		    # Monomer is an exception
		    for c1 in range(chain_lens[0]):
		        for c2 in range(chain_lens[0]):
		            dists[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[c1][c2]
		            
		    for c1 in range(chain_lens[1]):
		        for c2 in range(chain_lens[1]):
		            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[chain_lens[0]+c1][chain_lens[0]+c2]
		    
		    for c1 in range(chain_lens[1]):
		        for c2 in range(chain_lens[0]):
		            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[chain_lens[0] + c1][c2]
		    
		    for c1 in range(chain_lens[0]):
		        for c2 in range(chain_lens[1]):
		            dists[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[c1][chain_lens[0] + c2]

		    print('The distance is the distance between the two alpha atoms in the specified residues respectively.')
		    print('The interaction is between chain {} and chain {}.'.format(inter_pairs[0], inter_pairs[1]))
			return dists



		if Type == 'beta':

			chain_id1 = inter_pairs[0]
			chain_id2 = inter_pairs[1]

			chain_ids = [chain_id1, chain_id2]
		    list_PDBIndex1 = dict_chainPDBIndex[chain_id1]
		    list_PDBIndex2 = dict_chainPDBIndex[chain_id2]

		    uniprot_seq_begs = [dict_chainUniRange[chain_id1][0],dict_chainUniRange[chain_id2][0]]
		    uniprot_seq_ends = [dict_chainUniRange[chain_id1][1],dict_chainUniRange[chain_id2][1]]
		    
		    uniprot_ids = [self.uni1, self.uni2]
		    uniprot_lens = [dict_chainUniRange[chain_id1][1]-dict_chainUniRange[chain_id1][0]+1, dict_chainUniRange[chain_id2][1]-dict_chainUniRange[chain_id2][0]+1] #######

		    chain_lens = [uniprot_seq_ends[0] - uniprot_seq_begs[0]+1, uniprot_seq_ends[1] - uniprot_seq_begs[1]+1]
		    conc_len = sum(chain_lens)


		    # distances
		    distances = -1.0 * np.ones((conc_len, conc_len))

		    for i in range(conc_len):
		        if i < chain_lens[0]:
		            chain_id = chain_ids[0]
		            res_ind_in_chain = list_PDBIndex1[i]
		        else:
		            chain_id = chain_ids[1]
		            res_ind_in_chain = list_PDBIndex1[i - chain_lens[0]]

		        try:
		            res1 = structure[0][chain_id][res_ind_in_chain]
		            if 'CB' in res1 or res1.get_name() == "GLY":
		                distances[i][i] = .0
		                for j in range(conc_len):
		                    if j < i:
		                        distances[i][j] = distances[j][i]
		                    elif j > i:

		                        if j < chain_lens[0]:
		                            chain_id2 = chain_ids[0]
		                            res_ind_in_chain2 = list_PDBIndex2[j]
		                        else:
		                            chain_id2 = chain_ids[1]
		                            res_ind_in_chain2 = list_PDBIndex2[j - chain_lens[0]]

		                        try:
		                            res2 = structure[1][chain_id2][res_ind_in_chain2]

		                            if 'CB' in res2 or res2.get_name() == "GLY":

		                            	coord1 = None
		                            	if res1.get_name() == "GLY":
		                            		coord1 = getGlyCbPos(res1)
		                            	else:
		                            		coord1 = res1['CB'].coord

		                            	coord2 = None
		                            	if res2.get_name() == "GLY":
		                            		coord2 = getGlyCbPos(res2)
		                            	else:
		                            		coord2 = res2['CB'].coord


                                        diff = coord1 - coord2
                                        d = np.sqrt(np.sum(diff * diff)) # TODO: check whether the Vector object behaves as a list/numpy array

		                                distances[i][j] = d

		                            else:
		                                pass

		                        except Exception:
		                            pass

		            else:
		                pass



		        except Exception:
		            pass
		    
		    # complete distances matrix
		    comp_len = sum(uniprot_lens)
		    dists = -1.0 * np.ones((comp_len, comp_len))
		    
		    # (0, unibeg[0]-1, uniend[0]-1, length[0]+unibeg[1]-1, length[0]+unibeg[1]-1, length[0]+uniend[1]-1)
		    
		    # Monomer is an exception
		    for c1 in range(chain_lens[0]):
		        for c2 in range(chain_lens[0]):
		            dists[uniprot_seq_begs[0] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[c1][c2]
		            
		    for c1 in range(chain_lens[1]):
		        for c2 in range(chain_lens[1]):
		            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[chain_lens[0]+c1][chain_lens[0]+c2]
		    
		    for c1 in range(chain_lens[1]):
		        for c2 in range(chain_lens[0]):
		            dists[uniprot_lens[0] + uniprot_seq_begs[1] -1 + c1][uniprot_seq_begs[0] -1 + c2] = distances[chain_lens[0] + c1][c2]
		    
		    for c1 in range(chain_lens[0]):
		        for c2 in range(chain_lens[1]):
		            dists[uniprot_seq_begs[0] -1 + c1][uniprot_lens[0] + uniprot_seq_begs[1] -1 + c2] = distances[c1][chain_lens[0] + c2]

		    print('The distance is the distance between the two beta atoms in the specified residues respectively.')
		    print('The interaction is between chain {} and chain {}.'.format(inter_pairs[0], inter_pairs[1]))
			return dists















			

