import atomium
import uniChainMapping
import 

class ProtPairs:
	def __init__(self, Input = None, re_path = 'data'): # for every row, [uniprot1, uniprot2, pdbid]
		self.re_path = re_path
		if not Input:
			dt = self.loadData()
		else:
			dt = Input
		self.dataList = dt
		self.PDBid = dt[2]
		self.uni1 = dt[0]
		self.uni2 = dt[1]

	def loadData(self):
		pass
# May 28
# TODO1: which domains are interacting with each other (write the class template)
# TODO2: The atom availability statistics of the pdb
# TODO3: Complete the complex class

# June 4
# TODO1: write the helper functions to find the matching relationship between uniprot ids and chain ids (using APIs)
# TODO2: find the mapping from chain ids to the seq



class AtomStatistics(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getAtomStatForEachUniprot(self): # return [[chain_id4Uni1,'Atom Availability Statisctics'], [chain_id4Uni2,'Atom Availability Statisctics']]
		pass

class Complex(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getComplexInfoForPDB(self): # return 'Ass' # 'AsymmetricUnit' # 'NoEvidence'
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
		dict_chainSeq = {}
		uniprot_list = [self.uni1, self.uni2]
		mapp = uniChainMapping(self.PDBid, uniprot_list)

		pdb_id = self.PDBid


		pdbl = PDBList()

		pdbl.retrieve_pdb_file(pdb_id)

		parser = MMCIFParser()

		path_to_cif = pdb_id[1]+pdb_id[2]+'/'+pdb_id+'.cif'

		structure = parser.get_structure(pdb_id, path_to_cif)

		if mapp == None:
			return None
		else:
			for chain_list in mapp:
				for chain in chain_list:
					dict_chainSeq[chain] = structure[0][chain_id].get_residues()

		return dict_chainSeq




class Domain(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getInteractingDomainInfoForPDB(self): # return # 'interacting doamin pairs' # 'NoEvidence'
		pass


class Ang(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getAngInfoForResiduePairs(self, Type = 'all'): # (Type = 'all' # 'dihedral' # 'planar')
		pass

class Dist(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super().__init__(Input, re_path)

	def getDistInfoForResiduePairs(self, Type = 'closest'): # (Type = 'closest' # 'alpha' # 'beta')
		pass

