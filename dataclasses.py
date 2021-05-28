import atomium

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

# TODO1: which domains are interacting with each other (write the class template)
# TODO2: The atom availability statistics of the pdb
# TODO3: Complete the complex class

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

	def getSeqInfoForEachUniprot(self): #(return [[chain_id4Uni1, sequences4CorChains], [chain_id4Uni2, sequences4CorChains]])
		pass

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
