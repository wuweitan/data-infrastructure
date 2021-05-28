class ProtPairs:
	def __init__(self, Input = None, re_path = 'data'): # for every row, [uniprot1, uniprot2, pdbids]
		self.re_path = re_path
		if not Input:
			dt = self.loadData()
		else:
			dt = Input
		self.dataList = dt

	def loadData(self):
		pass

# TODO1: which domains are interacting with each other (write the class template)
# TODO2: The atom availability statistics of the pdb
# TODO3: Complete the complex class

class AtomStatistics(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getAtomStatForEachUniprot(self): # return [[chain_id4Uni1,'Atom Availability Statisctics'], [chain_id4Uni2,'Atom Availability Statisctics']]

class Complex(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getComplexInfoForPDB(self): # return 'Ass' # 'AsymmetricUnit' # 'NoEvidence'
		pass


class Seq(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getSeqInfoForEachUniprot(self): #(return [[chain_id4Uni1, sequences4CorChains], [chain_id4Uni2, sequences4CorChains]])
		pass

class Domain(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getInteractingDomainInfoForPDB(self): # return # 'interacting doamin pairs' # 'NoEvidence'
		pass


class Ang(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getAngInfoForResiduePairs(self, Type = 'all'): # (Type = 'all' # 'dihedral' # 'planar')
		pass

class Dist(ProtPairs):
	def __init__(self, Input = None, re_path = 'data'):
		super(Seq, self).__init__(Input, re_path)

	def loadData(self):
		pass

	def getDistInfoForResiduePairs(self, Type = 'closest'): # (Type = 'closest' # 'alpha' # 'beta')
		pass

