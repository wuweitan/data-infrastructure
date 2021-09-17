import numpy as np
import pickle


class Ontology:
	def __init__(self, path):
		self.path  = path
		self.go_graph = {}
		self.alt_id = {}
		self.obsolete = []

	def read_file(self):

		with open(self.path, "r") as f:
			for line in f:
	
				if("[Term]" in line):
					in_goterm_or_not = 1
					

				if(line=='\n'):
					in_goterm_or_not = 0


				if in_goterm_or_not == 1:
					if(line[0:3] == "id:"):
						current_goid = line[4:14]
						self.go_graph[current_goid] = {}
						self.alt_id[current_goid] = current_goid
						self.go_graph[current_goid]['father']=[]
						self.go_graph[current_goid]['child'] = []

					if(line[0:5] == 'name:'):
						self.go_graph[current_goid]['name'] = line[6:].strip('\n')

					if(line[0:9] == 'namespace'):
						self.go_graph[current_goid]['type'] = line[11]

					if(line[0:6] == 'alt_id'):
						self.alt_id[line[8:18]] = current_goid

					if(line[0:4] == 'is_a'):
						self.go_graph[current_goid]['father'].append(line[6:16])

					if(line[0:21] == "relationship: part_of"):
						self.go_graph[current_goid]['father'].append(line[22:32])				



					if("is_obsolete: true" in line):
						del self.go_graph[current_goid]
						self.obsolete.append(current_goid)
	def check_root_node(self):

		root_node=[]
		for go in self.go_graph:
			if self.go_graph[go]['father']==[]:
				root_node.append(go)
		return root_node

	def delete_cross_link(self):
		for go in self.go_graph:
			# delete cross-ontology:
			new_father = []
			for child_go in self.go_graph[go]['father']:
				if self.go_graph[go]['type'] == self.go_graph[child_go]['type']:
					new_father.append(child_go)


			self.go_graph[go]['father'] = new_father

			for child_go in self.go_graph[go]['father']:
				self.go_graph[child_go]['child'].append(go)

	def query_node(self, go):
		return self.graph[go]

	def dfs(self, node):
		global num
		for go  in self.go_graph[node]['child']:
			if 'label' not in self.go_graph[go]:
				self.go_graph[go]['label']=1
				num+=1
				dfs(go, self.go_graph)

	def check_isolated_node(self)
		is_node=[]
		for go in self.go_graph:

			if 'label' not in self.go_graph[go]:
				is_node.append(go)

		return is_node

