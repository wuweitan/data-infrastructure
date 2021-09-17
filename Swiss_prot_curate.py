import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import amino_acid
import sys


class Swiss_Prot:
	def __init__(self, file_path):
		# arguments:  file_path:  The path to the downloaded Uniport file. Please use the txt file format.

		self.file_path = file_path

	def read_file(self): 
	# The scripts for processing the UniProt file. 
	# return:
	#           A list contains all proteins in the file.
	#           Each entry of the list is a dictionary that stores the information of that proteins
	#           

	self.seq_SP = []
	self.ac={}
	self.id={}

	with open(self.file_path, "r") as f:
		num_line=0

		for line in f:
			if line[0:2]=='ID':
				self.seq_SP.append({})
				self.seq_SP[-1]['GO'] =[]
				self.seq_SP[-1]['ID'] = line.split()[1]
				self.seq_label = 0
				self.id[self.seq_SP[-1]['ID']] = len(self.seq_SP)-1


			elif(line[0:2]=='AC'):
				ac = line[5:].split(';')[0]
				self.seq_SP[-1]['ac'] =ac
				self.ac[ac] = len(self.seq_SP)-1

			elif line[0:2]=='DT' and "UniProtKB/Swiss-Prot" in line:
				self.seq_SP[-1]['date'] =  line[5:16]



			elif line[0:7] == 'DR   GO':
				entry = [line[9:19], line[21]]
				
				self.seq_SP[-1]['GO'].append(entry)

			elif line=='//\n':
				self.seq_SP[-1]['seq'] = temp_seq

				# ----------------if the instance does not have any GO terms or date; or it contains non-natural
				# amino acids,  we remove it.
				if self.seq_SP[-1]['GO'] ==[] or 'date' not in self.seq_SP[-1] or amino_acid.Nature_seq(self.seq_SP[-1]['seq'])==False:
					self.seq_SP.pop(-1)
				

			elif seq_label == 1:
				temp_seq += line.strip('\n').replace(' ','') 

			elif line[0:2]=='SQ':
				self.seq_label=1
				temp_seq=''

			num_line+=1

	def query_ac(self, ac):

		# query the protein using the accession number 
		return self.ac[ac]
	
	def query_id(self, id):

		# query the protein using the ID in Uniprot
		return self.id[id]
	