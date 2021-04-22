import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import amino_acid


with open("alt_id.pickle", "rb") as f:
	alt_id = pickle.load(f)


with open("obsolete.pickle", "rb") as f:
	obsolete = pickle.load(f)

with open("mf_go_all.pickle", "rb") as f:
	mf_go_graph = pickle.load(f)

cafa_criteria = {'EXP': 5,
'IDA': 5,
'IPI': 5,
'IMP': 5,
'IGI': 5, 
'IEP': 3,
'TAS': 4,
'IC': 4,
}
evidence_code_rank={
	'EXP': 5,
'IDA': 5,
'HDA': 5,
'IPI': 5,
'IMP': 5,
'HMP': 5,
'IGI': 5,
'HGI': 5,
'IEP': 3,
'HEP': 3,
'ISS': 2,
'ISO': 2,
'ISA':0,
'ISM': 0,
'IGC':3,
'IBA': 2,
'IBD': 2,
'IKR': 2,
'IRD': 2,
'RCA': 3 ,
'TAS': 4,
'NAS': 1.5,
'IC': 4,
'ND':0,
'IEA':2, 
'NR':1
}

evidence_code_stat={}
for i in evidence_code_rank:
	evidence_code_stat[i]=0

evidence_level=2


file_path = "Swiss_prot_12022020/uniprot_sprot.dat"
len_limit=200


def swiss_prot_curate(): # Swiss Prot data
	

	seq_SP = {}

	with open(file_path, "r") as f:
		num_line=0

		for line in f:
			if line[0:2]=='ID':
				
				seq_SP[line.split()[1]]={'GO':[]}
				idd = line.split()[1]
				seq_label=0

			elif(line[0:2]=='AC'):
				ac = line[5:].split(';')[0]
				seq_SP[idd]['ac'] =ac
				

			elif line[0:2]=='DT' and "UniProtKB/Swiss-Prot" in line:
				seq_SP[idd]['date'] =  line[5:16]



			elif line[0:7] == 'DR   GO':

				evidence_code = line.split(';')[-1][1:4].strip(':')
				#print (idd, evidence_code)
				assert evidence_code in evidence_code_stat

				if evidence_code_rank[evidence_code] >=2:
				#if evidence_code in cafa_criteria: #and line[9:19] in on:

				
					if alt_id[line[9:19]] in mf_go_graph:
						seq_SP[idd]['GO'].append( alt_id[line[9:19]])


			elif line=='//\n':
				seq_SP[idd]['seq'] = temp_seq

				# ----------------if the instance does not have any GO terms or date; or it contains non-natural
				# amino acids or the length is above 500,  we remove it.
				if seq_SP[idd]['GO'] ==[] or 'date' not in seq_SP[idd] or \
				amino_acid.Nature_seq(seq_SP[idd]['seq'])==False:
					del seq_SP[idd]
				

			elif seq_label == 1:
				temp_seq += line.strip('\n').replace(' ','') 

			elif line[0:2]=='SQ':
				seq_label=1
				temp_seq=''

			num_line+=1
			
	return seq_SP

seq_SP = swiss_prot_curate()


for i in seq_SP:
	if seq_SP[i]['GO']==[]:
		print ('go no', seq_SP[i]['ac'])
	if 'date' not in seq_SP[i]:
		print ('date no', seq_SP[i]['ac'])




print (len(seq_SP))

#with open("seq_ec2_mfo_l"+str(len_limit)+".pkl", "wb") as f:
with open("seq_ecCAFA_mfo_all.pkl", "wb") as f:
	pickle.dump(seq_SP, f)