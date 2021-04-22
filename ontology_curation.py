import numpy as np
import pickle

path  = "go-basic-12082020.obo"



go_graph = {}
alt_id = {}
obsolete = set()

in_goterm_or_not =0


with open(path, "r") as f:
	for line in f:
	

		if("[Term]" in line):
			in_goterm_or_not = 1
			

		if(line=='\n'):
			in_goterm_or_not = 0


		if in_goterm_or_not == 1:
			if(line[0:3] == "id:"):
				current_goid = line[4:14]
				go_graph[current_goid] = {}
				alt_id[current_goid] = current_goid
				go_graph[current_goid]['father']=[]
				go_graph[current_goid]['child'] = []	

			if(line[0:5] == 'name:'):
				go_graph[current_goid]['name'] = line[6:].strip('\n')

			if(line[0:9] == 'namespace'):
				go_graph[current_goid]['type'] = line[11]

			if(line[0:6] == 'alt_id'):
				alt_id[line[8:18]] = current_goid
			
			if(line[0:4] == 'def:'):
				go_graph[current_goid]['def'] = line.split('\"')[1]


			if(line[0:4] == 'is_a'):
					go_graph[current_goid]['father'].append(line[6:16])

			if(line[0:21] == "relationship: part_of"):
					go_graph[current_goid]['father'].append(line[22:32])				



			if("is_obsolete: true" in line):
				del go_graph[current_goid]
				obsolete.add(current_goid)


for go in go_graph:
	
	if go_graph[go]['father']==[]:
		print ("no father:", go, go_graph[go]['type'])



for go in go_graph:
	# delete cross-ontology:
	new_father = []
	for child_go in go_graph[go]['father']:
		if go_graph[go]['type'] == go_graph[child_go]['type']:
			new_father.append(child_go)


	go_graph[go]['father'] = new_father

	for child_go in go_graph[go]['father']:
		go_graph[child_go]['child'].append(go)


print (len(go_graph))


# ontolop MF, BP, CC

MF_go_graph={}
BP_go_graph={}
CC_go_graph={}

for go in go_graph:
	if go_graph[go]['type'] == 'm':
		MF_go_graph[go] = go_graph[go]
	elif go_graph[go]['type'] == 'b':
		BP_go_graph[go] = go_graph[go]
	else:
		CC_go_graph[go] = go_graph[go]



def dfs(node, go_graph):
	global num
	for go  in go_graph[node]['child']:
		if 'label' not in go_graph[go]:
			go_graph[go]['label']=1
			num+=1
			dfs(go, go_graph)

num=1
dfs('GO:0003674', MF_go_graph)
print ('MF', num, len(MF_go_graph))
num=1
dfs('GO:0008150', BP_go_graph)
print ('BP', num, len(BP_go_graph))
num=1
dfs('GO:0005575', CC_go_graph)
print ('CC', num, len(CC_go_graph))

# judge never been touched:
def isolated_node(go_graph):
	is_node=[]
	for go in go_graph:

		if 'label' not in go_graph[go]:
			is_node.append(go)

		else:
			del go_graph[go]['label']

	return is_node
# assign index 

print ('isolated_node:', len(isolated_node(MF_go_graph)))
print ('isolated_node:', len(isolated_node(BP_go_graph)))
print ('isolated_node:', len(isolated_node(CC_go_graph)))


# def count_instance(i, go_graph, instance_on_node):  # count the instance on each node
	
# 	instance_on_node[i]=1

# 	for fa in go_graph[i]['father']:
# 		count_instance(fa, go_graph, instance_on_node)


# go_father ={}

# for go in MF_go_graph:
# 	go_father[go]=[]
# 	instance_on_node={}
# 	count_instance(go, MF_go_graph, instance_on_node)
# 	for i in instance_on_node:
# 		go_father[go].append(i)

# for go in BP_go_graph:
# 	go_father[go]=[]
# 	instance_on_node={}
# 	count_instance(go, BP_go_graph, instance_on_node)
# 	for i in instance_on_node:
# 		go_father[go].append(i)

# for go in CC_go_graph:
# 	go_father[go]=[]
# 	instance_on_node={}
# 	count_instance(go, CC_go_graph, instance_on_node)
# 	for i in instance_on_node:
# 		go_father[go].append(i)

print (MF_go_graph['GO:0003824'])


fname = open("mf_go_name.txt", "w")
fdesp = open("mf_go_des.txt","w")
fgo = open("mf_go.txt","w")
for i in MF_go_graph:

	fname.write(MF_go_graph[i]['name'].replace('/',' or ')+'\n')
	fdesp.write(MF_go_graph[i]['def'].replace('/',' or ')+'\n')
	fgo.write(i+'\n')



with open("mf_go_all.pickle", "wb") as f:
	pickle.dump(MF_go_graph, f)

with open("bp_go_all.pickle", "wb") as f:
	pickle.dump(BP_go_graph, f)

with open("cc_go_all.pickle", "wb") as f:
	pickle.dump(CC_go_graph, f)

with open("alt_id.pickle", "wb") as f:
	pickle.dump(alt_id, f)

with open("obsolete.pickle", "wb") as f:
	pickle.dump(obsolete, f)
