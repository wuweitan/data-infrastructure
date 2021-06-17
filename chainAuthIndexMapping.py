def chainAuthIndexMapping(pdb_id, chain_id):
	"""

	find out the list of author defined indices for the specified chain id

	Input
	1. pdb_id
	2. chain_id (asym id)

	Output
	1. list of author definded indices for the chain id

	"""

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
	res_idxMap_json=res_idxMap.json()
	auth_pdbSeq_mapping=res_idxMap_json['data']['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers']['auth_to_entity_poly_seq_mapping']

	return auth_pdbSeq_mapping