import Bio.PDB
import numpy as np
import re,os,requests,sys,random
import json
from json import JSONEncoder
import matplotlib.pyplot as plt

def chainUniRangeMapping(pdb_id, unpAcc_list):
  """

  Find out the mapping from the corresponding chains of 2 uniprot ids to the index range of the uniprot 
  through RCSB PDB APIs.


  Input:
  1. unpAcc_list
  2. pdb_id

  Output:
  1. dict_chainUniRange (mapping between chain ids and uniprot index range) ({chain_id : [[uniprot_seq_beg, uniprot_seq_end]]})

  """





  """
  generate mapping between natural chain id and author defined chain id
  
  Input:
   1. pdb_id

  Output:
   2. asymIds_list
  """
  asym_mapping_dict = {}
  rcsbBase_url = "https://data.rcsb.org/graphql"
  rcsb1d_url = "https://1d-coordinates.rcsb.org/graphql"
  # query entity ids
  entityIds_query = '''
    {{entries(entry_ids: ["{}"]) {{
        rcsb_entry_container_identifiers {{
          polymer_entity_ids}}
      }}
    }}  
  '''.format(pdb_id)
  res_entityIds = requests.post(rcsbBase_url,json={'query':entityIds_query})
  if res_entityIds.status_code != 200:
    return None
  else:
    try:
      entityIds_list = res_entityIds.json()['data']['entries'][0]['rcsb_entry_container_identifiers']['polymer_entity_ids']
      for ent_id in entityIds_list:
        # query asym_ids, auth_asym_ids
        asymIds_query = '''
          {{polymer_entities(entity_ids:["{}_{}"])
              {{rcsb_polymer_entity_container_identifiers {{
                  asym_ids
                  auth_asym_ids}}
                entity_poly {{
                  pdbx_strand_id}}
              }}
          }}
        '''.format(bpdb_id,ent_id)
        res_asymIds = requests.post(rcsbBase_url,json={'query':asymIds_query})
        if res_asymIds.status_code != 200:
          return None
        else:
          rec_asymIds_json = res_asymIds.json()
          asymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['asym_ids']
          #authAsymIds_list = rec_asymIds_json['data']['polymer_entities'][0]['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] # auth_asym_ids may not have the right order of ids
          pdbx_strandId_list = re.split(r',',rec_asymIds_json['data']['polymer_entities'][0]['entity_poly']['pdbx_strand_id'])
          assert len(asymIds_list) == len(pdbx_strandId_list), "asym_ids length not same with auth_asym_ids"

  """

  Input:
  1. asymIds_list (chain id list)
  2. unpAcc_list
  3. pdb_id

  Output:
  1. dict_chainUniRange (mapping between chain ids and uniprot index range)

  """

  dict_chainUniRange = {}

  for chain_id in asymIds_list:
    pdb_instance = '{}.{}'.format(pdb_id,chain_id)

    query_align = '''
    {{alignment(from:PDB_INSTANCE,to:UNIPROT,queryId:"{}"){{
      query_sequence
      target_alignment {{
        target_id
        target_sequence
        aligned_regions {{
          query_begin
          query_end
          target_begin
          target_end}}
      }}
     }}
    }}
    '''.format(pdb_instance)

    res_align = requests.post(rcsb1d_url,json={'query':query_align})

    if res_align.status_code != 200:
      return None
    else:
      res_align_json=res_align.json()

      for d in res_align_json['data']['alignment']['target_alignment']:
        if d['target_id'] == unpAcc_list[0]:
          unp_seq=d['target_sequence']
          aligned_regions=[d['aligned_regions'][target_begin], d['aligned_regions'][target_end]]
          dict_chainUniRange[chain_id] = aligned_regions

        if d['target_id'] == unpAcc_list[1]:
          unp_seq=d['target_sequence']
          aligned_regions=[d['aligned_regions'][target_begin], d['aligned_regions'][target_end]]
          dict_chainUniRange[chain_id] = aligned_regions

      if unp_seq is None: 
        # no such unpAcc under this pdb,
        pass

  return dict_chainUniRange