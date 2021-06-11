import os 
from data_load import dict_load, dict_save

def remove_char(string, char):
    return ''.join(string.split(char))

### Information
#SabDab 
kind = 'SAbDab'
SabDab_dict = {}
with open('../Information/%s/%s_info.txt'%(kind,kind),'r') as rf:
    lines = rf.readlines()[1:]
for line in lines:
    line = [remove_char(c,' ') for c in line.strip('\n').split('\t')]
    pdb = line[0].upper()
    SabDab_dict[pdb] = []
    heavy_chains = line[2].split(',')
    light_chains = line[3].split(',')
    antigen_chains = line[4].split(',')
    for i,hc in enumerate(heavy_chains):
        SabDab_dict[pdb].append((hc, light_chains[i], antigen_chains[i]))
print('%d pdbs in SAbDab'%len(SabDab_dict.keys()))

#abYbank
map_dict = {}
with open('../Dataset/abYbank/AbDb_chainMapping.dat','r') as mf:
    lines = mf.readlines()
for line in lines:
    line = line.strip('\n').split(',')
    comp = line[0]
    map_dict[comp] = {}
    for ch in line[1:]:
        ch = ch.split(':')
        map_dict[comp][ch[0]] = ch[1]

kind = 'abYbank'
abYbank_dict = {}
with open('../Information/%s/%s_info.txt'%(kind,kind),'r') as rf:
    lines = rf.readlines()[1:]
for line in lines:
    line = [remove_char(c,' ') for c in line.strip('\n').split('\t')]
    comp = line[0]
    pdb = line[0].split('_')[0].upper()
    hc = map_dict[comp][line[2]]
    lc = map_dict[comp][line[3]]
    antigen = '|'.join(line[4].split(','))
    if not pdb in abYbank_dict.keys():
        abYbank_dict[pdb] = [(hc, lc, antigen)]
    else:
        abYbank_dict[pdb].append((hc, lc, antigen))
print('%d pdbs in abYbank'%len(abYbank_dict.keys()))

#IMGT
IMGT_dict = dict_load('../Information/IMGT/IMGT_info_dict.pickle')
unsele_pdb = []
for pdb in IMGT_dict.keys():
    unsele_chain = []
    for chain in IMGT_dict[pdb]['Seq_info'].keys():
        sele_flag = True
        if not 'CDR1' in IMGT_dict[pdb]['Seq_info'][chain].keys() or '#' in IMGT_dict[pdb]['Seq_info'][chain]['CDR1']:
            sele_flag = False
        elif not 'CDR2' in IMGT_dict[pdb]['Seq_info'][chain].keys() or '#' in IMGT_dict[pdb]['Seq_info'][chain]['CDR2']:
            sele_flag = False
        elif not 'CDR3' in IMGT_dict[pdb]['Seq_info'][chain].keys() or '#' in IMGT_dict[pdb]['Seq_info'][chain]['CDR3']:
            sele_flag = False
        if not sele_flag:
            unsele_chain.append(chain)

    for chain in unsele_chain:
        del IMGT_dict[pdb]['Seq_info'][chain]
        IMGT_dict[pdb]['antibody'].remove(chain)
    if len(IMGT_dict[pdb]['antibody']) == 0:
        unsele_pdb.append(pdb)

for pdb in unsele_pdb:
    del IMGT_dict[pdb]
print('%d pdbs in IMGT'%len(IMGT_dict.keys()))

_save = dict_save(IMGT_dict, '../Information/IMGT/IMGT_info_dict_sele.pickle')

#Cov3D
CoV3D_dict = dict_load('../Information/CoV3D/CoV3D_info_dict.pickle')
unsele_pdb = []
for pdb in CoV3D_dict.keys():
    if not ('H' in CoV3D_dict[pdb].keys() and 'H1' in CoV3D_dict[pdb].keys() and 'H2' in CoV3D_dict[pdb].keys() and 'H3' in CoV3D_dict[pdb].keys() and\
    'L' in CoV3D_dict[pdb].keys() and 'L1' in CoV3D_dict[pdb].keys() and 'L2' in CoV3D_dict[pdb].keys() and 'L3' in CoV3D_dict[pdb].keys()):
        unsele_pdb.append(pdb)

for pdb in unsele_pdb:
    del CoV3D_dict[pdb]
print('%d pdbs in CoV3D'%len(CoV3D_dict.keys()))

_save = dict_save(IMGT_dict, '../Information/CoV3D/CoV3D_info_dict_sele.pickle')

### clear intersection info ###

print()
print('Intersections:')

inder_dict = dict_load('../Information/Intersection/inter_dict.pickle')

for kind in inder_dict:
    for label in inder_dict[kind]:
        if 'IMGT' in label:
            unsele_pdb = []
            for pdb in inder_dict[kind][label]:
                if not pdb.upper() in IMGT_dict.keys():
                    unsele_pdb.append(pdb)
            for pdb in unsele_pdb:
                inder_dict[kind][label].remove(pdb)

        if 'CoV3D' in label:
            unsele_pdb = []
            for pdb in inder_dict[kind][label]:
                if not pdb.lower() in CoV3D_dict.keys():
                    unsele_pdb.append(pdb)
            for pdb in unsele_pdb:
                inder_dict[kind][label].remove(pdb)

        print('%d pdbs in %s.'%(len(inder_dict[kind][label]), label))

_save = dict_save(inder_dict, '../Information/Intersection/inter_dict_sele.pickle')

### Arrange the information ###

union_dict = {'All':{}, 'Pre':{}, 'Pre+':{}, 'CoV3D':{}}

### IMGT

for label in inder_dict[3].keys():
    if 'IMGT' in label and 'SAbDab' in label and 'abYbank' in label:
        for pdb in inder_dict[3][label]:
            sele_comp = [[],[]]
            # SabDab
            for comp in SabDab_dict[pdb]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                flag = True
                if not (hc in IMGT_dict[pdb]['antibody'] and lc in IMGT_dict[pdb]['antibody']):
                    flag = False
                if flag:
                    for ch in an:
                        if not ch in IMGT_dict[pdb]['antigen']:
                            flag = False
                            break
                if flag:
                    sele_comp[0].append((hc,lc))
                    sele_comp[1].append(an)
            # abYbank
            for comp in abYbank_dict[pdb]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                flag = True
                if not (hc in IMGT_dict[pdb]['antibody'] and lc in IMGT_dict[pdb]['antibody']):
                    flag = False
                if flag:
                    for ch in an:
                        if not ch in IMGT_dict[pdb]['antigen']:
                            flag = False
                            break
                if flag:
                    if (hc,lc) in sele_comp[0]:
                        index = sele_comp[0].index((hc,lc))
                        if an != sele_comp[1][index]:
                            print('Antigen chains in SAbDab (%s) and abYbank (%s) of %s do not match. Take abYbank as the result.'%(sele_comp[1][index], an, pdb))
                            sele_comp[1][index] = an
                    else:
                        sele_comp[0].append((hc,lc))
                        sele_comp[1].append(an)
            ### CDR info  
            union_dict['Pre'][pdb] = []
            for i, antibody in enumerate(sele_comp[0]):  
                hc = sele_comp[0][i][0]
                lc = sele_comp[0][i][1]
                comp_info = {'heavy':{'chain':hc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][hc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][hc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][hc]['CDR3']},
                             'light':{'chain':lc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][lc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][lc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][lc]['CDR3']},
                             'antigen':sele_comp[1][i]}
                union_dict['Pre'][pdb].append(comp_info)
             
            
for label in inder_dict[2].keys():
    # SabDa
    if 'IMGT' in label and 'SAbDab' in label:
        for pdb in inder_dict[2][label]:
            if pdb in union_dict['Pre'].keys():
                continue
            sele_comp = [[],[]]
            # SabDab
            for comp in SabDab_dict[pdb]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                flag = True
                if not (hc in IMGT_dict[pdb]['antibody'] and lc in IMGT_dict[pdb]['antibody']):
                    flag = False
                if flag:
                    for ch in an:
                        if not ch in IMGT_dict[pdb]['antigen']:
                            flag = False
                            break
                if flag:
                    sele_comp[0].append((hc,lc))
                    sele_comp[1].append(an)
            ### CDR info  
            union_dict['Pre'][pdb] = []
            for i, antibody in enumerate(sele_comp[0]):
                hc = sele_comp[0][i][0]
                lc = sele_comp[0][i][1]
                comp_info = {'heavy':{'chain':hc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][hc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][hc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][hc]['CDR3']},
                             'light':{'chain':lc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][lc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][lc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][lc]['CDR3']},
                             'antigen':sele_comp[1][i]}
                union_dict['Pre'][pdb].append(comp_info)

    # abYbank
    if 'IMGT' in label and 'abYbank' in label:
        for pdb in inder_dict[2][label]:
            if pdb in union_dict['Pre'].keys():
                continue
            sele_comp = [[],[]]
            for comp in abYbank_dict[pdb]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                flag = True
                if not (hc in IMGT_dict[pdb]['antibody'] and lc in IMGT_dict[pdb]['antibody']):
                    flag = False
                if flag:
                    for ch in an:
                        if not ch in IMGT_dict[pdb]['antigen']:
                            flag = False
                            break
                if flag:
                    sele_comp[0].append((hc,lc))
                    sele_comp[1].append(an)
            ### CDR info  
            union_dict['Pre'][pdb] = []
            for i, antibody in enumerate(sele_comp[0]):
                hc = sele_comp[0][i][0]
                lc = sele_comp[0][i][1]
                comp_info = {'heavy':{'chain':hc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][hc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][hc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][hc]['CDR3']},
                             'light':{'chain':lc,
                                      'CDR1':IMGT_dict[pdb]['Seq_info'][lc]['CDR1'],
                                      'CDR2':IMGT_dict[pdb]['Seq_info'][lc]['CDR2'],
                                      'CDR3':IMGT_dict[pdb]['Seq_info'][lc]['CDR3']},
                             'antigen':sele_comp[1][i]}
                union_dict['Pre'][pdb].append(comp_info)
        
### CoV3D 

CoV3D_alone_list = []

for pdb in CoV3D_dict.keys():
    pdb_name = pdb.upper()
    if pdb_name in union_dict['Pre'].keys():
        for comp in union_dict['Pre'][pdb_name]:
            # heavy chain
            if comp['heavy']['chain'] in CoV3D_dict[pdb]['H']:
                comp['heavy']['CDR1'] = CoV3D_dict[pdb]['H1']
                comp['heavy']['CDR2'] = CoV3D_dict[pdb]['H2']
                comp['heavy']['CDR3'] = CoV3D_dict[pdb]['H3']
            else:
                print('Heavy chain %s of %s is not in CoV3D but %s is.'%(union_dict['Pre'][pdb_name]['heavy']['chain'], pdb_name, pdb_name))
            ### light chain
            if comp['light']['chain'] in CoV3D_dict[pdb]['L']:
                comp['light']['CDR1'] = CoV3D_dict[pdb]['L1']
                comp['light']['CDR2'] = CoV3D_dict[pdb]['L2']
                comp['light']['CDR3'] = CoV3D_dict[pdb]['L3']
            else:
                print('Light chain %s of %s is not in CoV3D but %s is.'%(union_dict['Pre'][pdb_name]['light']['chain'], pdb_name, pdb_name))

        union_dict['CoV3D'][pdb_name] = union_dict['Pre'][pdb_name]

    elif pdb_name in abYbank_dict.keys() or pdb_name in SabDab_dict.keys():    
        sele_comp = [[],[]]
        # abYbank
        if pdb_name in abYbank_dict.keys():
            for comp in abYbank_dict[pdb_name]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                if (hc in CoV3D_dict[pdb]['H'] and lc in CoV3D_dict[pdb]['L']):
                    sele_comp[0].append((hc,lc))
                    sele_comp[1].append(an)
        # SabDab 
        if pdb_name in SabDab_dict.keys():
            for comp in SabDab_dict[pdb_name]:
                hc = comp[0]
                lc = comp[1]
                an = set(comp[2].split('|'))
                if (hc in CoV3D_dict[pdb]['H'] and lc in CoV3D_dict[pdb]['L']) and not (hc,lc) in sele_comp[0]:
                    sele_comp[0].append((hc,lc))
                    sele_comp[1].append(an)
        ### CDR info  
        union_dict['CoV3D'][pdb_name] = []
        for i, antibody in enumerate(sele_comp[0]):
            hc = sele_comp[0][i][0]
            lc = sele_comp[0][i][1]
            comp_info = {'heavy':{'chain':hc,
                                  'CDR1':CoV3D_dict[pdb]['H1'],
                                  'CDR2':CoV3D_dict[pdb]['H2'],
                                  'CDR3':CoV3D_dict[pdb]['H3']},
                         'light':{'chain':lc,
                                  'CDR1':CoV3D_dict[pdb]['L1'],
                                  'CDR2':CoV3D_dict[pdb]['L2'],
                                  'CDR3':CoV3D_dict[pdb]['L3']},
                         'antigen':sele_comp[1][i]}
            union_dict['CoV3D'][pdb_name].append(comp_info)

        union_dict['Pre+'][pdb_name] = union_dict['CoV3D'][pdb_name] 
     
    else: # CoV3D alone
        union_dict['CoV3D'][pdb_name] = []
        for i, hc in enumerate(CoV3D_dict[pdb]['H']):
            lc = CoV3D_dict[pdb]['L'][i]
            comp_info = {'heavy':{'chain':hc,
                                  'CDR1':CoV3D_dict[pdb]['H1'],
                                  'CDR2':CoV3D_dict[pdb]['H2'],
                                  'CDR3':CoV3D_dict[pdb]['H3']},
                         'light':{'chain':lc,
                                  'CDR1':CoV3D_dict[pdb]['L1'],
                                  'CDR2':CoV3D_dict[pdb]['L2'],
                                  'CDR3':CoV3D_dict[pdb]['L3']}}
            union_dict['CoV3D'][pdb_name].append(comp_info)
        CoV3D_alone_list.append(pdb_name)

for pdb in union_dict['Pre'].keys():
    union_dict['Pre+'][pdb] = union_dict['Pre'][pdb]
for pdb in union_dict['Pre+'].keys():
    union_dict['All'][pdb] = union_dict['Pre+'][pdb]
for pdb in union_dict['CoV3D'].keys():
    if not pdb in union_dict['All'].keys():
        union_dict['All'][pdb] = union_dict['CoV3D'][pdb]

for set_kind in ['All', 'Pre', 'Pre+', 'CoV3D']:
    pdb_num = len(union_dict[set_kind].keys())
    comp_num = 0
    for pdb in union_dict[set_kind].keys():
        comp_num += len(union_dict[set_kind][pdb])
    print('%s: %d pbds, %d complexes.'%(set_kind, pdb_num, comp_num))

print('%d pdbs in CoV3D without known antigen chains.'%len(CoV3D_alone_list))

_save = dict_save(union_dict, '../Information/CDR_info/CDR_info.pickle')
_save = dict_save(CoV3D_alone_list, '../Information/CDR_info/CoV3D_unknown_antigen.list')
