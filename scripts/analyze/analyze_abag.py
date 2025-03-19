import string
import sys
from pathlib import Path
from collections import defaultdict
from glob import glob
import numpy as np
import copy
import json

to1letter = {
    "UNK": "X",
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
class QuaternionBase: #from Galaxy.core.quaternion
    def __init__(self):
        self.q = []
    def __repr__(self):
        return self.q
    def rotate(self):
        if 'R' in dir(self):
            return self.R
        #
        self.R = np.zeros((3,3))
        #
        self.R[0][0] = self.q[0]**2 + self.q[1]**2 - self.q[2]**2 - self.q[3]**2
        self.R[0][1] = 2.0*(self.q[1]*self.q[2] - self.q[0]*self.q[3])
        self.R[0][2] = 2.0*(self.q[1]*self.q[3] + self.q[0]*self.q[2])
        #
        self.R[1][0] = 2.0*(self.q[1]*self.q[2] + self.q[0]*self.q[3])
        self.R[1][1] = self.q[0]**2 - self.q[1]**2 + self.q[2]**2 - self.q[3]**2
        self.R[1][2] = 2.0*(self.q[2]*self.q[3] - self.q[0]*self.q[1])
        #
        self.R[2][0] = 2.0*(self.q[1]*self.q[3] - self.q[0]*self.q[2])
        self.R[2][1] = 2.0*(self.q[2]*self.q[3] + self.q[0]*self.q[1])
        self.R[2][2] = self.q[0]**2 - self.q[1]**2 - self.q[2]**2 + self.q[3]**2
        return self.R
    
class QuaternionQ(QuaternionBase):
    def __init__(self, q):
        self.q = q

def change_chain(name, pdb):
    # change the chain name in the output file by name in the input file (e.g. 1a0m_H_L_A.pdb -> 1a0m_H_L_A_rechain.pdb)
    chains = []
    _, hchain, lchain, agchain = name.split('_')
    for i in [hchain, lchain, agchain]:
        if i != '#':
            for j in i:
                chains.append(j)
                
    chain_alphabet = list(string.ascii_uppercase)
    chain_dict = dict(zip(chain_alphabet, chains))
    newline = []
    with open(pdb) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                newline.append(line[:21] + chain_dict[line[21]] + line[22:])
    new_filename = Path(pdb).stem + '_rechain.pdb'
    f_out = open(new_filename, 'w')
    f_out.writelines(newline)
    f_out.close()
    
def AF_get_chain_dict(name):
    #AF_dir = '/home/yubeen/KIDDS/benchmark/positive/AF_Multimer'
    AF_dir = '/home/yubeen/KIDDS/benchmark/positive/AF_Monomer/antibody'
    chain_json = open(f'{AF_dir}/{name}/msas/chain_id_map.json')
    json_dict = json.load(chain_json)
    chain_dict = {}
    for k, v in json_dict.items():
        chain_dict[k] = v['description'].strip('_')[-1]
    return chain_dict
def get_chothia_numbered_rechain(name, output_pdb, chothia_numbered_pdb, seq_numbered_pdb):
    # change the chain name and chain numbering to chothia
    chains = []
    _, hchain, lchain, agchain = name.split('_')
    
    #for AF-based
    #chain_dict = AF_get_chain_dict(name)
    #print(chain_dict)
    
    # for RF
    for i in [hchain, lchain, agchain]:
        if i != '#':
            for j in i:
                chains.append(j)
                
    chain_alphabet = list(string.ascii_uppercase)
    chain_dict = dict(zip(chain_alphabet, chains))
    
    #for IgFold (H, L for heavy and light chain)
    #chain_dict = {'H': hchain, 'L': lchain}
    
    # for alphafold (A, B for heavy and light chain)
    #chain_dict = {'A': hchain, 'B': lchain, 'C': agchain}
    
    #print(chain_dict)
    
    seq_pdb = ''
    seq_chothia = ''
    seq_seq = ''
    chain_resno_pdb = defaultdict(list)
    chain_resno_chothia = defaultdict(list)
    chain_resno_seq = defaultdict(list)
    
    #print(output_pdb)
    with open(output_pdb) as f:
        lines = f.readlines()
        #print(lines)
        for line in lines:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                if (line[22:26].strip() + line[26].strip()) in chain_resno_pdb[line[21]]:
                    continue
                seq_pdb += to1letter[line[17:20]]
                resno = line[22:26].strip() + line[26].strip()
                chain_resno_pdb[chain_dict[line[21]]].append((resno,  to1letter[line[17:20]]))
    with open(chothia_numbered_pdb) as f:
        lines2 = f.readlines()
        for line in lines2:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                if (line[22:26].strip() + line[26].strip()) in chain_resno_chothia[line[21]]:
                    continue
                seq_chothia += to1letter[line[17:20]]
                resno = line[22:26].strip() + line[26].strip()
                chain_resno_chothia[line[21]].append((resno,  to1letter[line[17:20]]))
    
    with open(seq_numbered_pdb) as f:
        lines3 = f.readlines()
        for line in lines3:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                if (line[22:26].strip() + line[26].strip()) in chain_resno_seq[line[21]]:
                    continue
                seq_seq += to1letter[line[17:20]]
                resno = line[22:26].strip() + line[26].strip()
                chain_resno_seq[line[21]].append((resno,  to1letter[line[17:20]]))
    
    #print('seq_pdb', seq_pdb)
    #print('seq_chothia', seq_chothia)
    #print('seq_seq', seq_seq)
    assert seq_chothia == seq_seq    
    
    chain_resno_to_chain_resno = defaultdict(dict)
    for k, v in chain_resno_seq.items():
        for i, vs in enumerate(v):
            if vs[1] == chain_resno_chothia[k][i][1]:
                chain_resno_to_chain_resno[k][vs[0]] = chain_resno_chothia[k][i][0]
                #chain_resno_to_chain_resno[k][chain_resno_chothia[k][i][0]] = vs[0] 
            else:
                sys.exit()
    #print(chain_resno_to_chain_resno)
    # for AF2_unpaired (has index gap)
#    new_dict = {}
#    print(len(chain_resno_pdb[hchain]))
#    for k, v in chain_resno_to_chain_resno[lchain].items():
#        new_key = len(chain_resno_pdb[hchain]) + int(k) + 200
#        new_dict[str(new_key)] = v
#    chain_resno_to_chain_resno[lchain] = new_dict
#    print(chain_resno_to_chain_resno)
    newline_pdb = []
    with open(output_pdb) as f:
        f.seek(0)
        lines = f.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                chain_name = chain_dict[line[21]]
                if line[22:26].strip() in chain_resno_to_chain_resno[chain_name]: 
                    new_resno = chain_resno_to_chain_resno[chain_name][line[22:26].strip()]
                    new_chain = chain_name
                    if new_resno[-1].isalpha():
                        newline_pdb.append(f'{line[:21]}{new_chain}{new_resno:>5s}{line[27:]}')
                    else:
                        newline_pdb.append(f'{line[:21]}{new_chain}{new_resno:>4s} {line[27:]}')
            else:
                newline_pdb.append(line)
    
    new_filename = f'{Path(output_pdb).parent}/{Path(output_pdb).stem}_renum.pdb'
    f_out = open(new_filename, 'w')
    f_out.writelines(newline_pdb)
    f_out.close()
    

    
def get_loop_rmsd(chothia_numbered_ref, model, name):
    hchain = name.split('_')[1]
    lchain = name.split('_')[2]

    H = [list(range(26, 33)), list(range(52, 57)), list(range(95, 103))]
    L = [list(range(24, 35)), list(range(50, 57)), list(range(89, 98))]
    H = sum(H, [])
    L = sum(L, [])
    loop_dict_H = defaultdict(list)
    loop_dict_L = defaultdict(list)
    loop_dict_H['H1'] = list(range(26, 33))
    loop_dict_H['H2'] = list(range(52, 57))
    loop_dict_H['H3'] = list(range(95, 103))
    loop_dict_L['L1'] = list(range(24, 35))
    loop_dict_L['L2'] = list(range(50, 57))
    loop_dict_L['L3'] = list(range(89, 98))
    
    # first get coordinates without loop & calculate rotation matrix
    ref_coord = []
    model_coord = []
    loop_coord_H_ref, loop_coord_H_model = [], []
    loop_coord_L_ref, loop_coord_L_model = [], []
    loop_coord_dict_H_ref = defaultdict(list)
    loop_coord_dict_H_model = defaultdict(list)
    loop_coord_dict_L_ref = defaultdict(list)
    loop_coord_dict_L_model = defaultdict(list)
    
    with open(chothia_numbered_ref) as f_ref:
        lines = f_ref.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                if line[16].strip() != '' and line[16].strip() != 'A':
                    continue
                if line[21] == hchain and line[12:16].strip() == 'CA':
                    resno = int(line[22:26])
                    if resno in H:
                        loop_coord_H_ref.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                        for k, v in loop_dict_H.items():
                            if resno in v:
                                loop_coord_dict_H_ref[k].append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                                
                    else:
                        ref_coord.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                
    with open(chothia_numbered_ref) as f_ref:
        f_ref.seek(0)
        lines = f_ref.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                if line[16].strip() != '' and line[16].strip() != 'A':
                    continue
                       
                if line[21] == lchain and line[12:16].strip() == 'CA':
                    resno = int(line[22:26])
                    if resno in L:
                        loop_coord_L_ref.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                        for k, v in loop_dict_L.items():
                            if resno in v:
                                loop_coord_dict_L_ref[k].append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                    else:
                        ref_coord.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))   
    with open(model) as f_model:
        lines = f_model.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                if line[16].strip() != '' and line[16].strip() != 'A':
                    continue
                if line[21] == hchain and line[12:16].strip() == 'CA':
                    resno = int(line[22:26])
                    if resno in H:
                        loop_coord_H_model.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                        for k, v in loop_dict_H.items():
                            if resno in v:
                                loop_coord_dict_H_model[k].append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                        
                    else:
                        model_coord.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                

    with open(model) as f_model:
        f_model.seek(0)
        lines = f_model.readlines()
        for line in lines:
            if line.startswith('ATOM'):
                if line[16].strip() != '' and line[16].strip() != 'A':
                    continue
                if line[21] == lchain and line[12:16].strip() == 'CA':
                    resno = int(line[22:26])
                    if resno in L:
                        loop_coord_L_model.append(np.array([float(line[30:38]), \
                                                    float(line[38:46]), \
                                                    float(line[46:54])]))                    
                        for k, v in loop_dict_L.items():
                            if resno in v:
                                loop_coord_dict_L_model[k].append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
                    else:
                        model_coord.append(np.array([float(line[30:38]), \
                                                  float(line[38:46]), \
                                                  float(line[46:54])]))
    
    assert len(model_coord[0]) == len(ref_coord[0])
    model_coord = np.array(model_coord) #(n, 3)
    ref_coord = np.array(ref_coord) #(n, 3)
    frame_rmsd, (t, R) = ls_rmsd(model_coord, ref_coord)
    
    # get whole H loop rmsd
    loop_H_rmsd = calc_rmsd(loop_coord_H_ref, loop_coord_H_model, R, t)
    # get whole L loop rmsd
    loop_L_rmsd = calc_rmsd(loop_coord_L_ref, loop_coord_L_model, R, t)
    
    # get each loop rmsd
    loop_H_rmsd_dict = defaultdict(float)
    loop_L_rmsd_dict = defaultdict(float)
    for k, v in loop_coord_dict_L_ref.items():
        loop_L_rmsd_dict[k] = calc_rmsd(v, loop_coord_dict_L_model[k], R, t)
    for k, v in loop_coord_dict_H_ref.items():
        loop_H_rmsd_dict[k] = calc_rmsd(v, loop_coord_dict_H_model[k], R, t)
    return loop_H_rmsd_dict, loop_L_rmsd_dict, loop_H_rmsd, loop_L_rmsd, frame_rmsd

def calc_rmsd(ref, model, R, t):
    assert len(ref) == len(model)
    length = len(ref)
    ref = np.array(ref).transpose() #(3, n)
    model = np.array(model).transpose() #(3, n)
    t = t.reshape(3, 1)
    aligned_model = np.einsum('ij, jk -> ik', R, model) + t
    rmsd = np.sqrt(((aligned_model - ref)**2).sum(0).sum()/length)
    return rmsd

def ls_rmsd(_X, _Y): #from Galaxy.utils.subPDB
    # Kabsch algorithm & turn into quaternion
    
    X = copy.copy(_X) #(n, 3)
    Y = copy.copy(_Y) #(n, 3)
    n = float(len(X))

    X_cntr = X.transpose().sum(1)/n
    Y_cntr = Y.transpose().sum(1)/n
    X -= X_cntr
    Y -= Y_cntr
    Xtr = X.transpose()
    Ytr = Y.transpose()
    X_norm = (Xtr*Xtr).sum()
    Y_norm = (Ytr*Ytr).sum()
    
    Rmatrix = np.zeros(9).reshape(3,3)
    for i in range(3):
        for j in range(3):
            Rmatrix[i][j] = Xtr[i].dot(Ytr[j])
    
    S = np.zeros(16).reshape((4,4))
    S[0][0] =  Rmatrix[0][0] + Rmatrix[1][1] + Rmatrix[2][2]
    S[1][0] =  Rmatrix[1][2] - Rmatrix[2][1]
    S[0][1] =  S[1][0]
    S[1][1] =  Rmatrix[0][0] - Rmatrix[1][1] - Rmatrix[2][2]
    S[2][0] =  Rmatrix[2][0] - Rmatrix[0][2]
    S[0][2] =  S[2][0]
    S[2][1] =  Rmatrix[0][1] + Rmatrix[1][0]
    S[1][2] =  S[2][1]
    S[2][2] = -Rmatrix[0][0] + Rmatrix[1][1] - Rmatrix[2][2]
    S[3][0] =  Rmatrix[0][1] - Rmatrix[1][0]
    S[0][3] =  S[3][0]
    S[3][1] =  Rmatrix[0][2] + Rmatrix[2][0]
    S[1][3] =  S[3][1]
    S[3][2] =  Rmatrix[1][2] + Rmatrix[2][1]
    S[2][3] =  S[3][2]
    S[3][3] = -Rmatrix[0][0] - Rmatrix[1][1] + Rmatrix[2][2]
    #
    eigl,eigv = np.linalg.eigh(S)
    q = eigv.transpose()[-1]
    sU = QuaternionQ(q).rotate()
    sT = Y_cntr - sU.dot(X_cntr)
    #
    rmsd = np.sqrt(max(0.0, (X_norm + Y_norm - 2.0*eigl[-1]))/n)
    return rmsd, (sT,sU)    
#def get_capri_criteria(ref):
                       
                
if __name__ == '__main__':
    #RoseTTAFold
    #files = glob('/home/yubeen/ab_ag_prediction/RF2_230104/developing/inference/trained/abag_test/*/*_final_1_unrelaxed.pdb')
#    for p in range(16):
    files = glob(f'/home/yubeen/ab_ag_prediction/RF2_230104/inference_0607/abag_test_best_3/*/*_final_1_unrelaxed.pdb')
    DB = '/home/yubeen/ab_ag_prediction/training_set/ab_ag/DB_final'
    DB_test = '/home/yubeen/ab_ag_prediction/test_set/DB_final_test_67'
    DB_test_42 = '/home/yubeen/ab_ag_prediction/test_set/DB_final_test_42/DB_final'
    # IgFold
    #files = glob('/home/yubeen/KIDDS/benchmark/positive/IgFold/????_?_?_?.pdb')
    # AF_multimer
    #files = glob('/home/yubeen/KIDDS/benchmark/positive/AF_Multimer/*/ranked_0.pdb') 
    # AF_multimer_mono (only antibody)
    #files = glob('/home/yubeen/KIDDS/benchmark/positive/AF_Monomer/antibody/*/ranked_0.pdb')
    # AF2_unpaired
    #files = glob('/home/yubeen/KIDDS/benchmark/positive/AF2_unpaired/*/*_unrelaxed_model_1.pdb')
    
    print(len(files))
    f_out_final = open(f'loop_rmsd_ab_RF_best_0611.dat', 'w')
    f_out_final.write('ID frame H_total H1 H2 H3 L_total L1 L2 L3\n')
    count = 0
    sum_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = float)
    H3_value = []
    for i in files:
        count += 1
        name = Path(i).parent.stem
        print(i)
        print(name)
        hchain = name.split('_')[1]
        lchain = name.split('_')[2]
        if lchain == '#':
            continue
        agchain = name.split('_')[3]
        if Path(i).parts[-3] == 'abag_test_67':
            print(i)
            chothia_renumbered_file = f'{DB_test}/{name}/{name}.pdb'
            seq_renumbered_file = f'{DB_test}/{name}/{name}_renum.pdb'
        elif Path(i).parts[-3] == 'abag_test_42':
            print(i)
            chothia_renumbered_file = f'{DB_test_42}/{name}/{name}.pdb'
            seq_renumbered_file = f'{DB_test_42}/{name}/{name}_renum.pdb'
 
        else:
            chothia_renumbered_file = f'{DB}/{name}/{name}.pdb'
            seq_renumbered_file = f'{DB}/{name}/{name}_renum.pdb'
        get_chothia_numbered_rechain(name, i, chothia_renumbered_file, seq_renumbered_file)
        print('success', name)
        if Path(i).parts[-3] == 'abag_test':
            ref_file = f'{DB_test}/{name}/{name}.pdb'
        elif Path(i).parts[-3] == 'abag_test_42':
            ref_file = f'{DB_test_42}/{name}/{name}.pdb'
        else:
            ref_file = f'{DB}/{name}/{name}.pdb'
        model_file = Path(Path(i).parent, f'{Path(i).stem}_renum.pdb')
        print(model_file)
        loop_H_rmsd_dict, loop_L_rmsd_dict, loop_H_rmsd, loop_L_rmsd, frame_rmsd = \
            get_loop_rmsd(ref_file, model_file, name)
        
        H1_rmsd, H2_rmsd, H3_rmsd  = loop_H_rmsd_dict['H1'], loop_H_rmsd_dict['H2'], loop_H_rmsd_dict['H3']
        L1_rmsd, L2_rmsd, L3_rmsd  = loop_L_rmsd_dict['L1'], loop_L_rmsd_dict['L2'], loop_L_rmsd_dict['L3']
        sum_array += np.array([frame_rmsd, loop_H_rmsd, H1_rmsd, H2_rmsd, H3_rmsd, loop_L_rmsd, L1_rmsd, L2_rmsd,\
            L3_rmsd])
        H3_value.append(H3_rmsd)
        
        f_out_final.write(f'{name} {frame_rmsd:.4f} {loop_H_rmsd:.4f} {H1_rmsd:.4f} {H2_rmsd:.4f} {H3_rmsd:.4f}\
 {loop_L_rmsd:.4f} {L1_rmsd:.4f} {L2_rmsd:.4f} {L3_rmsd:.4f}\n')
    f_out_final.write(f'Average    ')
    sum_array = sum_array/count
    for i in sum_array:
        f_out_final.write(f'{i:.4f} ')
    f_out_final.close()
    print(H3_value)
    H3_value = np.array(H3_value)
    print(count)
    print(len(H3_value[H3_value < 1.5])/count)
    print(len(H3_value[H3_value < 2.0])/count)

