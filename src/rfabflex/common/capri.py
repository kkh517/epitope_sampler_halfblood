import copy
import numpy as np
from collections import defaultdict
import torch

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

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}

# full sc atom representation (Nx14)
bb_idx = {" N  " : 0, " CA " : 1, " C  " : 2, " O  " : 3}
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
    ]

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
    rmsd = np.sqrt(max(0.0, (X_norm + Y_norm - 2.0 * eigl[-1]))/n)
    return rmsd, (sT,sU)    

class PDB:
    def __init__ (self, pdb_fn):
        self.pdb_fn = pdb_fn
    
    def read(self):
        sequence = defaultdict(str)
        coord_bb = defaultdict(dict)  #chain: {idx: coordinate [3]}
        coord_all_atom = defaultdict(dict) #chain: {idx: coordinate [14, 3]}
        idx_atm = defaultdict(lambda: defaultdict(list))
        coord_bb = defaultdict(defaultdict)
        idx_total = defaultdict(set)
        with open(self.pdb_fn) as f_pdb:
            lines = f_pdb.readlines()
            for line in lines:
                if line.startswith('ATOM'):
                    chain = line[21]
                    #xyz = np.full((14, 3), np.nan, dtype=np.float32)
                    resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
                    idx_total[chain].add(resNo)
                    if resNo not in coord_all_atom[chain]:
                        coord_all_atom[chain][resNo] = np.full((14, 3), np.nan, dtype=np.float32)
                    for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]][:14]):
                        if tgtatm == atom:
                            coord_all_atom[chain][resNo][i_atm, :] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    #coord_all_atom[chain][resNo] = xyz
                    for i in [' CA ', ' N  ', ' C  ', ' O  ']:
                        if atom == i:
                            idx_atm[chain][i].append(resNo)
                    if atom == ' CA ':
                        sequence[chain] += aa
                    if atom == ' CA ' or atom == ' N  ' or atom == ' C  ' or atom == ' O  ':
                        if resNo not in coord_bb[chain]:
                            coord_bb[chain][resNo] = np.full((4, 3), np.nan, dtype = np.float32)
                        coord_bb[chain][resNo][bb_idx[atom], : ] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        
        return sequence, idx_total, idx_atm, coord_bb, coord_all_atom       
# [L, 14, 3]                
def get_interface_idx(coord, idx, receptor_chain, ligand_chain, device='cpu', iface_cutoff=10.0):
    chain_order = []
    receptor_length = 0
    ligand_length = 0
    for i in receptor_chain:
        chain_order.append(i)
        receptor_length += len(idx[i])
    for i in ligand_chain:
        chain_order.append(i)
        ligand_length += len(idx[i])
    
    xyz_tot = []
    idx_match = defaultdict(dict)
    count = 0
    for i in chain_order:
        xyz_chain = np.full((len(idx[i]), 14, 3), np.nan, dtype=np.float32)
        for j, k in enumerate(idx[i]):
            xyz_chain[j, :, :] = coord[i][k]
        xyz_chain = torch.from_numpy(xyz_chain)
        xyz_tot.append(xyz_chain)
        idx_match[i] = dict(zip(list(range(count, count+len(idx[i]))), idx[i]))
        count += len(idx[i])
    xyz_tot = torch.cat(xyz_tot, dim = 0)
    xyz_tot = xyz_tot.to(device=device)    #[total_len, 14, 3]
    
    dist = xyz_tot[:, None, :, None, :] - xyz_tot[None, :, None, :, :]
    dist = (dist**(2)).sum(dim = -1)
    dist = (dist)**(0.5) #[L, L, 14, 14]
    dist = dist.view(*dist.shape[:2], -1)
    dist = torch.nan_to_num(dist, nan=100.0)
    dist = torch.min(dist, dim = -1)[0]
    dist_dict = defaultdict(dict)
#    for i, j in enumerate(dist):
#        print(i, j)
#        for k, v in idx_match.items():  #A:{0:447, 1:448..}
#            print(k, v)
#            if i in v.keys():
#                dist_dict[k] = j[v[i]]
#                break
    mask = torch.le(dist, iface_cutoff) #[L, L]
    mask[:receptor_length, :receptor_length] = False #False for intra region
    mask[receptor_length:, receptor_length:] = False
    
    interface_lists = torch.unique(torch.where(mask==True)[0])
    interface_pair = torch.where(mask==True)
    
    interface_final = defaultdict(set)
    interface_pair_list = set()
    for i in interface_lists:
        i = i.item()
        for k, v in idx_match.items(): #k:chain v:{idx1:resno1, idx2:resno2,...}
            if i in v.keys():
                interface_final[k].add(v[i])
                
    for i, j in zip(interface_pair[0], interface_pair[1]):
        i = i.item()
        j = j.item()
        if i < j:
            continue
        chain_i, chain_j = None, None
        idx_i, idx_j = None, None
        for k, v in idx_match.items():
            if i in v:
                chain_i = k
                idx_i = v[i]
            if j in v:
                chain_j = k
                idx_j = v[j]
        interface_pair_list.add((chain_i, idx_i, chain_j, idx_j))                
    
    return interface_final, interface_pair_list

def calc_rmsd(ref, model, R, t):
    assert len(ref) == len(model)
    length = len(ref)
    ref = np.array(ref).transpose() #(3, n)
    model = np.array(model).transpose() #(3, n)
    t = t.reshape(3, 1)
    aligned_model = np.einsum('ij, jk -> ik', R, model) + t
    rmsd = np.sqrt(((aligned_model - ref)**2).sum(0).sum()/length)
    return rmsd

def get_capri(model_pdb, ref_pdb, receptor_chain, ligand_chain):
    
    model_seq, model_idx_tot, model_idx_atm, model_coord_bb, model_coord_all = PDB(model_pdb).read()
    ref_seq, ref_idx_tot, ref_idx_atm, ref_coord_bb, ref_coord_all = PDB(ref_pdb).read()
    
    # should only consider overlapped residues (reference does not have to be complete)
    idx_overlap = defaultdict(dict)
    for ch in model_idx_atm.keys():
        for atm in model_idx_atm[ch].keys():
            overlap = list(set(model_idx_atm[ch][atm]) & set(ref_idx_atm[ch][atm]))
            idx_overlap[ch][atm] = overlap
    #print(idx_overlap)
    model_rec_bb, ref_rec_bb = [], []
    model_lig_bb, ref_lig_bb = [], []
    

    for i in receptor_chain:
        for atm in idx_overlap[i].keys():
            for idx in idx_overlap[i][atm]:
                model_rec_bb.append(model_coord_bb[i][idx][bb_idx[atm]])
                ref_rec_bb.append(ref_coord_bb[i][idx][bb_idx[atm]])
    
    for i in ligand_chain:
        for atm in idx_overlap[i].keys():
            for idx in idx_overlap[i][atm]:
                model_lig_bb.append(model_coord_bb[i][idx][bb_idx[atm]])
                ref_lig_bb.append(ref_coord_bb[i][idx][bb_idx[atm]])
    
    model_rec_bb = np.array(model_rec_bb)
    ref_rec_bb = np.array(ref_rec_bb)
    model_lig_bb = np.array(model_lig_bb)
    ref_lig_bb = np.array(ref_lig_bb)
    
    model_rec_bb = model_rec_bb.reshape(-1, 3)
    ref_rec_bb = ref_rec_bb.reshape(-1, 3)
    model_lig_bb = model_lig_bb.reshape(-1, 3)
    ref_lig_bb = ref_lig_bb.reshape(-1, 3)
    
    #print('model_shape', model_rec_bb.shape)
    #print('ref_shape', ref_rec_bb.shape)
    assert model_rec_bb.shape == ref_rec_bb.shape
    assert model_lig_bb.shape == ref_lig_bb.shape
    
    #1. Calculate l-rmsd
    # Get receptor aligned R, T matrix
    receptor_rmsd, (t_rec, R_rec) = ls_rmsd(model_rec_bb, ref_rec_bb)
    l_rmsd = calc_rmsd(ref_lig_bb, model_lig_bb, R_rec, t_rec)
    
    #2. Calculate i-rmsd
    # Get interface region of refernce 
    model_interface_bb = []
    ref_interface_bb = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    ref_idx_interface, ref_pair_interface = get_interface_idx(ref_coord_all, ref_idx_tot, receptor_chain, ligand_chain, \
            iface_cutoff = 10.0, device = device)
    
    count_ori = defaultdict(int)
    for k, v in ref_idx_interface.items(): #k: chain #v: interface residue index
        sort_idx = sorted(v)
        for atm in idx_overlap[k].keys(): #if both exists in receptor and ligan
            for idx in sort_idx:
                if idx in idx_overlap[k][atm]:
                    model_interface_bb.append(model_coord_bb[k][idx][bb_idx[atm]])
                    ref_interface_bb.append(ref_coord_bb[k][idx][bb_idx[atm]])

    model_interface_bb = np.array(model_interface_bb)
    ref_interface_bb = np.array(ref_interface_bb)
    
    model_interface_bb = model_interface_bb.reshape(-1, 3)
    ref_interface_bb = ref_interface_bb.reshape(-1, 3)
    
    i_rmsd, (t_inf, R_inf) = ls_rmsd(model_interface_bb, ref_interface_bb)
    
    model_coord_all_overlap = defaultdict(dict)
    for ch, idxs in ref_coord_all.items():
        for idx in idxs:
            model_coord_all_overlap[ch][idx] = model_coord_all[ch][idx]

    #3. Calculate fNAT
    ref_idx_interface, ref_pair_interface = get_interface_idx(ref_coord_all, ref_idx_tot, receptor_chain, ligand_chain, iface_cutoff = 5.0)
    model_idx_interface, model_pair_interface = get_interface_idx(model_coord_all_overlap, ref_idx_tot, receptor_chain, ligand_chain, iface_cutoff = 5.0)
    
    pair_both = set(ref_pair_interface) & set(model_pair_interface)
    f_nat = float(len(pair_both))/len(set(ref_pair_interface))
    
    return l_rmsd, i_rmsd, f_nat
