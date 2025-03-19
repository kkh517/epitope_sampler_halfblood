#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, defaultdict
# from rfabflex.inference.featurizing import TemplFeaturize, merge_a3m_antibody, MSABlockDeletion
import importlib
import json
import rfabflex.common.util as util
from rfabflex.data.parsers import parse_a3m, parse_pdb_antibody
from rfabflex.data.ffindex import *
from rfabflex.common.chemical import INIT_CRDS
from rfabflex.common.kinematics import xyz_to_t2d, get_init_xyz_compl
from rfabflex.model.RoseTTAFoldModel import RoseTTAFoldModule
sys.path.append('/home/kkh517/Github/rf-abag-templ/scripts/training')
from data_loader_kh import(
    featurize_antibody_complex_kh,
    CustomTemplFeaturize_kh,
    get_epi_full,
    merge_a3m_antibody,
    MSABlockDeletion,
    get_tplts,
    MSAFeaturize,
)


model_dir = '/home/yubeen/ab_ag_prediction/RF2_230104/inference_0607/models'
def get_msa(a3mfilename, max_seq=8000):
    msa, ins = parse_a3m(a3mfilename, max_seq=max_seq)
    return {"msa": torch.Tensor(msa), "ins": torch.Tensor(ins)}


def get_args():
    DB = "/home/yubeen/RF2/pdb100_2021Mar03/pdb100_2021Mar03"
    import argparse
    parser = argparse.ArgumentParser(
        description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument(
        "-item",
        required=True,
        help="complex name [PDB]_[H]_[L]_[AG] (e.g. 1A2K_H_L_A)")
    parser.add_argument(
        "-a3m_fn",
        required=True,
        nargs='+',
        help="MSA file for each subunit (if you want to enrich pMSA w/ unpaired MSA)")
    parser.add_argument("-out_prefix", required=True,
                        help="prefix for output file")
    parser.add_argument("-hhr_fn", default=[], nargs='+',
                        help="Input hhr file")
    parser.add_argument("-atab_fn", default=[], nargs='+',
                        help="Input atab file")
    parser.add_argument("-templ_pt", default=[], nargs='+',
                        help="Input featurized template file")
    parser.add_argument("-db", default=DB, required=False,
                        help="HHsearch database [%s]" % DB)
    parser.add_argument("-use_cpu", default=False, action='store_true',
                        help="Use CPU")
    parser.add_argument("-max_cycle", default=4, type=int,
                        help="maximum cycle")
    parser.add_argument("-calc_loop_loss", default=False, action='store_true',
                        help="Calculate loop FAPE loss")
    parser.add_argument("-script_dir", required=True, 
                        help="script path for models.json and model weight")
    parser.add_argument("-n_templ", required=False, type= int, default = 0,
                        help="number of template to use")
    args = parser.parse_args()
    return args

def read_inputs(
        a3m_fn_s, 
        tplt_params_in, 
        homo_list, 
        chain_properties, 
        n_templ=0, 
        max_seq=10000,
        device='cuda:0',
        item=None,):

    # merge all MSAs
    a3m_s = list()
    for a3m_fn in a3m_fn_s:
        a3m = get_msa(a3m_fn, max_seq=8000)
        a3m_s.append(a3m)
    
    a3m = merge_a3m_antibody(a3m_s, homo_list)

    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    N_final_dict = a3m['N_final_dict']

    a3m = merge_a3m_antibody(a3m_s, homo_list)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    print('msa_orig', msa.shape)
    #print('msa_orig', msa_orig.shape)
    N_final_dict = a3m['N_final_dict']

    if len(msa) > 5:
        msas, inss = [], []
        msas.append(msa[None, 0])
        inss.append(ins[None, 0])
        for i in np.arange(len(N_final_dict)):
            sN, eN = N_final_dict[i][0], N_final_dict[i][1]
            if eN - sN > 50:
                msa_del, ins_del = MSABlockDeletion(msa[sN:eN], ins[sN:eN])
                msas.append(msa_del)
                print('msa_del', msa_del.shape)
                inss.append(ins_del)
            else:
                msas.append(msa[sN:eN])
                inss.append(ins[sN:eN])
        msa = torch.cat(msas, dim=0)
        ins = torch.cat(inss, dim=0)
    
    sample = len(msa)
    if sample > 3000:
        msa_sel = torch.randperm(sample-1)[:2999]
        msa = torch.cat((msa[:1, :], msa[1:, :][msa_sel]), dim=0)
        ins = torch.cat((ins[:1, :], ins[1:, :][msa_sel]), dim=0)
    
    msa_orig = msa
    ins_orig = ins
    # #msa = a3m['msa']
        # if a3m['msa'].shape[0] > max_seq // len(a3m_fn_s):
        #     a3m['msa'], a3m['ins'] = MSABlockDeletion(a3m['msa'], a3m['ins'])
        # a3m_s.append(a3m)
    
    #query = np.concatenate([a3m_single['msa'][0]
    #                        for a3m_single in a3m_s])[None]  # (1, L)
    # a3m = merge_a3m_antibody(a3m_s, homo_list)
    # #print(a3m)
    # #for i in a3m['msa']:
    # #    print(i)
    # #print(a3m['L_s'])
    # msa_orig = a3m['msa'].long()
    # ins_orig = a3m['ins'].long()
    # sample = len(msa_orig)
    # if sample > 3000:
    #     msa_sel = torch.randperm(sample-1)[:2999]
    #     msa_orig = torch.cat((msa_orig[:1, :], msa_orig[1:, :][msa_sel]), dim=0)
    #     ins_orig = torch.cat((ins_orig[:1, :], ins_orig[1:, :][msa_sel]), dim=0)
 
    
    # tplts = []
    # for i in tplt_params_in:
    #     print('i',i)
    #     tplts.append(torch.load(i))
    print('tplt_param_in\n',tplt_params_in)
    tplts = get_tplts(item, tplt_params_in,validation=True, inference=True)

    L_s = a3m['L_s']
    print('L_s', L_s)
    print('tplts', tplts)
    #print(len(tplts))
    template_features = []
    for i, tplt in enumerate(tplts):
        # template_features.append(TemplFeaturize(tplt, L_s[i], offset=0, npick=n_templ))
        template_feature = CustomTemplFeaturize_kh(tplts, tplt, L_s, i, None, None, print_log=True)  # print_log=True) for debuggin
        if template_feature == None: continue
        template_features.append(template_feature)

    xyz_t = []
    for i, template_feature in enumerate(template_features):
        seq_tmp = torch.argmax(template_feature[1][0][:, :21], dim = 1)
        if i == 0:
            xyz_t.append(template_feature[0])
        else:
            xyz_t.append(util.random_rot_trans(template_feature[0]))
            
    xyz_t = torch.cat(xyz_t, dim=1)
    t1d = torch.cat([template_feature[1] for template_feature in template_features], dim = 1)
    mask_t = torch.cat([template_feature[2] for template_feature in template_features], dim = 1)    
      
    idx = torch.arange(sum(L_s))
    L_start = []
    L_end = []
    L_start.append(0)
    temp = 0
    for i in L_s:
        temp += i
        L_start.append(temp)
        L_end.append(temp)
    L_start = L_start[:-1]


    for (i, j) in zip(L_start, L_end):
        count = 0
        idx[i:j] += count
        count += 200
    
    
    same_chain = torch.zeros((sum(L_s), sum(L_s))).long()
    for (i, j) in zip(L_start, L_end):
        same_chain[i:j, i:j] = 1

    same_chain = same_chain.unsqueeze(0).to(device)
    # print(same_chain)
    count = 0
    for (i, j) in zip(L_start, L_end):
        idx[i:j] += count
        count += 100


    chain_idx = torch.zeros(sum(L_s)).long()
    for k, (i, j) in enumerate(zip(L_start, L_end)):
        chain_idx[i:j] = chain_properties[k]
    chain_idx = chain_idx.unsqueeze(0).to(device)
    print(chain_idx)
    xyz = xyz_t[0].clone().unsqueeze(0).to(device)
    #xyz = xyz_t[0].clone()
    mask_prev = mask_t[0].unsqueeze(0).clone().to(device)
    mask_recycle = mask_prev[:, :, :3].bool().all(dim=-1)
    mask_recycle = mask_recycle[:, :, None] * \
        mask_recycle[:, None, :]
    print('mask_recycle', mask_recycle.shape)
    print('same_chain', same_chain.shape)
    #mask_recycle = same_chain.float()[None] * mask_recycle.float()
    mask_recycle = same_chain.float() * mask_recycle.float()
    for i in mask_recycle:
        print(i)
    mask_recycle = mask_recycle.to(device)
    # template features
    xyz_t = xyz_t.float().unsqueeze(0).to(device)
    print('xyz_t', xyz_t.shape)
    t1d = t1d.float().unsqueeze(0).to(device)
    mask_t = mask_t.unsqueeze(0).to(device)
    print('t1d', t1d.shape)
    print('mask_t', mask_t.shape)
    
    mask_t_2d = mask_t[:, :, :, :3].all(dim=-1) #(B, T, L)
    mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None] #(B, T, L, L)
    print('mask_t_2d', mask_t_2d.shape)
    mask_t_2d = mask_t_2d.float() * same_chain.float()[:, None] #(B, T, L, L)
    t2d = xyz_to_t2d(xyz_t, mask_t_2d)
    
    assert item != None, "item should be given"

    pdb = parse_pdb_antibody(f'/public_data/ml/antibody/PDB_Ag_Ab/200430/pdb/{item}/{item}_renum.pdb', item, antibody=True)
    xyz_new = [torch.full((i, 14, 3), np.nan).float() for i in L_s]
    mask_new = [torch.full((i, 14), np.nan).float() for i in L_s]
    print(len(pdb['xyz']))
    for i, _ in enumerate(L_s):
        for j, idx in enumerate(pdb['idx'][i]):
            # xyz_new[i][idx-1, : , :] = torch.from_numpy(pdb['xyz'][i][j])
            # print(pdb['xyz'][i][j])
            # print(idx)
            # print(i, j, idx)
            # print(type(pdb['xyz'][i]))
            xyz_new[i][idx-1, :, :] = torch.from_numpy(pdb['xyz'][i][j])
            mask_new[i][idx-1, :] = pdb['mask'][i][j]

    xyz_new = torch.cat(xyz_new).unsqueeze(0) 
    mask_new = torch.cat(mask_new).unsqueeze(0)

    pdb['xyz'] = xyz_new
    pdb['mask'] = mask_new
    pdb['mask'] = torch.nan_to_num(pdb['mask'])

    xyz_true = INIT_CRDS.reshape(1,1,27,3).repeat(1,sum(L_s),1,1)
    mask_true = torch.full((1, sum(L_s), 27), False)
    xyz_true[:,:,:14] = pdb['xyz']
    xyz_true = torch.nan_to_num(xyz)
    mask_true[:,:,:14]= pdb['mask']

    if len(L_s) == 2:
        interface_split = L_s
    elif len(L_s) == 3:
        interface_split = [L_s[0]+L_s[1], L_s[2]]
    print('interface_split',interface_split)

    epitope_full = get_epi_full(xyz_true[0], mask_true[0], interface_split, item, cutoff = 10.0)
    print(f'{item} epitope_full', epitope_full)
    seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, sum(L_s))
    alpha, _, alpha_mask, _ = util.get_torsions(
        xyz_t.reshape(-1, sum(L_s), 27, 3),
        seq_tmp,
        util.torsion_indices.to(device),
        util.torsion_can_flip.to(device),
        util.reference_angles.to(device),
        mask_in = mask_t.reshape(-1, sum(L_s), 27))
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))

    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1, -1, sum(L_s), 10, 2)
    alpha_mask = alpha_mask.reshape(1, -1, sum(L_s), 10, 1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-
                        1).reshape(1, -1, sum(L_s), 30)

    # chain_idx = list()
    # for i_sub, L in enumerate(L_s):
    #     chain_idx.extend([i_sub] * L)
    # chain_idx = torch.tensor(chain_idx).long().to(device)
    # same_chain = (chain_idx[None] == chain_idx[:, None])[None]

    # initialize coordinates with first template
    xyz_t = xyz_t[:, :, :, 1]

    # print(msa_orig.shape)
    # print(ins_orig.shape)
    # print(xyz.shape)
    # print(xyz_t.shape)
    # print(t1d.shape)
    # print(t2d.shape)
    # print(alpha_t.shape)
    # print(L_s)

    # featurizeMSA
    # for i, msai in enumerate(msa_orig):
    #     seq, msa_seed_orig,msa_seed, msa_extra, mask_msa = MSAFeaturize(msai, ins_orig[i], loader_param, L_s =L_s[i])
    #     seq_list.append(seq)
    #     msa_
    loader_param={"MAXLAT":128, "MAXCYCLE":4, "MAXSEQ":1024}
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa_orig, ins_orig, loader_param, L_s =L_s)


    msa_latent = msa_seed.to(device)
    msa_full = msa_extra.to(device)
    # seq = seq.to(device)
    # idx = idx.to(device)
    
    return msa_latent, msa_full, (xyz, xyz_t, t1d, t2d, alpha_t, mask_recycle, mask_t_2d), L_s, chain_idx, epitope_full, same_chain, seq, idx



def check_bond_geom(pdb_fn=None, xyz=None, L_s=[], BOND_CUTOFF2=25.0):
    if pdb_fn is not None:
        BB_ATOMS = ["N", "CA", "C"]
        xyz = [[], [], []]
        # read_backbone atoms
        with open(pdb_fn) as fp:
            for line in fp:
                if not line.startswith("ATOM"):
                    continue
                atom = line[12:16].strip()
                if atom not in BB_ATOMS:
                    continue
                idx = BB_ATOMS.index(atom)
                xyz[idx].append([float(line[30:38]), float(
                    line[38:46]), float(line[46:54])])
        xyz = np.stack(xyz, axis=1)  # (L, 3, 3)
        xyz = torch.tensor(xyz)
    else:
        if xyz is None:
            sys.stderr.write(
                "ERROR: pdb or xyz coordinate should be given to check bond geometry\n")
            sys.exit(1)

    if len(L_s) < 1:
        L_s = [xyz.shape[0]]

    start_L = 0
    for L_sub in L_s:
        print(start_L, L_sub, xyz[start_L + 1:start_L + L_sub, 0].shape, \
            xyz[start_L:start_L + L_sub - 1, 2].shape)
        CN_dist = torch.square(
            xyz[start_L + 1:start_L + L_sub, 0] - xyz[start_L:start_L + L_sub - 1, 2])
        start_L += L_sub
        if (CN_dist > BOND_CUTOFF2).any():
            return True
    return False


def _run_model(
        model,
        out_prefix,
        msa_orig,
        ins_orig,
        templ,
        chain_idx,
        epitope_full,
        seq,
        idx,
        same_chain,
        L_s=[],
        device='cuda:0',
        max_iter=1,
        max_cycle=15,
        max_seq=10000):
    pdb_s = list()
    plddt_s = list()

    for i_iter in range(max_iter):
        start_time = time.time()
        sys.stdout.write("INFO: Trial %02d/%02d\n" % (i_iter, max_iter))
        # Run with template
        xyz_init, xyz_t, t1d, t2d, alpha_t, mask_recycle, mask_t = templ
        msa = msa_orig.long().to(device)  # (N, L)
        ins = ins_orig.long().to(device)
        # if msa_orig.shape[0] > max_seq:
        #     msa, ins = MSABlockDeletion(msa_orig, ins_orig)
        #     msa = torch.tensor(msa).long().to(device)  # (N, L)
        #     ins = torch.tensor(ins).long().to(device)
        # else:
        #     msa = torch.tensor(msa_orig).long().to(device)  # (N, L)
        #     ins = torch.tensor(ins_orig).long().to(device)

        # Run with insertion statistics
        prefix = "%s_%02d_w_ins" % (out_prefix, i_iter)
        if os.path.exists("%s_raw.pdb" % prefix):
            is_bond_violated = check_bond_geom(
                pdb_fn="%s_raw.pdb" %
                prefix, L_s=L_s)
            pdb_s.append(os.path.abspath("%s_raw.pdb" % prefix))
            if is_bond_violated:
                plddt_s.append(np.load("%s.npz" % prefix)['lddt'].mean() - 1.0)
            else:
                plddt_s.append(np.load("%s.npz" % prefix)['lddt'].mean())
        else:
            sys.stdout.write(
                "INFO: Running template-based prediction with insertion statistics\n")
            print('msa', msa.shape)
            print('ins', ins.shape)
            print('t1d', t1d.shape)
            print('t2d', t2d.shape)
            print('xyz_t', xyz_t.shape)
            print('xyz_init', xyz_init.shape)
            print('alpha_t', alpha_t.shape)
            print('chain_idx', chain_idx.shape)
            print('mask_recycle', mask_recycle.shape)
            print('mask_t', mask_t.shape)
            print('epitope_full', epitope_full.shape)
            # xyz_pred, plddt, prob_s, pae = model(
            #     msa_latent =msa, msa_full =ins, seq=seq,xyz = xyz_init, idx= idx, t1d = t1d, t2d=t2d, xyz_t = xyz_t,  alpha_t =alpha_t, mask_t = mask_t, chain_idx =chain_idx, same_chain = same_chain, mask_recycle = mask_recycle, epitope_info = epitope_full[None], ) #L_s=L_s, device=device, max_cycle=max_cycle)
            # print('msa device', msa.device)

            msa = msa.float() 
            ins = ins.to(msa.device) ; seq = seq.to(msa.device) ;  xyz_init = xyz_init.to(msa.device) ; xyz_t = xyz_t.to(msa.device) ; t1d = t1d.to(msa.device) ; t2d = t2d.to(msa.device) ; alpha_t = alpha_t.to(msa.device) ; mask_t = mask_t.to(msa.device) ; chain_idx = chain_idx.to(msa.device) ; same_chain = same_chain.to(msa.device) ; mask_recycle = mask_recycle.to(msa.device) ; epitope_full = epitope_full.to(msa.device)
            print('idx', idx)
            msa, pair, state, xyz_pred, alpha, _, epitope_info = model(
                msa_latent =msa, msa_full =ins, seq=seq,xyz = xyz_init, idx= idx, t1d = t1d, t2d=t2d, xyz_t = xyz_t,  alpha_t =alpha_t, mask_t = mask_t, chain_idx =chain_idx, same_chain = same_chain, mask_recycle = mask_recycle, epitope_info = epitope_full[None], return_raw = True) #L_s=L_s, device=device, max_cycle=max_cycle)
            is_bond_violated = check_bond_geom(xyz=xyz_pred[:, :3], L_s=L_s)
            print('xyz_pred', xyz_pred.shape)
            print('msa_output', msa.shape) 
            util.write_pdb(
                msa[0],
                xyz_pred,
                L_s=L_s,
                Bfacts=plddt,
                prefix="%s_raw" %
                prefix)
            np.savez_compressed(
                "%s.npz" %
                (prefix),
                dist=prob_s[0],
                lddt=plddt.numpy(),
                pae = pae.numpy())
            if is_bond_violated:
                plddt_s.append(plddt.mean().item() - 1.0)
            else:
                plddt_s.append(plddt.mean().item())
            pdb_s.append(os.path.abspath("%s_raw.pdb" % (prefix)))
            sys.stdout.write(
                "Run time: %.2f sec\n" %
                (time.time() - start_time))

        # Run without insertion statistics (it sometimes help)
        prefix = "%s_%02d_wo_ins" % (out_prefix, i_iter)
        if os.path.exists("%s_raw.pdb" % prefix):
            is_bond_violated = check_bond_geom(
                pdb_fn="%s_raw.pdb" %
                prefix, L_s=L_s)
            pdb_s.append(os.path.abspath("%s_raw.pdb" % prefix))
            if is_bond_violated:
                plddt_s.append(np.load("%s.npz" % prefix)['lddt'].mean() - 1.0)
            else:
                plddt_s.append(np.load("%s.npz" % prefix)['lddt'].mean())
        else:
            sys.stdout.write(
                "INFO: Running template-based prediction without insertion statistics\n")
            xyz_pred, plddt, prob_s, pae = model(msa, torch.zeros_like(
                ins), t1d, t2d, xyz_t, xyz_init, alpha_t, chain_idx, mask_recycle, mask_t, L_s=L_s, device=device, max_cycle=max_cycle)
            is_bond_violated = check_bond_geom(xyz=xyz_pred[:, :3], L_s=L_s)
            util.write_pdb(
                msa[0],
                xyz_pred,
                L_s=L_s,
                Bfacts=plddt,
                prefix="%s_raw" %
                prefix)
            np.savez_compressed(
                "%s.npz" %
                (prefix),
                dist=prob_s[0],
                lddt=plddt.numpy(),
                pae=pae.numpy())
            if is_bond_violated:
                plddt_s.append(plddt.mean().item() - 1.0)
            else:
                plddt_s.append(plddt.mean().item())
            pdb_s.append(os.path.abspath("%s_raw.pdb" % (prefix)))
            sys.stdout.write(
                "Run time: %.2f sec\n" %
                (time.time() - start_time))

    max_idx = np.argmax(plddt_s)
    print(plddt_s, max_idx)
    os.system("ln -sf %s %s_raw.%.4f.pdb" %
              (pdb_s[max_idx], out_prefix, plddt_s[max_idx]))
    os.system("ln -sf %s.npz %s_raw.npz" %
              ("_".join(pdb_s[max_idx].split("_")[:-1]), out_prefix))

    return pdb_s[max_idx], plddt_s[max_idx]


def get_inputs(item, a3m_fn, hhr_fn=[], atab_fn=[], templ_pt=[], negative=False):

    def get_dict(info_dict, chain_dict):
        return_dict = defaultdict(list)
        for key, value in chain_dict.items():
            for chain in value:
                return_dict[key].append(info_dict[chain])
        return return_dict
    def get_sequence(a3m_dict):
        print(a3m_dict)
        fasta_dict = defaultdict(str)
        for key, value in a3m_dict.items():
            query_sequence = get_msa(value)['msa'][0]
            fasta_dict[key] = query_sequence
        return fasta_dict
    
    def find_indices(l, value):
        return [index for index,
            item in enumerate(l) if item.equal(value)]
    
    # get chain_dictionary {H: H, L: L, AG: AB}

    chain_dict = defaultdict(str)
    all_chain_list = []
    if negative:
        for (i, j) in zip(['H', 'L'], item.split(':')[0].split('_')[1:-1]):
            if j != '#':
                chain_dict[i] = j
                for k in j:
                    all_chain_list.append(k)
        agchain = item.split(':')[1].split('_')[-1]
        if agchain != '#':
            chain_dict['AG'] = agchain
            for k in agchain:
                all_chain_list.append(k)
    else:
        for (i, j) in zip(['H', 'L', 'AG'], item.split('_')[1:]):
            if j != '#':
                chain_dict[i] = j
                for k in j:
                    all_chain_list.append(k)
        
    a3m_dict = defaultdict(str)

    templ_pt_dict = defaultdict(str)
    for (i, j, k) in zip(all_chain_list, a3m_fn, templ_pt):
        a3m_dict[i] = j
        templ_pt_dict[i] = k
    # get a3m information
    #a3m_fn_s = [get_msa(i) for i in a3m_fn)]
    #print(a3m_dict)
    #print(templ_pt_dict)
    #print(chain_dict) 
    a3m = get_dict(a3m_dict, chain_dict)
    
    if templ_pt != []:
        template_params = get_dict(templ_pt_dict, chain_dict)
    else:
        pass # implement later
    
    # get sequence dictionary {H:[seq1, seq2], L:[seq1, seq2], AG:[seq1, seq2]}
    fasta_dict = get_sequence(a3m_dict)
    #print(fasta_dict)
    #print(a3m)
    #print(template_params)
    seq_dict = get_dict(fasta_dict, chain_dict)
    #print(seq_dict)
    homo_dict = defaultdict(list)
    for key, value in seq_dict.items():
        homo_list = []
        for seq in value:
            same = find_indices(value, seq)
            if same not in homo_list:
                homo_list.append(same)
        homo_dict[key] = homo_list
    # get template information
    print(homo_dict)
    a3m_in = []
    tplt_params_in = []
    length_dict = defaultdict(int)
    for k, v in homo_dict.items():
        count = 0
        for idx in v:
            a3m_in.append(a3m[k][idx[0]])
            for idx2 in idx:
                # print('tplts atom',template_params[k][idx2])
                tplt_params_in.append(template_params[k][idx2])
                count += 1
        length_dict[k] = count
    
    print(length_dict)
    for key in ['H', 'L']:
        if key not in length_dict:
            length_dict[key] = 0
    
    print(length_dict)
    h_length = length_dict['H']
    hl_length = length_dict['H'] + length_dict['L']
    ag_length = length_dict['AG']
    
    homo_list_final = []
    for k, v in homo_dict.items():
        if k == 'H':
            homo_list_final.extend(v)
        elif k == 'L':
            for i in v:
                homo_list_final.append([j + h_length for j in i])
        else:
            for i in v:
                homo_list_final.append([j + hl_length for j in i])
    
    chain_properties = []
    if hl_length == 1:
        if h_length == 0:
            chain_properties.append(1)
        else:
            chain_properties.append(0)
    else:
        chain_properties.extend([0,1])
    
    chain_properties.extend(list(range(2, 2+ ag_length)))
    print('chain_properites', chain_properties)
    print('homo_list_final', homo_list_final)
    return a3m_in, tplt_params_in, homo_list_final, chain_properties
                

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.use_cpu:
        device = "cpu"

    # Read template database
    #FFDB = args.db
    #FFindexDB = namedtuple("FFindexDB", "index, data")
    # ffdb = FFindexDB(read_index(FFDB + '_pdb.ffindex'),
    #                  read_data(FFDB + '_pdb.ffdata'))

    a3m_in, tplt_params_in, homo_list, chain_properties = get_inputs(
        args.item, args.a3m_fn, args.hhr_fn, args.atab_fn, args.templ_pt)
    msa_latent, msa_full, (xyz, xyz_t, t1d, t2d, alpha_t, mask_recycle, mask_t_2d), L_s, chain_idx, epitope_full, same_chain, seq, idx = read_inputs(a3m_in, tplt_params_in, homo_list, chain_properties, \
        n_templ=args.n_templ, device=device, item=args.item) #ffdb - include later
    templ = (xyz, xyz_t, t1d, t2d, alpha_t, mask_recycle, mask_t_2d)
    print('final_shape', msa_latent.shape)
    Nseq = msa_latent.shape[0]
    NREPLICATES = 1
    if Nseq < 30:
        NREPLICATES = 3
        MAX_CYCLE = 20
    else:
        NREPLICATES = 1
        MAX_CYCLE = 15
    if args.max_cycle > 0:
        MAX_CYCLE = args.max_cycle

    # models to use
    model_s = json.load(open("%s/models.json" % args.script_dir))
    pdb_s = list()
    plddt_s = list()
    for model_name in model_s:
        sys.stdout.write("INFO: Running model %s\n" % model_name)
        model_info = model_s[model_name]
        if "full_bigSE3" in model_name:
            model_info['model_param']['aamask'] = util.allatom_mask.to(device)
            model_info['model_param']['ljlk_parameters'] = util.ljlk_parameters.to(
                device)
            model_info['model_param']['lj_correction_parameters'] = util.lj_correction_parameters.to(
                device)
            model_info['model_param']['num_bonds'] = util.num_bonds.to(device)
        # RF_model = importlib.import_module('RoseTTAFoldModel_inference')
        # RF_model = importlib.import_module('RoseTTAFoldModel')
        model = RoseTTAFoldModule(
            **model_info['model_params']).to(device)
        model.eval()
        #
        for i_m, weight_fn in enumerate(model_info['weight_fn']):
            sys.stdout.write("INFO: Loading checkpoint file, %s\n" % weight_fn)
            out_prefix = "%s.%s.model%d" % (
                args.out_prefix, model_name, i_m + 1)
            checkpoint = torch.load(
                (f"{weight_fn}"),
                map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            best_pdb, plddt = _run_model(
                model, out_prefix, msa_latent, msa_full, templ, chain_idx, epitope_full,seq,idx,same_chain, L_s=L_s, device=device, max_iter=NREPLICATES, max_cycle=MAX_CYCLE)
            pdb_s.append(best_pdb)
            plddt_s.append(plddt)
        del model
        torch.cuda.empty_cache()

    rank = np.argsort(plddt_s)[::-1]
    for i_r, r in enumerate(rank):
        os.system("ln -sf %s %s_final_%d_unrelaxed.pdb" %
                  (pdb_s[r], args.out_prefix, i_r + 1))
