import sys
sys.path.append('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/src')
from rfabflex.common.util import center_and_realign_missing, random_rot_trans, writepdb, aa2num, aa2long, write_pdb
from rfabflex.common.chemical import INIT_CRDS
from rfabflex.data.parsers import parse_a3m, parse_pdb, parse_pdb_antibody, parse_templates
from rfabflex.common.kinematics import get_init_xyz_compl
import pickle
from itertools import product, permutations
import logging
import time
import csv
import random
import copy
import numpy as np
import torch.distributed as dist
from torch.utils import data
import torch
from collections import defaultdict, OrderedDict
import json
from glob import glob
import os
from tqdm import tqdm



# abag stands for antibody-antigen complex
ABAG_DIR = "/home/kkh517/Github/rf-abag-templ/DB/test_json/cluster_info"
ABAG_DB = "/public_data/ml/antibody/PDB_Ag_Ab/200430/pdb"
# gp stands for general protein
# GP_DIR = "/home/yubeen/cluster_info"
GP_DIR = "/public_data/ml/antibody/loop_PPI"
# GP_DIR = "/public_data/ml/antibody/loop_PPI/from_kisti"
NEG_DIR = "/home/yubeen/rf_ab_ag/DB/ab_ag/neg_interface"
VAL_DIR = "/home/kkh517/Github/rf-abag-templ/DB/test_json/cluster_info"
# ABAG_MSA = "/public_data/ml/antibody/PDB_Ag_Ab/200430/msa"
# ABAG_MSA="/home/kkh517/Github/rf-abag-templ/DB/real_final_set_copy/H_L_A"
ABAG_MSA = "/home/kkh517/flexible_pdbs" # for flexible docking
# ABAG_MSA = '/home/kkh517/antibody_pdb_woCDR' # woCDR
ABAG_MSA2 = '/home/kkh517/antibody_pdb_2' # for rigid docking
# ABAG_MSA2 = '/home/kkh517/antibody_pdb_woCDR' # for rigid docking
FFDB = "/home/yubeen/RF2/pdb100_2021Mar03/pdb100_2021Mar03"
# TM_DB = '/home/kkh517/submit_files/Project/halfblood/TM_score.csv'
TM_DB = '/home/kkh517/submit_files/TM_score.csv'

def __get_logger():
    __logger = logging.getLogger('logger')
    __logger.setLevel(logging.INFO)

    return __logger


logger = __get_logger()
def put_CDR_mask(xyz_t: torch.Tensor, f1d_t:torch.Tensor, mask_t:torch.Tensor, CDR_mask:torch.Tensor):
    """
    Put CDR mask on the input tensor

    ___ input ___
    xyz_t  : template xyz coords [B, L, N, 3]
    f1d_t  : t1d template [B, L, 22]
    mask_t : mask_t [B, L, 27]
    """
    L = xyz_t.shape[1]
    xyz_init = (INIT_CRDS.reshape(1,1,27,3).repeat(1,L,1,1))
    seq_init = torch.full((L,),20).long()
    conf_init = torch.full((L,),0).long()
    seq_init = torch.nn.functional.one_hot(seq_init,21).float()
    f1d_t_init = torch.cat((seq_init, conf_init[:,None]), -1).unsqueeze(0)
    # print(f" L : {L} // xyz_t : {xyz_t.shape} // f1d_t : {f1d_t.shape} // mask_t : {mask_t.shape} // CDR_mask : {CDR_mask.shape} // seq_init : {seq_init.shape} // f1d_t_init : {f1d_t_init.shape}")
    # xyz_t = torch.where(CDR_mask == 1,xyz_init,xyz_t)
    # breakpoint()
    # if which index that CDR_mask is 1, f1d_t should be replaced with f1d_t_init
    CDR_mask_f1d = CDR_mask.unsqueeze(-1).repeat(1,1,22)
    f1d_t = torch.where(CDR_mask_f1d == 1,f1d_t_init,f1d_t)
    
    CDR_mask_xyz = CDR_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,27,3)
    xyz_t = torch.where(CDR_mask_xyz == 1,xyz_init,xyz_t)
    CDR_mask_mask = CDR_mask.unsqueeze(-1).repeat(1,1,27)
    mask_t = torch.where(CDR_mask_mask == 1,torch.full_like(mask_t,0),mask_t)
    # breakpoint()
    # xyz_t = center_and_realign_missing(xyz_t[0], mask_t[0])
    # xyz_t = xyz_t.unsqueeze(0)
    # mask_t = torch.where(CDR_mask == 1,torch.full_like(mask_t,0),mask_t)    
    

    return xyz_t, f1d_t, mask_t

def set_data_loader_params(args):
    """Get parameters of the arguments

    Args:
        args (object)

    Returns:
        params (dict): Dictionary of the parameters
    """
    PARAMS = {
       "GP_LIST": f"{GP_DIR}/list.loopPPI.0.7.csv",    # yb
        "NEGATIVE_LIST": f"{ABAG_DIR}/negative_train.json",  
        "L_LIST": f"{ABAG_DIR}/only_l_woantigen_70_loop_final_train.json", # yb
        "H_LIST": f"{ABAG_DIR}/only_h_woantigen_70_loop_final_train.json", # yb
        "HL_LIST": f"{ABAG_DIR}/both_woantigen_70_loop_final_train.json", # yb
        "L_AG_LIST": f"{ABAG_DIR}/only_l_wantigen_70_loop_final_train.json", # yb
        # "L_AG_LIST": "/home/kkh517/Github/rf-abag-templ/DB/test_json/l_a_train.json", # yb
        "H_AG_LIST": f"{ABAG_DIR}/only_h_wantigen_70_loop_final_train.json", # yb
        "HL_AG_LIST": f"{ABAG_DIR}/both_wantigen_70_loop_final_train.json", # yb
        "VAL_L_LIST": f"{VAL_DIR}/only_l_woantigen_70_loop_final_val.json", # yb
        "VAL_H_LIST": f"{VAL_DIR}/only_h_woantigen_70_loop_final_val.json", # yb
        "VAL_HL_LIST": f"{VAL_DIR}/both_woantigen_70_loop_final_val.json", # yb
        "VAL_L_AG_LIST": f"{VAL_DIR}/only_l_wantigen_70_loop_final_val.json", # yb
        "VAL_H_AG_LIST": f"{VAL_DIR}/only_h_wantigen_70_loop_final_val.json", # yb
        "VAL_HL_AG_LIST": f"{VAL_DIR}/both_wantigen_70_loop_final_val.json", # yb
        # "TEST_HL_AG_LIST": f"/home/kkh517/Github/rf-abag-templ/DB/test_json/cluster_info/test_set2.json", # kh
        # "TEST_HL_AG_LIST" : "/home/kkh517/test_set_id.json", # new_test
        # "TEST_HL_AG_LIST" : "/home/kkh517/submit_files/Project/epitope_sampler_halfblood/test_set_id.json", # iitp
        # "TEST_HL_AG_LIST" : "/home/kkh517/submit_files/Project/inference_lj/test_set_id.json",
        "TEST_HL_AG_LIST" : "/home/kkh517/submit_files/Project/epitope_sampler_halfblood/gpu01_dict.json", # new_test gpu01
        # "TEST_HL_AG_LIST" : "/home/kkh517/submit_files/Project/epitope_sampler_halfblood/gpu02_dict.json", # new_test gpu02
        # "TEST_HL_AG_LIST" : "/home/kkh517/submit_files/Project/epitope_sampler_halfblood/ag_dict.json",
        "TEST_ALL_TRAIN" : '/home/kkh517/Github/rf-abag-templ/DB/test_json/cluster_info/train_all_id.json',
        "VAL_NEG": f"{ABAG_DIR}/negative_val.json", 
        "NEGATIVE_LIST_TEST": f"{ABAG_DIR}/negative_train_test.json", 
        # "HOMO_LIST": f"/home/yubeen/cluster_info/homo_list.json", # iitp
        # "HOMO_LIST" : f"/home/kkh517/Github/rf-abag-templ/DB/test_json/cluster_info/homo_list3.json",
        "HOMO_LIST" : "/home/kkh517/homo_list.json", # new_test
        "INTERFACE_CA_10": f"{NEG_DIR}/true_interface_10.json",
        "INTERFACE_CA_12": f"{NEG_DIR}/true_interface_12.json",
        "TM_DB" : TM_DB,
        "GP_DIR": GP_DIR,
        "AB_DIR": ABAG_DB,
        "AB_MSA_DIR": ABAG_MSA, # flexible
        "AG_MSA_DIR": ABAG_MSA2, # rigid
        "NEG_DIR": NEG_DIR,
        "VAL_DIR": VAL_DIR, 
        "FFDB": FFDB,
        "MINTPLT": 0,
        "MAXTPLT": 1, # yb 5 -> 1
        "MINSEQ": 1, 
        "MAXSEQ": 1024,
        "MAXLAT": 128,
        "CROP": 256,
        "DATCUT": "2020-Apr-30",
        "RESCUT": 10.0, # yb 10.0 -> 5.0
        "BLOCKCUT": 5,
        "PLDDTCUT": 70.0,
        "SCCUT": 90.0,
        "ROWS": 1,
        "SEQID": 95.0,
        # "MAXCYCLE": args.maxcycle,
        "MAXCYCLE" : 4,
        "CDR_DICT" : "/home/kkh517/CDR_pdb_dict.pt"
        # "CDR_DICT" : "/home/kkh517/benchmark_set_after210930/testset_CDR.pkl"
    }
    # if type(PARAMS['MAXCYCLE']) == 'NoneType':
    #     PARAMS['MAXCYCLE'] = 4
    # for param in PARAMS:
    #     if hasattr(args, param.lower()):
    #         PARAMS[param] = getattr(args, param.lower()) # for only debugging
    return PARAMS


def MSABlockDeletion(msa, ins, nb=5):
    """
    Down-sample given MSA by randomly delete blocks of sequences
    Input: MSA/Insertion having shape (N, L)
    output: new MSA/Insertion with block deletion (N', L)
    """
    N, _ = msa.shape
    block_size = max(int(N * 0.1), 1)
    block_start = np.random.randint(low=0, high=N, size=nb)  # (nb)
    to_delete = block_start[:, None] + np.arange(block_size)[None, :]
    to_delete = np.unique(np.clip(to_delete, 0, N - 1))
    # mask = np.ones(N, np.bool)
    mask = np.ones(N, bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]


def cluster_sum(data, assignment, N_seq, N_res):
    """Get statistics from clustering results (clustering extra sequences with seed sequences)

    Args:
        data : One hot encoding of extra sequences [N_extra, L, 22]
        assignment: Most same sequence in N_clust  [N_extra]
        N_seq: Number of cluster                   N_clust
        N_res: Number of residues                  L

    Returns:
        csum: Cluster statistics                   [N_cluster, L, 22]
    """

    csum = torch.zeros(N_seq,
                       N_res,
                       data.shape[-1]).to(data.device).scatter_add(0,
                                                                   assignment.view(-1,
                                                                                   1,
                                                                                   1).expand(-1,
                                                                                             N_res,
                                                                                             data.shape[-1]),
                                                                   data.float())
    # For each extra sequences, sum to the nearest sequence in N_clust
    return csum


def MSAFeaturize(msa, ins, params, eps=1e-6, L_s=[]):
    """Get MSA features

    Args:
        msa: full MSA information (after Block deletion if necessary)
        ins: full insertion information
        MAXCYCLE: number of recycle

    Returns:
        b_seq: masked sequence                        [MAXCYCLE, L]
        b_msa_clust: clusterd one-hot encoded MSA     [MAXCYCLE, N_clust, L]
        b_msa_seed: properties of the seed sequence   [MAXCYCLE, N_clust, L, 48]
            - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
            - profile of clustered sequences (22)
            - insertion statistics (2)
            - N-term or C-term? (2)
        b_msa_extra: properties of the extra sequence [MAXCYCLE, N_extra, L, 25]
            - aatype of extra sequence (22)
            - insertion info (1)
            - N-term or C-term? (2)
        b_mask_pos: postition of masking (15%)        [MAXCYCLE, N_clust, L]

    """

    N, L = msa.shape

    term_info = torch.zeros((L, 2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0, 0] = 1.0  # flag for N-term
        term_info[-1, 1] = 1.0  # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            L_chain = L_chain.long()
            term_info[start, 0] = 1.0  # flag for N-term
            term_info[start + L_chain - 1, 1] = 1.0  # flag for C-term
            start += L_chain

    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0)

    # Nclust sequences will be selected randomly as a seed MSA (aka latent MSA)
    # - First sequence is always query sequence
    # - the rest of sequences are selected randomly
    Nclust = torch.tensor(min(N, params["MAXLAT"]), device=msa.device)
    L = torch.tensor(L, device=msa.device)
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()

    for i_cycle in range(params["MAXCYCLE"]):  # 4

        sample = torch.randperm(N - 1, device=msa.device)
        msa_clust = torch.cat(
            (msa[:1, :], msa[1:, :][sample[: Nclust - 1]]), dim=0)
        ins_clust = torch.cat(
            (ins[:1, :], ins[1:, :][sample[: Nclust - 1]]), dim=0)

        # 15% random masking
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05] * 20 + [0.0]], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
        probs = 0.1 * random_aa + 0.1 * raw_profile + 0.1 * same_aa
        probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)

        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < 0.15
        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone())

        # get extra sequenes
        if N - Nclust >= params["MAXSEQ"]:  # there are enough extra sequences
            Nextra = params["MAXSEQ"]
            msa_extra = torch.cat(
                (msa_masked[:1, :], msa[1:, :][sample[Nclust - 1:]]), dim=0)
            ins_extra = torch.cat(
                (ins_clust[:1, :], ins[1:, :][sample[Nclust - 1:]]), dim=0)
            extra_mask = torch.full(
                msa_extra.shape, False, device=msa_extra.device)
            extra_mask[0] = mask_pos[0]
        elif N - Nclust < 1:  # no extra sequences, use all masked seed sequence as extra one
            Nextra = Nclust
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        # it has extra sequences, but not enough to maxseq. Use mixture of seed
        # (except query) & extra
        else:
            Nextra = min(N, params["MAXSEQ"])
            msa_add = msa[1:, :][sample[Nclust - 1:]]
            ins_add = ins[1:, :][sample[Nclust - 1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)

        N_extra_pool = msa_extra.shape[0]

        # 1. one_hot encoded aatype: msa_clust_onehot
        msa_clust_onehot = torch.nn.functional.one_hot(
            msa_masked, num_classes=22)  # (N_clust, L, 22)
        msa_extra_onehot = torch.nn.functional.one_hot(
            msa_extra, num_classes=22)   # (N_extra, L, 22)

        # clustering (assign remaining sequences to their closest cluster by
        # Hamming distance
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20)
        # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(
            ~extra_mask, msa_extra != 20)  # (N_extra, L)
        # get number of identical tokens for each pair of sequences (extra vs
        # seed)
        temp_extra = (count_extra[:, :, None] *
                      msa_extra_onehot).view(N_extra_pool, -1)
        temp_clust = (count_clust[:, :, None] *
                      msa_clust_onehot).view(Nclust, -1)

        temp_extra_float = temp_extra.float()
        temp_clust_float = temp_clust.float()
        # print(temp_extra_float.type())
        agreement = torch.matmul(
            temp_extra_float,  # (N_extra, L, 1) * (N_extra, L, 22)
            temp_clust_float.T,
        )  # (N_extra_pool, Nclust)
        assignment = torch.argmax(
            agreement, dim=-1
        )  # map each extra seq to the closest seed seq

        # 2. cluster profile -- ignore masked token when calculate profiles
        count_extra = ~extra_mask  # only consider non-masked tokens in extra seqs
        count_clust = ~mask_pos  # only consider non-masked tokens in seed seqs
        cluster_temp = count_extra[:, :, None] * msa_extra_onehot
        msa_clust_profile = cluster_sum(
            count_extra[:, :, None] * msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:, :, None] * msa_clust_onehot
        count_profile = cluster_sum(
            count_extra[:, :, None], assignment, Nclust, L).view(Nclust, L)  #
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:, :, None]

        # 3. insertion statistics
        msa_clust_del = cluster_sum(
            (count_extra *
             ins_extra)[
                :,
                :,
                None],
            assignment,
            Nclust,
            L).view(
            Nclust,
            L)
        msa_clust_del += count_clust * ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0 / np.pi) * \
            torch.arctan(ins_clust.float() / 3.0)  # (from 0 to 1)
        # (from 0 to 1)
        msa_clust_del = (2.0 / np.pi) * \
            torch.arctan(msa_clust_del.float() / 3.0)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)

        # seed MSA features (one-hot aa, cluster profile, ins statistics,
        # terminal info)
        msa_seed = torch.cat(
            (
                msa_clust_onehot,  # [N_clust, L, 22]
                msa_clust_profile,  # [N_clust, L, 22]
                ins_clust,  # [N_clust, L, 2]
                term_info[None].expand(Nclust, -1, -1),  # [N_clust, L, 2]
            ),
            dim=-1,
        )  # [N_clust, L, 48]

        # extra MSA features (one-hot aa, insertion, terminal info)
        # (from 0 to 1)
        ins_extra = (2.0 / np.pi) * \
            torch.arctan(ins_extra[:Nextra].float() / 3.0)
        msa_extra = torch.cat(
            (
                msa_extra_onehot[:Nextra],  # [N_extra, L, 22]
                ins_extra[:, :, None],  # [N_extra, L, 1]
                term_info[None].expand(Nextra, -1, -1),  # [N_extra, L, 2]
            ),
            dim=-1,
        )  # [N_extra, L, 25]

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)

    b_seq = torch.stack(b_seq)  # [MAXCYCLE, L]
    b_msa_clust = torch.stack(b_msa_clust)  # [MAXCYCLE, N_clust, L]
    b_msa_seed = torch.stack(b_msa_seed)  # [MAXCYCLE, N_clust, L, 48]
    b_msa_extra = torch.stack(b_msa_extra)  # [MAXCYCLE, N_extra, L, 25]
    b_mask_pos = torch.stack(b_mask_pos)  # [MAXCYCLE, N_clust, L]

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos


def TemplFeaturize(
    tplt,
    qlen,
    params,
    offset=0,
    npick=1,
    npick_global=None,
    pick_top=True,
    random_noise=5.0,
):
    """Feturize template

    Args:
        tplt (dict): Template features
        qlen (int): Total number of residue
        params (dict): Dictionary of parameters
        offset (int, optional): Offset for multiple chains. Defaults to 0.
        npick (int, optional): Number of templates to use. Defaults to 1.
        npick_global (int, optional): Number of templates to use. Defaults to None.
        pick_top (bool, optional): _description_. Defaults to True.
        random_noise (float, optional): _description_. Defaults to 5.0.

    Returns:
        xyz: xyz coordinate of the templates [npick_globl, L_ch, 27, 3]
        t1d: template sequence information (21) + alignment probability(1) [npick_global, L_ch, 22]
        mask_t: template masking information (True: Atom exist) [npick_global, L_ch, 27]

    """
    if npick_global is None:
        npick_global = max(npick, 1)
    seqID_cut = params["SEQID"]

    ntplt = len(tplt["ids"])
    if (ntplt < 1) or (
            npick < 1):  # no templates in hhsearch file or not want to use templ - return fake templ
        xyz = (INIT_CRDS.reshape(1, 1, 27, 3).repeat(npick_global, qlen, 1, 1)  # [npick_global, L_ch, 27, 3]
               + torch.rand(npick_global, qlen, 1, 3) * random_noise)  # add random noise to the structure
        t1d = torch.nn.functional.one_hot(torch.full(
            (npick_global, qlen), 20).long(), num_classes=21).float()  # all gaps
        # [npick_global, L_ch, 21]
        # [npick_global, L_ch, 1]
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)  # [npick_global, L_ch, 22]
        # [npick_global, L_ch, 27]
        mask_t = torch.full((npick_global, qlen, 27), False)
        return xyz, t1d, mask_t

    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt["f0d"][:, 0] < seqID_cut)[0]
        #print('tplt_valid_idx', tplt_valid_idx)
        if tplt_valid_idx.numel() == 1:
            tplt["ids"] = np.array([np.array(tplt["ids"])[tplt_valid_idx]])
        else:
            tplt["ids"] = np.array(tplt["ids"])[tplt_valid_idx]
        #print('ids', tplt["ids"])
    else:
        tplt_valid_idx = torch.arange(len(tplt["ids"]))

    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt["ids"])
    npick = min(npick, ntplt)
    if npick < 1:  # no templates -- return fake templ
        xyz = (
            INIT_CRDS.reshape(1, 1, 27, 3).repeat(npick_global, qlen, 1, 1)
            + torch.rand(npick_global, qlen, 1, 3) * random_noise
        )
        t1d = torch.nn.functional.one_hot(
            torch.full((npick_global, qlen), 20).long(), num_classes=21
        ).float()  # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global, qlen, 27), False)
        return xyz, t1d, mask_t

    if not pick_top:  # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else:  # only consider top 50 templates
        sample = torch.randperm(min(50, ntplt))[:npick]
        #print('sample',sample)

    xyz = (INIT_CRDS.reshape(1, 1, 27, 3).repeat(npick_global, qlen, 1, 1)
           + torch.rand(1, qlen, 1, 3) * random_noise)
    # True for valid atom, False for missing atom
    mask_t = torch.full((npick_global, qlen, 27), False)
    t1d = torch.full((npick_global, qlen), 20).long()
    t1d_val = torch.zeros((npick_global, qlen)).float()

    for i, nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt["qmap"][:, 1] == tplt_idx)[0]
        pos = tplt["qmap"][sel, 0] + offset
        xyz[i, pos, :14] = tplt["xyz"][sel]
        mask_t[i, pos, :14] = tplt["mask"][sel].bool()
        # 1-D features: alignment confidence
        t1d[i, pos] = tplt["seq"][sel]
        t1d_val[i, pos] = tplt["f1d"][sel, 2]  # alignment confidence
        mask_val = mask_t[i][:, :3].all(dim=-1)
        exist_in_xyz = torch.where(mask_val)[0]
        #print('exist in xyz', exist_in_xyz)
        #if exist_in_xyz.numel() != 0:
        xyz[i] = center_and_realign_missing(xyz[i], mask_t[i])

    t1d = torch.nn.functional.one_hot(t1d, num_classes=21).float()
    t1d = torch.cat((t1d, t1d_val[..., None]), dim=-1)

    return xyz, t1d, mask_t

def TemplFeaturize_kh(
        tplt, # template pdb path
        # qlen,
        L,
        params,
        # p_antibody,
        offset=0,
        npick=1,
        npick_global=None,
        pick_top=True,
        random_noise=5.0,
        template_dict=None):
    """Feturize template

    Args:
        tplt (dict): Template features
        qlen (int): Total number of residue
        params (dict): Dictionary of parameters
        offset (int, optional): Offset for multiple chains. Defaults to 0.
        npick (int, optional): Number of templates to use. Defaults to 1.
        npick_global (int, optional): Number of templates to use. Defaults to None.
        pick_top (bool, optional): _description_. Defaults to True.
        random_noise (float, optional): _description_. Defaults to 5.0.

    Returns:
        xyz: xyz coordinate of the templates [npick_globl, L_ch, 27, 3]
        t1d: template sequence information (21) + alignment probability(1) [npick_global, L_ch, 22]
        mask_t: template masking information (True: Atom exist) [npick_global, L_ch, 27]

    """
    # check the subdirectories of the template directory
    def get_init_xyz(qlen,random_noise=5.0):
        # xyz = torch.full((qlen, 27, 3), np.nan).float()
        xyz = (INIT_CRDS.reshape(1, 27, 3).repeat(qlen, 1, 1)  # [L_ch, 27, 3]
               + torch.rand(qlen, 1, 3) * random_noise) # for SE(3) transformer...
        seq = torch.full((qlen,), 20).long() # all gaps  
        conf = torch.full((qlen,), 0.0).float()
        same_chain = torch.full((1,qlen,qlen),True).bool()
        return xyz, seq, conf, same_chain

    # first initialize the xyz, seq, conf, same_chain
    qlen = L
    xyz, seq, conf, same_chain = get_init_xyz(qlen)
    # templ_fn = templ_fns[0]
    assert tplt == None, "wrong templfeaturize..."
    xyz_nan = xyz.unsqueeze(0).unsqueeze(0)
    xyz_init = get_init_xyz_compl(xyz_nan,same_chain)
    xyz_unsq = xyz_init.squeeze(0)
    xyz = xyz_unsq
    # print('xyz_shape',xyz.shape)
    for seq_idx in range(qlen):
        seq[seq_idx] = 20
    seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
    t1d = torch.cat((seq, conf[:,None]), -1)
    mask = torch.full((qlen,27),False).bool() # (qlen, 27)#missing region 파악 안하는 중... 나중에 debugging 필요
    # print('t1d_shape',t1d[None].shape)
    # print('mask_shape',mask[None].shape)
    return xyz, t1d[None], mask[None]


def get_train_valid_set(params):
    """Get train and validation set

    Returns:
        Dictionary of training set & weights of each set
    """
    # compile antibody examples

    # 1. compile only l antibody
    # f = open(params['L_LIST'], encoding='utf-8')
    # l_train = json.load(f)
    # f.close()
    with open(params['L_LIST'], 'r') as f:
        l_train = json.load(f)


    # 2. compile only h antibody
    # f = open(params['H_LIST'], encoding='utf-8')
    # h_train = json.load(f)
    # f.close()
    with open(params['H_LIST'], 'r') as f:
        h_train = json.load(f)
    
    # f_val = open(params['VAL_H_LIST'], encoding='utf-8')
    # h_val = json.load(f_val)
    # f_val.close()
    with open(params['VAL_H_LIST'], 'r') as f:
        h_val = json.load(f)
    # 3. compile both antibody
    # f = open(params['HL_LIST'], encoding='utf-8')
    # hl_train = json.load(f)
    # f.close()
    # f_val = open(params['VAL_HL_LIST'], encoding='utf-8')
    # hl_val = json.load(f_val)
    # f_val.close()
    with open(params['HL_LIST'], 'r') as f:
        hl_train = json.load(f)
    with open(params['VAL_HL_LIST'], 'r') as f:
        hl_val = json.load(f)

    # 4. compile only l & antigen
    # f = open(params['L_AG_LIST'], encoding='utf-8')
    # l_ag_train = json.load(f)
    # f.close()
    with open(params['L_AG_LIST'], 'r') as f:
        l_ag_train = json.load(f)
    # 5. compile only h & antigen
    # f = open(params['H_AG_LIST'], encoding='utf-8')
    # h_ag_train = json.load(f)
    # f.close()
    # f_val = open(params['VAL_H_AG_LIST'], encoding='utf-8')
    # h_ag_val = json.load(f_val)
    # f_val.close()
    with open(params['H_AG_LIST'], 'r') as f:
        h_ag_train = json.load(f)
    with open(params['VAL_H_AG_LIST'], 'r') as f:
        h_ag_val = json.load(f)

    # 6. compile both & antigen
    # f = open(params['HL_AG_LIST'], encoding='utf-8')
    # hl_ag_train = json.load(f)
    # f.close()
    # f_val = open(params['VAL_HL_AG_LIST'], encoding='utf-8')
    # hl_ag_val = json.load(f_val)
    # f_val.close
    # f_test = open(params['TEST_HL_AG_LIST'], encoding='utf-8')
    # hl_ag_test = json.load(f_test)
    # f_test.close()
    with open(params['HL_AG_LIST'], 'r') as f:
        hl_ag_train = json.load(f)
    with open(params['VAL_HL_AG_LIST'], 'r') as f:
        hl_ag_val = json.load(f)
    with open(params['TEST_HL_AG_LIST'], 'r') as f:
        hl_ag_test = json.load(f)
    
    # Remove key with value ['7uij_H_L_CD']
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['7uij_H_L_CD']} # new a3m is required for chain D
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['8kae_N_#_R']} # no a3m for R
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['8vyl_E_#_ACBD']} # D doesn't start with Valine on a3m
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['8eln_L_#_IJ']} # TODO
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['8vzo_B_D_A']} # no msa for antigen
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['8djm_H_L_BA']} # TODO
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['9ima_C_D_AB']} # TODO
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['7y9t_D_#_AB']} # TODO
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v != ['7z12_B_C_A']} # TODO

    print(f"TEST_HL_AG_LIST {params['TEST_HL_AG_LIST']}")
    # print('hl_ag_test len',len(hl_ag_test["000"]))
    # erase the test set with the key doesn't start with integer
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if k[0].isdigit()}
    # complex_list = ['2fjg', '2vis', '3eo1', '3hmx', '3wd5', '4fp8', '4fqi', '4gxu', '5hys', '5kov', '5whk', '5wux', '5y9j', '6ey6']
    # hl_ag_test = {k: v for k, v in hl_ag_test.items() if v[0][:4] not in complex_list}
    with open(params['TEST_ALL_TRAIN'], 'r') as f_all:
        train_all = json.load(f_all)


    # f = open(params['NEGATIVE_LIST'], encoding='utf-8')
    # neg_train = json.load(f)
    # f.close()
    with open(params['NEGATIVE_LIST'], 'r') as f:
        neg_train = json.load(f)
    #neg_train = hl_ag_train #negative set is the same as hl_ag_train
    # f_val = open(params['VAL_NEG'], encoding='utf-8')
    # neg_val = json.load(f_val)
    # f_val.close()
    with open(params['VAL_NEG'], 'r') as f:
        neg_val = json.load(f)


    len_l = len(l_train.keys())
    len_h = len(h_train.keys())

    
    len_hl = len(hl_train.keys())
    len_l_ag = len(l_ag_train.keys())
    len_h_ag = len(h_ag_train.keys())
    len_hl_ag = len(hl_ag_train.keys())

    total_len = len_l + len_h + len_hl + len_l_ag + len_h_ag + len_hl_ag
    # ab_nums = len_l, len_h, len_hl

    #gp_weight = 2/3
    #neg_weight = 0.25
    #neg_weight = 0.0
    #ab_weight = 1 - gp_weight - neg_weight
    l_weight = float(len_l) / float(total_len)
    h_weight = float(len_h) / float(total_len)
    hl_weight = float(len_hl) / float(total_len)
    l_ag_weight = float(len_l_ag) / float(total_len)
    h_ag_weight = float(len_h_ag) / float(total_len)
    hl_ag_weight = float(len_hl_ag) / float(total_len)

    weights = l_weight, h_weight, hl_weight, l_ag_weight, h_ag_weight, hl_ag_weight

    # 7. general protein set
    with open(params['GP_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # 0: pdb id separated by :
        # 1: date
        # 2: resolution
        # 3: MSA hash
        # 4: cluster_idx
        # 5: lenA:lenB
        rows = [[r[0], r[3], int(r[4]), [int(plen)
                                         for plen in r[5].split(':')]] for r in reader]
    gp_train = defaultdict(list)

    for r in rows:
        gp_train[r[2]].append((r[0], r[1], r[-1]))  # (ids, MSA_hash, len_list)

    return gp_train, l_train, h_train, hl_train, \
        l_ag_train, h_ag_train, hl_ag_train, neg_train,\
        h_val, hl_val, h_ag_val, hl_ag_val,\
        hl_ag_test,  neg_val, weights
        # hl_ag_test, train_all, neg_val, weights
        

# slice long chains
def get_crop(l, mask, device, crop_size, unclamp=False):
    """_summary_

    Args:
        l (_type_): _description_
        mask (_type_): _description_
        device (_type_): _description_
        crop_size (_type_): _description_
        unclamp (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    sel = torch.arange(l, device=device)
    if l <= crop_size:
        return sel

    size = crop_size

    mask = ~(mask[:, :3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]

    # bias it toward N-term.. (follow what AF did.. but don't know why)
    if unclamp:
        x = np.random.randint(len(exists)) + 1
        res_idx = exists[torch.randperm(x)[0]].item()
    else:
        res_idx = exists[torch.randperm(len(exists))[0]].item()
    lower_bound = max(0, res_idx - size + 1)
    upper_bound = min(l - size, res_idx + 1)
    start = np.random.randint(lower_bound, upper_bound)
    return sel[start: start + size]

def get_negative_crop(
        xyz,
        mask,
        sel,
        len_s,
        params,
        label,
        false_interface=None, 
        true_interface_12=None,
        true_interface_10_ab=None,
        true_interface_10_ag=None,
        cutoff=10.0,
        eps=1e-6):
    
    device = xyz.device
    complex1_ca = xyz[:len_s[0], 1]
    complex2_ca = xyz[len_s[0]:, 1]
    
    def get_choice(true_interface, mask):
        choice = np.random.choice(true_interface, 1)
        while not mask[choice]:
            choice = np.random.choice(true_interface, 1)
        return choice
    
    def get_topk(choice, length, topk, xyz_ca, mask, eps, exclude=None):
        distance = (
                torch.cdist(xyz_ca, xyz_ca[choice][None]).reshape(-1)
                + torch.arange(length, device=xyz.device)*eps)
        if exclude is not None:
            mask[exclude] = False
        cond = mask * mask[choice]
        distance[~cond] = 99999.9
        #less_than_25 = torch.where(distance < 50)[0]
        _, idx = torch.topk(distance, topk, largest=False)
        
        return idx
        #combined = torch.cat((less_than_25, idx))
        #uniques, counts = combined.unique(return_counts=True)
        #intersection = uniques[counts > 1]
        #return intersection
    
    if false_interface == None: # if not false_interface
        complex1_choice = get_choice(true_interface_10_ab['antibody'], mask[:len_s[0], 1])
        complex2_choice = get_choice(true_interface_10_ag['antigen'], mask[len_s[0]:, 1])
    
        # get length for each chain
        if len_s[1] <= int(params["CROP"]/2):
            complex1_length = params["CROP"] - len_s[1]
            complex2_length = len_s[1]
        else:
            complex1_length = min(len_s[0], int(params["CROP"]/2))
            complex2_length = params["CROP"] - complex1_length

        complex1_topk = get_topk(complex1_choice, len_s[0], complex1_length, \
                xyz[:len_s[0], 1], mask[:len_s[0], 1], eps)
        complex2_topk = get_topk(complex2_choice, len_s[1], complex2_length, \
                xyz[len_s[0]:, 1], mask[len_s[0]:, 1], eps)

        idx = torch.cat((complex1_topk, complex2_topk + len_s[0]))
        #print('idx', idx)
        sel, _ = torch.sort(sel[idx])
        return sel
    
    else: #if false_interface
        # print('false_interface', false_interface)
        # print(true_interface_10_ab['antibody'])
        complex1_choice = get_choice(true_interface_10_ab['antibody'], mask[:len_s[0], 1])
        #print('complex1_choice', complex1_choice)
        # print('mask', all (not mask[len_s[0]:, 1][index] for index in false_interface))
        if all (not mask[len_s[0]:, 1][index] for index in false_interface): #all false interface is masked
            print('all false interface is masked!!!')
            complex2_choice = get_choice(list(range(len_s[1])), mask[len_s[0]:, 1])
        else:
            complex2_choice = get_choice(false_interface, mask[len_s[0]:, 1])
        #print('complex2_choice', complex2_choice)
        epitope_excluded_length = len_s[1] - len(true_interface_12)
        #print('epitope_excluded_length', epitope_excluded_length)
        if epitope_excluded_length <= int(params["CROP"]/2):
            complex1_length = params["CROP"] - epitope_excluded_length
            complex2_length = epitope_excluded_length
        else:
            complex1_length = min(len_s[0], int(params["CROP"]/2))
            complex2_length = params["CROP"] - complex1_length
        
        complex1_topk = get_topk(complex1_choice, len_s[0], complex1_length, \
                xyz[:len_s[0], 1], mask[:len_s[0], 1], eps)
        #print('complex1_topk', complex1_topk)
        complex2_topk = get_topk(complex2_choice, len_s[1], complex2_length, \
                xyz[len_s[0]:, 1], mask[len_s[0]:, 1], eps, exclude=true_interface_12)
        #print('complex2_topk', complex2_topk)
        idx = torch.cat((complex1_topk, complex2_topk + len_s[0]))
        #print('idx', idx)
        sel, _ = torch.sort(sel[idx])
        return sel
        

def get_complex_crop(len_s, mask, device, params):
    """_summary_

    Args:
        len_s (_type_): _description_
        mask (_type_): _description_
        device (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)

    n_added = 0
    n_remaining = sum(len_s)
    preset = 0
    sel_s = list()

    for i, length in enumerate(len_s):
        n_remaining -= length
        crop_max = min(params["CROP"] - n_added, length)
        crop_min = min(length, max(0, params["CROP"] - n_added - n_remaining))

        if i == 0:
            crop_max = min(crop_max, params["CROP"] - 5)
        crop_size = np.random.randint(crop_min, crop_max + 1)
        n_added += crop_size

        mask_chain = ~(mask[preset: preset + length, :3].sum(dim=-1) < 3.0)
        # check if backbone exist
        exists = mask_chain.nonzero(as_tuple=True)[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        # radomly select from backbone exist 
        lower_bound = max(0, res_idx - crop_size + 1)
        upper_bound = min(length - crop_size, res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + preset
        sel_s.append(sel[start: start + crop_size])
        preset += length

    return torch.cat(sel_s)
def get_surface(
        params,
        len_s,
        item,
):
    """
    """
    antigen = item.split('_')[-1]
    # rsa_file = params['AB_MSA_DIR'] +f'/{item}/{antigen}/{item}_rank_001_{antigen}.rsa' # for flexible
    # if item[:4].upper() == '5VPG': item = "3RVW"+item[4:]
    
    # rsa_file = f"/home/kkh517/Github/antibody_benchmark/pdbs/{item[:4].upper()}_l_u.rsa"
    # if item[:4] in ['3hmx','4fqi','4gxu','5kov','5wux','5y9j']:
    # rsa_file = f'/home/kkh517/Github/antibody_benchmark/re_index_ag_2/{item[:4].upper()}_l_u.rsa'
    # rsa_file = glob(params['AB_MSA_DIR'] +f'/{item}/{antigen}/*.rsa')[0] # for rigid
    rsa_file = f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/alphafold2.rsa' # new_test
    print('rsa_file', rsa_file)
    rsa = {} ; surface_residues = []
    surface_residues = torch.zeros(sum(len_s))
    pdb_id = item[:4].lower()
    # shift = json.load(open("/home/kkh517/pdb_shift_dict.json"))[pdb_id]
    with open(rsa_file, 'r') as f:
       for line in f:
            if not line.startswith("RES"):
               continue
            x = line.split()
            chain = x[2]
            # if pdb_id in ['2vis', '3hmx','3wd5','4fqi','4gxu','5kov','5wux','5y9j']:
            #     if chain != 'A':
            #         continue
            # elif pdb_id == '4fqi':
            #     if chain != 'B' or chain != 'A':
            #         continue

            key = f"{x[1]}_{antigen}_{x[3]}"
            rsa[key] = float(x[5])
            if float(x[5]) > 40: # hyper parameter! can be changed 20 ~ 30
                # print('surface residue', x[1], x[3])
                try:
                    surface_residues[int(x[3])+len_s[0]-1] = True
                except Exception as e:
                    print('surface residue error', e)
                    # print('line', line)
                    continue
            # else:
            #     surface_residues.append(0.0)
    # print('surface_residues', surface_residues.shape)
    return surface_residues
def get_epi_full(
        xyz,
        mask,
        len_s,
        item,
        cutoff=10.0,):
    """_summary_

    Args:
        xyz (_type_): _description_
        mask (_type_): _description_
        sel (_type_): _description_
        len_s (_type_): _description_
        params (_type_): _description_
        label (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 10.0.
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    if len(len_s) == 0:
        print('no interface -> return zero tensor')
        return torch.zeros(xyz.shape[0])
    device = xyz.device
    # get interface residue
    # print('spatial crop', len_s)
    cond = torch.cdist(xyz[: len_s[0], 1], xyz[len_s[0]:, 1]) < cutoff
    # cond = torch.cdist(xyz[len_s[0] : , 1], xyz[len_s[0]:, 1]) < cutoff
    # print('cond check', cond.sum())
    cond = cond.to(device); mask = mask.to(device)
    cond = torch.logical_and(cond, mask[: len_s[0], None, 1] * mask[None, len_s[0]:, 1])
        #cond, mask[len_s[:1].sum()+1 : , None, 1] * mask[None, :len_s[:1].sum()+1:, 1]
    # )
    # print('cond check again', cond.sum())
    i, j = torch.where(cond)
    # j = j + len_s[0]
    # print('i',i.shape)
    # print('j',j.shape)
    # for n in range(i.shape[0]):
    #     print(n)
    #     print(i[n],j[n])
    epitope_ifaces = j + len_s[0]
    # if len(ifaces) < 1:
    #     print("ERROR: no iface residue????", label)
    #     return get_complex_crop(len_s, mask, device, params)
    epi_full = torch.zeros(sum(len_s))
    j = torch.unique(j)
    # print(f"{item} : epitope_ifaces {j+1}")
    # epitope_ifaces = torch.Tensor([269, 304, 337, 359, 388, 389]).long() # iitp
    # print(f"epitope_ifaces {epitope_ifaces}")
    epi_full[epitope_ifaces] = 1
    # print('epi_full_idxs', epitope_ifaces)
    # print('epi_full check', epi_full.sum())
    if epi_full.sum() == 0:
        print(f'no interface?? {item}')
        return torch.zeros(xyz.shape[0])
    return epi_full
    # distance = (
    #     torch.cdist(xyz[:, 1], xyz[cnt_idx, 1][None]).reshape(-1)
    #     + torch.arange(len(xyz), device=xyz.device) * eps
    # )
    # cond = mask[:, 1] * mask[cnt_idx, 1]
    # distance[~cond] = 999999.9
    # _, idx = torch.topk(distance, params["CROP"], largest=False)

    # sel, _ = torch.sort(sel[idx])
    # return sel
def get_spatial_crop_new(
        xyz,
        mask,
        sel,
        L_start_sel,
        L_end_sel,
        len_s,
        params,
        label,
        cutoff=10.0,
        eps=1e-6):
    """_summary_

    Args:
        xyz (_type_): _description_
        mask (_type_): _description_
        sel (_type_): _description_
        len_s (_type_): _description_
        params (_type_): _description_
        label (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 10.0.
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    device = xyz.device

    # get interface residue
    cond = torch.cdist(xyz[L_start_sel[0]: L_end_sel[0], 1],
                       xyz[L_start_sel[1]:L_end_sel[1], 1]) < cutoff
    cond = torch.logical_and(cond,
                             mask[L_start_sel[0]: L_end_sel[0],
                                  None,
                                  1] * mask[None,
                                            L_start_sel[1]:L_end_sel[1],
                                            1])
    i, j = torch.where(cond)
    ifaces = torch.cat([i + L_start_sel[0], j + L_start_sel[1]])
    if len(ifaces) < 1:
        print("ERROR: no iface residue????", label)
        return get_spatial_crop(xyz, mask, sel, len_s, params, label)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    distance = (
        torch.cdist(xyz[:, 1], xyz[cnt_idx, 1][None]).reshape(-1)
        + torch.arange(len(xyz), device=xyz.device) * eps
    )
    cond = mask[:, 1] * mask[cnt_idx, 1]
    distance[~cond] = 999999.9
    _, idx = torch.topk(distance, params["CROP"], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel
def get_spatial_crop_kh(
        xyz, #xyz[0] [L, 27, 3]
        mask, #mask[0] [L, ]
        sel, #torch.arange(sum(L_s))
        len_s, #interface_split
        params = {"CROP": 256},
        label='ABCD',#item
        cutoff=10.0,
        eps=1e-6):
    """_summary_

    Args:
        xyz (_type_): _description_
        mask (_type_): _description_
        sel (_type_): _description_
        len_s (_type_): _description_
        params (_type_): _description_
        label (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 10.0.
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    device = xyz.device

    # get interface residue
    # print('spatial crop', len_s)
    cond = torch.cdist(xyz[: len_s[0], 1], xyz[len_s[0]:, 1]) < cutoff
    # cond = torch.cdist(xyz[len_s[0] : , 1], xyz[len_s[0]:, 1]) < cutoff
    cond = torch.logical_and(
        # cond, mask[: len_s[0], None, 1] * mask[None, len_s[0]:, 1]
        cond, mask[len_s[0] : , None, 1] * mask[None, :len_s[0]:, 1]
    )
    i, j = torch.where(cond)
    # print('i', i)
    # print('j', j)
    ifaces = torch.cat([i, j + len_s[0]])
    # print('ifaces', len(ifaces))
    if len(ifaces) < 1:
        print("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    distance = (
        torch.cdist(xyz[:, 1], xyz[cnt_idx, 1][None]).reshape(-1)
        + torch.arange(len(xyz), device=xyz.device) * eps
    )
    cond = mask[:, 1] * mask[cnt_idx, 1]
    # print('cond.shape', cond.shape)
    distance[~cond] = 999999.9
    # print(distance)
    _, idx = torch.topk(distance, params["CROP"], largest=False)

    sel, _ = torch.sort(sel[idx])
    # print('sel',sel)
    # from sel, take random 5 index bigger than len_s[0]
    epitope_length = random.randint(1,5)
    # print('random', epitope_length)
    epitope = sel[torch.randperm(len(sel))[:epitope_length]]
    # print('epitope', epitope)
    # make [B,L] tensor, if index is in epitope, 1, else 0
    # crop_epitope = torch.zeros(params["CROP"])
    # crop_epitope.scatter_(dim=0, index=epitope, value =1)
    # arange_len_s = torch.arange(sum(len_s)) # [L]
    arange_len_s = torch.arange(params["CROP"]) # [256] which is cropping size....
    crop_epitope = torch.where(
        # torch.isin(arange_len_s, epitope), torch.ones_like(arange_len_s), torch.zeros_like(arange_len_s)
        torch.isin(sel, epitope), torch.ones_like(sel), torch.zeros_like(sel)
    )   
    # print('crop_epitope', crop_epitope[None].shape)
    # print('epitope', epitope)
    return sel, crop_epitope[None]

def get_spatial_crop(
        xyz,
        mask,
        sel,
        len_s,
        params,
        label,
        cutoff=10.0,
        eps=1e-6):
    """_summary_

    Args:
        xyz (_type_): _description_
        mask (_type_): _description_
        sel (_type_): _description_
        len_s (_type_): _description_
        params (_type_): _description_
        label (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 10.0.
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    device = xyz.device

    # get interface residue
    # print('spatial crop', len_s)
    cond = torch.cdist(xyz[: len_s[0], 1], xyz[len_s[0]:, 1]) < cutoff
    # cond = torch.cdist(xyz[len_s[0] : , 1], xyz[len_s[0]:, 1]) < cutoff
    cond = torch.logical_and(
        cond, mask[: len_s[0], None, 1] * mask[None, len_s[0]:, 1]
        #cond, mask[len_s[:1].sum()+1 : , None, 1] * mask[None, :len_s[:1].sum()+1:, 1]
    )
    i, j = torch.where(cond)
    ifaces = torch.cat([i, j + len_s[0]])
    if len(ifaces) < 1:
        print("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    distance = (
        torch.cdist(xyz[:, 1], xyz[cnt_idx, 1][None]).reshape(-1)
        + torch.arange(len(xyz), device=xyz.device) * eps
    )
    cond = mask[:, 1] * mask[cnt_idx, 1]
    distance[~cond] = 999999.9
    _, idx = torch.topk(distance, params["CROP"], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel


# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer):
    # N, L = msa_orig.shape[:2]
    msa = torch.cat([msa_orig for imer in range(nmer)], dim=1)
    ins = torch.cat([ins_orig for imer in range(nmer)], dim=1)
    return msa, ins


def merge_a3m_antibody(a3m, homo_list):

    L_s = []
    N_s = []
    msa_mono_s = []
    ins_mono_s = []

    for a3ms in a3m:
        L_s.append(a3ms['msa'].shape[1])
        N_s.append(a3ms['msa'].shape[0])
        msa_mono_s.append(a3ms['msa'])
        ins_mono_s.append(a3ms['ins'])

    #print('N_s', N_s)
    #print('L_s', L_s)
    N_start, N_end = [], []
    N_start.append(1)
    temp = 1
    for i in N_s:
        temp += i - 1
        N_start.append(temp)
        N_end.append(temp)
    sum_Ns = temp
    N_start = N_start[:-1]
    N_final = [(i, j) for (i, j) in zip(N_start, N_end)]

    L_s_sorted = []
    L_s_dict = defaultdict(int)
    N_final_dict = defaultdict(tuple)
    msa_dict = defaultdict(list)
    ins_dict = defaultdict(list)
    chain_count = 0
    
    for i, ch in enumerate(homo_list):   # TO DO : homomer에 대해서도 heteromer로 바꿔서 block diagonal로 돌리기
        # ex) homo_list = [[0], [1, 2], [3, 5], [4]]
        for j in ch:
            L_s_dict[j] = L_s[i]
            N_final_dict[j] = N_final[i]
            msa_dict[j] = msa_mono_s[i]
            ins_dict[j] = ins_mono_s[i]
            chain_count += 1

    L_s_sorted = OrderedDict(sorted(L_s_dict.items()))
    L_s = [v for k, v in L_s_sorted.items()]
    L_start, L_end = [], []
    L_start.append(0)
    temp = 0
    for i in L_s:
        temp += i
        L_start.append(temp)
        L_end.append(temp)
    sum_Ls = temp
    L_start = L_start[:-1]
    L_final = [(i, j) for (i, j) in zip(L_start, L_end)]

    L_final_dict = defaultdict(tuple)
    for ch in homo_list:
        for j in ch:
            L_final_dict[j] = L_final[j]

    # N_final_dict = {0:(0, #of chain0), 1:(#of chain0, #of chain0+#of chain1)}

    # get query of msa, ins
    query_msa = []
    query_ins = []
    for i in list(range(chain_count)):
        query_msa.append(msa_dict[i][0])
        query_ins.append(ins_dict[i][0])

    query_msa = torch.cat(query_msa)
    query_ins = torch.cat(query_ins)
    # get initial matrix
    msa_orig = torch.full((sum_Ns, sum_Ls), 20)
    ins_orig = torch.full((sum_Ns, sum_Ls), 0)

    msa_orig[0, :] = query_msa
    ins_orig[0, :] = query_ins

    for i in list(range(chain_count)):
        sN, eN = N_final_dict[i][0], N_final_dict[i][1]
        sL, eL = L_final_dict[i][0], L_final_dict[i][1]
        msa_orig[sN:eN, sL:eL] = msa_dict[i][1:]
        ins_orig[sN:eN, sL:eL] = ins_dict[i][1:]

    return {
        'msa': msa_orig,
        'ins': ins_orig,
        'L_s': L_s,
        'N_final_dict': N_final_dict}


def merge_a3m_antibody_old(a3m):

    L_s = []
    N_s = []
    msa_mono_s = []
    ins_mono_s = []

    for a3ms in a3m:
        L_s.append(a3ms['msa'].shape[1])
        N_s.append(a3ms['msa'].shape[0])
        msa_mono_s.append(a3ms['msa'])
        ins_mono_s.append(a3ms['ins'])

    # merge msa
    query = torch.cat([msa_mono[0]
                      for msa_mono in msa_mono_s]).unsqueeze(0)  # (1, L)
    # print('query', query)
    #print('query length', len(query[0]))
    msa_orig = [query]

    left_pad = 0
    right_pad = sum(L_s)
    for i, L in enumerate(L_s):
        if N_s[i] < 2:  # strange, should adjust pad number?
            continue
        right_pad -= L
        extra = torch.nn.functional.pad(
            msa_mono_s[i][1:], (left_pad, right_pad), "constant", 20)
        msa_orig.append(extra)
        left_pad += L
    msa_orig = torch.cat(msa_orig, dim=0)

    # merge insertion
    query = torch.cat([ins_mono[0] for ins_mono in ins_mono_s]).unsqueeze(0)
    ins_orig = [query]
    left_pad = 0
    right_pad = sum(L_s)
    for i, L in enumerate(L_s):
        if N_s[i] < 2:
            continue
        right_pad -= L
        extra = torch.nn.functional.pad(
            ins_mono_s[i][1:], (left_pad, right_pad), "constant", 20)
        ins_orig.append(extra)
        left_pad += L
    ins_orig = torch.cat(ins_orig, dim=0)

    return {'msa': msa_orig, 'ins': ins_orig, 'L_s': L_s}

# Generate input features for single-chain


def featurize_single_chain(
        a3m,
        template_params,
        params,
        item,
        cluster,
        unclamp=False,
        pick_top=True,
        random_noise=5.0,
        validation=False):
    item = a3m['label']
    pdb = parse_pdb(params['AB_DIR'] + f'/{item}/{item}_renum.pdb')
    # tplt = torch.load(template_params)
    
    tplt = None
    # tplt = parse_templates(template_params)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    if len(msa) > params['BLOCKCUT']:
        msa_del, ins_del = MSABlockDeletion(msa[1:], ins[1:])
        msa = torch.cat((msa[None, 0], msa_del), dim=0)
        ins = torch.cat((ins[None, 0], ins_del), dim=0)
    
    sample = len(msa)
    if sample > 3000:
        msa_sel = torch.randperm(sample-1)[:2999]
        msa = torch.cat((msa[:1, :], msa[1:, :][msa_sel]), dim=0)
        ins = torch.cat((ins[:1, :], ins[1:, :][msa_sel]), dim=0)

    #print('msa datatype', msa.dtype)
    #print('ins datatype', ins.dtype)
    # set size of the pdb ['xyz'] and ['mask'] with ['idx']
    L = msa.shape[1]
    xyz_new = torch.full((1, L, 14, 3), np.nan).float()
    mask_new = torch.full((1, L, 14), np.nan).float()

    #print('shape idx', pdb['idx'].shape)
    #print(pdb['idx'])
    #print('length', L)
    for i, idx in enumerate(pdb['idx']):
        xyz_new[0][idx - 1, :, :] = pdb['xyz'][i]
        mask_new[0][idx - 1, :] = pdb['mask'][i]

    pdb['xyz'] = xyz_new
    pdb['mask'] = mask_new

    # seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

    # get template features
    # print('mintplt', params['MINTPLT'], 'maxtpl', params['MAXTPLT'])
    ntempl = np.random.randint(params["MINTPLT"], params["MAXTPLT"] + 1)
    if validation:
        ntempl = 0
    xyz_t, f1d_t, mask_t = TemplFeaturize_kh(
        tplt,
        msa.shape[1],
        params,
        npick=ntempl,
        offset=0,
        pick_top=pick_top,
        random_noise=random_noise,
    )

    # get ground-truth structures
    idx = torch.arange(L)
    xyz = INIT_CRDS.reshape(1, 1, 27, 3).repeat(1, len(idx), 1, 1)
    #print('pdbxyz size', pdb['xyz'].shape)
    xyz[:, :, :14] = pdb["xyz"]
    mask = torch.full((1, len(idx), 27), False)
    mask[:, :, :14] = pdb["mask"]
    xyz = torch.nan_to_num(xyz)

    # Residue cropping
    sel = get_crop(
        len(idx), mask, msa.device, params["CROP"], unclamp=unclamp
    )
    # seq = seq[:, crop_idx]
    # msa_seed_orig = msa_seed_orig[:, :, crop_idx]
    # msa_seed = msa_seed[:, :, crop_idx]
    # msa_extra = msa_extra[:, :, crop_idx]
    # mask_msa = mask_msa[:, :, crop_idx]
    # xyz_t = xyz_t[:, crop_idx]
    # f1d_t = f1d_t[:, crop_idx]
    # mask_t = mask_t[:, crop_idx]
    # xyz = xyz[crop_idx]
    # mask = mask[crop_idx]
    # idx = idx[crop_idx]

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    same_chain = torch.ones(L, L).long()
    chain_idx = torch.zeros(L)

    epi_full = torch.zeros(L)

    # print ("featurize_single", ntempl, xyz_t.shape, msa_seed.shape, msa_extra.shape)
    # print('output item', item)
    return (
        # seq.long(),
        cluster,
        item,
        sel,
        torch.tensor([]),
        # params,
        msa,
        ins,
        # msa_seed_orig.long(),
        # msa_seed.float(),
        # msa_extra.float(),
        # mask_msa,
        xyz.float(),
        mask,
        idx.long(),
        xyz_t.float(),
        f1d_t.float(),
        mask_t,
        xyz_prev.float(),
        mask_prev,
        same_chain,
        chain_idx,
        unclamp,
        False,
        [], #interface_split
        epi_full, #epi_full
        validation,

    )


def merge_a3m_hetero(a3mA, a3mB, L_s, orig={}):
    # merge msa
    if 'msa' in orig:
        msa = [orig['msa']]
    else:
        query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]
                          ).unsqueeze(0)  # (1, L)
        msa = [query]

    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(
            a3mA['msa'][1:], (0, L_s[1]), "constant", 20)  # pad gaps
        msa.append(extra_A)

    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(
            a3mB['msa'][1:], (L_s[0], 0), "constant", 20)
        msa.append(extra_B)

    msa = torch.cat(msa, dim=0)

    if 'ins' in orig:
        ins = [orig['ins']]
    else:
        query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]
                          ).unsqueeze(0)  # (1, L)
        ins = [query]

    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(
            a3mA['ins'][1:], (0, L_s[1]), "constant", 0)  # pad gaps
        ins.append(extra_A)

    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(
            a3mB['ins'][1:], (L_s[0], 0), "constant", 0)
        ins.append(extra_B)

    ins = torch.cat(ins, dim=0)
    N_final_dict = {0: (1, a3mA['msa'].shape[0]), 1: (
        a3mA['msa'].shape[0], a3mA['msa'].shape[0] + a3mB['msa'].shape[0] - 1)}
    return {'msa': msa, 'ins': ins, 'N_final_dict': N_final_dict}


def loader_complex_gp(
        cluster,
        item,
        msa_hash,
        L_s,
        params,
        negative=False,
        pick_top=True,
        unclamp=False,
        random_noise=5.0):
    # print('input item', item)
    pdb_pair = item
    pMSA_hash = msa_hash
    print(f'{item} L_s',L_s)
    # get MSA, insertion feature
    msaA_id, msaB_id = pMSA_hash.split('_')
    # a3mA_fn = params['GP_DIR'] + f'/a3m/{msaA_id[:3]}/{msaA_id}.a3m'
    # a3mB_fn = params['GP_DIR'] + f'/a3m/{msaB_id[:3]}/{msaB_id}.a3m'
    a3mA_fn = params['GP_DIR'] + f'/loop_PPI_a3m/{msaA_id[:3]}/{msaA_id}.a3m' # check
    a3mB_fn = params['GP_DIR'] + f'/loop_PPI_a3m/{msaB_id[:3]}/{msaB_id}.a3m' # check
    a3mA = get_msa(a3mA_fn, msaA_id, max_seq=params['MAXSEQ'] * 2)
    a3mB = get_msa(a3mB_fn, msaB_id, max_seq=params['MAXSEQ'] * 2)
    a3m = merge_a3m_hetero(a3mA, a3mB, L_s)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    N_final_dict = a3m['N_final_dict']
    # load csv file into dictionary
    TM_dict = {}
    with open('/home/kkh517/submit_files/gp_TMscore.csv', mode='r') as infile:
        reader = csv.reader(infile)
        TM_dict = {rows[0]: rows[1] for rows in reader}
    
    if len(msa) > params['BLOCKCUT']:
        msas, inss = [], []
        msas.append(msa[None, 0])
        inss.append(ins[None, 0])

        for i in np.arange(len(N_final_dict)):
            sN, eN = N_final_dict[i][0], N_final_dict[i][1]
            if eN - sN > 50:
                msa_del, ins_del = MSABlockDeletion(msa[sN:eN], ins[sN:eN])
                msas.append(msa_del)
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
    #    msa = msa[msa_sel]
    #    ins = ins[msa_sel]

    # get template features
    # tpltA_fn = params['GP_DIR'] + f'/templ_pt/{msaA_id[:3]}/{msaA_id}.pt'
    # tpltB_fn = params['GP_DIR'] + f'/templ_pt/{msaB_id[:3]}/{msaB_id}.pt'
    # tpltA_fn = params['GP_DIR'] + f'/loop_PPI_pdb/{msaA_id[:3]}/{msaA_id}/{msaA_id}_reference.pdb' # for rigid
    # tpltB_fn = params['GP_DIR'] + f'/loop_PPI_pdb/{msaB_id[:3]}/{msaB_id}/{msaB_id}_reference.pdb' # for rigid
    try:
        tpltA_fn = random.choice(glob(params['GP_DIR'] + f'/from_kisti/loop_PPI_pdb/{msaA_id[:3]}/{msaA_id}_unrelaxed_rank_00*.pdb')) # for flexible
    except Exception as e:
        print(f'no template found {e}')
        tpltA_fn = None
    try:
        tpltB_fn = random.choice(glob(params['GP_DIR'] + f'/from_kisti/loop_PPI_pdb/{msaB_id[:3]}/{msaB_id}_unrelaxed_rank_00*.pdb')) # for flexible
    except Exception as e:
        print(f'no template found {e}')
        tpltB_fn = None
        
    # if float(TM_dict[tpltA_fn]) < 0.8 :
    #     tpltA_fn = params['GP_DIR'] + f'/loop_PPI_pdb/{msaA_id[:3]}/{msaA_id}/{msaA_id}_reference.pdb' # for rigid
    # if float(TM_dict[tpltB_fn]) < 0.8 :
    #     tpltB_fn = params['GP_DIR'] + f'/loop_PPI_pdb/{msaB_id[:3]}/{msaB_id}/{msaB_id}_reference.pdb' # for rigid
    # tpltA = torch.load(tpltA_fn)
    # tpltB = torch.load(tpltB_fn)
    #ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT'] + 1)
    #ntemplB = np.random.randint(0, params['MAXTPLT'] + 1 - ntemplA)
    # print('mintplt', params['MINTPLT'], 'maxtpl', params['MAXTPLT'])
    ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    ntemplB = ntemplA
    #print('general L_s', L_s)
    # xyz_t_A, f1d_t_A, mask_t_A = TemplFeaturize(
    #     tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(
    #         1, max(
    #             ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    tplts_dict = {'H': [tpltA_fn], 'L': [tpltB_fn], 'AG' : None}
    # xyz_t_A, f1d_t_A, mask_t_A = CustomTemplFeaturize_kh(item,None, tpltA_fn, L_s,0,params=None, p_antibody=None)
    # #for ii, x in enumerate(xyz_t_A):
    # #    writepdb(
    # #        f'check_template/{item}_first_{ii}.pdb',
    # #        xyz_t_A[ii],
    # #        msa[0][:L_s[0]],
    # #        mask_t_A[ii])
    # # xyz_t_B, f1d_t_B, mask_t_B = TemplFeaturize(
    # #     tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(
    # #         1, max(
    # #             ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    # xyz_t_B, f1d_t_B, mask_t_B = CustomTemplFeaturize_kh(item,None, tpltB_fn, L_s, 1,params=None,p_antibody=None)
    # #print(xyz_t_B)
    # #print(f1d_t_B)
    # #print(mask_t_B)
    # #for ii, x in enumerate(xyz_t_B):
    # #    writepdb(
    # #        f'check_template/{item}_second_{ii}.pdb',
    # #        xyz_t_B[ii],
    # #        msa[0][L_s[0]:],
    # #        mask_t_B[ii])
    # xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B, random_noise=10.0)), dim=1)
    # f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1)
    # mask_t = torch.cat((mask_t_A, mask_t_B), dim=1)
    # print('tplts_dict', tplts_dict)
    xyz_t, f1d_t, mask_t = CustomTemplFeaturize_kh(item,tplts_dict, L_s,antibody=False)
    # get initial coordinates

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdb = parse_pdb_antibody(
        params['GP_DIR'] +
        # f'/pdb/{pdbA_id[1:3]}/{pdb_pair}.pdb',
        f'/loop_PPI_ans/{pdbA_id[1:3]}/{pdb_pair}.pdb', # is this right??
        pdb_pair,
        antibody=False)

    xyz_new = [torch.full((i, 14, 3), np.nan).float() for i in L_s]
    mask_new = [torch.full((i, 14), np.nan).float() for i in L_s]

    for i, _ in enumerate(L_s):
        for j, idx in enumerate(pdb['idx'][i]):
            xyz_new[i][idx - 1, :, :] = torch.from_numpy(pdb['xyz'][i][j])
            mask_new[i][idx - 1, :] = pdb['mask'][i][j]

    xyz_new = torch.cat(xyz_new)
    mask_new = torch.cat(mask_new)
    xyz_new = xyz_new.unsqueeze(0)
    mask_new = mask_new.unsqueeze(0)

    pdb['xyz'] = xyz_new
    pdb['mask'] = mask_new
    pdb['mask'] = torch.nan_to_num(pdb['mask'])

    xyz = INIT_CRDS.reshape(1, 1, 27, 3).repeat(1, sum(L_s), 1, 1)
    mask = torch.full((1, sum(L_s), 27), False)
    xyz[:, :, :14] = pdb['xyz']
    xyz = torch.nan_to_num(xyz)
    mask[:, :, :14] = pdb['mask']

    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 100  # to let network know about chain breaks

    # indicator for which residues are in same chain
    same_chain = torch.zeros((sum(L_s), sum(L_s))).long()
    same_chain[:L_s[0], :L_s[0]] = 1
    same_chain[L_s[0]:, L_s[0]:] = 1

    chain_idx = torch.zeros(sum(L_s)).long()
    chain_idx[:L_s[0]] = 0
    chain_idx[L_s[0]:] = 1
    # get epitope
    interface_split = L_s

    epi_info = get_epi_full(xyz[0],mask[0],interface_split,item)


    # Do cropping

    if sum(L_s) > params['CROP']:
        sel = get_spatial_crop(
            xyz[0], mask[0], torch.arange(
                sum(L_s)), L_s, params, item)
        # sel, epitope_crop = get_spacial_crop_kh(xyz[0], mask[0], torch.arange(sum(L_s)), L_s, params, item)
    else:
        sel = torch.arange(sum(L_s))
        # epitope_crop = ... need to make new function to get epitope without cropping

    #for ii, x in enumerate(xyz):
    #    writepdb(f'check_parser/{item}_{ii}.pdb', xyz[ii], msa[0], mask[ii])
    
    #print('xyz_t', xyz_t.shape)
    #print('mask_t', mask_t.shape)
    #for ii, x in enumerate(xyz_t):
    #    writepdb(
    #        f'check_template/{item}_{ii}.pdb',
    #        xyz_t[ii],
    #        msa[0],
    #        mask_t[ii])

    # print('output item', item)
    return (
        cluster,
        item,
        sel,
        torch.tensor(L_s),
        msa,
        ins,
        xyz.float(), # true_crds
        mask,   #mask_crds
        idx.long(), #idx_pdb
        xyz_t.float(),
        f1d_t.float(),
        mask_t,
        xyz_prev.float(),
        mask_prev,
        same_chain,
        chain_idx,
        unclamp,
        negative,
        [],
        epi_info,
        False, #validation is False
    )

def CustomTemplFeaturize_kh(item,tplts, L_s, antibody=True):
    """
    Returns a xyz, t1d, mask_t
    
    Parameters
    ----------
    tplts : dict ; dict of templates path
        tplts['H'] : template H path or template HL path or None
        tplts['L'] : template L path or None
        tplts['AG'] : template AG path or None

    Returns
    -------
    xyz : torch.tensor ; (B,L,N,3)
    t1d : torch.tensor ; (B,L,N,14)
    mask_t : torch.tensor ; (B,L,N)
    """
    # if (item == '4gxu_M_N_AB') or (item == '3hmx_H_L_AB'):
    #     cdr_mask=~torch.load('/home/kkh517/CDR_pdb_dict.pt')[item[:-1]]['CDR'][0].bool().unsqueeze(-1)
    # else:
    #     cdr_mask=~torch.load('/home/kkh517/CDR_pdb_dict.pt')[item]['CDR'][0].bool().unsqueeze(-1)
    with open('/home/kkh517/benchmark_set_after210930/testset_CDR.pkl', 'rb') as f: # new_test
        testset_CDR = pickle.load(f)
    cdr_mask = ~testset_CDR[item]['CDR'].bool().unsqueeze(-1)
    cdr_mask = torch.ones_like(cdr_mask) # for rigid body


    # print(f"cdr_mask {cdr_mask.shape}")
    def get_init_xyz(qlen,random_noise=5.0):
        # xyz = torch.full((qlen, 27, 3), np.nan).float()
        xyz = (INIT_CRDS.reshape(1, 27, 3).repeat(qlen, 1, 1)  # [L_ch, 27, 3]
               + torch.rand(qlen, 1, 3) * random_noise) # for SE(3) transformer...
        seq = torch.full((qlen,), 20).long() # all gaps  
        conf = torch.full((qlen,), 0.0).float()
        same_chain = torch.full((1,qlen,qlen),True).bool()
        return xyz, seq, conf, same_chain
    def get_xyz_t1d_mask(tplt, xyz, seq,conf,L_s,chain_count=0, antibody=None):
        xyz_for_mask = torch.full((xyz.shape[0], 27,3),np.nan).float()
        # pdb_id = tplt.split('/')[-1].split('_')[0].lower()
        pdb_id = tplt.split('/')[-2]
        # shift_dict = json.load(open(f'/home/kkh517/pdb_shift_dict.json'))
        # if pdb_id in shift_dict.keys() and chain_count != 0:
            # shift = shift_dict[pdb_id]
            # print('shift :', shift)
        # else:
        #     print("Couldn't find shift for ", pdb_id)
        #     shift = 0
        with open(tplt) as lines:
            temp_lines = lines.readlines()
            atom_lines = [line for line in temp_lines if line[:4] == 'ATOM']
            # print(f"atom_lines {atom_lines}")
            prev_chain = atom_lines[0][21] ; before_chain_len = 0
            for line in temp_lines:
                if line[:4] == 'ATOM':
                    chain = line[21]
                    resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
                    if chain != prev_chain:
                        before_chain_len += L_s[chain_count]
                        # print('before_chain_len :', before_chain_len)
                        chain_count += 1
                        prev_chain = chain
                        # print('befor_chain_len :', before_chain_len)
                    resNo += before_chain_len
                    aa_idx = aa2num[aa] if aa in aa2num.keys() else 20
                    idx = resNo-1
                    # print('idx, chain :', idx, chain)
                    for i_atm, tgtatm in enumerate(aa2long[aa_idx]):
                        if tgtatm == atom:
                            try:
                                conf[idx] = float(line[60:66])/100.0
                                seq[idx] = aa_idx
                                # if conf[idx] < 0.8: continue 
                                if conf[idx] < 0.8: conf[idx] = 0.3
                                xyz[idx, i_atm,:] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                                xyz_for_mask[idx, i_atm,:] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                                
                                
                            except Exception as e:
                                # print('error :', e)
                                print(f'{pdb_id} line :', line)
                                assert 1 == 0, f"{e}"
                                # print("")
                                continue
                    # seq[idx] = aa_idx
                    # conf[idx] = float(line[60:66])/100.0
        
        # if "renum_chain" in str(tplt): # for too big templates
        #     print('renum_chain')
        #     xyz = xyz + (torch.rand_like(xyz)+ torch.ones_like(xyz)) * 0.5
        #     conf = torch.ones_like(conf) * 0.9

        mask = torch.logical_not(torch.isnan(xyz_for_mask[:,:,0]))
        # Feb 6 update
        # if antibody != None:
        #     if item.split('_')[2] == '#':
        #         mask = mask * cdr_mask[:L_s[0]]
        #     else:
        #         mask = mask * cdr_mask[:sum(L_s[:2])]
            # print(f"item : {item}, mask : {mask.shape}, cdr_mask : {cdr_mask.shape}")
        # print(f"item : {item} xyz : {xyz.shape}, mask : {mask.shape}")
        # breakpoint()
        # if (item.startswith('8djm')) & (antibody == None) : breakpoint()
        try:
            xyz = center_and_realign_missing(xyz, mask)
        except Exception as e:
            print(f"{item} line 1869 error \n{e}")
        # xyz = center_and_realign_missing(xyz, mask)

        seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
        t1d = torch.cat((seq, conf[:,None]), dim=1)
        # print('t1d :', t1d.shape)
        # print('mask :', mask.shape)
        return xyz[None], t1d[None], mask[None]
    
    def get_blank_template(qlen, xyz, seq, conf):
        seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
        # xyz = xyz.unsqueeze(0)
        t1d = torch.cat((seq, conf[:,None]), -1)
        mask = torch.full((qlen,27), False).bool()
        return xyz[None], t1d[None], mask[None]
    # if len(L_s) == 5:
    #     L_s[3], L_s[4] = L_s[4], L_s[3]
    #     print("changed L_s : ", L_s)
    
    if antibody:
        h_chain = item.split('_')[1] ; l_chain = item.split('_')[2] ; ag_chain = item.split('_')[3]
    else:
        h_chain = "H" ; l_chain = "L" ; ag_chain = "#"
    if len(h_chain) ==1 and len(l_chain)==1 and (h_chain == '#' or l_chain == '#'):
        ab_len = L_s[0]; ab_chain = 1
        ag_len = sum(L_s[1:]) ;ag_chain = len(L_s)-1
    else:
        ab_len = L_s[0] + L_s[1] ; ab_chain = 2
        ag_len = sum(L_s[2:]) ; ag_chain = len(L_s)-2
    #featurize antibody
    assert ab_chain >=0 and ag_chain >=0, "chain should be non-negative integer"
    # print('ab_chain :', ab_chain)
    # print('ag_chain :', ag_chain)
    xyz_list = [] ; t1d_list = [] ; mask_list = []; ab_tplts = []; count = 0
    # xyz, seq, conf, same_chain = get_init_xyz(ab_len)
    if tplts['H'] is not None:
        for i, tplt in enumerate(tplts['H']): # i =2 for monomer with HH, i=1 for HL complex, monomer
            ab_tplts.append(tplt)
    if tplts['L'] is not None:
        for i, tplt in enumerate(tplts['L']):
            ab_tplts.append(tplt)
    
    if len(ab_tplts) != 0:
        for i_tplt, tplt in enumerate(ab_tplts):
            chain = h_chain + l_chain
            chain = chain.replace('#','')
            # print('chain :', chain)
            if len(chain) == 1:
                # print('qlen is :', L_s[i_tplt])
                xyz, seq, conf, same_chain = get_init_xyz(L_s[i_tplt])
                xyz, t1d, mask_t = get_xyz_t1d_mask(tplt, xyz, seq, conf,L_s,chain_count=0,antibody=antibody)
                if i_tplt != 1:
                    xyz_list.append(xyz) 
                else:
                    xyz_list.append(random_rot_trans(xyz, random_noise=0.0))
                t1d_list.append(t1d) ; mask_list.append(mask_t)
            elif len(chain) > 1 :
                xyz, seq, conf, same_chain = get_init_xyz(sum(L_s[:len(chain)]))
                xyz, t1d, mask_t = get_xyz_t1d_mask(tplt, xyz, seq, conf,L_s,chain_count=0,antibody=antibody)
                xyz_list.append(xyz) ; t1d_list.append(t1d) ; mask_list.append(mask_t)
    else : # no antibody template
        xyz, seq, conf, same_chain = get_init_xyz(sum(L_s[:ab_chain]))
        xyz, t1d, mask_t = get_blank_template(ab_len, xyz, seq, conf)
        xyz_list.append(xyz) ; t1d_list.append(t1d) ; mask_list.append(mask_t)
    if tplts['AG'] is not None: #antigen template case
        # chain = tplts['AG'][0].split('/')[-1].split('_')[-1].split('.')[0]
        
        chain = item.split('_')[3]   
        # print('chain :', chain)
        for i_tplt, tplt in enumerate(tplts['AG']):
            if len(chain) == 1:
                # print('tplt :', tplt)
                xyz, seq, conf, same_chain = get_init_xyz(L_s[ab_chain + i_tplt])
                xyz, t1d, mask_t = get_xyz_t1d_mask(tplt, xyz, seq, conf,L_s,chain_count=ab_chain)
                xyz_list.append(random_rot_trans(xyz,random_noise=0.0)) ; t1d_list.append(t1d) ; mask_list.append(mask_t)
            elif len(chain) > 1:
                # print('L_s :', L_s[ab_chain:ab_chain+len(chain)])
                # xyz, seq, conf, same_chain = get_init_xyz(sum(L_s[ab_chain:ab_chain+len(chain)]))
                xyz, seq, conf, same_chain = get_init_xyz(sum(L_s[ab_chain:]))
                # print('xyz :', xyz.shape)
                xyz, t1d, mask_t = get_xyz_t1d_mask(tplts['AG'][0], xyz, seq, conf,L_s,chain_count=ab_chain)
                xyz_list.append(xyz) ; t1d_list.append(t1d) ; mask_list.append(mask_t)
    else: # no antigen template
        # print('does it happen when no antigen?')
        xyz, seq, conf, same_chain = get_init_xyz(sum(L_s[ab_chain:]))
        xyz, t1d, mask_t = get_blank_template(ag_len, xyz, seq, conf)
        # print(xyz)
        xyz_list.append(xyz) ; t1d_list.append(t1d) ; mask_list.append(mask_t)
    # for i in range(len(xyz_list)):
    #     print(xyz_list[i].shape)
    #     print(t1d_list[i].shape)
    #     print(mask_list[i].shape)
    xyz = torch.cat(xyz_list, dim=1) ; t1d = torch.cat(t1d_list, dim=1) ; mask = torch.cat(mask_list, dim=1)
    # print('count :', len(xyz_list))
    # print(L_s)
    # centering the xyz again
    xyz = xyz - torch.mean(xyz, dim=(1,2), keepdim=True)


    # put_cdr_mask
    cdr_mask = ~cdr_mask.bool()[0]
    xyz, t1d, mask = put_CDR_mask(xyz, t1d, mask, cdr_mask)

    return xyz, t1d, mask
    
def TemplFeaturize_Tlqkf(item,tplts, L_s, antibody=True):
    """
    Returns a xyz, t1d, mask_t
    
    Parameters
    ----------
    tplts : dict ; dict of templates path
        tplts['H'] : template H path or template HL path or None
        tplts['L'] : template L path or None
        tplts['AG'] : template AG path or None

    Returns
    -------
    xyz : torch.tensor ; (B,L,N,3)
    t1d : torch.tensor ; (B,L,N,14)
    mask_t : torch.tensor ; (B,L,N)
    """
    if (item == '4gxu_M_N_AB') or (item == '3hmx_H_L_AB'):
        cdr_mask=~torch.load('/home/kkh517/CDR_pdb_dict.pt')[item[:-1]]['CDR'][0].bool().unsqueeze(-1)
    else:
        cdr_mask=~torch.load('/home/kkh517/CDR_pdb_dict.pt')[item]['CDR'][0].bool().unsqueeze(-1)
    # with open('/home/kkh517/benchmark_set_after210930/testset_CDR.pkl', 'rb') as f: # new_test
    #     testset_CDR = pickle.load(f)
    # cdr_mask = ~testset_CDR[item]['CDR'].bool().unsqueeze(-1)
    # cdr_mask = torch.ones_like(cdr_mask) # for rigid body


    def get_init_xyz(qlen,random_noise=5.0):
        # xyz = torch.full((qlen, 27, 3), np.nan).float()
        xyz = (INIT_CRDS.reshape(1, 27, 3).repeat(qlen, 1, 1)  # [L_ch, 27, 3]
               + torch.rand(qlen, 1, 3) * random_noise) # for SE(3) transformer...
        seq = torch.full((qlen,), 20).long() # all gaps  
        conf = torch.full((qlen,), 0.0).float()
        same_chain = torch.full((1,qlen,qlen),True).bool()
        return xyz, seq, conf, same_chain
    def get_xyz_t1d_mask(tplt, xyz, seq,conf,L_s,chain_count=0, antibody=None):
        xyz_for_mask = torch.full((xyz.shape[0], 27,3),np.nan).float()
        pdb_id = tplt.split('/')[-2]
        with open(tplt) as lines:
            temp_lines = lines.readlines()
            atom_lines = [line for line in temp_lines if line[:4] == 'ATOM']
            prev_chain = atom_lines[0][21] ; before_chain_len = 0
            for line in temp_lines:
                if line[:4] == 'ATOM':
                    chain = line[21]
                    resNo, atom, aa = int(line[22:26]), line[12:16], line[17:20]
                    if chain != prev_chain:
                        before_chain_len += L_s[chain_count]
                        # print('before_chain_len :', before_chain_len)
                        chain_count += 1
                        prev_chain = chain
                        # print('befor_chain_len :', before_chain_len)
                    resNo += before_chain_len
                    aa_idx = aa2num[aa] if aa in aa2num.keys() else 20
                    idx = resNo-1
                    # print('idx, chain :', idx, chain)
                    for i_atm, tgtatm in enumerate(aa2long[aa_idx]):
                        if tgtatm == atom:
                            try:
                                conf[idx] = float(line[60:66])/100.0
                                seq[idx] = aa_idx
                                # if conf[idx] < 0.8: continue 
                                if conf[idx] < 0.8: conf[idx]=0.1
                                xyz[idx, i_atm,:] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                                xyz_for_mask[idx, i_atm,:] = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                                
                                
                            except Exception as e:
                                print(f'{pdb_id} line :', line)
                                assert 1 == 0, f"{e}"
                                continue
        mask = torch.logical_not(torch.isnan(xyz_for_mask[:,:,0]))
        if antibody != None:
            if item.split('_')[2] == '#':
                mask = mask * cdr_mask[:L_s[0]]
            else:
                mask = mask * cdr_mask[:sum(L_s[:2])]
        try:
            xyz = center_and_realign_missing(xyz, mask)#.unsqueeze(0)
        except Exception as e:
            print(f"{item} line 1869 error \n{e}")
        # xyz = center_and_realign_missing(xyz, mask)

        seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
        t1d = torch.cat((seq, conf[:,None]), dim=1)
        return xyz[None], t1d[None], mask[None]
    
    def get_blank_template(qlen, xyz, seq, conf):
        seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
        # xyz = xyz.unsqueeze(0)
        t1d = torch.cat((seq, conf[:,None]), -1)
        mask = torch.full((qlen,27), False).bool()
        return xyz[None], t1d[None], mask[None]


    xyz_list = [] ; t1d_list = [] ; mask_list = []; ab_tplts = []; count = 0
    
    tplts[0] = '/home/kkh517/antibody_pdb_2/6oge_C_B_A/6oge_C_B_A_renum.pdb' #update
    xyz, seq, conf, same_chain = get_init_xyz(sum(L_s))
    xyz, t1d, mask = get_xyz_t1d_mask(tplts[0], xyz, seq, conf,L_s,)
    # xyz = torch.cat(xyz_list, dim=1) ; t1d = torch.cat(t1d_list, dim=1) ; mask = torch.cat(mask_list, dim=1)
    # print('count :', len(xyz_list))
    # print(L_s)
    # centering the xyz again
    xyz = xyz - torch.mean(xyz, dim=(1,2), keepdim=True)


    # put_cdr_mask
    # cdr_mask = ~cdr_mask.bool()[0]
    
    # xyz, t1d, mask = put_CDR_mask(xyz, t1d, mask, cdr_mask)
    # xyz = center_and_realign_missing(xyz[0], mask[0]).unsqueeze(0)
    return xyz, t1d, mask

def get_tplts(item,params,ab_tplt = None, ag_tplt = None,p_antibody=1, p_antibody_cut = 0.05, tm_dict = None):
    """
    Returns a list of templates path

    Parameters
    ----------
    item : str ; "1A2Y_H_L_A"
    params : dict ; parameters for 'AB_MSA_DIR'
    ab_tplt : str ; antibody template type
    ag_tplt : str ; antigen template type
    p_antibody : float ; probability of antibody
    p_antibody_cut : float ; probability of antibody cut-off
    

    Returns
    -------
    tplts : dict ; dict of templates path
        tplts['H'] : template H path or template HL path or None
        tplts['L'] : template L path or None
        tplts['AG'] : template AG path or None
    """
    def random_pick():
        r = random.random()
        if r < 0.333: return 'no tplt'
        elif r < 0.666: return 'monomer'
        else: return 'complex'
    def check_pdb(list):
        if len(list) == 0: return False
        path = os.path.dirname(list[0])
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith('.pdb'):
                    return True
        return False
    # if ab_tplt is None: ab_tplt = random_pick() # change if you want to check
    if ab_tplt is None: ab_tplt = 'complex' # change if you want to check
    # # if ag_tplt is None: ag_tplt = random_pick() # change if you want to check
    if ag_tplt is None: ag_tplt = 'complex' # change if you want to check
    # print(f'ab_tplt : {ab_tplt}, ag_tplt : {ag_tplt}')
    ab_tplts_path = params['AB_MSA_DIR']+f'/{item}' # antibody -> flexible docking
    # ab_tplts_path = params['AG_MSA_DIR']+f'/{item}' # antibody -> bound docking
    # flex_or_rigid = random.random()
    # if flex_or_rigid < 0.5:
    #     ag_tplts_path = params['AG_MSA_DIR']+f'/{item}' # antigen -> rigid docking
    # else:
    ag_tplts_path = params['AB_MSA_DIR']+f'/{item}' # antigen -> flexible docking
    # ag_tplts_path = f'/home/kkh517/Github/antibody_benchmark/pdbs/{item[:4].upper()}_l_u.pdb'
    # if item in ['3hmx','3wd5','4fqi','4gxu','5kov','5wux','5y9j']:
    if item[:4].upper() == '5VPG': item = "3RVW"+item[4:]
    # ag_tplts_path = f'/home/kkh517/Github/antibody_benchmark/re_index_ag_2/{item[:4].upper()}_l_u.pdb'

    tplts = {'H':[],'L':[],'AG':[]}
    h_chain = item.split('_')[1].strip() ; l_chain = item.split('_')[2].strip() ; ag_chain = item.split('_')[3].strip()
    if ab_tplt == 'monomer':
        # Heavy chain section
        for h in h_chain:
            h_candidate = glob(ab_tplts_path+f'/{h}/*.pdb')
            # print('h_candidate',h_candidate)
            if check_pdb(h_candidate):
                H = random.choice(h_candidate)
                tplts['H'].append(H)
            else:
                tplts['H'] = None
        # Light chain section
        for l in l_chain:
            l_candidate = glob(ab_tplts_path+f'/{l}/*.pdb')
            if check_pdb(l_candidate):
                L = random.choice(l_candidate)
                tplts['L'].append(L)
            else:
                tplts['L'] = None


        # if only monobody case, change to no tplt ## updated part. 
        if (h_chain == '#' or l_chain == '#') and ag_chain == '#':
            tplts['H'] = None
            tplts['L'] = None
    elif ab_tplt == 'complex':
        # if only antibody case, change to monomer
        if (p_antibody <= p_antibody_cut) or ag_chain == '#':
            return get_tplts(item,params,ab_tplt = 'monomer', ag_tplt = ag_tplt,p_antibody=p_antibody, p_antibody_cut = p_antibody_cut, tm_dict=tm_dict)



        if len(h_chain)>1:         # HH complex section
            h_candidate = glob(ab_tplts_path+f'/{h_chain}/*.pdb')
            if check_pdb(h_candidate):
                H = random.choice(h_candidate)
                tplts['H'].append(H)
            else:
                return get_tplts(item,params,ab_tplt = 'monomer', ag_tplt = ag_tplt,p_antibody=p_antibody, p_antibody_cut = p_antibody_cut, tm_dict=tm_dict)
            tplts['L'] = None
        elif len(l_chain)>1:       # LL complex section
            l_candidate = glob(ab_tplts_path+f'/{l_chain}/*.pdb')
            if check_pdb(l_candidate):
                L = random.choice(l_candidate)
                tplts['L'].append(L)
            else:
                return get_tplts(item,params,ab_tplt = 'monomer', ag_tplt = ag_tplt,p_antibody=p_antibody, p_antibody_cut = p_antibody_cut, tm_dict=tm_dict)
            tplts['H'] = None
        else: # HL complex section
            if h_chain == '#' or l_chain == '#':
                return get_tplts(item,params,ab_tplt = 'monomer', ag_tplt = ag_tplt,p_antibody=p_antibody, p_antibody_cut = p_antibody_cut, tm_dict=tm_dict)
            ab_chain = h_chain + l_chain
            abag_candidate = glob(ab_tplts_path+f'/{ab_chain}/*.pdb')
            if check_pdb(abag_candidate):
                AB = random.choice(abag_candidate)
                tplts['H'].append(AB)
            else:
                tplts['H'] = None
            tplts['L'] = None    
    elif ab_tplt == 'no tplt': # No template case
        tplts['H'] = None
        tplts['L'] = None
    
    # antibody_cut
    if p_antibody <= p_antibody_cut:
        print('only antibody case')
        tplts['AG'] = None
    else:
        # Antigen section
        if ag_tplt == 'monomer':
            for ag in ag_chain:
                print(f'{item} ag : {ag}')
                ag_candidate = glob(ag_tplts_path+f'/{ag}/*.pdb')
                if check_pdb(ag_candidate) and tplts['AG'] is not None:
                    AG = random.choice(ag_candidate)
                    # Check if AG has 0.8 or higher TM score
                    # if flex_or_rigid > 0.5 and tm_dict is not None: # if flexible docking, check the TM score
                        # tm_score = float(tm_dict[AG])
                    # else:
                    # tm_score = 1.0
                    # if AG in tm_dict.keys():
                    if tm_dict is not None:
                        tm_score = float(tm_dict[AG])
                    else:
                        tm_score = 0.0
                    # tm_score = 1.0
                    if tm_score >= 0.8: 
                        tplts['AG'].append(AG)
                    else:
                        ag_candidate = glob(params['AG_MSA_DIR']+f'/{item}'+f'/{ag}/*.pdb') # if the TM score is lower than 0.8, use rigid template
                        AG = random.choice(ag_candidate)
                        tplts['AG'].append(AG)

                else:
                    tplts['AG'] = None
                    # return get_tplts(item,params,ab_tplt = ab_tplt, ag_tplt = 'complex',p_antibody=p_antibody, p_antibody_cut = p_antibody_cut)
        elif ag_tplt == 'complex':
            ag_candidate = glob(ag_tplts_path+f'/{ag_chain}/*.pdb')
            # AG = ag_tplts_path
            # if True:
            if check_pdb(ag_candidate):
                AG = random.choice(ag_candidate)
                # if flex_or_rigid > 0.5 and tm_dict is not None: # if flexible docking, check the TM score
                #     tm_score = float(tm_dict[AG])
                # else:
                # tm_score = 1.0
                # if AG in tm_dict.keys():
                #     tm_score = float(tm_dict[AG])
                # else:
                #     print('No TM score for ', AG)
                #     tm_score = 0.0
                # tm_score=1.0
                # if tm_score >= 0.8:  # change later
                #     tplts['AG'].append(AG)
                # else:
                #     ag_candidate = glob(params['AG_MSA_DIR']+f'/{item}'+f'/{ag_chain}/*.pdb') # if the TM score is lower than 0.8, use rigid template
                #     AG = random.choice(ag_candidate)
                #     tplts['AG'].append(AG)         
                tplts['AG'].append(AG)   
            
            else:
                return get_tplts(item,params,ab_tplt = ab_tplt, ag_tplt = 'monomer',p_antibody=p_antibody, p_antibody_cut = p_antibody_cut, tm_dict=tm_dict)
            # tplts['AG'].append(ag_tplts_path)
        elif ag_tplt == 'no tplt':
            tplts['AG'] = None

    return tplts


def featurize_antibody_complex_kh(
        a3m,
        template_params,
        params,
        item,
        cluster,
        homo_list,
        chain_properties,
        p_antibody,
        p_antibody_cut=0.05,
        false_interface = None,
        true_interface_12 = None,
        true_interface_10_ab = None,
        true_interface_10_ag = None,
        pick_top=True,
        random_noise=5.0,
        negative=False,
        inference=False,
        unclamp=False,
        validation=False):  # p_interface_cut = 1.0):

    a3m = merge_a3m_antibody(a3m, homo_list)
    # print('homo_list',homo_list)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    N_final_dict = a3m['N_final_dict']

    if len(msa) > params['BLOCKCUT']:
        msas, inss = [], []
        msas.append(msa[None, 0])
        logger.info('msa zero', msa[None, 0].shape)
        inss.append(ins[None, 0])
        logger.info('ins zero', ins[None, 0].shape)
        for i in np.arange(len(N_final_dict)):
            sN, eN = N_final_dict[i][0], N_final_dict[i][1]
            if eN - sN > 50:
                logger.info('msa shape', i, msa[sN:eN].shape)
                logger.info('ins shape', i, ins[sN:eN].shape)
                msa_del, ins_del = MSABlockDeletion(msa[sN:eN], ins[sN:eN])
                logger.info('msa del', i, msa_del.shape)
                logger.info('ins del', i, ins_del.shape)
                msas.append(msa_del)
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
 
    # read csv file into dictionary
    # tm_dict = {}
    # # read csv file into dictionary
    # with open(params['TM_DB'], mode='r') as infile:
    #     reader = csv.reader(infile)
    #     for rows in reader:
    #         if rows[0] == 'template':
    #             continue
    #         tm_dict[rows[0]] = rows[1]

    
    
    # print('params',params)
    # tplts = get_tplts(item, params,p_antibody = p_antibody,p_antibody_cut =p_antibody_cut, tm_dict = tm_dict) # tplts has pdb file path
    # tplts = get_tplts(item,params)
    # tplts = {'H': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/new_ab_block.pdb'], 'L': None, 'AG': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/new_ag_block.pdb']} # new_test
    # tplts = {""}
    # tplts = {'H': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/af3_new_ab_block.pdb'], 'L': None, 'AG': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/{item}_af3_ag_block.pdb']} # new_test
    tplts = {'H': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/af3_new_ab_block.pdb'], 'L': None, 'AG': [f'/home/kkh517/alphafold2.3_ab_benchmark/{item}/{item}_af3_ag_block_new.pdb']} # new_test
    # tplts = {'H': [f'/home/yubeen/alphafold2.3_monomer/{item}/{item}_ab/ranked_0.pdb'], 'L':None, 'AG':[f"/home/yubeen/alphafold2.3_monomer/{item}/{item}_ag/ranked_0.pdb"]}
    # tplts = {'H': ["/home/kkh517/submit_files/Project/epitope_sampler_halfblood/iitp_inputs/ab_block.pdb"], 'L':None, 'AG':["/home/kkh517/submit_files/Project/epitope_sampler_halfblood/iitp_inputs/ag_block.pdb"]} #iitp

    print(f"{item} {tplts}")
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT'] + 1)
    
    if validation:
        ntempl = 0
    
    L_s = a3m['L_s']
    
    # print(f'{item} L_s', L_s)
    # print(f'{item} tplts', tplts)
    xyz_t, f1d_t, mask_t = CustomTemplFeaturize_kh(item,tplts, L_s)
    # CDR_dict= torch.load('/home/kkh517/CDR_pdb_dict.pt')
    # CDR_mask = CDR_dict[item]['CDR']
    # xyz_t, f1d_t, mask_t = put_CDR_mask(xyz_t, f1d_t, mask_t, CDR_mask)
    
    # nan check
    # print('xyz_t nan check', torch.isnan(xyz_t).sum())
    # print('f1d_t nan check', torch.isnan(f1d_t).sum())
    # print('mask_t nan check', torch.isnan(mask_t).sum())

    # print('xyz_t shape', xyz_t.shape, 'L_s.sum()',sum(L_s), 'item', item)

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    # read reference pdb
    if negative:
        ab_item = item.split(':')[0]
        ag_item = item.split(':')[1]
        #print('item', item)
        pdb = {}
        ab_pdb = parse_pdb_antibody(
            params['AB_DIR'] +
            f'/{ab_item}/{ab_item}_renum.pdb',
            ab_item)
        ag_pdb = parse_pdb_antibody(
            params['AB_DIR'] +
            f'/{ag_item}/{ag_item}_renum.pdb',
            ag_item)
        ab_length = 0
        ab_ch_count_real = 0  # antibody chain count in antibody
        total_ch_count = 0  # total chain count in antigen
        ag_ch_count_real = 0  # antibody chain count in antigen
        for i in ab_item.split('_')[1:3]:
            if i != '#':
                ab_ch_count_real += len(i)
        #print('ab_ch_count_real', ab_ch_count_real)

        for i, ch in enumerate(ag_item.split('_')[1:]):
            if ch != '#':
                total_ch_count += len(ch)
                if i < 2:
                    ag_ch_count_real += len(ch)
        #print('total_ch_count', total_ch_count)
        #print('ag_ch_count_real', ag_ch_count_real)
        xyz_new = [torch.full((i, 14, 3), np.nan).float()
                   for i in L_s[:ab_ch_count_real]]
        xyz_new_ag = [torch.full((i, 14, 3), np.nan).float()
                      for i in L_s[ab_ch_count_real:]]
        mask_new = [torch.full((i, 14), np.nan).float()
                    for i in L_s[:ab_ch_count_real]]
        mask_new_ag = [torch.full((i, 14), np.nan).float()
                       for i in L_s[ab_ch_count_real:]]

        for i in range(ab_ch_count_real):
            for j, idx in enumerate(ab_pdb['idx'][i]):
                xyz_new[i][idx - 1, :, :] = torch.from_numpy(ab_pdb['xyz'][i][j])
                mask_new[i][idx - 1, :] = ab_pdb['mask'][i][j]
        for i in range(ag_ch_count_real, total_ch_count):
            #print(i)
            for j, idx in enumerate(ag_pdb['idx'][i]):
                xyz_new_ag[i - ag_ch_count_real][idx - 1, :,:] = torch.from_numpy(ag_pdb['xyz'][i][j])
                mask_new_ag[i - ag_ch_count_real][idx - 1, :] = ag_pdb['mask'][i][j]

        for i in xyz_new_ag:
            xyz_new.append(i)
        for i in mask_new_ag:
            mask_new.append(i)

    else:
        if item == '3hmx_H_L_AB': item = '3hmx_H_L_A'
        if item == '4gxu_M_N_AB': item = '4gxu_M_N_A'
        
        # pdb = parse_pdb_antibody(params['AB_DIR'] + f'/{item}/{item}_renum.pdb',item) #iitp
        pdb = parse_pdb_antibody(f"/home/kkh517/benchmark_set_after210930_ag/{item}/new.pdb", item) # new_test
        # print(f"true_pdb \n/home/kkh517/benchmark_set_after210930_ag/{item}/new.pdb")
        xyz_new = [torch.full((i, 14, 3), np.nan).float() for i in L_s]
        mask_new = [torch.full((i, 14), np.nan).float() for i in L_s]
        # modify length
        for i, _ in enumerate(L_s):
            for j, idx in enumerate(pdb['idx'][i]):
                xyz_new[i][idx - 1, :, :] = torch.from_numpy(pdb['xyz'][i][j])
                mask_new[i][idx - 1, :] = pdb['mask'][i][j]

    total_list = []
    for i in homo_list:
        total_list.extend(i)

    combination_list = []
    original_list = []

    for i in homo_list:
        if len(i) != 1:
            combination_list.append(list(permutations(i)))
            original_list.append(i)

    product_list = list(product(*combination_list))
    initial_list = list(range(len(total_list)))

    new_lists = []

    for comb in product_list:
        new_list = initial_list
        for src, tgt in zip(comb, original_list):
            for idx, j in enumerate(tgt):
                new_list[j] = src[idx]
        new_lists.append(copy.deepcopy(new_list))

    xyz_shuffled = []
    mask_shuffled = []

    for shuffled_idx in new_lists:
        xyz_alt = []
        mask_alt = []
        for j in shuffled_idx:
            xyz_alt.append(xyz_new[j])
            mask_alt.append(mask_new[j])
        xyz_alt = torch.cat(xyz_alt)
        mask_alt = torch.cat(mask_alt)
        xyz_shuffled.append(copy.deepcopy(xyz_alt))
        mask_shuffled.append(copy.deepcopy(mask_alt))

    pdb['xyz'] = torch.stack(xyz_shuffled)
    pdb['mask'] = torch.stack(mask_shuffled)
    pdb['mask'] = torch.nan_to_num(pdb['mask'])
    assert len(pdb['xyz'].shape) == 4
    assert len(pdb['mask'].shape) == 3
    xyz = INIT_CRDS.reshape(1, 1, 27, 3).repeat(len(new_lists), sum(L_s), 1, 1)
    # mask = torch.full((len(new_lists), sum(L_s), 27), False)
    xyz[:, :, :14] = pdb['xyz']
    xyz = torch.nan_to_num(xyz)
    mask = torch.full((len(new_lists), sum(L_s), 27), False)
    mask[:, :, :14] = pdb['mask']

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

    count = 0
    for (i, j) in zip(L_start, L_end):
        idx[i:j] += count
        count += 100

    chain_idx = torch.zeros(sum(L_s)).long()
    for k, (i, j) in enumerate(zip(L_start, L_end)):
        chain_idx[i:j] = chain_properties[k]

    ab_length = 0 ; ab_length2 = 0
    if (np.array(chain_properties) > 1).sum() == 0:
        ab_ch_count = 1 ; ab_ch_count2 = 1  # if only antibody, check interface btw hchain and lchain
        if chain_properties[-1] == 1 and len(chain_properties) > 1: # if only antibody case with light chain complex, interface should be consdiered.
            ab_ch_count2 = 2 # no, if only antibody, takes both of them.
    else:
        ab_ch_count = (np.array(chain_properties) < 2).sum()
        ab_ch_count2 = ab_ch_count

    for i in range(ab_ch_count):
        ab_length += L_s[i]

    for j in range(ab_ch_count2):
        ab_length2 += L_s[j]

    ag_length = sum(L_s) - ab_length ; ag_length2 = sum(L_s) - ab_length2
    interface_split = [ab_length, ag_length]
    interface_split_LRMSD = [ab_length2, ag_length2]
    # print('L_s', L_s)
    # print('interface_split', interface_split)
    #print('ab_length', ab_length)
    #print('ag_length', ag_length) 
    same_chain = torch.zeros((sum(L_s), sum(L_s))).long() # TODO : L_S = [heavy_len, light_len, ag_len] 
    chain_length = [ab_length, ag_length]
    # if negative:
    #     same_chain[0:ab_length, 0:ab_length] = 1
    #     total_length = sum(L_s)
    #     same_chain[ab_length: total_length, ab_length:total_length] = 1
    # else:
    #     for (i, j) in zip(L_start, L_end):
    #         same_chain[i:j, i:j] = 1
    same_chain[0:ab_length, 0:ab_length] = 1
    total_length = sum(L_s)
    same_chain[ab_length: total_length, ab_length:total_length] = 1
    # get_epi_full
    # print(f"xyz shape {xyz.shape}")
    # print(f"interface_split {interface_split}")
    epi_full = get_epi_full(xyz[0], mask[0], interface_split, item, cutoff=10.0)
    # epi_full = get_surface(params, interface_split, item)
    # epi_full [B, L]
    #putting the epitope center
    # breakpoint()
    if epi_full.sum() != 0:
        epi_idx = epi_full.nonzero().squeeze()
        new_epi_idx = epi_idx
        epi_t = torch.full((xyz_t.shape[1],),False)
        # print('interface_split',interface_split)
        # print('epi_t',epi_t.shape) # what is wrong with u?
        # print('new_epi_idx',new_epi_idx)
        epi_t[new_epi_idx] = True
        # print('epi_t',epi_t.shape)
        xyz_cen = xyz_t - xyz_t[:,:,1].mean(dim=1)[:,None,None,:]
        epitope_xyz = xyz_cen[0][new_epi_idx]
        # print('epitope_xyz',epitope_xyz.shape)
        epitope_center = epitope_xyz[:,1].mean(dim=0)
        xyz_t[:,:interface_split[0]] += epitope_center
    else: epi_t = None
    # write pdb of templates
    try:
        
        seq = msa[0].long()
        # print('seq',seq.shape)
        # atoms = xyz_t[0][:,:3] # pdb for only backbone
        atoms = xyz_t[0] # pdb for all atoms
        # print('atoms',atoms.shape)
        # atoms = atoms * mask_t[0][:,:3].unsqueeze(-1)
        # print('mask_t[0].unsqueeze(-1)',mask_t[0].unsqueeze(-1).shape)
        # print('atoms',atoms.shape)
        # print('xyz_t',xyz_t.shape)
        # print('L_s',L_s)
        # os.makedirs(f"/home/kkh517/templates_check/feb5", exist_ok=True)
        print(f'writing templates into pdb... {item}')
        os.makedirs(f"templates_check", exist_ok=True)
        write_pdb(seq, atoms, L_s=L_s, Bfacts=epi_t,prefix=f"templates_check/templates_{item}") #ProjectName
        # write_pdb(seq, atoms, L_s=L_s, Bfacts=None,prefix=f"/home/kkh517/templates_check/feb5/{item}") #ProjectName
    except Exception as e:
        print(f"{e} error occured")
        print("passing pdb writing")
    if validation:
        sel = torch.arange(sum(L_s))

    elif sum(L_s) > params["CROP"]:
        
        if inference:
            sel = torch.arange(sum(L_s))
        else:
            if negative:
                if false_interface == None:
                    sel = get_negative_crop(xyz[0], mask[0], torch.arange(sum(L_s)), interface_split, \
                        params, item, true_interface_10_ab=true_interface_10_ab,
                        true_interface_10_ag=true_interface_10_ag)
                else:
                    sel = get_negative_crop(xyz[0], mask[0], torch.arange(sum(L_s)), interface_split, \
                        params, item, false_interface=false_interface, true_interface_12=true_interface_12, \
                        true_interface_10_ab=true_interface_10_ab)
            else:
                sel = get_spatial_crop(xyz[0], mask[0], torch.arange(sum(L_s)), interface_split, params, item)
                # sel, epitope_crop = get_spatial_crop(xyz[0], mask[0], torch.arange(sum(L_s)), interface_split, params, item)

    else:
        sel = torch.arange(sum(L_s))

    if p_antibody <= p_antibody_cut and len(L_s)== 2:
        interface_split = []
        for i in range(2):
            interface_split.append(L_s[i])
    elif p_antibody <= p_antibody_cut and len(L_s)== 1:
        interface_split = []
    
    return (
        # seq.long(),
        cluster,
        item,
        sel,
        torch.tensor(L_s),
        msa,
        ins,
        xyz.float(), # true_crds
        mask,   #mask_crds
        idx.long(),#idx_pdb
        xyz_t.float(),
        f1d_t.float(),
        mask_t,
        xyz_prev.float(),
        mask_prev,
        same_chain,
        chain_idx,
        unclamp,
        negative,
        interface_split,
        epi_full,
        validation, 
    )



def find_indices(l, value, start_ind):
    return [index + start_ind for index,
            item in enumerate(l) if item.equal(value)]


def get_msa(a3mfilename, item, max_seq=8000):
    msa, ins = parse_a3m(a3mfilename, max_seq=max_seq)
    return {"msa": torch.tensor(msa), "ins": torch.tensor(ins), "label": item}\


    
def loader_complex_antibody_kh(
        cluster,
        item, # xxxx_H_L_A
        params,
        unclamp=False,
        p_antibody_cut=0.05,
        negative=False,
        false_interface=None,
        inference=False,
        validation=False):
    """Load antibody complex

    Args:
        item (str): Antibody training ID (ex. 6abc_H_L_A)
        params (dict): loader parameters
        negative (bool, optional): Negative set or not.  Defaults to False.
        pick_top (bool, optional): Pick top templates or not. Defaults to True.

    Returns:
        _type_: _description_
    """
    # print('negative', negative)
    # print('input item', item)
    # item='6oge_C_B_A' ## needs to erase!!!!!!!!!!
    if negative:
        cluster_ab = cluster.split(':')[0]
        cluster_ag = cluster.split(':')[1]
        ab = item.split(':')[0]
        ag = item.split(':')[1]
        hchain = ab.split('_')[1]
        lchain = ab.split('_')[2]
        ag_chain = ag.split('_')[3]
    
    else:
        print(f'item {item}')
        hchain = item.split('_')[1]
        lchain = item.split('_')[2]
        ag_chain = item.split('_')[3]
    # print('hchain', hchain, 'lchain', lchain, 'ag_chain', ag_chain)
    with open(params['HOMO_LIST']) as homo_list_json:
        homo_list_dict = json.load(homo_list_json) # what is params['HOMO_LIST']??
    
#    with open(params['INTERFACE_CA_10']) as interface_ca_10_json:
#        interface_ca_10_dict = json.load(interface_ca_10_json)
    homo_list_dict['5hys_C_D_JK'] = {"H": [[0]], "L": [[0]], "AG": [[0], [1]]}
    homo_list_dict['2i25_N_#_L'] ={"H": [[0]], "AG": [[0]]} # for test set
    # homo_list_dict['3hmx_H_L_A'] = {"H": [[0]], "L": [[0]], "AG":[[0],[1]]}
    # homo_list_dict['3wd5_H_L_A'] = {"H": [[0]], "L": [[0]], "AG":[[0,1,2]]}
    # homo_list_dict['4fqi_H_L_BA'] = {"H": [[0]], "L": [[0]], "AG":[[0,2,4],[1,3,5]]}
    # homo_list_dict['4gxu_M_N_A'] = {"H": [[0]], "L": [[0]], "AG":[[0,2,4],[1,3,5]]}
    # homo_list_dict['4gxu_M_N_AB'] = {"H": [[0]], "L": [[0]], "AG":[[0,2,4],[1,3,5]]}
    # homo_list_dict['5kov_C_c_A'] = {"H": [[0]], "L": [[0]], "AG":[[0, 1]]}
    # homo_list_dict['5wux_H_L_E'] = {"H": [[0]], "L": [[0]], "AG":[[0,1,2]]}
    # homo_list_dict['5y9j_H_L_A'] = {"H": [[0]], "L": [[0]], "AG":[[0,1,2]]}
    # homo_list_dict['2vis_B_A_C'] = {"H": [[0]], "L": [[0]], "AG":[[0,1,2]]}
#    with open(params['INTERFACE_CA_12']) as interface_ca_12_json:
#        interface_ca_12_dict = json.load(interface_ca_12_json)
#
    if negative:
        homo_list = defaultdict(list)
        if hchain != '#':
            homo_list['H'] = homo_list_dict[ab]['H']
        if lchain != '#':
            homo_list['L'] = homo_list_dict[ab]['L']
        if ag_chain != '#':
            homo_list['AG'] = homo_list_dict[ag]['AG']
        #print('homo_list', homo_list)
    else:
        homo_list = homo_list_dict[item]
    a3m = defaultdict(list)
    template_params = defaultdict(list)
    # change here to takeit
    if hchain != '#':
        if negative:
            for i in hchain:
                try:
                    a3m['H'].append(get_msa(a3m_tmp, ab))
                except Exception as e:
                    print(e)
                    a3m_tmp = params['AB_MSA_DI'] + f'/{ab}/{i}/t000_.msa0.a3m' # needs to take care of params too
                    a3m['H'].append(get_msa(a3m_tmp, ab))
                template_params_temp = params['AB_MSA_DIR'] + \
                    f'/{ab}/{i}/' # make params['AB_MSA_DIR'] as /home/kkh517/Github/rf-abag-templ/DB/real_final_set/H_L_A/1a2y_B_A_C
                template_params['H'].append(template_params_temp) # now params has directory for each monomers

        else:
            for i in hchain:
                # a3m_tmp = params['AG_MSA_DIR'] + f'/{item}/{i}/t000_.msa0.a3m' #iitp
                # chain_dictionary_path = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/chain_id_map.json"
                # chain_dictionary_path = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/chain_id_map.json" #gpu01
                # parse dictionary from _path
                # with open(chain_dictionary_path, 'r') as f:
                #     chain_dictionary = json.load(f)
                # dictionary_key = [key for key, value in chain_dictionary.items() if value['description'] == i][0]
                # a3m_tmp = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m" # new_test
                # a3m_tmp = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                
                a3m_tmp = f"/home/yubeen/af3_msa/{item.lower().replace('#','')}/{item.lower().replace('#','')}_{i}.a3m"
                try:
                    a3m['H'].append(get_msa(a3m_tmp, item))
                except Exception as e:
                    print(e)
                    a3m_tmp = params['AB_MSA_DIR'] + f'/{item}/{i}/t000_.msa0.a3m'
                    a3m['H'].append(get_msa(a3m_tmp, item))
                template_params_temp = params['AB_MSA_DIR'] + \
                    f'/{item}/{i}/'
                template_params['H'].append(template_params_temp)

    if lchain != '#':
        if negative:
            for i in lchain:
                a3m_tmp = params['AB_MSA_DIR'] + f'/{ab}/{i}/t000_.msa0.a3m'
                a3m['L'].append(get_msa(a3m_tmp, ab))
                template_params_temp = params['AB_MSA_DIR'] + \
                    f'/{ab}/{i}/'
                template_params['L'].append(template_params_temp)

        else:
            for i in lchain:
                # dictionary_key = [key for key, value in chain_dictionary.items() if value['description'] == i][0]
                # a3m_tmp = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m" # new_test
                # a3m_tmp = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                a3m_tmp = f"/home/yubeen/af3_msa/{item.lower().replace('#','')}/{item.lower().replace('#','')}_{i}.a3m"
                # a3m_tmp = params['AG_MSA_DIR'] + f'/{item}/{i}/t000_.msa0.a3m' #iitp
                try:
                    a3m['L'].append(get_msa(a3m_tmp, item))
                except Exception as e:
                    print(e)
                    a3m_tmp = params['AB_MSA_DIR'] + f'/{item}/{i}/t000_.msa0.a3m'
                    a3m['L'].append(get_msa(a3m_tmp, item))
                template_params_temp = params['AB_MSA_DIR'] + \
                    f'/{item}/{i}/'
                template_params['L'].append(template_params_temp)
        if len(lchain) >=2:
            template_params_temp=params['AB_MSA_DIR']+\
                f'/{item}/{i}/'

    if ag_chain != '#':
        if item.startswith('3hmx'): 
            ag_chain = "AB"
            item = "3hmx_H_L_AB"
        if item.startswith('4gxu'):
            ag_chain ="AB"
            item = "4gxu_M_N_AB"
        if negative:
            for i in ag_chain:
                a3m_tmp = params['AG_MSA_DIR'] + f'/{ag}/{i}/t000_.msa0.a3m'
                a3m['AG'].append(get_msa(a3m_tmp, item))
                template_params_temp = params['AG_MSA_DIR'] + \
                    f'/{ag}/{i}/templ_info.pt'
                template_params['AG'].append(template_params_temp)
        else:
                # item = "3hmx_H_L_AB"
            for i in ag_chain:
                print(f'ag_chain {i}')
                # dictionary_key = [key for key, value in chain_dictionary.items() if value['description'] == i][0]
                # if item.startswith('8vyl'):
                #     if dictionary_key == 'D': dictionary_key = 'C'
                #     if dictionary_key == 'E': dictionary_key = 'A'
                # if item.startswith('8pnu'):
                #     if dictionary_key == 'C' or dictionary_key == 'D' : dictionary_key = 'B'
                # if item.startswith('7y9t'):
                #     if dictionary_key == 'A' : dictionary_key ='B'
                # if item.startswith('7uij'):
                #     if dictionary_key == 'C'   : dictionary_key = 'D'
                #     elif dictionary_key == 'D' : dictionary_key = 'C'
                # a3m_tmp = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m" # new_test
                # a3m_tmp = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                a3m_tmp = f"/home/yubeen/af3_msa/{item.lower().replace('#','')}/{item.lower().replace('#','')}_{i}.a3m"
                
                # a3m_tmp = params['AG_MSA_DIR'] + f'/{item}/{i}/t000_.msa0.a3m'
                try:
                    
                    a3m['AG'].append(get_msa(a3m_tmp, item))
                    print(f"antigen a3m : \n{a3m_tmp}")
                except Exception as e:
                    print(e)
                    # change the dictionary_key for the another key which has same value['sequence']
                    # for key, value in chain_dictionary.items(): #new_test
                    #     if (value['sequence'] == chain_dictionary[dictionary_key]['sequence']) and (dictionary_key != key):
                    #         dictionary_key = key
                            # break
                    
                    # try:
                    #     dictionary_key = [key for key, value in chain_dictionary.items() if value['description'] == ag_chain[-1]][0]
                    #     # a3m_tmp = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                    #     a3m_tmp = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                    #     a3m['AG'].append(get_msa(a3m_tmp, item))
                    # except Exception as e2:
                    #     print(e2)
                    #     dictionary_key = [key for key, value in chain_dictionary.items() if value['description'] == ag_chain[0]][0]
                    #     # a3m_tmp = f"/home/kkh517/alphafold2.3_ab_benchmark/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                    #     a3m_tmp = f"/home/yubeen/afm_msa/alphafold2.3/{item}/msas/{dictionary_key}/bfd_uniclust_hits.a3m"
                    #     a3m['AG'].append(get_msa(a3m_tmp, item))
                    # print(f"new a3m file : {a3m_tmp}")
                template_params_temp = params['AB_MSA_DIR'] + \
                    f'/{item}/{i}/'
                template_params['AG'].append(template_params_temp)
        if len(ag_chain) >=2:
            template_params_temp=params['AG_MSA_DIR']+\
                f'/{item}/{i}/'
            template_params['AG'].append(template_params_temp)
    # print('a3m',a3m)
    # print('template_params',template_params)
    # print('cluster',cluster)
    p_antibody = np.random.rand() ; p_antibody_cut = 0.05
    if validation:
        p_antibody_cut = -10 # always p_antibody >> p_antibody_cut
        print(f'validation set should  p_antibody >> p_antibody_cut : {p_antibody} // p_antiobdy_cut : {p_antibody_cut}')

    # 1. only l chain or only h chain
    if len(homo_list) == 1:
        for k, v in homo_list.items():
            # if single chain 
            if len(v) == 1 and len(v[0]) == 1:
                return featurize_single_chain(
                    a3m[k][0], template_params[k][0], params, item, cluster, unclamp=unclamp, validation=validation)
            # if multiple chain
            else:
                a3m_in = []
                tplt_params_in = [template_params[k][0], template_params[k][1]]
                for idx in v:  # [[0, 1]] or [[0], [1]]
                    a3m_in.append(a3m[k][idx[0]])
                return featurize_antibody_complex_kh(
                    a3m_in,
                    tplt_params_in,
                    params,
                    item,
                    cluster,
                    homo_list=v,
                    chain_properties=[
                        0,
                        1],
                    p_antibody=100,
                    p_antibody_cut=p_antibody_cut,
                    negative=negative,
                    validation=validation)
    else:
        
        a3m_in = []
        tplt_params_in = []
        length_dict = defaultdict(int)
        for k, v in homo_list.items():  # [[0, 1]] or [[0], [1]]
            count = 0
            for idx in v:
                for idx2 in idx:
                    count += 1
            length_dict[k] = count #total length

        if (p_antibody > p_antibody_cut) or negative or inference or validation: # for what??
            # print(f"this should be validation case : validation is {validation}")
            # print(f"or p_antibody is bigger than p_antibody _cut {p_antibody > p_antibody_cut}")
            for key in ['H', 'L', 'AG']:
                if key in homo_list:
                    for value in homo_list[key]:
                        # print(f'value {value} {a3m[key]}')
                        a3m_in.append(a3m[key][value[0]])
                    for tp in template_params[key]:
                        tplt_params_in.append(tp)

        else:
            for key in ['H', 'L']:
                if key in homo_list:
                    for value in homo_list[key]:
                        a3m_in.append(a3m[key][value[0]])
                    for tp in template_params[key]:
                        tplt_params_in.append(tp)
            
        # get length of H, L, AG
        h_length = length_dict['H']
        hl_length = length_dict['H'] + length_dict['L']
        ag_length = length_dict['AG']

        # get homo_list  ex)[[0], [1, 2], [3, 5], [4]]
        homo_list_final = []
        for k, v in homo_list.items():
            if k == 'H':
                homo_list_final.extend(v)
            elif k == 'L':
                for i in v:
                    homo_list_final.append([j + h_length for j in i])
            else:
                for i in v:
                    if p_antibody > p_antibody_cut or negative or inference or validation:
                        homo_list_final.append([j + hl_length for j in i])

        chain_properties = []

        # get chain properties (0 for H, 1 for L, 2~ for antigen)
        if hl_length == 1:
            if h_length == 0:
                chain_properties.append(1)
            else:
                chain_properties.append(0)
        else:
            chain_properties.extend([0, 1])

        if p_antibody > p_antibody_cut or negative or inference or validation: #only antibody case
            # chain_properties.extend(list(range(2, 2 + ag_length)))
            chain_properties.extend([2] * ag_length)

        # print(f"chain_properties_item {item} {chain_properties}")
        #print('homo_list_final', homo_list_final)
        #print('homo_list', homo_list)
        if negative:
            if validation:
                return featurize_antibody_complex_kh(
                    a3m_in,
                    tplt_params_in,
                    params,
                    item,
                    cluster,
                    homo_list_final,
                    chain_properties,
                    p_antibody,
                    p_antibody_cut = p_antibody_cut,
                    negative = negative,
                    inference = inference,
                    validation = validation)
            elif ab == ag and not false_interface == None: #incorrect interface - need false interface info
                print('cluster',cluster)
                print('item',item)
                true_interface_12 = interface_ca_12_dict[cluster_ag][ag]
                true_interface_10 = interface_ca_10_dict[cluster_ab][ab]
                return featurize_antibody_complex_kh(
                    a3m_in,
                    tplt_params_in,
                    params,
                    item,
                    cluster,
                    homo_list_final,
                    chain_properties,
                    p_antibody,
                    p_antibody_cut=p_antibody_cut,
                    false_interface = false_interface, #false epitope of antigen
                    true_interface_12 = true_interface_12, #CA 12A cutoff true epitope of antigen
                    true_interface_10_ab = true_interface_10, #CA 10A cutoff true interface (for antibody)
                    negative= negative,
                    inference = inference
                    
                )
            else:
                true_interface_10_ab= interface_ca_10_dict[cluster_ab][ab]
                true_interface_10_ag= interface_ca_10_dict[cluster_ag][ag]
                return featurize_antibody_complex_kh(
                    a3m_in,
                    tplt_params_in,
                    params,
                    item,
                    cluster,
                    homo_list_final,
                    chain_properties,
                    p_antibody,
                    p_antibody_cut=p_antibody_cut,
                    false_interface = None,
                    true_interface_12 = None,
                    true_interface_10_ab = true_interface_10_ab, #CA 10A cutoff true interface (for antibody, antigen)
                    true_interface_10_ag = true_interface_10_ag,
                    negative= negative,
                    inference = inference
                    
                ) 
        else:
            return featurize_antibody_complex_kh(
                a3m_in,
                tplt_params_in,
                params,
                item,
                cluster,
                homo_list_final,
                chain_properties,
                p_antibody,
                p_antibody_cut=p_antibody_cut,
                unclamp=unclamp,
                negative=negative,
                inference = inference,
                validation = validation)


class Dataset(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, validation=False):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.validation = validation
        #self.unclamp_cut = unclamp_cut
        # self.p_homo_cut = p_homo_cut

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        if self.validation:
            sel_idx = 0
            out = self.loader(
                ID,
                self.item_dict[ID][sel_idx],
                self.params,
                # p_homo_cut=self.p_homo_cut,
            )
        else:
            out = self.loader(
                ID,
                self.item_dict[ID][sel_idx],
                self.params,
                # p_homo_cut=self.p_homo_cut,
            )
        return out


class DatasetComplex_antibody(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, negative=False, inference=False, validation=False, unclamp=False):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.negative = negative
        self.inference = inference
        self.validation = validation
        self.unclamp = unclamp

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        # print(self.item_dict[ID][sel_idx])

        if self.inference:
            sel_idx = np.random.randint(0, len(self.item_dict[ID]))
            out = self.loader(ID, self.item_dict[ID][sel_idx], self.params, negative=self.negative, inference=True)
        
        elif self.validation and not self.negative:
            sel_idx = 0
            # print("ID",ID)
            # print("index",index)
            # print("Ids",self.IDs)
            # print("item_dict",self.item_dict.keys())
            out = self.loader(ID, self.item_dict[ID][sel_idx],
                              self.params, validation=self.validation, unclamp=self.unclamp)
        elif self.negative:
            out = self.loader(ID, self.item_dict[ID],
                self.params, negative=self.negative, validation=self.validation)

        # print('out', out[1])
        return out


class DistilledDataset(data.Dataset):
    def __init__(self,
                 gp_IDs, gp_loader, gp_dict,
                 l_IDs, l_loader, l_dict,
                 h_IDs, h_loader, h_dict,
                 hl_IDs, hl_loader, hl_dict,
                 l_ag_IDs, l_ag_loader, l_ag_dict,
                 h_ag_IDs, h_ag_loader, h_ag_dict,
                 hl_ag_IDs, hl_ag_loader, hl_ag_dict,
                 neg_IDs, neg_loader, neg_dict, interface_correctness, incorrect_interface_info,
                 params):
        # p_homo_cut=0.5):

        # general protein
        self.gp_IDs = gp_IDs
        self.gp_dict = gp_dict
        self.gp_loader = gp_loader
        self.gp_len = len(gp_IDs)

        # only l
        self.l_IDs = l_IDs
        self.l_loader = l_loader
        self.l_dict = l_dict
        self.l_len = len(l_IDs)
        # only h
        self.h_IDs = h_IDs
        self.h_loader = h_loader
        self.h_dict = h_dict
        self.h_len = len(h_IDs)
        # hl
        self.hl_IDs = hl_IDs
        self.hl_loader = hl_loader
        self.hl_dict = hl_dict
        self.hl_len = len(hl_IDs)
        # l w/ antigen
        self.l_ag_IDs = l_ag_IDs
        self.l_ag_loader = l_ag_loader
        self.l_ag_dict = l_ag_dict
        self.l_ag_len = len(l_ag_IDs)
        # h w/antigen
        self.h_ag_IDs = h_ag_IDs
        self.h_ag_loader = h_ag_loader
        self.h_ag_dict = h_ag_dict
        self.h_ag_len = len(h_ag_IDs)
        # hl w/antigen
        self.hl_ag_IDs = hl_ag_IDs
        self.hl_ag_loader = hl_ag_loader
        self.hl_ag_dict = hl_ag_dict
        self.hl_ag_len = len(hl_ag_IDs)
        # neg
        self.neg_IDs = neg_IDs
        self.neg_loader = neg_loader
        self.neg_dict = neg_dict
        self.neg_len = len(neg_IDs)
        
        self.params = params
        self.unclamp_cut = 0.9
        self.interface_correctness = interface_correctness
        self.incorrect_interface_info = incorrect_interface_info
        # self.p_homo_cut = p_homo_cut
        self.total_len = self.l_len + self.h_len + self.hl_len + self.l_ag_len + \
            self.h_ag_len + self.hl_ag_len + self.gp_len + self.neg_len

    def __len__(self):
        return self.l_len + self.h_len + self.hl_len + self.l_ag_len + \
            self.h_ag_len + self.hl_ag_len + self.gp_len + self.neg_len

    def __getitem__(self, index):
        p_unclamp = np.random.rand()
        if index >= self.total_len - self.neg_len:  # from negative set
            ID = self.neg_IDs[index - self.total_len + self.neg_len]
            #print('negative ID', ID)
            if_crct = self.interface_correctness[ID]
            print('interface_correctness', if_crct)
            # check if there is false interface in the cluster we chose
            ids = [pdb_id for pdb_id, v in if_crct.items() if v == False]
            # probability to chose incorrect sample if exist
            p_choose_incorrect = np.random.rand()
            false_interface = None
            if len(ids) == 0: # if there is no incorrect items in the cluster
                antibody_part_idx = np.random.randint(0, len(self.neg_dict[ID]))
                antibody_item = self.neg_dict[ID][antibody_part_idx]
                other_keys = list(self.neg_dict.keys())
                remove_keys = set()
                # remove if one cluster is same
                for i in other_keys:
                    if i.split('_')[0] == ID.split('_')[0] or \
                        i.split('_')[1] == ID.split('_')[1] or \
                        i.split('_')[2] == ID.split('_')[2]:
                        remove_keys.add(i)
                #print('remove_keys', remove_keys)
                for i in remove_keys:
                    other_keys.remove(i)

                antigen_part_ID = np.random.choice(other_keys)
                antigen_part_idx = np.random.randint(0, len(self.neg_dict[antigen_part_ID]))
                antigen_item = self.neg_dict[antigen_part_ID][antigen_part_idx]
                    
            else:
                if p_choose_incorrect > 0: #choose incorrect interface
                    antibody_part_idx = np.random.randint(0, len(ids))
                    antibody_item = ids[antibody_part_idx]
                    antigen_item = antibody_item
                    antigen_part_ID = ID
                    false_interface = self.incorrect_interface_info[ID][antibody_item]
                    if len(false_interface) == 0:
                        false_interface = None
                        print('false interface length is zero!!!!!')
                        antibody_part_idx = np.random.randint(0, len(self.neg_dict[ID]))
                        antibody_item = self.neg_dict[ID][antibody_part_idx]
                        other_keys = list(self.neg_dict.keys())
                        remove_keys = set()

                        for i in other_keys:
                            if i.split('_')[0] == ID.split('_')[0] or \
                                i.split('_')[1] == ID.split('_')[1] or \
                                i.split('_')[2] == ID.split('_')[2]:
                                remove_keys.add(i)
                                
                        #print('remove_keys', remove_keys)
                        for i in remove_keys:
                            other_keys.remove(i)
                        antigen_part_ID = np.random.choice(other_keys)
                        antigen_part_idx = np.random.randint(0, len(self.neg_dict[antigen_part_ID]))
                        antigen_item = self.neg_dict[antigen_part_ID][antigen_part_idx]
                else: #choose from other clusters
                    antibody_part_idx = np.random.randint(0, len(self.neg_dict[ID]))
                    antibody_item = self.neg_dict[ID][antibody_part_idx]
                    other_keys = list(self.neg_dict.keys())
                    remove_keys = set()

                    for i in other_keys:
                        if i.split('_')[0] == ID.split('_')[0] or \
                            i.split('_')[1] == ID.split('_')[1] or \
                            i.split('_')[2] == ID.split('_')[2]:
                            remove_keys.add(i)
                            
                    #print('remove_keys', remove_keys)
                    for i in remove_keys:
                        other_keys.remove(i)
                    antigen_part_ID = np.random.choice(other_keys)
                    antigen_part_idx = np.random.randint(0, len(self.neg_dict[antigen_part_ID]))
                    antigen_item = self.neg_dict[antigen_part_ID][antigen_part_idx]
                    
            # just one negative complex in one cluster
            #print(f'negative ID: {antigen_item}')
            out = self.neg_loader(f'{ID}:{antigen_part_ID}', \
                    f'{antibody_item}:{antigen_item}', self.params, negative=True, false_interface=false_interface)

        elif index >= self.total_len - self.neg_len - self.hl_ag_len:  # from hl_ag set
            ID = self.hl_ag_IDs[index - self.total_len + self.neg_len + self.hl_ag_len]
            sel_idx = np.random.randint(0, len(self.hl_ag_dict[ID]))
            #print(f'hl_ag ID: {self.hl_ag_dict[ID][sel_idx]}')
            # always unclamp for antibody-antigen complex
            out = self.hl_ag_loader(ID, self.hl_ag_dict[ID][sel_idx], self.params, unclamp=True)
            #if p_unclamp > self.unclamp_cut:
            #    out = self.hl_ag_loader(self.hl_ag_dict[ID][sel_idx], self.params, unclamp=True)
            #else:
            #    out = self.hl_ag_loader(self.hl_ag_dict[ID][sel_idx], self.params, unclamp=False)

        elif index >= self.total_len - self.neg_len - self.hl_ag_len - self.h_ag_len:  # from h_ag_set
            ID = self.h_ag_IDs[index - self.total_len + self.neg_len + self.hl_ag_len + self.h_ag_len]
            # print('ID', ID)
            sel_idx = np.random.randint(0, len(self.h_ag_dict[ID]))
            # print('length', len(self.h_ag_dict[ID]))
            # print('sel_idx', sel_idx)
            # print('idx', self.h_ag_dict[ID][sel_idx])
            # print("loader", self.h_ag_loader)
            #print(f'h_ag ID: {self.h_ag_dict[ID][sel_idx]}')
            out = self.h_ag_loader(ID, self.h_ag_dict[ID][sel_idx], self.params, unclamp=True)

            #if p_unclamp > self.unclamp_cut:
            #    out = self.h_ag_loader(self.h_ag_dict[ID][sel_idx], self.params, unclamp=True)
            #else:
            #    out = self.h_ag_loader(self.h_ag_dict[ID][sel_idx], self.params, unclamp=False)

        elif index >= self.gp_len + self.l_len + self.h_len + self.hl_len:  # from l_ag_set
            # print('from_l_ag_set')
            ID = self.l_ag_IDs[index - self.gp_len - self.l_len - self.h_len - self.hl_len]
            sel_idx = np.random.randint(0, len(self.l_ag_dict[ID]))
            #print(f'l_ag ID: {self.l_ag_dict[ID][sel_idx]}')
            out = self.l_ag_loader(ID, self.l_ag_dict[ID][sel_idx], self.params, unclamp=True)
            #if p_unclamp > self.unclamp_cut:
            #    out = self.l_ag_loader(self.l_ag_dict[ID][sel_idx], self.params, unclamp=True)
            #else:
            #    out = self.l_ag_loader(self.l_ag_dict[ID][sel_idx], self.params, unclamp=False)

        elif index >= self.gp_len + self.l_len + self.h_len:  # from hl set
            # print('from_hl_set')
            ID = self.hl_IDs[index - self.gp_len - self.l_len - self.h_len]
            sel_idx = np.random.randint(0, len(self.hl_dict[ID]))
            #print(f'hl ID: {self.hl_dict[ID][sel_idx]}')
            if p_unclamp > self.unclamp_cut:
                out = self.hl_loader(ID, self.hl_dict[ID][sel_idx], self.params, unclamp=True)
            else:
                out = self.hl_loader(ID, self.hl_dict[ID][sel_idx], self.params, unclamp=False)

        elif index >= self.gp_len + self.l_len:  # from h set
            # print('from_h_set')
            ID = self.h_IDs[index - self.gp_len - self.l_len]
            sel_idx = np.random.randint(0, len(self.h_dict[ID]))
            #print(f'h ID: {self.h_dict[ID][sel_idx]}')
            if p_unclamp > self.unclamp_cut:
                out = self.h_loader(
                    ID, 
                    self.h_dict[ID][sel_idx],
                    self.params,
                    unclamp=True)  # p_homo_cut=self.p_homo_cut)
            else:
                out = self.h_loader(
                    ID, 
                    self.h_dict[ID][sel_idx],
                    self.params,
                    unclamp=False)  # p_homo_cut=self.p_homo_cut)
        elif index >= self.gp_len:  # from l set
            # print('from_l_set')
            ID = self.l_IDs[index - self.gp_len]
            sel_idx = np.random.randint(0, len(self.l_dict[ID]))
            #print(f'l ID: {self.l_dict[ID][sel_idx]}')
            if p_unclamp > self.unclamp_cut:
                out = self.l_loader(
                    ID,
                    self.l_dict[ID][sel_idx],
                    self.params,
                    unclamp=True)
            else:
                out = self.l_loader(
                    ID,
                    self.l_dict[ID][sel_idx],
                    self.params,
                    unclamp=False)
        else:
            ID = self.gp_IDs[index]
            sel_idx = np.random.randint(0, len(self.gp_dict[ID]))
            #logger.info(f'gp ID: {self.gp_dict[ID][sel_idx][0]}')
            #print(f'gp ID: {self.gp_dict[ID][sel_idx][0]}') 
            if p_unclamp > self.unclamp_cut:
                out = self.gp_loader(
                    ID,
                    self.gp_dict[ID][sel_idx][0],
                    self.gp_dict[ID][sel_idx][1],
                    self.gp_dict[ID][sel_idx][2],
                    self.params,
                    unclamp=True,
                    negative=False)
            else:
                out = self.gp_loader(
                    ID,
                    self.gp_dict[ID][sel_idx][0],
                    self.gp_dict[ID][sel_idx][1],
                    self.gp_dict[ID][sel_idx][2],
                    self.params,
                    unclamp=False,
                    negative=False)

        return out


class DistributedWeightedSampler(data.Sampler):
    def __init__(self, dataset, ab_weights, num_example_per_epoch=2648,
                 fraction_ab=0.25, fraction_gp=0.5,
                 num_replicas=None, rank=None, replacement=False):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        assert num_example_per_epoch % num_replicas == 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_ab_per_epoch = int(round(num_example_per_epoch * fraction_ab))
        self.num_gp_per_epoch = int(round(num_example_per_epoch * fraction_gp))
        self.num_neg_per_epoch = num_example_per_epoch - \
            self.num_ab_per_epoch - self.num_gp_per_epoch

        # get each count of l, h, hl, l_ag, h_ag, hl_ag
        self.l_weight, self.h_weight, self.hl_weight, \
            self.l_ag_weight, self.h_ag_weight, self.hl_ag_weight = ab_weights
        # self.num_l, self.num_h, self.num_hl = len(self.dataset.l_IDs), len(self.dataset.h_IDs), len(self.dataset.hl_IDs)
        # self.num_l_ag, self.num_h_ag, self.num_hl_ag = abag_nums
        # self.num_gp = gp_num
        # self.num_neg = neg_num

        self.num_l_per_epoch = int(
            round(
                self.num_ab_per_epoch *
                self.l_weight))
        self.num_h_per_epoch = int(
            round(
                self.num_ab_per_epoch *
                self.h_weight))
        self.num_hl_per_epoch = int(
            round(
                self.num_ab_per_epoch) *
            self.hl_weight)
        self.num_l_ag_per_epoch = int(
            round(self.num_ab_per_epoch * self.l_ag_weight))
        self.num_h_ag_per_epoch = int(
            round(self.num_ab_per_epoch * self.h_ag_weight))
        self.num_hl_ag_per_epoch = self.num_ab_per_epoch - self.num_l_ag_per_epoch - \
            self.num_h_ag_per_epoch - self.num_hl_per_epoch - self.num_h_per_epoch - self.num_l_per_epoch

        # print (self.num_ab_per_epoch, self.num_abag_per_epoch, self.num_gp_per_epoch, self.num_neg_per_epoch)
        # print (self.num_l, self.num_h, self.num_hl)
        print(
            self.num_l_per_epoch,
            self.num_h_per_epoch,
            self.num_hl_per_epoch)
        # print (self.num_l_ag, self.num_h_ag, self.num_hl_ag)
        print(
            self.num_l_ag_per_epoch,
            self.num_h_ag_per_epoch,
            self.num_hl_ag_per_epoch)
        print(
            self.num_ab_per_epoch,
            self.num_gp_per_epoch,
            self.num_neg_per_epoch)

        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        # self.ab_weights = ab_weights
        # self.abag_weights = abag_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))
        # print('indices length', len(indices))
        sel_indices = torch.tensor((), dtype=int)
        print('gp_IDs', len(self.dataset.gp_IDs), self.num_gp_per_epoch)
        print('neg_IDs',
                    len(self.dataset.neg_IDs),
                    self.num_neg_per_epoch)
        print('l_IDS', len(self.dataset.l_IDs),
                    self.num_l_per_epoch, self.l_weight)
        print('h_IDs', len(self.dataset.h_IDs),
                    self.num_h_per_epoch, self.h_weight)
        print('hl_IDs', len(self.dataset.hl_IDs),
                    self.num_hl_per_epoch, self.hl_weight)
        print('l_ag_IDs', len(self.dataset.l_ag_IDs),
                    self.num_l_ag_per_epoch, self.l_ag_weight)
        print('h_ag_IDs', len(self.dataset.h_ag_IDs),
                    self.num_h_ag_per_epoch, self.h_ag_weight)
        print('hl_ag_IDs', len(self.dataset.hl_ag_IDs),
                    self.num_hl_ag_per_epoch, self.hl_ag_weight)
        if (self.num_gp_per_epoch > 0):
            gp_sampled = torch.randperm(len(self.dataset.gp_IDs), generator=g)[
                :self.num_gp_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[gp_sampled]))

        if (self.num_l_per_epoch > 0):
            l_sampled = torch.randperm(len(self.dataset.l_IDs), generator=g)[
                :self.num_l_per_epoch]
            sel_indices = torch.cat(
                (sel_indices, indices[l_sampled + len(self.dataset.gp_IDs)]))

        if (self.num_h_per_epoch > 0):
            h_sampled = torch.randperm(len(self.dataset.h_IDs), generator=g)[
                :self.num_h_per_epoch]
            sel_indices = torch.cat(
                (sel_indices, indices[h_sampled + len(self.dataset.gp_IDs) + len(self.dataset.l_IDs)]))

        if (self.num_hl_per_epoch > 0):
            hl_sampled = torch.randperm(len(self.dataset.hl_IDs), generator=g)[
                :self.num_hl_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[hl_sampled + len(
                self.dataset.gp_IDs) + len(self.dataset.l_IDs) + len(self.dataset.h_IDs)]))

        if (self.num_l_ag_per_epoch > 0):
            l_ag_sampled = torch.randperm(len(self.dataset.l_ag_IDs), generator=g)[
                :self.num_l_ag_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[l_ag_sampled + len(self.dataset.gp_IDs) + len(
                self.dataset.l_IDs) + len(self.dataset.h_IDs) + len(self.dataset.hl_IDs)]))

        if (self.num_h_ag_per_epoch > 0):
            h_ag_sampled = torch.randperm(len(self.dataset.h_ag_IDs), generator=g)[
                :self.num_h_ag_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[h_ag_sampled +
                                                          len(self.dataset.gp_IDs) +
                                                          len(self.dataset.l_IDs) +
                                                          len(self.dataset.h_IDs) +
                                                          len(self.dataset.hl_IDs) +
                                                          len(self.dataset.l_ag_IDs)]))

        if (self.num_hl_ag_per_epoch > 0):
            hl_ag_sampled = torch.randperm(len(self.dataset.hl_ag_IDs), generator=g)[
                :self.num_hl_ag_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[hl_ag_sampled +
                                                          len(self.dataset.gp_IDs) +
                                                          len(self.dataset.l_IDs) +
                                                          len(self.dataset.h_IDs) +
                                                          len(self.dataset.hl_IDs) +
                                                          len(self.dataset.l_ag_IDs) +
                                                          len(self.dataset.h_ag_IDs)]))
        if (self.num_neg_per_epoch > 0):
            assert self.num_neg_per_epoch <= len(self.dataset.neg_IDs)
            neg_sampled = torch.randperm(len(self.dataset.neg_IDs), generator=g)[
                :self.num_neg_per_epoch]
            sel_indices = torch.cat((sel_indices, indices[neg_sampled +
                                                          len(self.dataset.gp_IDs) +
                                                          len(self.dataset.l_IDs) +
                                                          len(self.dataset.h_IDs) +
                                                          len(self.dataset.hl_IDs) +
                                                          len(self.dataset.l_ag_IDs) +
                                                          len(self.dataset.h_ag_IDs) +
                                                          len(self.dataset.hl_ag_IDs)]))

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]
        print('sel_indices length', len(sel_indices))
        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        print('total_size', self.total_size)
        print('rank', self.rank)
        print('\n')
        print('num_replicas', self.num_replicas)
        print('len(indices)', len(indices))
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == "__main__":
    args=None
    loader_param = set_data_loader_params(args) 
    (
            gp_train,
            l_train,
            h_train,
            hl_train,#
            l_ag_train,
            h_ag_train,
            hl_ag_train,
            neg_train,#
            h_val,
            hl_val,
            h_ag_val,
            hl_ag_val,#
            hl_ag_test,
            neg_val,
            weights,#
        ) = get_train_valid_set(loader_param)
    negative_keys_train = list(neg_train.keys()) + list(neg_train.keys())
    interface_correctness = defaultdict(dict)
    incorrect_interface_info = defaultdict(lambda: defaultdict(list))
    train_set = DistilledDataset(
            list(gp_train.keys()),
            loader_complex_gp,
            gp_train,
            list(l_train.keys()),
            loader_complex_antibody_kh,
            l_train,
            list(h_train.keys()),
            loader_complex_antibody_kh,
            h_train,
            list(hl_train.keys()),
            loader_complex_antibody_kh,
            hl_train,
            list(l_ag_train.keys()),
            loader_complex_antibody_kh,
            l_ag_train,
            list(h_ag_train.keys()),
            loader_complex_antibody_kh,
            h_ag_train,
            list(hl_ag_train.keys()),
            loader_complex_antibody_kh,
            hl_ag_train,
            negative_keys_train,
            loader_complex_antibody_kh,
            neg_train,
            interface_correctness,
            incorrect_interface_info,
            loader_param,
        )
    world_size = 1
    # hl_ag_val = {'994_355_328': ['6lz9_H_L_B'],'1612_78_566': ['5nh3_I_M_B'],'460_434_256/321': ['1rvf_H_L_13'],'1546_4_321/339': ['7eaj_F_L_13']}
    # world_size = int(os.environ["SLURM_NTASKS"])
    # hl_ag_test.pop('61')
    # hl_ag_test.pop('16') # 4gxu
    # hl_ag_test.pop('25') # 3wd5
    # hl_ag_test.pop('11') # 5kov
    n_hl_ag_val = (len(hl_ag_val.keys()) // world_size) * world_size
    n_h_ag_val = (len(h_ag_val.keys()) // world_size) * world_size
    # n_train_all = (len(train_all.keys()) // world_size) * world_size
    n_hl_ag_test = (len(hl_ag_test.keys()) // world_size) * world_size
    # print('n_all_train is',n_train_all)
    print('n_hl_ag_val is',n_hl_ag_val)
    # n_l_ag_train = (len(l_ag_train.keys()) // world_size) * world_size
    # temp_list = ['101_639', '101_601']
    # print('hl_ag_val.keys()',list(hl_ag_val.keys()))
    # print('hl_ag_val',hl_ag_val)
    valid_hl_ag_set = DatasetComplex_antibody(
            list(hl_ag_val.keys())[: n_hl_ag_val],
            # list(l_train.keys())[: n_l_ag_train],
            # temp_list,
            loader_complex_antibody_kh,
            hl_ag_val,
            # l_ag_train,
            loader_param,# same with valid_param
            validation=True,
        )
    valid_h_ag_set = DatasetComplex_antibody(
            list(h_ag_val.keys())[: n_h_ag_val],
            loader_complex_antibody_kh,
            h_ag_val,
            loader_param,
            validation=True
    )
    # train_all_set = DatasetComplex_antibody(
    #         list(train_all.keys())[: n_train_all],
    #         loader_complex_antibody_kh,
    #         train_all,
    #         loader_param,
    #         validation=True,
    #     )
    # print('hl_ag_test',hl_ag_test)
    test_hl_ag_set = DatasetComplex_antibody(
            list(hl_ag_test.keys())[: n_hl_ag_test],
            loader_complex_antibody_kh,
            hl_ag_test,
            loader_param,
            validation=True,
        )
    # rank=int(os.environ["SLURM_PROCID"])
    rank=0
    valid_hl_ag_sampler = data.distributed.DistributedSampler(
            valid_hl_ag_set, num_replicas=world_size, rank=rank
        )
    valid_h_sampler = data.distributed.DistributedSampler(
            valid_h_ag_set, num_replicas=world_size, rank=rank
    )
    # train_all_sampler = data.distributed.DistributedSampler(
    #         train_all_set, num_replicas=world_size, rank=rank
    # )
    LOAD_PARAM2 = {"shuffle": False, "num_workers": 0, "pin_memory": True}
    valid_h_ag_loader = data.DataLoader(
            valid_h_ag_set, sampler=valid_h_sampler, **LOAD_PARAM2
    )
    valid_hl_ag_loader = data.DataLoader(
            valid_hl_ag_set, sampler=valid_hl_ag_sampler, **LOAD_PARAM2
    )
    # train_all_loader = data.DataLoader(
    #         train_all_set, sampler=train_all_sampler, **LOAD_PARAM2
    # )
    test_hl_ag_loader = data.DataLoader(
            test_hl_ag_set, **LOAD_PARAM2
    )
    
    # for inputs in tqdm(valid_hl_ag_loader):
    #     print(inputs[0])
    # for inputs in tqdm(valid_h_ag_loader):
    #     print(inputs[0])
    # for inputs in tqdm(train_all_loader):
    for inputs in tqdm(test_hl_ag_loader):
        try:
            print('#'*50)
            print(f"item : {inputs[1]}")
            print(f"size : {torch.sum(inputs[3])}")
        except Exception as e:
            print(f"{e} error")
            continue
