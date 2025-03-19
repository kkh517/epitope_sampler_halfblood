"""
Test script for measure execution time for RF_ABAG
"""

import random
import sys
import os
from contextlib import ExitStack
import time
from copy import deepcopy
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import os
# distributed data parallel
sys.path.append('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/src')
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# import wandb


from data_loader_rigid2 import (
    get_train_valid_set,
    loader_complex_antibody_kh,
    loader_complex_gp,
    Dataset,
    DatasetComplex_antibody,
    DistilledDataset,
    DistributedWeightedSampler,
    MSAFeaturize,
)
from rfabflex.common.kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d
from rfabflex.model.RoseTTAFoldModel import RoseTTAFoldModule
from loss_jan30 import *
from rfabflex.common.util import *
from rfabflex.model.util_module import XYZConverter
from scheduler import get_stepwise_decay_schedule_with_warmup
import wandb


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# torch.autograd.set_detect_anomaly(True)
ProjectName = ""
USE_AMP = True #Falseautomated mixed precision #nan occur?? #False when debugging
N_PRINT_TRAIN = 8
# BATCH_SIZE = 1 * torch.cuda.device_count()

# num structs per epoch
# must be divisible by #GPUs
# N_EXAMPLE_PER_EPOCH = 10560
# N_EXAMPLE_PER_EPOCH = 8
N_EXAMPLE_PER_EPOCH = 2256
LOAD_PARAM = {"shuffle": False, "num_workers": 4, "pin_memory": True} # "num_workers": 3~4
LOAD_PARAM2 = {"shuffle": False, "num_workers": 4, "pin_memory": True} # "num_workers": 3~4

# wandb.init(project='RF_ABAG',  name='Loader_updated')



def add_weight_decay(model, l2_coeff):
    """If the parameter requires L2 normalization, \
            add weight_decay paramter (add to the loss)
    Args:
        model (nn.Module):
        l2_coeff (float): weight decay parameter

    Returns:
        [dict1, dict2]
        dict1 = {"params": parameters with no decay, "weight_decay":0.0}
        dict2 = {"params": parameters with decay, "weight_decay":l2_coeff}

    """

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": l2_coeff},
    ]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA(nn.Module):
    """Return model for training/non-training

    Args:
        nn.Module

    Returns:
        nn.Module


    """

    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print(
                "EMA update should only be called during training",
                file=sys.stderr,
                flush=True,
            )
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1.0 - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)


class Trainer:
    def __init__(
        self,
        model_name="RF2_apr23",
        n_epoch=100,
        lr=1.0e-4,
        l2_coeff=1.0e-2,
        port=None,
        interactive=False,
        model_param={},
        loader_param={},
        loss_param={},
        batch_size=1,
        accum_step=8,
        maxcycle=4,
        crop_size=256,
        wandb=False,
        wandb_name=None,
    ):
        self.model_name = model_name  # "BFF"
        # self.model_name = "%s_%d_%d_%d_%d"%(model_name, model_param['n_module'],
        #                                    model_param['n_module_str'],
        #                                    model_param['d_msa'],
        #                                    model_param['d_pair'])
        #
        self.n_epoch = n_epoch
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        #
        self.model_param = model_param
        self.loader_param = loader_param
        self.valid_param = deepcopy(loader_param)
        self.valid_param["SEQID"] = 150.0
        self.loss_param = loss_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size
        self.crop_size = crop_size

        # for all-atom str loss
        self.l2a = long2alt
        self.aamask = allatom_mask
        self.num_bonds = num_bonds
        self.ljlk_parameters = ljlk_parameters
        self.lj_correction_parameters = lj_correction_parameters
        self.hbtypes = hbtypes
        self.hbbaseatoms = hbbaseatoms
        self.hbpolys = hbpolys

        # from xyz to get xxxx or from xxxx to xyz
        self.xyz_converter = XYZConverter()

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.active_fn = nn.Softmax(dim=1)


        # self.incorrect_interface_info = defaultdict(defaultdict_list)

        self.maxcycle = maxcycle
        self.wandb = wandb
        if self.wandb:
            self.wandb_name = wandb_name
        if self.wandb:
            print("wandb is on!!!!!")
        print(model_param, loader_param, loss_param)

    def calc_loss(
        self,
        logit_s,  # logits_dist, omega, theta [B, 37, L, L]*3 , logits_phi [B, 19, L, L]
        label_s,  # Binned value of true structure     [B, L, L, 4]
        logit_aa_s,  # Prediction of masked residue       [B, 21, N*L]
        label_aa_s,  # msa                                [B, N_clust, L]
        mask_aa_s,  # masked position of MSA             [B, N_clust, L]
        logit_exp,  # predict experimentally resolved    [B, L]
        logit_epitope, #predict epitope                  [B, L]
        logit_pae,  # predict PAE                        [B, 64, L, L]
        p_bind,
        pred,  # predicted coordinates              [B, iteration, L, 27, 3]
        pred_tors,  # predicted torsion                  [B, iteration, L, 10, 2]
        true,  # true coordinate                    [B, L, 27, 3]
        mask_crds,  # maksed position of true coordinate [B, L, 27]
        mask_BB,  # missing backbone residue for true  [B, L]
        mask_2d,  # mask_BB to 2d feature              [B, L, L]
        same_chain,  # same_chain or not                  [B, L, L]
        pred_lddt,  # predicted LDDTS                    [B, 50, L]
        idx,  # idx of pdb                         [B, L]
        interface_split, # interface_split []
        epitope_info, # epitope_info [B, L]
        item = None,
        L_s = None,
        unclamp=False,
        negative=False,
        pred_prev_s=None,
        w_dist=1.0,
        w_aa=1.0,
        w_str=1.0,
        w_all=0.5,
        w_exp=1.0,
        w_epi=1.0,
        w_con=1.0,
        w_pae=1.0,
        w_lddt=1.0,
        w_blen=1.0,
        w_bang=1.0,
        w_bind=1.0,
        w_lj=0.0,
        w_hb=0.0,
        w_lrmsd=0.1,
        lj_lin=0.75,
        use_H=False,
        eps=1e-6,
        clashcut=0.0,
    ):
        B, L = true.shape[:2]
        seq = label_aa_s[:, 0].clone()
        # print('B should be 1... check this', B)
        assert B == 1  # fd - code assumes a batch size of 1

        loss_s = list()
        loss_dict = defaultdict(list)

        # col 0~3: c6d loss (distogram, orientogram prediction)
        loss = calc_c6d_loss(logit_s, label_s, mask_2d)

        tot_loss = w_dist * loss.sum()
        loss_s.append(loss.detach()) 
        # print("distogram loss \t", loss.detach()[0:1])
        # print("omega loss \t", loss.detach()[1:2])
        # print("theta loss \t", loss.detach()[2:3])
        # print("phi loss \t", loss.detach()[3:])
        
        loss_dict["distogram"] = loss.detach()[0:1]
        loss_dict["omega"] = loss.detach()[1:2]
        loss_dict["theta"] = loss.detach()[2:3]
        loss_dict["phi"] = loss.detach()[3:]

        # col 4: masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss = tot_loss + w_aa * loss
        loss_s.append(loss[None].detach())
        loss_dict["masked_aa"] = loss[None].detach()

        # print('masked_aa \t', loss[None].detach())
        # col 5: p_bind loss
        loss = torch.tensor(0.0, device=p_bind.device)
        if torch.sum(same_chain == 0) > 0:
            bce = torch.nn.BCELoss()
            target = torch.tensor([1.0], device=p_bind.device)
            if negative:
                target = torch.tensor([0.0], device=p_bind.device)
            # print('p_bind', p_bind.shape, p_bind)
            # print('target', target.shape, target)
            with torch.cuda.amp.autocast(enabled=False):
                loss = bce(p_bind.float(), target)
            tot_loss = tot_loss + w_bind * loss
        else:
            tot_loss = tot_loss + 0.0 * p_bind.sum()
        loss_s.append(loss[None].detach())
        loss_dict["p_bind"] = loss[None].detach()
        # print('p_bind \t', loss[None].detach())

        # update atom mask for structural loss calculation
        # calc lj for ground-truth --> ignore SC conformation from ground-truth if it makes clashes (lj > clashcut)
        xs_mask = self.aamask[seq]  # (B, L, 27)
        xs_mask[0, :, 14:] = False  # ignore hydrogens
        xs_mask *= mask_crds  # mask missing atoms & residues as well
        lj_nat = calc_lj(
            seq[0],
            true[0, ..., :3],
            self.aamask,
            same_chain[0],
            self.ljlk_parameters,
            self.lj_correction_parameters,
            self.num_bonds,
            lj_lin=lj_lin,
            use_H=False,
            negative=negative,
            reswise=True,
            atom_mask=xs_mask[0],
        )
        mask_clash = (lj_nat < clashcut) * mask_BB[
            0
        ]  # if False, the residue has clash (L)
        # ignore clashed side-chains
        xs_mask[:, :, 5:] *= mask_clash.view(1, L, 1)

        # col 6: experimentally resolved prediction loss
        loss = nn.BCEWithLogitsLoss()(logit_exp, mask_BB.float())
        tot_loss = tot_loss + w_exp * loss
        loss_s.append(loss[None].detach())
        loss_dict["exp_resolved"] = loss[None].detach()

        # print('exp_resolved \t', loss[None].detach())

        # col 7 : Epitope prediction loss
        loss = nn.BCEWithLogitsLoss()(logit_epitope, epitope_info.float())
        tot_loss = tot_loss + w_epi * loss
        loss_s.append(loss[None].detach())
        loss_dict["epitope"] = loss[None].detach()
        
        # print('epitope \t', loss[None].detach())
        # AllAtom loss
        # get ground-truth torsion angles
        (
            true_tors,
            true_tors_alt,
            tors_mask,
            tors_planar,
        ) = self.xyz_converter.get_torsions(true, seq, mask_in=xs_mask)
        # masking missing residues as well
        tors_mask *= mask_BB[..., None]  # (B, L, 10)

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, self.l2a[seq, :, None].repeat(1, 1, 1, 3), true)

        natRs_all, _n0 = self.xyz_converter.compute_all_atom(
            seq, true[..., :3, :], true_tors
        )
        natRs_all_alt, _n1 = self.xyz_converter.compute_all_atom(
            seq, true_alt[..., :3, :], true_tors_alt
        )
        predRs_all, pred_all = self.xyz_converter.compute_all_atom(
            seq, pred[-1], pred_tors[-1]
        )

        #  - resolve symmetry
        natRs_all_symm, nat_symm = resolve_symmetry(
            pred_all[0],
            natRs_all[0],
            true[0],
            natRs_all_alt[0],
            true_alt[0],
            xs_mask[0],
        )
        frame_mask = torch.cat(
            [mask_BB[0][:, None], tors_mask[0, :, :8]], dim=-1
        )  # only first 8 torsions have unique frames

        # Structural loss
        # 1. Backbone FAPE
        if unclamp:
            FAPE_Loss, intra_loss, inter_loss, str_loss, pae_loss = calc_str_loss(
                pred,
                true,
                logit_pae,
                mask_2d,
                same_chain,
                negative=negative,
                A=10.0,
                d_clamp=None,
            )
            with torch.cuda.amp.autocast(enabled=False):
                LADE_loss = calc_LocalAlignment_Backbone_Displacement_Loss(true,pred,mask_crds,mask_2d,same_chain,d_clamp=None, A=30, reduction='none')
                # ChADE_loss = get_ChainAlignment_Backbone_Displacement_Loss(true, pred, mask_2d, same_chain, interface_split, d_clamp=None)
        else:
            FAPE_Loss, intra_loss, inter_loss, str_loss, pae_loss = calc_str_loss(
                pred,
                true,
                logit_pae,
                mask_2d,
                same_chain,
                negative=negative,
                A=10.0,
                d_clamp=10.0,
            )
            with torch.cuda.amp.autocast(enabled=False):
                LADE_loss = calc_LocalAlignment_Backbone_Displacement_Loss(true,pred,mask_crds,mask_2d,same_chain,d_clamp=10.0, A=30, reduction='none')
                # ChADE_loss = get_ChainAlignment_Backbone_Displacement_Loss(true, pred, mask_2d, same_chain, interface_split, d_clamp=None)
        # tot_str = 0.3333 *(FAPE_Loss + LADE_loss + ChADE_loss)
        tot_str = 0.5 *(FAPE_Loss + LADE_loss)
        # tot_str = FAPE_Loss
        tot_loss = tot_loss + (1.0 - w_all) * w_str * tot_str
        # col 8: intra-chain FAPE 
        loss_s.append(intra_loss)
        # col 9: inter-chain FAPE
        loss_s.append(inter_loss)
        # col 10 : total FAPE
        loss_s.append(str_loss)

        # print('BB_FAPE \t', FAPE_Loss *0.5)#*0.3333)
        # print('LADE \t', LADE_loss *0.5)#*0.3333)
        # print('ChADE \t', ChADE_loss *0.3333)

        loss_dict["intra_FAPE"] = intra_loss
        loss_dict["inter_FAPE"] = inter_loss
        loss_dict["total_FAPE"] = str_loss
        # col 11 : PAE
        tot_loss = tot_loss + w_pae * pae_loss
        loss_s.append(pae_loss[None].detach())
        loss_dict["pae"] = pae_loss[None].detach()

        # print('pae \t', pae_loss[None].detach())
        # allatom fape and torsion angle loss
        if negative:  # inter-chain fapes should be ignored for negative cases
            L1 = same_chain[0, 0, :].sum()
            frame_maskA = frame_mask.clone()
            frame_maskA[L1:] = False
            xs_maskA = xs_mask.clone()
            xs_maskA[0, L1:] = False
            l_fape_A = compute_FAPE(
                predRs_all[0, frame_maskA][..., :3, :3],
                predRs_all[0, frame_maskA][..., :3, 3],
                pred_all[xs_maskA][..., :3],
                natRs_all_symm[frame_maskA][..., :3, :3],
                natRs_all_symm[frame_maskA][..., :3, 3],
                nat_symm[xs_maskA[0]][..., :3],
                eps=1e-4,
            )
            frame_maskB = frame_mask.clone()
            frame_maskB[:L1] = False
            xs_maskB = xs_mask.clone()
            xs_maskB[0, :L1] = False
            l_fape_B = compute_FAPE(
                predRs_all[0, frame_maskB][..., :3, :3],
                predRs_all[0, frame_maskB][..., :3, 3],
                pred_all[xs_maskB][..., :3],
                natRs_all_symm[frame_maskB][..., :3, :3],
                natRs_all_symm[frame_maskB][..., :3, 3],
                nat_symm[xs_maskB[0]][..., :3],
                eps=1e-4,
            )
            fracA = float(L1) / len(same_chain[0, 0])
            l_fape = fracA * l_fape_A + (1.0 - fracA) * l_fape_B
        else:
            l_fape = compute_FAPE(
                predRs_all[0, frame_mask][..., :3, :3],
                predRs_all[0, frame_mask][..., :3, 3],
                pred_all[xs_mask][..., :3],
                natRs_all_symm[frame_mask][..., :3, :3],
                natRs_all_symm[frame_mask][..., :3, 3],
                nat_symm[xs_mask[0]][..., :3],
                eps=1e-4,
            )
        l_tors = torsionAngleLoss(
            pred_tors, true_tors, true_tors_alt, tors_mask, tors_planar, eps=1e-10
        )
        tot_loss = tot_loss + w_all * w_str * (l_fape + l_tors)
        # col 12: all-atom FAPE
        loss_s.append(l_fape[None].detach())
        # col 13: torsion angle loss
        loss_s.append(l_tors[None].detach())

        loss_dict["all_atom_FAPE"] = l_fape[None].detach()
        loss_dict["torsion_angle"] = l_tors[None].detach()

        # print('all_atom_FAPE \t', l_fape[None].detach())
        # print('torsion_angle \t', l_tors[None].detach())
        # CA-LDDT
        ca_lddt = calc_lddt(
            pred[:, :, :, 1].detach(),
            true[:, :, 1],
            mask_BB,
            mask_2d,
            same_chain,
            negative=negative,
            epitope=False,
        )
        # col 14: CA-LDDT
        loss_s.append(ca_lddt.detach())
        loss_dict["ca_lddt"] = ca_lddt.detach()

        # print('ca_lddt \t', ca_lddt.detach())
        # allatom lddt loss
        lddt_loss, true_lddt = calc_allatom_lddt_w_loss(
            pred_all[0, ..., :14, :3].detach(), #(L, 14, 3)
            nat_symm[..., :14, :3], #(L, 14, 3)
            xs_mask[0, ..., :14], #(L, 14)
            pred_lddt, 
            idx[0], #(L)
            same_chain[0],
            negative=negative,
        )
        # col 15/16: all-atom LDDT, lddt_loss
        loss_s.append(true_lddt[None].detach())
        loss_dict["true_lddt"] = true_lddt[None].detach()
        tot_loss = tot_loss + w_lddt * lddt_loss
        loss_s.append(lddt_loss.detach()[None])
        loss_dict["lddt_loss"] = lddt_loss[None].detach()

        # print('true_lddt \t', true_lddt[None].detach())
        # print('lddt_loss \t', lddt_loss[None].detach())
        # bond geometry
        blen_loss, bang_loss = calc_BB_bond_geom(pred[-1, :, :], idx)
        if w_blen > 0.0:
            tot_loss = tot_loss + w_blen * blen_loss
        if w_bang > 0.0:
            tot_loss = tot_loss + w_bang * bang_loss

        # lj potential
        lj_loss = calc_lj(
            seq[0],
            pred_all[0, ..., :3],
            self.aamask,
            same_chain[0],
            self.ljlk_parameters,
            self.lj_correction_parameters,
            self.num_bonds,  # negative=negative,
            lj_lin=lj_lin,
            use_H=use_H,
        )
        if w_lj > 0.0:
            tot_loss = tot_loss + w_lj * lj_loss

        # hbond [use all atoms not just those in native]
        #hb_loss = calc_hb(
        #    seq[0],
        #    pred_all[0, ..., :3],
        #    self.aamask,
        #    self.hbtypes,
        #    self.hbbaseatoms,
        #    self.hbpolys,
        #)
        #if w_hb > 0.0:
        #    tot_loss = tot_loss + w_hb * hb_loss
        ## col 17, 18 ,19, 20: bond length, bond angle, lj, hb
        hb_loss = torch.zeros_like(lj_loss)
        loss_s.append(torch.stack((blen_loss, bang_loss, lj_loss, hb_loss)).detach())

        loss_dict["blen_loss"] = blen_loss[None].detach()
        loss_dict["band_loss"] = bang_loss[None].detach()
        loss_dict["lj_loss"] = lj_loss[None].detach()
        loss_dict["hb_loss"] = hb_loss[None].detach()

        # mask_inter_chain = mask_2d * ~(same_chain.bool())
        # inter_Lade_loss = (mask_inter_chain[None].float() * LADE_loss).sum(dim=(1,2,3)) / (mask_inter_chain.sum() + 1e-8)
        # print('inter_Lade_loss device',inter_Lade_loss.device)
        # loss_dict["inter_LADE"] = inter_Lade_loss
        # loss_s.append(inter_Lade_loss.detach())

        # lets make local lddt 
        # print('pred shape',pred[:,:,:,1].shape)
        # print('epitope_info shape',epitope_info.unsqueeze(0).unsqueeze(-1).shape)
        # epitope_pred = pred[:,:,:,1].detach() * epitope_info.unsqueeze(0).unsqueeze(-1)
        # eptiope_true = true[:,:,1] * epitope_info.unsqueeze(-1)
        # epitope_ca_lddt = calc_lddt(
        #     pred[:,:,:,1].detach(),
        #     true[:,:,1],
        #     mask_BB,
        #     mask_2d,
        #     same_chain,
        #     negative=negative,
        #     epitope=epitope_info,
        # )
        # loss_dict["epitope_ca_lddt"] = epitope_ca_lddt.detach()
        # loss_s.append(epitope_ca_lddt.detach())
        # print('epitope_ca_lddt device',epitope_ca_lddt.device)

        # get interface pae
        # interface_pae = 

        # print('interface split',len(interface_split))
        # Lrmsd_loss = getLigandRMSD_svd(pred, true, mask_crds, interface_split, reduction='mean')
        # Lrmsd_loss = get_ligandRMSD(pred, true, mask_crds, interface_split, reduction='mean',item=item, L_s = L_s)   
        # print('Lrmsd_loss',Lrmsd_loss.shape)
        # Lrmsd_loss = Lrmsd_loss.to(pred.device)
        # tot_loss = tot_loss + w_lrmsd * Lrmsd_loss
        # # col 21 : Lrmsd_loss
        # loss_s.append(Lrmsd_loss[None].detach())
        # loss_dict["Lrmsd_loss"]=Lrmsd_loss[None].detach()
        # loss_s.append(Lrmsd_loss[None].detach())
        # print('Lrmsd_loss device',Lrmsd_loss.device)


        if pred_prev_s is not None:  # not None in validation
            lddt_s = []
            for pred_prev in pred_prev_s:
                prev_lddt = calc_allatom_lddt(
                    pred_prev[0, :, :14, :3],
                    nat_symm[:, :14, :3],
                    xs_mask[0, :, :14],
                    idx[0],
                    same_chain[0],
                    negative=negative,
                )
                lddt_s.append(prev_lddt.detach())
            lddt_s.append(true_lddt.detach())
            lddt_s = torch.stack(lddt_s)
            return tot_loss, lddt_s, torch.cat(loss_s, dim=0), loss_dict

        # for i, loss in enumerate(loss_s):
        #     print(f"{i}th loss shape", loss.shape) #, f"\n{i}th loss", loss)  
        return tot_loss, true_lddt.detach(), torch.cat(loss_s, dim=0), loss_dict

    def calc_acc(self, prob, dist, idx_pdb, mask_2d):
        """_summary_

        Args:
            prob: Soft-maxed distogram                          (B, 37, L, L)
            dist: True coordinate distance (binned)             (B, L, L)
            idx_pdb (_type_): index                             (B, L)
            mask_2d (_type_): Masked residue for true structure (B, L, L)
            (Calculate loss without masked)

        Returns:
            _type_: _description_
        """
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1]  # (B, L)
        seqsep = (
            torch.abs(idx_pdb[:, :, None] - idx_pdb[:, None, :]) + 1
        )  # (B, L, L) -> get relative position
        mask = seqsep > 24  # masked true if sequence is far away
        mask = torch.triu(mask.float())  # Get uppder triangular matrix
        mask *= (
            mask_2d  # Get final masked structure (far away & have backbone structure)
        )
        #
        cnt_ref = dist < 20  # if distance is small enough
        cnt_ref = (
            cnt_ref.float() * mask
        )  # distance small & far away & have backbone structure
        #
        cnt_pred = (
            prob[:, :20, :, :].sum(dim=1) * mask
        )  # calculate proabability of small distance (less than 20)
        # (B, L, L)
        top_pred = torch.topk(
            cnt_pred.view(B, -1), L
        )  # Get top L residue pairs which has small distance distribution
        # (B, L)
        kth = top_pred.values.min(
            dim=-1
        ).values  # residue distance which has most lowest probability (distance largest)
        # B
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        tmp_pred = torch.stack(tmp_pred, dim=0)  # [B, L, L]
        tmp_pred = (
            tmp_pred.float() * mask
        )  # [B, L, L] (get top L residue pairs with small distance distribution)
        #
        condition = torch.logical_and(
            tmp_pred == cnt_ref, cnt_ref == torch.ones_like(cnt_ref)
        )  # Far away in the sequence but close in the distance
        # good distribution also
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        # distance small(bin) & far away in sequence & have backbone structure
        n_total_pred = (tmp_pred == torch.ones_like(tmp_pred)).float().sum() + 1e-9
        # distance small(proabability sum) & far away in sequence & have backbone structure
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0 * prec * recall / (prec + recall + 1e-9)

        return torch.stack([prec, recall, F1]), cnt_pred, cnt_ref

    def load_model(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        model_name,
        rank,
        suffix="best",
        resume_train=False,
    ):
        # chk_fn = f"/home/yubeen/rf_ab_ag/model_weight/{model_name}_{suffix}.pt"
        # chk_fn = (
            # f"/home/yubeen/rf-abag-templ/results/0919_test/models/weights/{model_name}.pt"
        # )
        # chk_fn = (f"/home/yubeen/rf_abag_templ/results/0919_test/models/weights/{model_name}.pt")
        # chk_fn = (f"/home/yubeen/rf_abag_templ/results/0919_test/models/weights/RF2_apr23.pt")
        # chk_fn = (f"/home/kkh517/submit_files/27_Nov_2021/models/RF2_apr23_20.pt")
        # chk_fn = '/home/kkh517/submit_files/20231205/models/RF2_apr23_2.pt'
        # chk_fn = '/home/kkh517/submit_files/20231212/models/RF2_apr23_3.pt'
        # chk_fn = '/home/kkh517/submit_files/20231217/models/RF2_apr23_4.pt'
        # chk_fn = "/home/kkh517/Github/rf-abag-templ/DB/pre-trained/RF2_apr23_state.pt"
        # chk_fn = "/home/kkh517/submit_files/epitope_valid/models/RF2_apr23_10.pt"
        # chk_fn = "/home/kkh517/submit_files/20231221/models/RF2_apr23_best.pt"
        # chk_fn = '/home/kkh517/submit_files/20231227/models/RF2_apr23_best.pt'
        # chk_fn = '/home/kkh517/submit_files/Project/centering_lrmsd/models/RF2_apr23_0.pt'
        # chk_fn = '/home/kkh517/submit_files/Project/OnlyRigid_1gpu/models/RF2_apr23_3.pt'
        # chk_fn = '/home/kkh517/submit_files/Project/Flexible_lj/models/RF2_apr23_best.pt'
        # chk_fn = '/home/kkh517/submit_files/Project/Flexible_lj/models/RF2_apr23_20.pt'
        # chk_fn = '/home/kkh517/submit_files/Project/Flexible_wo_lj/models/RF2_apr23_74.pt'
        chk_fn = ('/home/kkh517/submit_files/Project/halfblood/models/RF2_apr23_0.pt')
        print("print chk_fn",chk_fn)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            print("no model found", model_name)
            return -1, best_valid_loss
        else:
            print("loading model", f"{model_name}_{suffix}.pt")
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        # new_chk = {}
        # for param in model.module.model.state_dict():
        #    if param not in checkpoint['model_state_dict']:
        #        print ('missing',param)
        #        rename_model=True
        #    elif (checkpoint['model_state_dict'][param].shape == model.module.model.state_dict()[param].shape):
        #        new_chk[param] = checkpoint['model_state_dict'][param]
        #    else:
        #        print (
        #            'wrong size',param,
        #            checkpoint['model_state_dict'][param].shape,
        #             model.module.model.state_dict()[param].shape )

        # model.module.model.load_state_dict(checkpoint['final_state_dict'], strict=False)
        model.module.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.module.shadow.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if resume_train and (not rename_model):
            print(" ... loading optimization params")
            loaded_epoch = checkpoint["epoch"]
            print("loaded_epoch", loaded_epoch)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                # print (' ... loading scheduler params')
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            # if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description, ProjectName = ProjectName):
        # ProjectName =time.strftime("%Y%m%d", time.localtime(time.time())) 
        if ProjectName != "":
            os.makedirs(f"{ProjectName}", exist_ok=True)
        if not os.path.exists(f"{ProjectName}models"):
            os.mkdir(f"{ProjectName}models")
        # elif os.path.exists(f"{ProjectName}/models"):
        #     ProjectName = ProjectName+'_2'
            # os.mkdir(f"{ProjectName}/models")
        name = f"{model_name}_{description}.pt"
        return os.path.join(f"{ProjectName}models", name)

    def run_model_training(self, world_size, ProjectName = ProjectName):
        """Main entry function of training
           1) make sure ddp env vars set
           2) figure out if we launched using slurm or interactively
           #    - if slurm, assume 1 job launched per GPU
           #    - if interactive, launch one job for each GPU on node

        Args:
            world_size (int): _description_
        """
        if "MASTER_ADDR" not in os.environ:
            os.environ[
                "MASTER_ADDR"
            ] = "localhost"  # multinode requires this set in submit script
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12760" #f"{self.port}"

        if (
            not self.interactive
            and "SLURM_NTASKS" in os.environ
            and "SLURM_PROCID" in os.environ
        ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            print("Launched from slurm", rank, world_size)
            os.makedirs(f"{ProjectName}models", exist_ok=True)
            os.makedirs(f"{ProjectName}validation_loss_data", exist_ok=True)
            os.makedirs(f"{ProjectName}validation_pdb", exist_ok=True)
            self.train_model(rank, world_size)
        else:
            print("Launched from interactive")
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def inference(
        self,
        ddp_model,
        inference_loader,
        gpu,
        epoch,
        interface_correctness,
        incorrect_interface_info,
    ):
        with torch.no_grad():
            ddp_model.eval()
            for inputs in inference_loader:
                (
                    network_input,
                    xyz_prev,
                    mask_recycle,
                    true_crds,
                    mask_crds,
                    msa,
                    _,
                    _,
                    _,
                    L_s,
                    item,
                    cluster,
                    interface_split,
                    epitope_info,
                ) = self._prepare_input(inputs, gpu)

                N_cycle = self.maxcycle

                output_i = (None, None, None, xyz_prev, None, mask_recycle)

                for i_cycle in range(N_cycle):
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        if i_cycle < N_cycle - 1:
                            return_raw = True
                        else:
                            return_raw = False

                        input_i = self._get_model_input(
                            network_input, output_i, i_cycle, return_raw=return_raw
                        )
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            continue

                        (
                            _,
                            # logits_dist   [B, 37, L, L],
                            # logits_omega  [B, 37, L, L],
                            # logits_theta  [B, 37, L, L],
                            # logits_phi    [B, 19, L, L]
                            _,  # [B, 21, N*L]
                            _,  # [B, L]
                            _,  # [B, 64, L, L]
                            pred_crds,  # [iteration, B, L, 27, 3]
                            alphas,  # [iteration, B, L, 10, 2]
                            pred_lddts,  # [B, 50, L]
                        ) = output_i

                        plddt_unbinned = lddt_unbin(pred_lddts)
                        _, final_crds = self.xyz_converter.compute_all_atom(
                            msa[:, i_cycle, 0], pred_crds[-1], alphas[-1]
                        )
                        interface_correct, interface_list = self.get_interface(
                            final_crds[0], true_crds[0], L_s[0], item[0]
                        )

                        if not interface_correct:
                            interface_correctness[cluster[0]][item[0]] = False
                            incorrect_interface_info[cluster[0]][
                                item[0]
                            ] = interface_list
                        else:
                            interface_correctness[cluster[0]][item[0]] = True

                        write_pdb(
                            msa[:, i_cycle, 0][0],
                            final_crds[0],
                            L_s[0],
                            Bfacts=plddt_unbinned[0],
                            prefix=f"{item[0]}",
                        )
                        sys.stdout.write(f"\r{item[0]} inference done")
                        os.makedirs(
                            f"inference_pdb/inference_pdbs_{epoch}", exist_ok=True
                        )
                        os.system(
                            f"mv {item[0]}.pdb inference_pdb/inference_pdbs_{epoch}"
                        )

    def get_interface(self, final_crds, true_crds, L_s, item):
        def _get_interface_residue(xyz, L_s, ab_length, cutoff=6.0):
            dist = torch.zeros(sum(L_s), sum(L_s))
            dist = xyz[:, None, :, None, :] - xyz[None, :, None, :, :]
            dist = (dist ** (2)).sum(dim=-1)
            dist = (dist) ** (0.5)
            dist = dist.view(*dist.shape[:2], -1)
            dist = torch.nan_to_num(dist, nan=100.0)
            dist = torch.min(dist, dim=-1)[0]
            mask = torch.le(dist, cutoff)

            antibody_length = sum(L_s[:ab_length])
            antigen_length = sum(L_s[ab_length:])

            # antibody = list(range(antibody_length))
            # antigen = list(range(antibody_length, antibody_length + antigen_length))

            ab_start = 0
            ab_end = antibody_length
            ag_start = antibody_length
            ag_end = sum(L_s)

            mask[ab_start:ab_end, ab_start:ab_end] = False
            mask[ag_start:ag_end, ag_start:ag_end] = False
            mask[ab_start:ab_end, ag_start:ag_end] = False

            interface_lists = (
                torch.unique(torch.where(mask == True)[0]) - antibody_length
            )
            interface_lists = set(interface_lists.tolist())

            return interface_lists

        chains = []
        _, hchain, lchain, agchain = item.split("_")

        ab_length = len(hchain) + len(lchain)  # not included # cases
        for i in [hchain, lchain, agchain]:
            for j in i:
                chains.append(j)

        xyz_model = final_crds[:, :14, :]
        xyz_ref = true_crds[:, :14, :]

        inf_model = _get_interface_residue(xyz_model, L_s, ab_length)
        print(item)
        print(len(xyz_model))
        print(L_s)
        print(ab_length)
        print(inf_model)
        inf_ref = _get_interface_residue(xyz_ref, L_s, ab_length)
        interface_overlap_ratio = float(len(inf_model & inf_ref)) / len(inf_model)
        if interface_overlap_ratio < 0.1:
            return False, list(inf_model)
        else:
            return True, None

    def train_model(self, rank, world_size):
        """_summary_

        Args:
            rank (_type_): _description_
            world_size (_type_): _description_
        """

        print("running ddp on rank %d, world_size %d" % (rank, world_size))
        gpu = rank % torch.cuda.device_count()
        # gpu = int(os.environ["LOCAL_RANK"])
        init_process_group(backend="nccl", world_size=world_size, rank=rank)
        # os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu}'
        torch.cuda.set_device(f"cuda:{gpu}")
        # device = torch.device('cpu')
        # define dataset & data loader

        if self.wandb:
            if rank == 0:
                wandb_args = {}
                wandb_args["lr"] = (self.init_lr,)
                wandb.init(project="rf_ab_ag")
                wandb.config.update(wandb_args)
                wandb.run.name = self.wandb_name
                wandb.run.save()

        # move some global data to cuda device
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        self.xyz_converter = self.xyz_converter.to(gpu)
        self.num_bonds = self.num_bonds.to(gpu)
        self.ljlk_parameters = self.ljlk_parameters.to(gpu)
        self.lj_correction_parameters = self.lj_correction_parameters.to(gpu)
        self.hbtypes = self.hbtypes.to(gpu)
        self.hbbaseatoms = self.hbbaseatoms.to(gpu)
        self.hbpolys = self.hbpolys.to(gpu)

        # define model
        model = EMA(RoseTTAFoldModule(**self.model_param).to(gpu), 0.99)
        # model = EMA(RoseTTAFoldModule(**self.model_param).to(gpu), 0.999)

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        # ddp_model._set_static_graph() # required to use gradient checkpointing w/ shared parameters
        if rank == 0:
            print("# of parameters:", count_parameters(ddp_model))

        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        # optimizer = torch.optim.Adam(opt_params, lr=self.init_lr)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        # scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 15000, 0.95)
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 2000, 0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        # load model
        loaded_epoch, best_valid_loss = self.load_model(
            ddp_model,
            optimizer,
            scheduler,
            scaler,
            self.model_name,
            gpu,
            suffix="",
            resume_train = False, #False, #True, # for fine-tuning, set False
        )

        if loaded_epoch >= self.n_epoch:
            DDP.cleanup()
            return

        (
            gp_train,
            l_train,
            h_train,
            hl_train,
            l_ag_train,
            h_ag_train,
            hl_ag_train,
            neg_train,
            h_val,
            hl_val,
            h_ag_val,
            hl_ag_val,
            hl_ag_test,
            # train_all,
            neg_val,
            weights,
        ) = get_train_valid_set(self.loader_param)
        # print('hl_ag_test', hl_ag_test)
        negative_keys_train = list(neg_train.keys()) + list(neg_train.keys())
        self.n_train = N_EXAMPLE_PER_EPOCH
        # self.n_l_val = (len(l_val.keys())//world_size) * world_size
        self.n_h_val = (len(h_val.keys()) // world_size) * world_size
        self.n_hl_val = (len(hl_val.keys()) // world_size) * world_size
        # self.n_l_ag_val = (len(l_ag_val.keys())//world_size) * world_size
        self.n_h_ag_val = (len(h_ag_val.keys()) // world_size) * world_size
        self.n_hl_ag_val = (len(hl_ag_val.keys()) // world_size) * world_size
        self.n_hl_ag_test = (len(hl_ag_test.keys()) // world_size) * world_size
        self.neg_val = (len(neg_val.keys()) // world_size) * world_size
        self.negative_keys = (len(neg_train.keys()) // world_size) * world_size

        valid_h_set = Dataset(
            list(h_val.keys())[: self.n_h_val],
            loader_complex_antibody_kh,
            h_val,
            self.valid_param,
            validation=True,
        )

        valid_hl_set = DatasetComplex_antibody(
            list(hl_val.keys())[: self.n_hl_val],
            loader_complex_antibody_kh,
            hl_val,
            self.valid_param,
            validation=True,
        )

        valid_h_ag_set = DatasetComplex_antibody(
            list(h_ag_val.keys())[: self.n_h_ag_val],
            loader_complex_antibody_kh,
            h_ag_val,
            self.valid_param,
            validation=True,
            unclamp=True,
        )
        valid_hl_ag_set = DatasetComplex_antibody(
            list(hl_ag_val.keys())[: self.n_hl_ag_val],
            loader_complex_antibody_kh,
            hl_ag_val,
            self.valid_param,
            validation=True,
            unclamp=True,
        )
        valid_neg_set = DatasetComplex_antibody(
            list(neg_val.keys())[: self.neg_val],
            loader_complex_antibody_kh,
            neg_val,
            self.valid_param,
            negative=True,
            validation=True,\
        )

        # if inference: no selection of interface -> predict total structure
        valid_negative_inference = DatasetComplex_antibody(
            list(neg_train.keys())[: self.negative_keys],
            loader_complex_antibody_kh,
            neg_train,
            self.valid_param,
            inference=True,
        )
        # print('n_hl_ag_test', self.n_hl_ag_test)
        # print('list(hl_ag_test.keys())', list(hl_ag_test.keys())[: self.n_hl_ag_test])
        test_hl_ag_set = DatasetComplex_antibody(
            list(hl_ag_test.keys())[: self.n_hl_ag_test],
            loader_complex_antibody_kh,
            hl_ag_test,
            self.valid_param,
            validation=True,
            unclamp=True,
        )
        
        print ("define valid_sampler on rank: ", rank, world_size)
        valid_h_sampler = data.distributed.DistributedSampler(
            valid_h_set, num_replicas=world_size, rank=rank
        )
        valid_hl_sampler = data.distributed.DistributedSampler(
            valid_hl_set, num_replicas=world_size, rank=rank
        )
        valid_h_ag_sampler = data.distributed.DistributedSampler(
            valid_h_ag_set, num_replicas=world_size, rank=rank
        )
        valid_hl_ag_sampler = data.distributed.DistributedSampler(
            valid_hl_ag_set, num_replicas=world_size, rank=rank
        )
        test_hl_ag_sampler = data.distributed.DistributedSampler(
            test_hl_ag_set, num_replicas=world_size, rank=rank
        )
        valid_neg_sampler = data.distributed.DistributedSampler(
            valid_neg_set, num_replicas=world_size, rank=rank
        )
        valid_negative_inference_sampler = data.distributed.DistributedSampler(
            valid_negative_inference, num_replicas=world_size, rank=rank
        )

        valid_h_loader = data.DataLoader(
            valid_h_set, sampler=valid_h_sampler, **LOAD_PARAM
        )
        valid_hl_loader = data.DataLoader(
            valid_hl_set, sampler=valid_hl_sampler, **LOAD_PARAM2
        )
        valid_h_ag_loader = data.DataLoader(
            valid_h_ag_set, sampler=valid_h_ag_sampler, **LOAD_PARAM
        )
        valid_hl_ag_loader = data.DataLoader(
            valid_hl_ag_set, sampler=valid_hl_ag_sampler, **LOAD_PARAM
        )
        test_hl_ag_loader = data.DataLoader(
            test_hl_ag_set, sampler=test_hl_ag_sampler, **LOAD_PARAM
        )

        valid_neg_loader = data.DataLoader(
            valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM
        )
        valid_negative_inference_loader = data.DataLoader(
            valid_negative_inference,
            sampler=valid_negative_inference_sampler,
            **LOAD_PARAM,
        )
        interface_correctness = defaultdict(dict)
        
        incorrect_interface_info = defaultdict(lambda: defaultdict(list))
        # incorrect_interface_info = defaultdict(defaultdict_list())
        # incorrect_interface_info = self.incorrect_interface_info

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
            self.loader_param,
        )
        # negative train same as hl_ag_train
        # interface_info = {cluster_id: {pdb_id: interface_true_or_not}}

        train_sampler = DistributedWeightedSampler(
            train_set,
            ab_weights=weights,
            num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
            num_replicas=world_size,
            rank=rank,
            fraction_ab= 0.5,
            fraction_gp= 0.5,
        )
        train_loader = data.DataLoader(
            train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM
        )
        # _, _, _ = self.valid_pdb_cycle(
        #    ddp_model, valid_h_loader, rank, gpu, world_size, -1
        # )

        # _, _, _ = self.valid_ppi_cycle( #kh -> checking for right checkpoint
        #     ddp_model, valid_hl_loader, rank, gpu, world_size, -1
        # )

        # _, _, _ = self.valid_ppi_cycle( #kh
        #     ddp_model, valid_h_ag_loader, rank, gpu, world_size, -1
        # ) # --> checked for kh

        # _, _, _ = self.valid_ppi_cycle( #kh
        #     ddp_model, valid_hl_ag_loader, rank, gpu, world_size, -1
        # )
        # _, _, _ = self.valid_ppi_cycle( #kh
        #     ddp_model, test_hl_ag_loader, rank, gpu, world_size, -1
        # )


        for epoch in range(loaded_epoch + 1, self.n_epoch):
        # for epoch in range(wandb.config.epochs):
            # train_sampler.set_epoch(epoch)
            print('epoch', epoch)
            valid_h_sampler.set_epoch(epoch)
            valid_hl_sampler.set_epoch(epoch)
            valid_h_ag_sampler.set_epoch(epoch)
            valid_hl_ag_sampler.set_epoch(epoch)
            valid_negative_inference_sampler.set_epoch(epoch)
            valid_neg_sampler.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            # self.inference(ddp_model, valid_negative_inference_loader, gpu, epoch, interface_correctness, incorrect_interface_info)
            # print(interface_correctness)
            # print(incorrect_interface_info)
            # print(f'\n{epoch} epoch inference done!\n')
            # train_sampler.set_epoch(epoch)

            print(f'{epoch} train_cycle start!\n')
            train_tot, train_loss, train_acc = self.train_cycle(
                ddp_model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                rank,
                gpu,
                world_size,
                epoch,
            )
            print(f'{epoch} train_cycle done!\n')
            # _, _, _ = self.valid_pdb_cycle( # len = 8
            #     ddp_model, valid_h_loader, rank, gpu, world_size, epoch
            # )

            # _, _, _ = self.valid_ppi_cycle( # len = 56
            #     ddp_model, valid_hl_loader, rank, gpu, world_size, epoch
            # )

            # _, _, _ = self.valid_ppi_cycle( # take p_antibody case # len = 16
            #     ddp_model, valid_h_ag_loader, rank, gpu, world_size, epoch
            # )

            valid_tot, valid_loss, valid_acc = self.valid_ppi_cycle( # take p_antibody case # 104
                ddp_model, valid_hl_ag_loader, rank, gpu, world_size, epoch
            )

            # _, _, _ = self.valid_ppi_cycle(
            #     ddp_model, valid_neg_loader, rank, gpu, world_size, epoch
            # ) 
            # print('train_loss', train_loss)
            loss_key = ['distogram', 'omega', 'theta', 'phi', 'masked_aa', 'p_bind', 'exp_resolved', 'epitope', 'intra_FAPE', 'inter_FAPE', 'total_FAPE', 'pae', 'all_atom_FAPE', 'torsion_angle', 'ca_lddt', 'true_lddt', 'lddt_loss', 'blen_loss', 'band_loss', 'lj_loss', 'hb_loss','inter_LADE','epitope_ca_lddt','Lrmsd_loss']
            train_key = ['train_'+key for key in loss_key]
            valid_key = ['valid_'+key for key in loss_key]
            # print('train_loss\n', train_loss)
            # print('valid_loss\n', valid_loss)
            train_loss_s = []; valid_loss_s = []
            print('valid_loss length', len(valid_loss))
            for i in range(22):
                if i < 8: # distogram, omega, theta, phi, masked_aa, p_bind, exp_resolved, epitope
                    train_loss_s.append(train_loss[i]) ; valid_loss_s.append(valid_loss[i])
                elif i == 8: # intraFAPE
                    train_loss_s.append(np.mean(train_loss[8:52])) ; valid_loss_s.append(np.mean(valid_loss[8:52]))
                elif i == 9: # interFAPE
                    train_loss_s.append(np.mean(train_loss[52:96])) ; valid_loss_s.append(np.mean(valid_loss[52:96]))
                elif i == 10: # totalFAPE
                    train_loss_s.append(np.mean(train_loss[96:140])) ; valid_loss_s.append(np.mean(valid_loss[96:140]))
                elif 11 <= i <= 13: # pae, all_atom_FAPE, torsion_angle
                    train_loss_s.append(train_loss[140+i-11]) ; valid_loss_s.append(valid_loss[140+i-11])
                elif i == 14: # ca_lddt
                    train_loss_s.append(np.mean(train_loss[143:187])) ; valid_loss_s.append(np.mean(valid_loss[143:187]))
                elif i <= 20: # true_lddt, lddt_loss, blen_loss, band_loss, lj_loss, hb_loss
                    train_loss_s.append(train_loss[187+i-15]) ; valid_loss_s.append(valid_loss[187+i-15])
                elif i == 21: # inter_LADE
                    train_loss_s.append(np.mean(train_loss[193:237])) ; valid_loss_s.append(np.mean(valid_loss[193:237]))
                elif i == 22 : # epitope_ca_lddt
                    train_loss_s.append(train_loss[237]) ; valid_loss_s.append(valid_loss[237])
                elif i == 23: # Lrmsd_loss
                    train_loss_s.append(train_loss[238]) ; valid_loss_s.append(valid_loss[238])
                else: # ligandRMSD
                    # train_loss_s.append(train_loss[193:237]) ; valid_loss_s.append(valid_loss[193:237]) # ligandRMSD reduction = 'None'
                    
                    # train_loss_s.append(train_loss[193]) ; valid_loss_s.append(valid_loss[193]) # ligandRMSD reduction = 'mean'
                    pass
            train_loss_dict = dict(zip(train_key, train_loss_s))
            valid_loss_dict = dict(zip(valid_key, valid_loss_s))
            
            if rank == 0:
                # wandb.log({"train_loss": train_tot, "train_acc": train_acc, "valid_loss": valid_tot, "valid_acc": valid_acc, "epoch": epoch, "ca_lddt":train_loss[14]})
                # wandb.log({"train_loss": train_tot, "train_acc": train_acc, "valid_loss": valid_tot, "valid_acc": valid_acc, "epoch": epoch})
            # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "epoch": epoch})
            # wandb.log({"train_loss": train_loss, "epoch": epoch})
                wandb_dict = {"train_loss": train_tot, "train_acc": train_acc.mean(), "valid_loss": valid_tot, "valid_acc": valid_acc.mean(), "epoch": epoch}
                wandb_dict.update(train_loss_dict)
                wandb_dict.update(valid_loss_dict)
                wandb.log(wandb_dict)
            if rank == 0:  # save model
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save(
                        {
                            "epoch": epoch,
                            # 'model_state_dict': ddp_model.state_dict(),
                            "model_state_dict": ddp_model.module.shadow.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "best_loss": best_valid_loss,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "valid_loss": valid_loss,
                            "valid_acc": valid_acc,
                        },
                        self.checkpoint_fn(self.model_name, "best"),
                    )

                torch.save(
                    {
                        "epoch": epoch,
                        # 'model_state_dict': ddp_model.state_dict(),
                        "model_state_dict": ddp_model.module.shadow.state_dict(),
                        "final_state_dict": ddp_model.module.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "valid_loss": valid_loss,
                        "valid_acc": valid_acc,
                        "best_loss": best_valid_loss,
                    },
                    self.checkpoint_fn(self.model_name, f"{epoch}"),
                )
            train_sampler.set_epoch(epoch + 1)

        destroy_process_group()

    def _prepare_input(self, inputs, gpu):
        def get_5_epitope(epi_sel, validation=True):
            # epi_sel [1,1,L] -> [ 1, L] 
            epi_sel = epi_sel.squeeze(0).squeeze(0)
            # print('epi_sel.shape', epi_sel.shape)
            # if epi_sel.sum() == 0:
            #     print('is it monomer???')
            current_epi = torch.nonzero(epi_sel)
            # get random 5 from current_epi
            # print('current_epi.shape', current_epi.shape)
            if len(current_epi) == 0:
                return torch.zeros_like(epi_sel)[None]
            # pick random_index integer 1 to 5
            random_index = random.randint(1, 5)
            # minimum value between random_index and len(current_epi)
            index_value = min(random_index, len(current_epi))
            random_epi = torch.randperm(len(current_epi))[:index_value]
            epi_info = torch.zeros_like(epi_sel)
            
            epi_info[random_epi] = 1.0
            # print('result epi_info.shape', epi_info.shape)
            if not validation:
                return epi_info[None]
            else:
                # print('return_epi_sel', epi_sel.unsqueeze(0).shape)
                return epi_sel.unsqueeze(0)

            

        (
            # seq,
            # msa,
            # msa_masked,
            # msa_full,
            # mask_msa,
            cluster,
            item,
            sel,
            L_s,
            # params,
            msa,
            ins,
            true_crds,
            mask_crds,
            idx_pdb,
            xyz_t,
            t1d,
            mask_t,
            xyz_prev,
            mask_prev,
            same_chain,
            chain_idx,
            unclamp,
            negative,
            interface_split,
            epi_full,
            validation,
        ) = inputs
        # print('id', item)
        # print('xyz_t', xyz_t.shape)
        # print('msa', len(msa))
        # print('sel', len(sel[0]))
        # print('epi_full', epi_full.shape, item)
        # print('epi_full is empty?', epi_full.sum())
        assert len(sel[0]) >= xyz_t.shape[1]  # = L


        msa = msa.to(gpu, non_blocking=True)
        ins = ins.to(gpu, non_blocking=True)
        L_s = L_s.to(gpu, non_blocking=True)
        # print('xyz_t', xyz_t.shape)
        # print(msa.type())
        # print(msa.shape)
        # print(ins.type())
        # print(ins.shape)
        # print(L_s)
        # print(params)
        seq_list = []
        msa_seed_orig_list = []
        msa_seed_list = []
        msa_extra_list = []
        mask_msa_list = []
        true_crds_list = []
        mask_crds_list = []
        xyz_t_list = []
        t1d_list = []
        mask_t_list = []
        xyz_prev_list = []
        mask_prev_list = []
        idx_pdb_list = []
        same_chain_list = []
        chain_idx_list = []
        epi_info_list=[]
        # print('true_crds.shape[len(msa)(=i),:,sel[i]]',true_crds.shape)
        # print('xyz_t.shape[len(msa)(=i),:,sel[i]]',xyz_t.shape)
        # print('mask_t.shape[len(msa)(=i),:,sel[i]]',mask_t.shape)
        # print('msa length', len(msa))
        for i, msai in enumerate(msa): # len(sel[0]) = L
            # print('sel', sel.shape)
            # print('msa', msai.shape)
            # print('msa type', msa.dtype)
            # print('ins', ins[i].shape)
            # print('L_s', L_s)
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                msai, ins[i], self.loader_param, L_s=L_s[i]
            )

            # select the interface region
            # print('xyz_t', xyz_t.shape)
            seq_list.append(seq[:, sel[i]])
            msa_seed_orig_list.append(msa_seed_orig[:, :, sel[i]])
            msa_seed_list.append(msa_seed[:, :, sel[i]])
            msa_extra_list.append(msa_extra[:, :, sel[i]])
            mask_msa_list.append(mask_msa[:, :, sel[i]])
            true_crds_list.append(true_crds[i, :, sel[i]])
            mask_crds_list.append(mask_crds[i, :, sel[i]])
            xyz_t_list.append(xyz_t[i, :, sel[i]])
            t1d_list.append(t1d[i, :, sel[i]])
            mask_t_list.append(mask_t[i, :, sel[i]])
            xyz_prev_list.append(xyz_prev[i, sel[i]])
            mask_prev_list.append(mask_prev[i, sel[i]])
            idx_pdb_list.append(idx_pdb[i, sel[i]])
            same_chain_list.append(same_chain[i, sel[i]][:, sel[i]])
            chain_idx_list.append(chain_idx[i, sel[i]])
            epi_info_list.append(epi_full[:, sel[i]]) # [B, L]

        seq = torch.stack(seq_list)
        msa_seed_orig = torch.stack(msa_seed_orig_list)
        msa_seed = torch.stack(msa_seed_list)
        msa_extra = torch.stack(msa_extra_list)
        mask_msa = torch.stack(mask_msa_list)
        true_crds = torch.stack(true_crds_list)
        mask_crds = torch.stack(mask_crds_list)
        xyz_t = torch.stack(xyz_t_list)
        t1d = torch.stack(t1d_list)
        mask_t = torch.stack(mask_t_list)
        xyz_prev = torch.stack(xyz_prev_list)
        mask_prev = torch.stack(mask_prev_list)
        idx_pdb = torch.stack(idx_pdb_list)
        same_chain = torch.stack(same_chain_list)
        chain_idx = torch.stack(chain_idx_list)
        epitope_full_info = torch.stack(epi_info_list)

        # epitope_info = get_5_epitope(epitope_full_info, validation=validation)
        epitope_info = epitope_full_info.squeeze(0)
        # print('mask_t_list?',mask_t_list)
        # print('seq', seq.shape)
        # print('msa_seed_orig', msa_seed_orig.shape)
        # print('msa_seed', msa_seed.shape)
        # print('msa_extra', msa_extra.shape)
        # print('mask_msa', mask_msa.shape)
        # print('true_crds', true_crds.shape)
        # print('maks_crds', mask_crds.shape)
        # print('xyz_t', xyz_t.shape)
        # print('t1d', t1d.shape)
        # print('mask_t', mask_t.shape)
        # print('xyz_prev', xyz_prev.shape)
        # print('idx_pdb', idx_pdb.shape)
        # print('same_chain', same_chain.shape)
        # print('chain_idx', chain_idx.shape)
        # transfer inputs to device
        (
            B,
            _,
            N,
            L,
        ) = (
            msa_seed_orig.shape
        )  # if loaded, B dimension added in the front (1 in this case)

        idx_pdb = idx_pdb.to(gpu, non_blocking=True)  # (B, L)
        true_crds = true_crds.to(gpu, non_blocking=True)  # (B, N_homo, L, 27, 3)
        mask_crds = mask_crds.to(gpu, non_blocking=True)  # (B, N_homo, L, 27)
        chain_idx = chain_idx.to(gpu, non_blocking=True)  # (B, L)
        same_chain = same_chain.to(gpu, non_blocking=True)  # (B, L, L)

        xyz_t = xyz_t.to(gpu, non_blocking=True)  # (B, T, L, 27, 3)
        # print('xyz_t shape (B,T,L,27,3)',xyz_t.shape)
        t1d = t1d.to(gpu, non_blocking=True)  # (B, T, L, 22)
        mask_t = mask_t.to(gpu, non_blocking=True)  # (B, T, L, 27)

        xyz_prev = xyz_prev.to(gpu, non_blocking=True)  # (B, L, 27, 3)
        mask_prev = mask_prev.to(gpu, non_blocking=True)  # (B, L, 27)

        seq = seq.to(gpu, non_blocking=True)  # (B, _, L)
        msa = msa_seed_orig.to(gpu, non_blocking=True)  # (B, _, N_clust, L)
        msa_masked = msa_seed.to(gpu, non_blocking=True)  # (B, _, N_clust, L, 48)
        msa_full = msa_extra.to(gpu, non_blocking=True)  # (B, _, N_extra, L, 25)
        mask_msa = mask_msa.to(gpu, non_blocking=True)  # (B, _, N_clust, L)
        epitope_info = epitope_info.to(gpu, non_blocking=True).int() #

        # processing template features
        # print('mask_t shape?',mask_t.shape)
        mask_t_2d = mask_t[:, :, :, :3].all(dim=-1)  # (B, T, L) # 3개의 atom이 모두 존재하는지 여부
        mask_t_2d = mask_t_2d[:, :, None] * mask_t_2d[:, :, :, None]  # (B, T, L, L)
        mask_t_2d = (
            mask_t_2d.float()# (B, T, L, L)
            # * same_chain.float()[:, None]  -> template 간의 oreintation 지우지 않기 
        )  # (ignore inter-chain region)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)  # (B, T, L, L, 44)
        # get torsion angles from templates
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)  # (B*T, L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(
            xyz_t.reshape(-1, L, 27, 3), seq_tmp, mask_in=mask_t.reshape(-1, L, 27)
        )
        # alpha = [B * T, L, 10, 2] (cos, sin)
        # 0, 1, 2 (omega, phi, psi)
        # 3, 4, 5, 6, 7 (side chain torsion angle)
        # 7, 8, 9 (CB bend, CB twist, CG bend)
        alpha = alpha.reshape(B, -1, L, 10, 2)  # [B, T, L, 10, 2]
        alpha_mask = alpha_mask.reshape(B, -1, L, 10, 1)
        # alpha_mask = [B, T, L, 10, 1]
        # 3:7 -> 4개의 side chain torsion angle 존재 여부
        # 7&8 -> aa가 GLY인지의 여부
        # 8   -> aa가 GLY, ALA, UNK, MASK 인지의 여부

        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 30)
        # [B, T, L, 30]
        network_input = {}
        network_input["msa_latent"] = msa_masked  # (B, _, N_clust, L, 48)
        network_input["msa_full"] = msa_full  # (B, _, N_extra, L, 25)
        network_input["seq"] = seq  # (B, _, L)
        network_input["idx"] = idx_pdb  # (B, L)
        network_input["t1d"] = t1d  # (B, T, L, 22)
        network_input["t2d"] = t2d  # (B, T, L, L, 44)
        network_input["xyz_t"] = xyz_t[
            :, :, :, 1
        ]  # (B, T, L, 3) only get Ca coordinate
        network_input["alpha_t"] = alpha_t  # (B, T, L, 30)
        network_input["mask_t"] = mask_t_2d  # (B, T, L, L)
        network_input["same_chain"] = same_chain  # (B, L, L)
        network_input["chain_idx"] = chain_idx
        mask_recycle = mask_prev[:, :, :3].bool().all(dim=-1)
        mask_recycle = mask_recycle[:, :, None] * mask_recycle[:, None, :]  # (B, L, L)
        mask_recycle = same_chain.float() * mask_recycle.float()  # (B, L, L)

        # print('alpha_t', alpha_t.shape)
        # print('msa_latent', msa_masked.shape)
        # print('msa_full', msa_full.shape)
        # print('seq', seq.shape)
        # print('idx', idx_pdb.shape)
        # print('t1d', t1d.shape)
        # print('t2d', t2d.shape)
        # print('xyz_t', xyz_t.shape)
        # print('mask_t', mask_t_2d.shape)
        # print('same_chain', same_chain.shape)
        # print('chain_idx', chain_idx.shape)
        # print('xyz_prev', xyz_prev.shape)
        # print('mask_recycle', mask_recycle.shape)
        # print('true_crds', true_crds.shape)
        # print('mask_crds', mask_crds.shape)
        # print('msa', msa.shape)
        # print('mask_msa', mask_msa.shape)
        
        #for checking the clamp effect kh
        if validation:
            unclamp = True
        return (
            network_input,
            xyz_prev,  # (B, L, 27, 3)
            mask_recycle,  # (B, L, L)
            true_crds,  # (B, L, 27, 3)
            mask_crds,  # (B, L, 27)
            msa,  # (B, _, N_clust, L)
            mask_msa,  # (B, _, N_clust, L)
            unclamp,  # T/F
            negative,  # T/F
            L_s,
            item,
            cluster,
            interface_split,
            epitope_info,
        )

    def _get_model_input(
        self, network_input, output_i, i_cycle, return_raw=False, use_checkpoint=False
    ):
        """Get model input for RoseTTAFold module

        Args:
            network_input (dict): dictionary of various variables
            [msa_latent, msa_full, seq, idx, t1d, t2d, xyz_t, alpha_t, mask_t, same_chain]
            output_i (list): Last cycle's output [msa_prev, pair_prev, xyz_prev, state_prev, alpha]
            i_cycle (int): Cycle number
            return_raw (bool, optional): Return final structure. Defaults to False.
            use_checkpoint (bool, optional): Use checkpoint or not. Defaults to False.
            Use checkpoint in last cycle.
        Returns:
            _type_: _description_
        """

        input_i = {}
        for key in network_input:
            if key in ["msa_latent", "msa_full", "seq"]:
                input_i[key] = network_input[key][
                    :, i_cycle
                ]  # get specific cycle's input
            else:
                input_i[key] = network_input[key]
        msa_prev, pair_prev, state_prev, xyz_prev, alpha, mask_recycle, epitope_info = output_i # , epitope_crop = output_i
        input_i["msa_prev"] = msa_prev
        input_i["pair_prev"] = pair_prev
        input_i["state_prev"] = state_prev
        input_i["xyz"] = xyz_prev
        input_i["mask_recycle"] = mask_recycle
        input_i["return_raw"] = return_raw
        input_i["use_checkpoint"] = use_checkpoint
        input_i["epitope_info"] = epitope_info
        return input_i

    def _get_loss_and_misc(
        self,
        output_i,
        true_crds,
        mask_crds,
        same_chain,
        msa,
        mask_msa,
        idx_pdb,
        epitope_info,
        unclamp,
        negative,
        interface_split,
        pred_prev_s=None,
        return_cnt=False,
        item=None,
        L_s=None,
    ):
        """_summary_

        Args:
            output_i (dict): Predicted features
            true_crds: True coordinates [B, L, 27, 3]
            mask_crds: Masked coordinates [B, L, 27]
            same_chain: Same chain or not [B, L, L]
            msa: MSA                      [B, N_clust, L]
            mask_msa:                     [B, N_clust, L]
            idx_pdb (_type_):             [B, L]
            unclamp (bool): _description_
            negative (bool): _description_
            pred_prev_s (_type_, optional): _description_. Defaults to None.
            return_cnt (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        (
            logit_s,
            # logits_dist   [B, 37, L, L],
            # logits_omega  [B, 37, L, L],
            # logits_theta  [B, 37, L, L],
            # logits_phi    [B, 19, L, L]
            logit_aa_s,  # [B, 21, N*L]
            logit_exp,  # [B, L]
            logit_epitope, #[B, L]
            logit_pae,  # [B, 64, L, L]
            p_bind,
            pred_crds,  # [B, iteration, L, 27, 3]
            alphas,  # [B, iteration, L, 10, 2]
            pred_lddts,  # [B, 50, L]
        ) = output_i

        # find closest homo-oligomer pairs (find closest answer for homo)
        true_crds, mask_crds = resolve_equiv_natives(
            pred_crds[-1], true_crds, mask_crds
        )

        # processing labels for distogram orientograms
        mask_BB = ~(
            mask_crds[:, :, :3].sum(dim=-1) < 3.0
        )  # ignore residues having missing BB atoms for loss calculation
        mask_2d = (
            mask_BB[:, None, :] * mask_BB[:, :, None]  # (B, L, L)
        )  # ignore pairs having missing residues

        c6d = xyz_to_c6d(true_crds)  # (B, L, L, 4) - dist, omega, theta, phi 2D maps
        c6d = c6d_to_bins2(
            c6d, same_chain, negative=negative
        )  # (B, L, L, 4) -> binned value

        prob = self.active_fn(logit_s[0])  # distogram (B, 37, L, L)
        # print('is acc okay?')
        acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[..., 0], idx_pdb, mask_2d)
        
        # print('yes ok')
        # acc_s : [Preision, Recall, F1]
        # cnt_pred: [B, L, L] - proability sum of small distance (bin less than 20)
        # cnt_ref : [B, L, L] = distance small & far away in sequence & have backbone structure
        loss, lddt, loss_s, loss_dict = self.calc_loss(
            logit_s,
            c6d,
            logit_aa_s,
            msa,
            mask_msa,
            logit_exp,
            logit_epitope,
            logit_pae,
            p_bind,
            pred_crds,
            alphas,
            true_crds,
            mask_crds,
            mask_BB,
            mask_2d,
            same_chain,
            pred_lddts,
            idx_pdb,
            interface_split,
            epitope_info,
            unclamp=unclamp,
            negative=negative,
            pred_prev_s=pred_prev_s,
            item = item,
            L_s = L_s,
            **self.loss_param,

        )
        # print(f"ca_lddt loss : {loss_dict['ca_lddt']}")
        # print(f"loss[14] ? : {loss[14]}")

        if return_cnt:
            return loss, lddt, loss_s, acc_s, cnt_pred, cnt_ref, loss_dict

        return loss, lddt, loss_s, acc_s, loss_dict

    def train_cycle(
        self,
        ddp_model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        rank,
        gpu,
        world_size,
        epoch,
    ):
        # Turn on training mode
        ddp_model.train()

        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()

        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        input_count = 0 ; output_count = 0
        for inputs in train_loader:
            # print('inputs in train_loader', inputs[1])
            input_count += 1
            (
                network_input,
                xyz_prev,
                mask_recycle,
                true_crds,
                mask_crds,
                msa,
                mask_msa,
                unclamp,
                negative,
                L_s,
                item,
                cluster,
                interface_split,
                epitope_info,
            ) = self._prepare_input(inputs, gpu)
            # print('inputs', inputs)
            # print("input prepared")
            counter += 1  # count number of training inputs

            N_cycle = np.random.randint(1, self.maxcycle + 1)  # number of recycling

            output_i = (None, None, None, xyz_prev, None, mask_recycle, epitope_info)

            for i_cycle in range(N_cycle):
                with ExitStack() as stack:
                    if i_cycle < N_cycle - 1:
                        stack.enter_context(torch.no_grad())
                        stack.enter_context(ddp_model.no_sync())
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw = True
                        use_checkpoint = False
                    else:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw = False
                        use_checkpoint = True

                    input_i = self._get_model_input(
                        network_input,
                        output_i,
                        i_cycle,
                        return_raw=return_raw,
                        use_checkpoint=use_checkpoint,
                    )

                    output_i = ddp_model(**input_i)

                    if i_cycle < N_cycle - 1:
                        continue
                    # for i in range(len(output_i)): # for later on
                    #     check_nan = torch.isnan(output_i[i]).any()
                    #     print('check_nan', check_nan)
                    #     print('output_i[i]', output_i[i])
                        


                    
                    loss, _, loss_s, acc_s, loss_dict = self._get_loss_and_misc(
                        output_i,
                        true_crds,
                        mask_crds,
                        network_input["same_chain"],
                        msa[:, i_cycle],
                        mask_msa[:, i_cycle],
                        network_input["idx"],
                        epitope_info,
                        unclamp,
                        negative,
                        interface_split,
                        item=item,
                        L_s=L_s,
                        return_cnt=False,
                    )

            # print('loss', loss)
            chain_length = network_input["idx"].shape[1]
            length_weight = np.sqrt(float(chain_length) / self.crop_size)
            # print(f'{item} train_length_weight', length_weight)
            # print('length_weight', length_weight)
            loss = loss / self.ACCUM_STEP * length_weight
            output_count += 1
            
            print("item", item[0], float(loss * self.ACCUM_STEP))
            # print('output_count', output_count)
            print('loss value', loss)
            scaler.scale(loss).backward()
            # print('after backward')
            if counter % self.ACCUM_STEP == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = scale != scaler.get_scale()
                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()
                ddp_model.module.update()  # apply EMA
                # print('after step')

            # check parameters with no grad
            # if rank == 0:
            #    for n, p in ddp_model.named_parameters():
            #        if p.grad is None and p.requires_grad is True:
            # self.wandb_name = wandb_name

            #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model

            local_tot += loss.detach() * self.ACCUM_STEP
            if local_loss is None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += loss_s.detach() * length_weight
            local_acc += acc_s.detach()
            # print('after local_acc')
            train_tot += loss.detach() * self.ACCUM_STEP
            if train_loss is None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach() * length_weight
            train_acc += acc_s.detach()
            # print('after train_acc')
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated() / 1e9
                    end_event.record()
                    torch.cuda.synchronize()
                    train_time = start_event.elapsed_time(end_event)
                    local_tot /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc /= float(N_PRINT_TRAIN)

                    # local_tot = local_tot.cpu().detach()
                    # local_loss = local_loss.cpu().detach().numpy()
                    # local_acc = local_acc.cpu().detach().numpy()
                    local_tot = local_tot.detach().cpu()
                    local_loss = local_loss.detach().cpu().numpy()
                    local_acc = local_acc.detach().cpu().numpy()


                    sys.stdout.write(
                        "Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"
                        % (
                            epoch,
                            self.n_epoch,
                            counter * self.batch_size * world_size,
                            self.n_train,
                            train_time,
                            local_tot,
                            " ".join(["%8.4f" % l for l in local_loss]),
                            local_acc[0],
                            local_acc[1],
                            local_acc[2],
                            max_mem,
                        )
                    )
                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None
                    local_acc = None
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            # print('end_end_end_end')
        # print('after printing local')
        # write total train loss
        train_tot /= float(counter * world_size)
        train_loss /= float(counter * world_size)
        train_acc /= float(counter * world_size)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        # print('check1')
        # train_tot = train_tot.cpu().detach()
        # train_loss = train_loss.cpu().detach().numpy()
        # train_acc = train_acc.cpu().detach().numpy()
        train_tot = train_tot.detach().cpu()
        train_loss = train_loss.detach().cpu().numpy()
        train_acc = train_acc.detach().cpu().numpy()
        # print('check2')
        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write(
                "Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"
                % (
                    epoch,
                    self.n_epoch,
                    self.n_train,
                    self.n_train,
                    train_time,
                    train_tot,
                    " ".join(["%8.4f" % l for l in train_loss]),
                    train_acc[0],
                    train_acc[1],
                    train_acc[2],
                )
            )
            sys.stdout.flush()

        return train_tot, train_loss, train_acc

    def valid_pdb_cycle(
        self, ddp_model, valid_loader, rank, gpu, world_size, epoch, header="PDB", ProjectName = ProjectName
    ):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        counter = 0
        
        start_time = time.time()

        with torch.no_grad():  # no need to calculate gradient
            ddp_model.eval()  # change it to eval mode
            for inputs in valid_loader:
                (
                    network_input,
                    xyz_prev,
                    mask_recycle,
                    true_crds,
                    mask_crds,
                    msa,
                    mask_msa,
                    unclamp,
                    negative,
                    L_s,
                    item,
                    cluster,
                    interface_split,
                    epitope_info,
                ) = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle  # number of recycling

                output_i = (None, None, None, xyz_prev, None, mask_recycle, epitope_info) #, epitope_crop)
                pred_prev_s = list()
                for i_cycle in range(N_cycle):
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint = False
                        if i_cycle < N_cycle - 1:
                            return_raw = True
                        else:
                            return_raw = False

                        input_i = self._get_model_input(
                            network_input, output_i, i_cycle, return_raw=return_raw
                        )

                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            predTs = output_i[3]
                            pred_tors = output_i[4]
                            _, pred_all = self.xyz_converter.compute_all_atom(
                                msa[:, i_cycle, 0], predTs, pred_tors
                            )
                            pred_prev_s.append(pred_all.detach())
                            continue

                        (
                            loss,
                            lddt_s,
                            loss_s,
                            acc_s,
                            loss_dict,
                        ) = self._get_loss_and_misc(
                            output_i,
                            true_crds,
                            mask_crds,
                            network_input["same_chain"],
                            msa[:, i_cycle],
                            mask_msa[:, i_cycle],
                            network_input["idx"],
                            epitope_info,
                            unclamp,
                            negative,
                            interface_split,
                            pred_prev_s,
                        )
                        (
                            logit_s,
                            # logits_dist   [B, 37, L, L],
                            # logits_omega  [B, 37, L, L],
                            # logits_theta  [B, 37, L, L],
                            # logits_phi    [B, 19, L, L]
                            logit_aa_s,  # [B, 21, N*L]
                            logit_exp,  # [B, L]
                            logit_epitope, #[B, L]
                            logit_pae,  # [B, 64, L, L]
                            p_bind,
                            pred_crds,  # [iteration, B, L, 27, 3]
                            alphas,  # [iteration, B, L, 10, 2]
                            pred_lddts,  # [B, 50, L]
                        ) = output_i

                        for i in range(len(pred_crds)):
                            if i == 0 or i % 4 == 3:
                                plddt_unbinned = lddt_unbin(pred_lddts)

                                _, final_crds = self.xyz_converter.compute_all_atom(
                                    msa[:, i_cycle, 0], pred_crds[i], alphas[i]
                                )

                                write_pdb(
                                    msa[:, i_cycle, 0][0],
                                    final_crds[0],
                                    L_s[0],
                                    Bfacts=plddt_unbinned[0],
                                    prefix=f"{item[0]}_{i_cycle}_{i:02d}",
                                )
                                os.makedirs(
                                    f"{ProjectName}validation_pdb/validation_pdbs_{epoch}",
                                    exist_ok=True,
                                )
                                os.system(
                                    f"mv {item[0]}_{i_cycle}_{i:02d}.pdb {ProjectName}validation_pdb/validation_pdbs_{epoch}"
                                )

                chain_length = network_input["idx"].shape[1]
                length_weight = np.sqrt(float(chain_length) / self.crop_size)
                # print(f'{item} validation length_weight', length_weight)
                valid_tot += loss.detach()  # * length_weight
                print("item", item[0], float(loss))
                if valid_loss is None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                valid_loss += loss_s.detach()  # * length_weight
                for k, v in loss_dict.items():
                    loss_dict[k] = v  # * length_weight
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()
                
                
                with open(f"{ProjectName}validation_loss_data/{item[0]}_{epoch}.out", "w") as f_out:
                    f_out.write(f"model_weight {epoch}\n")
                    f_out.write(f"recycle_number {i_cycle}\n")
                    f_out.write(f"total_loss {valid_tot}\n")
                    for k, v in loss_dict.items():
                        f_out.write(f"{k} ")
                        for vs in v:
                            f_out.write(f"{vs} ")
                        f_out.write("\n")
        # print('counter', counter)
        # print('world_size', world_size)
        valid_tot /= float(counter * world_size)
        valid_loss /= float(counter * world_size)
        valid_acc /= float(counter * world_size)
        valid_lddt /= float(counter * world_size)

        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)

        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()

        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write(
                "%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %s | %.4f %.4f %.4f\n"
                % (
                    header,
                    epoch,
                    self.n_epoch,
                    counter * world_size,
                    counter * world_size,
                    train_time,
                    valid_tot,
                    " ".join(["%8.4f" % l for l in valid_loss]),
                    " ".join(["%8.4f" % l for l in valid_lddt]),
                    valid_acc[0],
                    valid_acc[1],
                    valid_acc[2],
                )
            )
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc

    def valid_ppi_cycle(
        self,
        ddp_model,
        valid_ppi_loader,
        rank,
        gpu,
        world_size,
        epoch,
        verbose=False,
        ProjectName = ProjectName,
    ):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        valid_inter = None
        counter = 0

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        start_time = time.time()
        with torch.no_grad():  # no need to calculate gradient
            ddp_model.eval()  # change it to eval mode
            for inputs in valid_ppi_loader:
                # print(inputs)
                (
                    network_input,
                    xyz_prev,
                    mask_recycle,
                    true_crds,
                    mask_crds,
                    msa,
                    mask_msa,
                    unclamp,
                    negative,
                    L_s,
                    item,
                    cluster,
                    interface_split,
                    epitope_info,
                ) = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle  # number of recycling

                output_i = (None, None, None, xyz_prev, None, mask_recycle, epitope_info) #, epitope_crop)
                pred_prev_s = list()
                for i_cycle in range(N_cycle):
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint = False
                        if i_cycle < N_cycle - 1:
                            return_raw = True
                        else:
                            return_raw = False

                        input_i = self._get_model_input(
                            network_input, output_i, i_cycle, return_raw=return_raw
                        )
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            predTs = output_i[3]
                            pred_tors = output_i[4]
                            _, pred_all = self.xyz_converter.compute_all_atom(
                                msa[:, i_cycle, 0], predTs, pred_tors
                            )
                            pred_prev_s.append(pred_all.detach())
                            continue

                        (
                            loss,
                            lddt_s,
                            loss_s,
                            acc_s,
                            cnt_pred,
                            cnt_ref,
                            loss_dict,
                        ) = self._get_loss_and_misc(
                            output_i,
                            true_crds,
                            mask_crds,
                            network_input["same_chain"],
                            msa[:, i_cycle],
                            mask_msa[:, i_cycle],
                            network_input["idx"],
                            epitope_info, #kh
                            unclamp,
                            negative,
                            interface_split,
                            pred_prev_s,
                            return_cnt=True,
                            item=item,
                            L_s = L_s,
                        )

                        (
                            logit_s,
                            # logits_dist   [B, 37, L, L],
                            # logits_omega  [B, 37, L, L],
                            # logits_theta  [B, 37, L, L],
                            # logits_phi    [B, 19, L, L]
                            logit_aa_s,  # [B, 21, N*L]
                            logit_exp,  # [B, L]
                            logit_epitope, #[B, L]
                            logit_pae,  # [B, 64, L, L]
                            p_bind,
                            pred_crds,  # [iteration, B, L, 27, 3]
                            alphas,  # [iteration, B, L, 10, 2]
                            pred_lddts,  # [B, 50, L]
                        ) = output_i

                        for i in range(len(pred_crds)):
                            if i == 0 or i % 4 == 3:
                                plddt_unbinned = lddt_unbin(pred_lddts)

                                _, final_crds = self.xyz_converter.compute_all_atom(
                                    msa[:, i_cycle, 0], pred_crds[i], alphas[i]
                                )

                                write_pdb(
                                    msa[:, i_cycle, 0][0],
                                    final_crds[0],
                                    L_s[0],
                                    Bfacts=plddt_unbinned[0],
                                    prefix=f"{item[0]}_{i_cycle}_{i:02d}",
                                )
                                os.makedirs(
                                    f"{ProjectName}validation_pdb/validation_pdbs_{epoch}",
                                    exist_ok=True,
                                )
                                os.system(
                                    f"mv {item[0]}_{i_cycle}_{i:02d}.pdb {ProjectName}validation_pdb/validation_pdbs_{epoch}"
                                )

                        # inter-chain contact prob
                        cnt_pred = cnt_pred * (1 - network_input["same_chain"]).float()
                        cnt_ref = cnt_ref * (1 - network_input["same_chain"]).float()
                        max_prob = cnt_pred.max()
                        if max_prob > 0.5:
                            if (cnt_ref > 0).any():
                                TP += 1.0
                            else:
                                FP += 1.0
                        else:
                            if (cnt_ref > 0).any():
                                FN += 1.0
                            else:
                                TN += 1.0
                        inter_s = torch.tensor(
                            [TP, FP, TN, FN], device=cnt_pred.device
                        ).float()

                chain_length = network_input["idx"].shape[1]
                length_weight = np.sqrt(float(chain_length) / self.crop_size)
                # print(f'{item} validation length weight', length_weight)
                valid_tot += loss.detach()  # * length_weight
                print("item", item[0], float(loss))
                if valid_loss is None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                    valid_inter = torch.zeros_like(inter_s.detach())
                valid_loss += loss_s.detach()  # * length_weight
                for k, v in loss_dict.items():
                    loss_dict[k] = v  # * length_weight
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()
                valid_inter += inter_s.detach()
                with open(f"{ProjectName}validation_loss_data/{item[0]}_{epoch}.out", "w") as f_out:
                    f_out.write(f"model_weight {epoch}\n")
                    f_out.write(f"recycle_number {i_cycle}\n")
                    f_out.write(f"total_loss {valid_tot}\n")
                    for k, v in loss_dict.items():
                        f_out.write(f"{k} ")
                        for vs in v:
                            f_out.write(f"{vs} ")
                        f_out.write("\n")
        valid_tot /= float(counter * world_size)
        valid_loss /= float(counter * world_size)
        valid_acc /= float(counter * world_size)
        valid_lddt /= float(counter * world_size)

        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)

        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        print("valid_loss", valid_loss)
        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write(
                "Hetero: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %s | %.4f %.4f %.4f\n"
                #"Hetero: [epoch/self.n_epoch] Batch: [counter * world_size/counter * world_size] Time: train_time | total_loss: valid_tot | %s | %s | %.4f %.4f %.4f\n"
                % (
                    epoch,
                    self.n_epoch,
                    counter * world_size,
                    counter * world_size,
                    train_time,
                    valid_tot,
                    " ".join(["%8.4f" % l for l in valid_loss]),
                    " ".join(["%8.4f" % l for l in valid_lddt]),
                    valid_acc[0],
                    valid_acc[1],
                    valid_acc[2],
                )
            )
            sys.stdout.flush()
            torch.cuda.empty_cache()

        return valid_tot, valid_loss, valid_acc


if __name__ == "__main__":
    from arguments import get_args

    args, model_param, loader_param, loss_param = get_args()
    sys.stdout.write("Loader_param:\n, %s\n" % loader_param)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mp.freeze_support()
    train = Trainer(
        model_name=args.model_name,
        interactive=args.interactive,
        n_epoch=args.num_epochs,
        lr=args.lr,
        l2_coeff=1.0e-4,
        port=args.port,
        model_param=model_param,
        loader_param=loader_param,
        loss_param=loss_param,
        batch_size=args.batch_size,
        accum_step=args.accum,
        maxcycle=args.maxcycle,
        crop_size=args.crop,
        wandb=args.wandb,
        wandb_name=args.wandb_name,
    )
    train.run_model_training(torch.cuda.device_count())
