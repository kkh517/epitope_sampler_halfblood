import torch
import torch.nn as nn
from rfabflex.model.Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from rfabflex.model.Track_module import IterativeSimulator
from rfabflex.model.AuxiliaryPredictor import (
    DistanceNetwork,
    MaskedTokenNetwork,
    ExpResolvedNetwork,
    LDDTNetwork,
    PAENetwork,
    BinderNetwork,
    EpitopeNetwork,
)
from rfabflex.common.util import INIT_CRDS
from opt_einsum import contract as einsum


class RoseTTAFoldModule(nn.Module):
    def __init__(
        self,
        n_extra_block=4,
        n_main_block=8,
        n_ref_block=4,
        d_msa=256,
        d_msa_full=64,
        d_pair=128,
        d_templ=64,
        n_head_msa=8,
        n_head_pair=4,
        n_head_templ=4,
        d_hidden=32,
        d_hidden_templ=64,
        p_drop=0.15,
        SE3_param_full={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
        SE3_param_topk={
            "l0_in_features": 32,
            "l0_out_features": 16,
            "num_edge_features": 32,
        },
    ):
        super(RoseTTAFoldModule, self).__init__()
        #
        # Input Embeddings
        d_state = SE3_param_topk["l0_out_features"]
        self.latent_emb = MSA_emb(
            d_msa=d_msa, d_pair=d_pair, d_state=d_state, p_drop=p_drop
        )
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25, p_drop=p_drop)
        self.templ_emb = Templ_emb(
            d_pair=d_pair,
            d_templ=d_templ,
            d_state=d_state,
            n_head=n_head_templ,
            d_hidden=d_hidden_templ,
            p_drop=0.25,
        )
        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            d_msa=d_msa,
            d_msa_full=d_msa_full,
            d_pair=d_pair,
            d_hidden=d_hidden,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            SE3_param_full=SE3_param_full,
            SE3_param_topk=SE3_param_topk,
            p_drop=p_drop,
        )
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)

        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)
        self.pae_pred = PAENetwork(d_pair)
        self.bind_pred = BinderNetwork()
        self.epitope_pred = EpitopeNetwork(d_msa, d_state)


    def forward(
        self,
        msa_latent=None,
        msa_full=None,
        seq=None,
        xyz=None, # updated in _get_model_input
        idx=None,
        t1d=None,
        t2d=None,
        xyz_t=None,
        alpha_t=None,
        mask_t=None,
        chain_idx=None,
        same_chain=None, # network_input until here
        msa_prev=None,
        pair_prev=None,
        state_prev=None,
        mask_recycle=None,
        epitope_info=None, # can be added in _get_model_input
        return_raw=False,
        return_full=False, # not added
        use_checkpoint=False,
        validation=False,
    ):
        """_summary_

        Args:
            msa_latent  : MSA one-hot encoding & cluster information. [B, N_clust, L, 48]
            msa_full    : Extra MSA feature.                          [B, N_extra, L, 25]
            seq         : Sequence information                        [B, L]
            xyz         : Previous xyz coordinates                    [B, L, 27, 3] (initially first template coords)
            idx         : Residue index                               [B, L]
            t1d         : Template 1D information                     [B, T, L, 22]
            t2d         : Template 2D information                     [B, T, L, L, 44]
            xyz_t       : Template Ca coordinates                     [B, T, L, 3]
            alpha_t     : Template torsion information                [B, T, L, 30]
            mask_t      : Template mask information                   [B, T, L, L]
            chain_idx   : chain index                                 [B, T, L, L] (0 for H, 1 for L, 2~ for antigen)
            same_chain  : 1 for same chain, 0 for different chain     [B, L, L]
            msa_prev    : msa                                         [B, L, 256]            (initially None)
            pair_prev   : previous pair representation                [B, L, L, d_pair] (initially None)
            state_prev  : previous state representation               [B, L, 32]        (initially None)
            mask_recycle: If masked or not for N, CA, C               [B, L, L]     (initially first template masking)
            return_raw (bool)       : Get the last structure. Defaults to False.
            return_full (bool)      : Return full information. Defaults to False.
            use_checkpoint (bool)   : Use checkpoint or not. Defaults to False.

        Returns:
            If return_raw, return [msa_prev, pair_prev, state_prev, xyz_prev, alpha, mask_recycle]
            or logits, logits_aa, logits_exp, logits_pae, xyz, alpha, lddt
        """

        B, N, L = msa_latent.shape[:3]
        dtype = msa_latent.dtype
        # print(mask_recycle)

        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, chain_idx, epitope_info)
        # msa_latent     [B, N_clust, L, d_model = 256]
        # pair           [B, L, L, d_pair = 128]
        # state          [B, L, d_state = 64]

        msa_full = self.full_emb(msa_full, seq, idx)
        msa_latent, pair, state = msa_latent.to(dtype), pair.to(dtype), state.to(dtype)
        # msa_full       [B, N_extra, L, d_model = 256]

        # Do recycling
        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:, 0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(
            seq, msa_prev, pair_prev, state_prev, xyz, mask_recycle
        )
        # msa_recycle    [B, L, d_model = 256]
        # pair_recycle   [B, L, L, d_pair = 128]
        # state_recycle  [B, L, d_state = 32]

        msa_latent[:, 0] = msa_latent[:, 0] + msa_recycle.reshape(B, L, -1)
        pair = pair + pair_recycle
        state = state + state_recycle
        #
        # add template embedding
        pair, state = self.templ_emb(
            t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, use_checkpoint=use_checkpoint
        )
        # pair       = [B, L, L, d_pair = 128]
        # state      = [B, L, d_state=64]

        # Predict coordinates from given inputs
        msa, pair, R, T, alpha, state = self.simulator(
            seq,
            msa_latent,
            msa_full,
            pair,
            xyz[:, :, :3],
            state,
            idx,
            use_checkpoint=use_checkpoint,
        )
        # add epitope information on state

        # update state with epitope information
        # state = state + nn.Embedding(2, 32)(crop_epitope) # not good idea.

        # msa        = [B, N_clust, L, d_model  256]
        # pair       = [B, L, L, d_pair = 128]
        # R          = [Iteration, B, L, 3, 3]
        # T          = [Iteration, B, L, 3]
        # alpha_s    = [Iteration, B, L, 10, 2]
        # state      = [B, L, 32]

        if return_raw:
            # get last structure
            xyz = einsum(
                "blij,blaj->blai", R[-1], xyz - xyz[:, :, 1].unsqueeze(-2)
            ) + T[-1].unsqueeze(-2)
            return msa[:, 0], pair, state, xyz, alpha[-1], None, epitope_info


        # predict masked amino acids
        logits_aa = self.aa_pred(msa)  # [B, 21, N*L]

        # predict distogram & orientograms
        logits = self.c6d_pred(pair)
        # logits_dist   = [B, 37, L, L]
        # logits_omega  = [B, 37, L, L]
        # logits_theta  = [B, 37, L, L]
        # logits_phi    = [B, 19, L, L]

        # Predict LDDT
        lddt = self.lddt_pred(state.view(B, L, -1))  # [B, 50, L]

        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:, 0], state)  # [B, L]

        # predict epitope
        logits_epitope = self.epitope_pred(msa[:, 0], state)  # [B, L]

        # predict PAE
        logits_pae = self.pae_pred(pair)  # [B, 64, L, L]

        # predict bind/no-bind
        p_bind = self.bind_pred(logits_pae, same_chain)

        # get all intermediate bb structures
        xyz = einsum(
            "rblij,blaj->rblai", R, xyz - xyz[:, :, 1].unsqueeze(-2)
        ) + T.unsqueeze(-2)
        # [iteration, B, L, 27, 3]
        if validation:
            # print("validation!!!")
            return (
                logits,
                logits_aa,
                logits_exp,
                logits_epitope,
                logits_pae,
                p_bind,
                xyz,
                alpha,
                lddt,
                msa[:, 0],
                pair,
                state,
                None,
            )
        return logits, logits_aa, logits_exp, logits_epitope, logits_pae, p_bind, xyz, alpha, lddt #, logits_epitope

