import torch
import torch.nn as nn
import numpy as np
import copy

from opt_einsum import contract as einsum
from rfabflex.common.chemical import aa2num
from rfabflex.common.util import rigid_from_3_points, write_pdb
from rfabflex.common.kinematics import get_dih
from rfabflex.common.scoring import HbHybType

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='none'):
        """
        Focal Loss 구현
        alpha: 클래스 가중치 (불균형 보정)
        gamma: 초점 조절 계수 (Hard Example을 더 강조)
        reduction: 'mean' (평균), 'sum' (합)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스 가중치 (None이면 일반 Focal Loss)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        p_t = torch.exp(-ce_loss)  # 모델이 예측한 확률값
        focal_loss = (1 - p_t) ** self.gamma * ce_loss  # Focal Loss 적용

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-5):
    # m = nn.Softmax(dim=-1)
    # distogram = logit_s[0]
    # distogram = distogram.permute(0, 2, 3, 1)
    # length = list(range(distogram.shape[1]))
    # for i in length:
    #     for j in length:
    #         print(list(m(distogram[0][i, j, :]).detach().cpu().numpy()))
    # print('probability')
    # distogram = torch.argmax(distogram, dim=-1)
    # for idx, i in enumerate(distogram[0]):
    #     print(idx, list(i.detach().cpu().numpy()))
    # print('answer')
    # for idx, j in enumerate(label_s[0][:, :, 0]):
    #     print(idx, list(j.detach().cpu().numpy()))
    # print('compare')
    # for idx, j in enumerate(distogram[0]):
    #     print(idx, j.detach().cpu().numpy() - label_s[0][idx, :, 0].detach().cpu().numpy())
    loss_s = list()
    for i, logit in enumerate(logit_s):
        loss = nn.CrossEntropyLoss(reduction="none")(
            logit, label_s[..., i]
        )  # (B, L, L)
        loss = (mask_2d * loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    # print('loss_s', loss_s)
    # print('mask_2d', mask_2d.sum())
    loss_s = torch.stack(loss_s)  # [4, 1]
    return loss_s


# use improved coordinate frame generation


def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
    I, B, L = N.shape[:3]
    Rs, Ts = rigid_from_3_points(
        N.view(I * B, L, 3),
        Ca.view(I * B, L, 3),
        C.view(I * B, L, 3),
        non_ideal=non_ideal,
        eps=eps,
    )
    Rs = Rs.view(I, B, L, 3, 3)
    Ts = Ts.view(I, B, L, 3)
    # t[0,1] = residue 0 -> residue 1 vector (I*B, L, L, 3)
    t = Ts[:, :, None] - Ts[:, :, :, None]
    return einsum("iblkj, iblmk -> iblmj", Rs, t)  # (I*B,L,L,3)

    # (I*B, L, 3, 3) *(I*B, L, L, 3 ) -> (I*B, L, L, 3)


def calc_str_loss(
    pred,
    true,
    logit_pae,
    mask_2d, # [B, L, L]
    same_chain, # [B, L, L]
    negative=False,
    d_clamp=10.0,
    d_clamp_inter=30.0,
    A=10.0, # about intra chain loss
    B=20.0,  # 20 from uni-fold paper # about inter chain loss

    gamma=1.0,
    eps=1e-6,
):
    """
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    """
    I = pred.shape[0]
    true = true.unsqueeze(0)  # [1, B, L, n_atom, 3]
    t_tilde_ij = get_t(true[:, :, :, 0], true[:, :, :, 1], true[:, :, :, 2], non_ideal=True)# (1,B,L,L,3)
    t_ij = get_t(pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2])  # (I,B,L,L,3)

    difference = torch.sqrt(torch.square(t_tilde_ij - t_ij).sum(dim=-1) + eps)  # (I,B,L,L)
    eij_label = difference[-1].clone().detach() # (L, L)


    if d_clamp is not None:
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = torch.where(same_chain.bool(), difference / A, difference / B)  # (I*B,L,L)
    # loss = difference / A  # (I, B, L, L)
    #
    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    mask_intra_chain = mask_2d * same_chain.bool() #(B, L, L)

    # print('same_chain', same_chain)
    # print('same_chain_size', same_chain.shape)
    if negative:
        mask_inter_chain = (same_chain.bool()) * ~(same_chain.bool())
        # print(mask_inter_chain)
    else:
        mask_inter_chain = mask_2d * ~(same_chain.bool()) #(B, L, L)
    loss_intra = (mask_intra_chain[None].float() * loss).sum(dim=(1, 2, 3)) / (mask_intra_chain.sum() + eps

    )  # (I)
    loss_inter = (mask_inter_chain[None].float() * loss).sum(dim=(1, 2, 3)) / (
        mask_inter_chain.sum() + eps
    )  # (I)
    # print('loss_intra', loss_intra)
    # print('loss_inter', loss_inter)
    loss = loss_intra + loss_inter
    # weighting loss
    w_loss = torch.pow(
        torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device)
    )
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()

    # calculate pae loss
    nbin = logit_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(
        bin_step,
        bin_step * (nbin - 1),
        nbin - 1,
        dtype=logit_pae.dtype,
        device=logit_pae.device,
    )
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
    pae_loss = torch.nn.CrossEntropyLoss(reduction="none")(logit_pae, true_pae_label)

    pae_loss = (pae_loss * mask).sum() / (mask.sum() + eps)
    return tot_loss, loss_intra.detach(), loss_inter.detach(), loss.detach(), pae_loss
    # tot_loss: weighted loss (more weight on recent structure)
    # loss - (I) loss for structures (number : iteration cycle number)
    # pae_loss : pae_loss


# resolve rotationally equivalent sidechains


def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm(
        xs[:, :, None, :] - xs[atm_mask, :][None, None, :, :], dim=-1
    )
    dists_nat = torch.linalg.norm(
        xsnat[:, :, None, :] - xsnat[atm_mask, :][None, None, :, :], dim=-1
    )
    dists_natalt = torch.linalg.norm(
        xsnat_alt[:, :, None, :] - xsnat_alt[atm_mask, :][None, None, :, :], dim=-1
    )

    drms_nat = torch.sum(torch.abs(dists_nat - dists), dim=(-1, -2))
    drms_natalt = torch.sum(torch.abs(dists_nat - dists_natalt), dim=(-1, -2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt < drms_nat

    Rsnat_symm[toflip, ...] = Rsnat_all_alt[toflip, ...]
    xs_symm[toflip, ...] = xsnat_alt[toflip, ...]

    return Rsnat_symm, xs_symm


# resolve "equivalent" natives


def resolve_equiv_natives(xs, natstack, maskstack):
    """
    xs: calculated structures [B, L, 27, 3]
    natstack: reference structures [B, N, L, 27, 3]
    maskstack: mask of reference structuers [B, N, L, 27]
    """
    if len(natstack.shape) == 4:
        return natstack, maskstack
    if natstack.shape[1] == 1:
        return natstack[:, 0, ...], maskstack[:, 0, ...]  # [B, L, 27, 3]
    dx = torch.norm(xs[:, None, :, None, 1, :] - xs[:, None, None, :, 1, :], dim=-1)
    # [B, 1, L, 1, ca, 3], [B, 1, 1, L, ca, 3]
    # [B, 1, L, L]
    dnat = torch.norm(
        natstack[:, :, :, None, 1, :] - natstack[:, :, None, :, 1, :], dim=-1
    )
    # [B, N, L, 1, 1, 3] - [B, N, 1, L, 1, 3] = [B, N, L, L]
    delta = torch.sum(torch.abs(dnat - dx), dim=(-2, -1))
    # [B, N]
    return natstack[:, torch.argmin(delta), ...], maskstack[:, torch.argmin(delta), ...]
    # [B, L, 27, 3], [B, L, 27, 3]


# torsion angle predictor loss
def torsionAngleLoss(alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8):
    I = alpha.shape[0]
    lnat = torch.sqrt(torch.sum(torch.square(alpha), dim=-1) + eps)
    anorm = alpha / (lnat[..., None])

    l_tors_ij = torch.min(
        torch.sum(torch.square(anorm - alphanat[None]), dim=-1),
        torch.sum(torch.square(anorm - alphanat_alt[None]), dim=-1),
    )

    l_tors = torch.sum(l_tors_ij * tors_mask[None]) / (torch.sum(tors_mask) * I + eps)
    l_norm = torch.sum(torch.abs(lnat - 1.0) * tors_mask[None]) / (
        torch.sum(tors_mask) * I + eps
    )
    l_planar = torch.sum(torch.abs(alpha[..., 0]) * tors_planar[None]) / (
        torch.sum(tors_planar) * I + eps
    )

    return l_tors + 0.02 * l_norm + 0.02 * l_planar


def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum("rji,rsj->rsi", Rs, xs[None, ...] - Ts[:, None, ...])
    xij_t = torch.einsum("rji,rsj->rsi", Rsnat, xsnat[None, ...] - Tsnat[:, None, ...])

    diff = torch.sqrt(torch.sum(torch.square(xij - xij_t), dim=-1) + eps)
    loss = (1.0 / Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss


def angle(a, b, c, eps=1e-6):
    """
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    """
    B, L = a.shape[:2]

    u1 = a - b
    u2 = c - b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B * L, 3)
    u2 = u2.reshape(B * L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(
        B, L, 1
    )  # (B,L,1)
    cos_theta = torch.matmul(u1[:, None, :], u2[:, :, None]).reshape(B, L, 1)

    return torch.cat([cos_theta, sin_theta], axis=-1)  # (B, L, 2)


def length(a, b):
    return torch.norm(a - b, dim=-1)

def torsion(a, b, c, d, eps=1e-6):
    # A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b - a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c - b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d - c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1)  # [B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)

    cos_angle = torch.matmul(t1[:, :, None, :], t2[:, :, :, None])[:, :, 0]
    sin_angle = torch.norm(u2, dim=-1, keepdim=True) * (
        torch.matmul(u1[:, :, None, :], t2[:, :, :, None])[:, :, 0]
    )

    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1) / (
        t1_norm * t2_norm + eps
    )  # [B,L,2]
    return cos_sin


def calc_BB_bond_geom(
    pred,
    idx,
    eps=1e-6,
    ideal_NC=1.329,
    ideal_CACN=-0.4415,
    ideal_CNCA=-0.5255,
    sig_len=0.02,
    sig_ang=0.05,
):
    """
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    """

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    B, L = pred.shape[:2]

    bonded = (idx[:, 1:] - idx[:, :-1]) == 1

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred = length(pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)  # (B, L-1)
    CN_loss = bonded * torch.clamp(
        torch.square(blen_CN_pred - ideal_NC) - sig_len**2, min=0.0
    )
    n_viol = (CN_loss > 0.0).sum()
    blen_loss = CN_loss.sum() / (n_viol + eps)

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:, :-1, 1], pred[:, :-1, 2], pred[:, 1:, 0]).reshape(
        B, L - 1
    )
    bang_CNCA_pred = cosangle(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(
        B, L - 1
    )
    CACN_loss = bonded * torch.clamp(
        torch.square(bang_CACN_pred - ideal_CACN) - sig_ang**2, min=0.0
    )
    CNCA_loss = bonded * torch.clamp(
        torch.square(bang_CNCA_pred - ideal_CNCA) - sig_ang**2, min=0.0
    )
    bang_loss = CACN_loss + CNCA_loss
    n_viol = (bang_loss > 0.0).sum()
    bang_loss = bang_loss.sum() / (n_viol + eps)

    return blen_loss, bang_loss


# Rosetta-like version of LJ (fa_atr+fa_rep)
# lj_lin is switch from linear to 12-6.  Smaller values more sharply
# penalize clashes
@torch.no_grad()
def get_Chain_Alignment_Rotation_matrix(X,Y,mask_2d, chain_mask):
    """
    rotation matrix (pred to true)
    X: set of points(B, L, N, 3)
    Y: set of points(I, B, L, N, 3)
    mask_2d : (B, L, L)
    chain_mask : (B, L, L)
    N = num of atoms
    """
    I, B, L, N, _ = Y.shape
    X = X.detach()
    Y = Y.detach()
    B, L = X.shape[:2]
    

    MASK = mask_2d.bool() * chain_mask.bool()  # (B, L, L)

    X = X.unsqueeze(-3).repeat(1, 1, L, 1, 1).transpose(1,2) # (B, L, L, N, 3)
    Y = Y.unsqueeze(-3).repeat(1, 1, 1, L, 1, 1).transpose(2,3) # (I, B, L, L, N, 3)
    X = X * MASK.unsqueeze(-1).unsqueeze(-1) # (B, L, L, N, 3)
    Y = Y * MASK.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (I, B, L, L, N, 3)

    X = X.reshape(B, L, L*N, 3) # (B, L, L*N, 3)
    Y = Y.reshape(I, B, L, L*N, 3) # (I, B, L, L*N, 3)

    valid_atom = N * MASK.sum(dim=-1)+1e-8 # (B, L)
    X_center = X.sum(dim=(-2)) / valid_atom.unsqueeze(-1) # (B, L, 3)
    Y_center = Y.sum(dim=(-2)) / valid_atom.unsqueeze(-1) # (I, B, L, 3)

    X = X - X_center.unsqueeze(-2) # (B, L, N*L, 3)
    Y = Y - Y_center.unsqueeze(-2) # (I, B, L, N*L, 3)
    X = X.reshape(B, L, L, N, 3) # (B, L, L, N, 3)
    Y = Y.reshape(I, B, L, L, N, 3) # (I, B, L, L, N, 3)
    X = X * MASK.unsqueeze(-1).unsqueeze(-1)
    Y = Y * MASK.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    X = X.reshape(B, L, L*N, 3) # (B, L, L*N, 3)
    Y = Y.reshape(I, B, L, L*N, 3)
    H = torch.einsum("bijx, abijy -> abixy", X, Y) # (I, B, L, 3, 3)
    I, B, L = H.shape[:3]
    H = H.reshape(I*B*L, 3, 3).float() # (I * B * L, 3, 3)
    U, S, V = torch.svd(H) # U : (I * B * L, 3, 3), S : (I * B * L, 3), V : (I * B * L, 3, 3)
    # Ensure a proper rotation matrix
    U = U.double() ; V = V.double()
    d = torch.sign(torch.det(V @ U.transpose(-1, -2))) # (I * B * L, 1)
    S = torch.eye(3).unsqueeze(0).repeat(I*B*L, 1, 1) # (I * B * L, 3, 3)
    S = S.to(X.device)
    # S[:, 2, 2] = d.squeeze(-1) # (I * B * L, 3, 3)
    S[:,2,2] = d
    V = V.half() ; U = U.half() ; S = S.half()
    R = V @ S @ U.transpose(-1, -2) # (I * B * L, 3, 3)
    R = R.reshape(I, B, L, 3, 3) # (I, B, L, 3, 3)
    return R.detach()

@torch.cuda.amp.autocast(enabled=False)
def get_ChainAlignment_Backbone_Displacement_Loss(true, pred, mask_2d, chain_mask, interface_split, d_clamp=10, d_clamp_inter=30, A=30, reduction='mean'):
    """
    Calculate Backbone ChADE loss
    true : set of points (B, L, N, 3)
    pred : set of points (I, B, L, N, 3)
    mask_2d : (B, L, L)
    chain_mask : (B, L, L)
    """
    I, B, L, N, _ = pred.shape
    assert B == 1
    if len(interface_split) <2:
        return torch.tensor(0.0).to(pred.device)
    true = true.detach()

    true_backbone=true[:,:,:3,:] # (B, L, 3, 3)
    pred_backbone=pred[:,:,:,:3,:] # (I, B, L, 3, 3)

    Chain_Alignment_Rotation_matrix = get_Chain_Alignment_Rotation_matrix(true_backbone, pred_backbone, mask_2d, chain_mask) # (I, B, L, 3, 3)
    t_true = true_backbone[:,None,:,:,:] - true_backbone[:,:,None,:,:] # (B, L, L, 3, 3)

    t_pred = pred_backbone[:,:,None, :,:,:] - pred_backbone[:,:,:,None,:,:] # (I, B, L, L, 3, 3)
    t_pred = t_pred.half() ; Chain_Alignment_Rotation_matrix = Chain_Alignment_Rotation_matrix.half()
    t_pred = torch.einsum("ablmij, abljk -> ablmik", t_pred, Chain_Alignment_Rotation_matrix) # (I, B, L, L, 3, 3)

    l2 = torch.square(t_true - t_pred).sum(dim=(-1)) # (I, B, L, L, 3)

    # Chain Length Penalty
    # chain_avg = chain_mask.sum(dim=-1) # (B, L)
    # chain_avg = torch.ones_like(chain_mask) / chain_avg.unsqueeze(-1) # (B, L, L)
    # idxs = torch.arange(L-1)
    # chain_num = L - torch.sum(chain_mask[:,idxs,idxs+1])

    # ignore intra-chain loss
    intra_chain_mask = ~chain_mask.bool()
    l2 = l2 * intra_chain_mask.unsqueeze(0).unsqueeze(-1) # (I, B, L, L, 3)

    if d_clamp is not None:
        clamp = torch.where(chain_mask.bool(), d_clamp**2, d_clamp_inter**2)
        clamp = clamp.unsqueeze(0).unsqueeze(-1).float() # (I, B, L, L, 1)
        l2 = torch.clamp(l2, max=clamp)
    l2 = torch.sum(l2,dim=(-1))
    l2 = l2 * mask_2d.unsqueeze(0).float() # (I, B, L, L)
    Displacement_loss = torch.sqrt(l2+1e-8) # (I, B, L, L)
    Displacement_loss = Displacement_loss / A
    Displacement_loss = Displacement_loss.sum(dim=(-1,-2)) / float(L) # (I, B)
    Displacement_loss = Displacement_loss.mean(dim=0) / chain_num # (B)
    Displacement_loss.to(pred.device)
    if reduction == 'none':
        return Displacement_loss
    elif reduction == 'mean':
        return Displacement_loss.mean() # (1)

@torch.no_grad()
def get_Local_Alignment_Rotation_matrix(X, Y,mask_2d, same_chain, d_local = 15.0):
    '''
    X: set of points (B, L, N, 3)
    Y: set of points (I, B, L, N, 3) 
    mask_2d : (B, L, L)
    chain_mask : (B, L, L)
    N = num of atoms
    '''
    I, B, L, N, _ = Y.shape
    X = X.detach()
    Y = Y.detach()
    B, L = X.shape[:2]
    X_Ca = X[:,:,1,:] # Ca, (B, L, 3)
    Y_Ca = Y[:,:,:,1,:] # Ca, (I, B, L, 3)
    # Get idx tensor for each residue that < d_local
    if d_local is None :
        d_local = 9999.0
    distance_map = torch.sqrt(torch.square(X_Ca[:, :, None, :] - X_Ca[:, None, :, :]).sum(dim=-1) +1e-8)# (M, L, L)
    MASK = torch.where(distance_map < d_local, torch.ones_like(distance_map), torch.zeros_like(distance_map)) # (B, L, L)
    MASK = MASK * mask_2d.bool() # (B, L, L)
    if same_chain is not None:
        MASK = MASK * same_chain.bool()
    X = X.unsqueeze(-3).repeat(1, 1, L, 1, 1).transpose(1,2) # (B, L, L, N, 3)
    Y = Y.unsqueeze(-3).repeat(1, 1, 1, L, 1, 1).transpose(2,3) # (I, B, L, L, N, 3)
    X = X * MASK.unsqueeze(-1).unsqueeze(-1) # (B, L, L, N, 3)
    Y = Y * MASK.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (I, B, L, L, N, 3)

    # Local Centering
    X = X.reshape(B, L, N*L, 3) # (B, L, N*L, 3)
    Y = Y.reshape(I, B, L, N*L, 3) # (I, B, L, N*L, 3)

    valid_atom = N * MASK.sum(dim=-1) # (B, L)
    # X_center = X.sum(dim=(-2)) / valid_atom.unsqueeze(-1) # (B, L, 3)
    # Y_center = Y.sum(dim=(-2)) / valid_atom.unsqueeze(-1) # (I, B, L, 3)
    X_center = X_Ca # psk update 240103
    Y_center = Y_Ca # psk update 240103
    
    X = X - X_center.unsqueeze(-2) # (B, L, N*L, 3)
    Y = Y - Y_center.unsqueeze(-2) # (I, B, L, N*L, 3)

    X = X.reshape(B, L, L, N, 3) # (B, L, L, N, 3)
    Y = Y.reshape(I, B, L, L, N, 3) # (I, B, L, L, N, 3)
    X = X * MASK.unsqueeze(-1).unsqueeze(-1)
    Y = Y * MASK.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    X = X.reshape(B, L, L*N, 3) # (B, L, L*N, 3)
    Y = Y.reshape(I, B, L, L*N, 3)
    H = torch.einsum("bijx, abijy -> abixy", X, Y) # (I, B, L, 3, 3)
    I, B, L = H.shape[:3]
    H = H.reshape(I*B*L, 3, 3) # (I * B * L, 3, 3)
    H = H.float()
    U, S, V = torch.svd(H) # U : (I * B * L, 3, 3), S : (I * B * L, 3), V : (I * B * L, 3, 3)
    # Ensure a proper rotation matrix
    d = torch.sign(torch.det(V @ U.transpose(-1, -2))) # (I * B * L, 1)
    S = torch.eye(3).unsqueeze(0).repeat(I*B*L, 1, 1) # (I * B * L, 3, 3)
    S = S.to(X.device)
    S[:, 2, 2] = d
    R = V @ S @ U.transpose(-1, -2) # (I * B * L, 3, 3)
    R = R.reshape(I, B, L, 3, 3) # (I, B, L, 3, 3)
    return R.detach() # (I, B, L, 3, 3)


@torch.cuda.amp.autocast(enabled=False)
def calc_LocalAlignment_Backbone_Displacement_Loss(true, pred, mask_crds, mask_2d,same_chain, d_local = 15.0, d_clamp = 10.0, d_clamp_inter=30.0, A = 3 * 10.0, reduction = "mean"):
    '''
    true: set of points (B, L, 27, 3)
    pred: set of points (I, B, L, 27, 3)
    mask_crds: (B, L, 27)
    chain_mask: (B, L, L)
    '''
    I, B, L, n_atoms, _ = pred.shape
    assert B == 1

    # true[~mask_crds] = 0.0
    # pred[:, ~mask_crds] = 0.0

    true = true.detach()
    # pred = pred[:,mask_BB].unsqueeze(1)
    true_backbone = true[:,:,:3,:] # (B, L, 3, 3)
    pred_backbone = pred[:,:,:,:3,:] # (I, B, L, 3, 3)

    # alignment for the first chain
    # true_backbone_chain = true_backbone[:,:chain_mask[0][0].sum(),:,:] # (B, L', 3, 3)
    # pred_backbone_chain = pred_backbone[:,:,:,chain_mask[0][0].sum(),:,:] # (I, B, L', 3, 3)

    Local_Alignment_Rotation_matrix = get_Local_Alignment_Rotation_matrix(true_backbone, pred_backbone,mask_2d, same_chain, d_local=d_local) # (I, B, L, 3, 3)
    # Local_Alignment_Rotation_matrix = get_Local_Alignment_Rotation_matrix(true_backbone_chain, pred_backbone_chain, d_clamp=d_clamp) # (I, B, L, 3, 3)
    t_true = true_backbone[:,None,:,:,:] - true_backbone[:,:,None,:,:] # (B, L, L, 27, 3)

    t_pred = pred_backbone[:,:,None,:,:,:] - pred_backbone[:,:,:,None,:,:] # (I, B, L, L, 27, 3)
    t_pred = torch.einsum('ablmij, abljk -> ablmik', t_pred, Local_Alignment_Rotation_matrix) # (I, B, L, L, 3, 3) #matrix for pred to true
    # t_map_list = [t_true[0], t_pred[0,0]]
    
    # Centering is not needed because we use displacement.
    l2 = torch.square(t_true - t_pred).sum(dim=-1) # (I, B, L, L, 3)

    if d_clamp is not None :
        clamp = torch.where(same_chain.bool(), d_clamp ** 2, d_clamp_inter ** 2) # (B, L, L)
        # clamp = clamp[None] # (1, B, L, L)

        clamp = clamp.unsqueeze(0).unsqueeze(-1).float() # (1, B, L, L, 1)
        l2 = torch.clamp(l2, max= clamp)
    l2 = torch.sum(l2,dim=(-1)) # (I, B, L, L)
    l2 = l2 * mask_2d.unsqueeze(0).float()
    Displacement_Loss = torch.sqrt(l2 +1e-8)# (I, B, L, L)
    Displacement_Loss = Displacement_Loss / A
    # visualize_2D_heatmap(Displacement_Loss[0,0], file_name = "Displacement_Loss_"+file_name, heatmap_dir = "LADE_visualize/")
    Displacement_Loss = Displacement_Loss.sum(dim=(-1,-2)) / float(L*L) # (I, B)
    Displacement_Loss = Displacement_Loss.mean(dim=0) # (B)
    if reduction == "none":
        return Displacement_Loss # (B)
    elif reduction == "mean":
        return Displacement_Loss.mean() # (1)
    # Actually, in my model batch size is always 1, so I don't need to consider batch size.


def calc_lj(
    seq,
    xs,
    aamask,
    same_chain,
    ljparams,
    ljcorr,
    num_bonds,
    use_H=False,
    negative=False,
    lj_lin=0.75,
    lj_hb_dis=3.0,
    lj_OHdon_dis=2.6,
    lj_hbond_hdis=1.75,
    lj_maxrad=-1.0,
    eps=1e-8,
    normalize=True,
    reswise=False,
    atom_mask=None,
):
    def ljV(dist, sigma, epsilon, lj_lin, lj_maxrad):
        linpart = dist < lj_lin * sigma
        deff = dist.clone()
        deff[linpart] = lj_lin * sigma[linpart]
        sd = sigma / deff
        sd2 = sd * sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += (
            epsilon[linpart]
            * (-12 * sd12[linpart] / deff[linpart] + 12 * sd6[linpart] / deff[linpart])
            * (dist[linpart] - deff[linpart])
        )
        if lj_maxrad > 0:
            sdmax = sigma / lj_maxrad
            sd2 = sd * sd
            sd6 = sd2 * sd2 * sd2
            sd12 = sd6 * sd6
            ljE = ljE - epsilon * (sd12 - 2 * sd6)
        return ljE

    L = xs.shape[0]

    # mask keeps running total of what to compute
    if atom_mask is not None:
        mask = atom_mask[..., None, None] * atom_mask[None, None, ...]
    else:
        aamask = aamask[seq]
        if not use_H:
            aamask[..., 14:] = False
        mask = aamask[..., None, None] * aamask[None, None, ...]

    # ignore CYS-CYS (disulfide bonds)
    is_CYS = seq == aa2num["CYS"]  # (L)
    is_CYS_pair = is_CYS[:, None] * is_CYS[None, :]
    is_CYS_pair = is_CYS_pair.view(L, 1, L, 1)
    mask *= ~is_CYS_pair

    if negative:
        # ignore inter-chains
        mask *= same_chain.bool()[:, None, :, None]

    idxes1r = torch.tril_indices(L, L, -1)
    mask[idxes1r[0], :, idxes1r[1], :] = False
    idxes2r = torch.arange(L)
    idxes2a = torch.tril_indices(27, 27, 0)
    mask[idxes2r[:, None], idxes2a[0:1], idxes2r[:, None], idxes2a[1:2]] = False

    # "countpair" can be enforced by making this a weight
    mask[idxes2r, :, idxes2r, :] *= num_bonds[seq, :, :] > 3  # intra-res
    mask[idxes2r[:-1], :, idxes2r[1:], :] *= (
        num_bonds[seq[:-1], :, 2:3] + num_bonds[seq[1:], 0:1, :] + 1 > 3  # inter-res
    )
    si, ai, sj, aj = mask.nonzero(as_tuple=True)
    ds = torch.sqrt(torch.sum(torch.square(xs[si, ai] - xs[sj, aj]), dim=-1) + eps)

    # hbond correction
    use_hb_dis = (
        ljcorr[seq[si], ai, 0] * ljcorr[seq[sj], aj, 1]
        + ljcorr[seq[si], ai, 1] * ljcorr[seq[sj], aj, 0]
    )
    use_ohdon_dis = (  # OH are both donors & acceptors
        ljcorr[seq[si], ai, 0] * ljcorr[seq[si], ai, 1] * ljcorr[seq[sj], aj, 0]
        + ljcorr[seq[si], ai, 0] * ljcorr[seq[sj], aj, 0] * ljcorr[seq[sj], aj, 1]
    )

    ljrs = ljparams[seq[si], ai, 0] + ljparams[seq[sj], aj, 0]
    ljrs[use_hb_dis] = lj_hb_dis
    ljrs[use_ohdon_dis] = lj_OHdon_dis

    if use_H:
        use_hb_hdis = (
            ljcorr[seq[si], ai, 2] * ljcorr[seq[sj], aj, 1]
            + ljcorr[seq[si], ai, 1] * ljcorr[seq[sj], aj, 2]
        )
        ljrs[use_hb_hdis] = lj_hbond_hdis

    # disulfide correction
    potential_disulf = ljcorr[seq[si], ai, 3] * ljcorr[seq[sj], aj, 3]

    ljss = torch.sqrt(ljparams[seq[si], ai, 1] * ljparams[seq[sj], aj, 1] + eps)
    ljss[potential_disulf] = 0.0


    ljval = ljV(ds, ljrs, ljss, lj_lin, lj_maxrad)

    if reswise:
        ljval_res = torch.zeros_like(mask.float())
        ljval_res[si, ai, sj, aj] = ljval
        # ljval_res[:,:4,:,:4] = 0.0 # ignore clashes btw backbones?
        ljval_res = ljval_res.sum(dim=(1, 3))
        ljval_res = ljval_res + ljval_res.permute(1, 0)
        return ljval_res.sum(dim=-1)

    if normalize:
        return torch.sum(ljval) / torch.sum(aamask[seq])
    else:
        return torch.sum(ljval)


def calc_hb(
    seq,
    xs,
    aamask,
    hbtypes,
    hbbaseatoms,
    hbpolys,
    hb_sp2_range_span=1.6,
    hb_sp2_BAH180_rise=0.75,
    hb_sp2_outer_width=0.357,
    hb_sp3_softmax_fade=2.5,
    threshold_distance=6.0,
    eps=1e-8,
    normalize=True,
):
    def evalpoly(ds, xrange, yrange, coeffs):
        v = coeffs[..., 0]
        for i in range(1, 10):
            v = v * ds + coeffs[..., i]
        minmask = ds < xrange[..., 0]
        v[minmask] = yrange[minmask][..., 0]
        maxmask = ds > xrange[..., 1]
        v[maxmask] = yrange[maxmask][..., 1]
        return v

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh, ah = (hbts[..., 0] >= 0).nonzero(as_tuple=True)
    ra, aa = (hbts[..., 1] >= 0).nonzero(as_tuple=True)
    D_xs = xs[rh, hbba[rh, ah, 0]][:, None, :]
    H_xs = xs[rh, ah][:, None, :]
    A_xs = xs[ra, aa][None, :, :]
    B_xs = xs[ra, hbba[ra, aa, 0]][None, :, :]
    B0_xs = xs[ra, hbba[ra, aa, 1]][None, :, :]
    hyb = hbts[ra, aa, 2]
    polys = hbpolys[hbts[rh, ah, 0][:, None], hbts[ra, aa, 1][None, :]]

    AH = torch.sqrt(torch.sum(torch.square(H_xs - A_xs), axis=-1) + eps)
    AHD = torch.acos(cosangle(B_xs, A_xs, H_xs))

    Es = polys[..., 0, 0] * evalpoly(
        AH, polys[..., 0, 1:3], polys[..., 0, 3:5], polys[..., 0, 5:]
    )
    Es += polys[..., 1, 0] * evalpoly(
        AHD, polys[..., 1, 1:3], polys[..., 1, 3:5], polys[..., 1, 5:]
    )

    Bm = 0.5 * (B0_xs[:, hyb == HbHybType.RING] + B_xs[:, hyb == HbHybType.RING])
    cosBAH = cosangle(Bm, A_xs[:, hyb == HbHybType.RING], H_xs)
    Es[:, hyb == HbHybType.RING] += polys[:, hyb == HbHybType.RING, 2, 0] * evalpoly(
        cosBAH,
        polys[:, hyb == HbHybType.RING, 2, 1:3],
        polys[:, hyb == HbHybType.RING, 2, 3:5],
        polys[:, hyb == HbHybType.RING, 2, 5:],
    )

    cosBAH1 = cosangle(
        B_xs[:, hyb == HbHybType.SP3], A_xs[:, hyb == HbHybType.SP3], H_xs
    )
    cosBAH2 = cosangle(
        B0_xs[:, hyb == HbHybType.SP3], A_xs[:, hyb == HbHybType.SP3], H_xs
    )
    Esp3_1 = polys[:, hyb == HbHybType.SP3, 2, 0] * evalpoly(
        cosBAH1,
        polys[:, hyb == HbHybType.SP3, 2, 1:3],
        polys[:, hyb == HbHybType.SP3, 2, 3:5],
        polys[:, hyb == HbHybType.SP3, 2, 5:],
    )
    Esp3_2 = polys[:, hyb == HbHybType.SP3, 2, 0] * evalpoly(
        cosBAH2,
        polys[:, hyb == HbHybType.SP3, 2, 1:3],
        polys[:, hyb == HbHybType.SP3, 2, 3:5],
        polys[:, hyb == HbHybType.SP3, 2, 5:],
    )
    Es[:, hyb == HbHybType.SP3] += (
        torch.log(
            torch.exp(Esp3_1 * hb_sp3_softmax_fade)
            + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
        )
        / hb_sp3_softmax_fade
    )

    cosBAH = cosangle(
        B_xs[:, hyb == HbHybType.SP2], A_xs[:, hyb == HbHybType.SP2], H_xs
    )
    Es[:, hyb == HbHybType.SP2] += polys[:, hyb == HbHybType.SP2, 2, 0] * evalpoly(
        cosBAH,
        polys[:, hyb == HbHybType.SP2, 2, 1:3],
        polys[:, hyb == HbHybType.SP2, 2, 3:5],
        polys[:, hyb == HbHybType.SP2, 2, 5:],
    )

    BAH = torch.acos(cosBAH)
    B0BAH = get_dih(
        B0_xs[:, hyb == HbHybType.SP2],
        B_xs[:, hyb == HbHybType.SP2],
        A_xs[:, hyb == HbHybType.SP2],
        H_xs,
    )

    d, m, l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like(B0BAH, m - 0.5)

    mask1 = BAH > np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH > np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:, hyb == HbHybType.SP2] += polys[:, hyb == HbHybType.SP2, 2, 0] * Echi

    tosquish = torch.logical_and(Es > -0.1, Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.0

    if normalize:
        return torch.sum(Es) / torch.sum(aamask[seq])
    else:
        return torch.sum(Es)



def calc_lddt(
    pred_ca, true_ca, mask_crds, mask_2d, same_chain, negative=False, eps=1e-6
):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]

    pred_dist = torch.cdist(pred_ca, pred_ca)  # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0)  # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0)  # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    delta = torch.abs(pred_dist - true_dist)  # (I, B, L, L)

    true_lddt = torch.zeros((I, B, L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += (
            0.25
            * torch.sum((delta <= distbin) * mask, dim=-1)
            / (torch.sum(mask, dim=-1) + eps)
        )

    true_lddt = mask_crds[None] * true_lddt
    true_lddt = true_lddt.sum(dim=(1, 2)) / (mask_crds.sum() + eps)
    return true_lddt


# fd allatom lddt


def calc_allatom_lddt_w_loss(
    P, Q, atm_mask, pred_lddt, idx, same_chain, negative=False, eps=1e-8
):
    # Inputs
    #  - P: predicted coordinates (L, 14, 3)
    #  - Q: ground truth coordinates (L, 14, 3)
    #  - atm_mask: valid atoms (L, 14)
    #  - idx: residue index (L)

    # distance matrix
    Pij = torch.square(
        P[:, None, :, None, :] - P[None, :, None, :, :]
    )  # (L, L, 14, 14, 3)
    Pij = torch.sqrt(Pij.sum(dim=-1) + eps)  # (L, L, 14, 14)
    Qij = torch.square(
        Q[:, None, :, None, :] - Q[None, :, None, :, :]
    )  # (L, L, 14, 14)
    Qij = torch.sqrt(Qij.sum(dim=-1) + eps)  # (L, L, 14, 14)

    # get valid pairs
    # only consider atom pairs within 15A
    pair_mask = torch.logical_and(Qij > 0, Qij < 15).float()
    # ignore missing atoms
    pair_mask *= (atm_mask[:, None, :, None] * atm_mask[None, :, None, :]).float()
    # ignore atoms within same residue
    pair_mask *= (
        idx[:, None, None, None] != idx[None, :, None, None]
    ).float()  # (L, L, 14, 14)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:, :, None, None]

    delta_PQ = torch.abs(Pij - Qij)  # (L, L, 14, 14)

    true_lddt = torch.zeros(P.shape[:2], device=P.device)  # (L, 14)
    for distbin in (0.5, 1.0, 2.0, 4.0):
        true_lddt += (
            0.25
            * torch.sum((delta_PQ <= distbin) * pair_mask, dim=(1, 3))
            / (torch.sum(pair_mask, dim=(1, 3)) + 1e-8)
        )
    true_lddt = true_lddt.sum(dim=-1) / (atm_mask.sum(dim=-1) + 1e-8)  # L


    res_mask = atm_mask.any(dim=-1)  # L
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(
        bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device
    )
    true_lddt_label = torch.bucketize(true_lddt[None], lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction="none")(pred_lddt, true_lddt_label)
    lddt_loss = (lddt_loss * res_mask[None]).sum() / (res_mask.sum() + eps)

    true_lddt = (res_mask * true_lddt).sum() / (res_mask.sum() + 1e-8)

    return lddt_loss, true_lddt


def calc_allatom_lddt(P, Q, atm_mask, idx, same_chain, negative=False, eps=1e-8):
    # Inputs
    #  - P: predicted coordinates (L, 14, 3)
    #  - Q: ground truth coordinates (L, 14, 3)
    #  - atm_mask: valid atoms (L, 14)
    #  - idx: residue index (L)

    # distance matrix
    Pij = torch.square(
        P[:, None, :, None, :] - P[None, :, None, :, :]
    )  # (L, L, 14, 14)
    Pij = torch.sqrt(Pij.sum(dim=-1) + eps)
    Qij = torch.square(
        Q[:, None, :, None, :] - Q[None, :, None, :, :]
    )  # (L, L, 14, 14)
    Qij = torch.sqrt(Qij.sum(dim=-1) + eps)

    # get valid pairs
    # only consider atom pairs within 15A
    pair_mask = torch.logical_and(Qij > 0, Qij < 15).float()
    # ignore missing atoms
    pair_mask *= (atm_mask[:, None, :, None] * atm_mask[None, :, None, :]).float()
    # ignore atoms within same residue
    pair_mask *= (
        idx[:, None, None, None] != idx[None, :, None, None]
    ).float()  # (L, L, 14, 14)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:, :, None, None]

    delta_PQ = torch.abs(Pij - Qij)  # (L, L, 14, 14)

    true_lddt = torch.zeros(P.shape[:2], device=P.device)  # (L, 14)
    for distbin in (0.5, 1.0, 2.0, 4.0):
        true_lddt += (
            0.25
            * torch.sum((delta_PQ <= distbin) * pair_mask, dim=(1, 3))
            / (torch.sum(pair_mask, dim=(1, 3)) + 1e-8)
        )
    true_lddt = true_lddt.sum(dim=-1) / (atm_mask.sum(dim=-1) + 1e-8)  # L

    res_mask = atm_mask.any(dim=-1)  # L

    true_lddt = (res_mask * true_lddt).sum() / (res_mask.sum() + 1e-8)

    return true_lddt

# get ligand rmsd
def getLigandRMSD(pred,true,mask, interface_split, reduction = 'none'):
    """
    Calculate the RMSD between two sets of vectors

    Parameters
    ----------
    pred : tensor (I, B, L, N, 3) : first set of vectors
    true : tensor (B, L, N, 3) : second set of vectors
    mask : tensor (B,L,N) : mask for true
    translate : bool : whether to translate X and Y to the origin
    reduction : str : whether to return the mean or list of the RMSD

    Returns
    -------
    RMSD : float : RMSD between P and Q
    """
    class QuaternionBase: #from Galaxy.core.quaternion
        def __init__(self):
            self.q = []
        def __repr__(self):
            return self.q
        def rotate(self):
            if 'R' in dir(self):
                return self.R
            #
            self.R = torch.zeros((3,3))
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

    def input_to_ca(pred, true):
        """
        Get only the CA atoms from the input

        Parameters
        ----------
        pred : (I, B, L, N, 3)
        true : (B, L, N, 3)

        Returns
        -------
        X : (I, L, 3)
        Y : (L, 3)
        """
        # get only ca atoms
        I, B, L, N, _ = pred.shape
        X = pred[:,:,:,:3,:].reshape(I, L, 3, 3) #(I, L, 3,3)
        Y = true[:,:,:3,:].squeeze() #(L,3, 3)
        X = X.to(pred.device)
        Y = Y.to(pred.device)
        return X, Y
    def getRMSDwithR(model, ref, R,t):
        """
        get RMSD from the model and ref with R matrix and translation vector t

        Parameters
        ----------
        model : tensor (I, L, 3) : first set of vectors
        ref : tensor (L, 3) : second set of vectors
        R : tensor (I, 3, 3) : rotation matrix
        t : tensor (I, 3) : translation vector
        """
        I,L,N = model.shape
        ref = ref.transpose(0,1).unsqueeze(0).expand(I,3,L) # (I, 3, L)
        model = model.transpose(1,2) # (I, 3, L)
        t = t.unsqueeze(-1).expand(I,3,L)
        aligned_model = torch.einsum("iab, ibc -> iac", R, model)
        aligned_model = aligned_model + t
        rmsd = torch.sqrt(((aligned_model - ref)**2).sum(dim=(-2,-1))/L)
        return rmsd, aligned_model
    
    def kabsch_rotate(X, Y):
        """
        Rotate matrix X unto Y using Kabsch algorithm

        Parameters
        ----------
        X : tensor (I, L, 3) : matrix to be rotated
        Y : tensor (L, 3) : matrix unto which X will be rotated

        Returns
        -------
        R : tensor (3, 3) : rotation matrix
        """
        I, L, _ = X.shape
        X_cntr = X.transpose(-1,-2).sum(dim=-1)/L # (I, 3)
        Y_cntr = Y.sum(dim=0)/L # (3)
        X -= X_cntr.unsqueeze(1) # (I, L, 3)
        Y -= Y_cntr.unsqueeze(0).expand(L, 3) # (L, 3)
        Xtr = X.transpose(1,2) # (I, 3, L)
        Ytr = Y.transpose(0,1) # (3, L)
        X_norm = torch.square(Xtr).sum(dim=1).sum(dim=1) # (I)  
        Y_norm = torch.square((Ytr.unsqueeze(0).expand(I,3,-1))).sum(dim=1).sum(dim=1) # (I)
        # Y_expand = Y.unsqueeze(0).expand(I,L,3) # (I, L, 3)
        Xtr = Xtr.to(torch.float64) ; Y = Y.to(torch.float64)
        # print('Xtr nan chec :',torch.isnan(Xtr).any())
        # print('Y nan check :',torch.isnan(Y).any())
        # print('Xtr inf check : ',torch.isinf(Xtr).any())
        # print('Y inf check : ',torch.isinf(Y).any())

        # print('Xtr \t',Xtr.shape)
        # for i in range(I):
        #     print('I : ',i)
        #     for j in range(L):
        #         print(Xtr[i,:,j])
        # print('Y \t',Y.shape)
        # for i in range(L):
        #     print(Y[i,:])
        # print('Y_expand\n',Y_expand)
        # Xtr = Xtr.float() ; Y_expand = Y.float()
        # Rmat = torch.matmul(Xtr, Y_expand) # (I, 3, 3)
        Rmat = torch.einsum("iab, bc -> iac", Xtr, Y) # (I, 3, 3)

        Rmat_flat = Rmat.reshape(I,9) 
        # print('Rmat\n',Rmat)
        mat = torch.full((9, 16), 0.0).float()
        mat[:,0] = torch.tensor([1,0,0,0,1,0,0,0,1])
        mat[:,1] = torch.tensor([0,0,0,0,0,1,0,-1,0])
        mat[:,2] = torch.tensor([0,0,-1,0,0,0,1,0,0])
        mat[:,3] = torch.tensor([0,1,0,-1,0,0,0,0,0])
        mat[:,4] = mat[:,1]
        mat[:,5] = torch.tensor([1,0,0,0,-1,0,0,0,-1])
        mat[:,6] = torch.tensor([0,1,0,1,0,0,0,0,0])
        mat[:,7] = torch.tensor([0,0,1,0,0,0,1,0,0])
        mat[:,8] = mat[:,2]
        mat[:,9] = mat[:,6]
        mat[:,10] = torch.tensor([-1,0,0,0,1,0,0,0,-1])
        mat[:,11] = torch.tensor([0,0,0,0,0,1,0,1,0])
        mat[:,12] = mat[:,3]
        mat[:,13] = mat[:,7]
        mat[:,14] = mat[:,11]
        mat[:,15] = torch.tensor([-1,0,0,0,-1,0,0,0,1])
        mat = mat.reshape(9,16)
        Rmat_flat = Rmat_flat.to(X.device).double() ; mat = mat.to(X.device).double()
        # print('Rmat_flat\n',Rmat_flat)
        # print('mat\n',mat)
        # S = torch.einsum("ik, kj -> ij", Rmat_flat, mat) # (I, 16)
        S = torch.matmul(Rmat_flat, mat) # (I, 16)
        S = S.reshape(I, 4, 4)
        # S = S.half()
        # print('S\n',S)
        eigl, eigv = torch.linalg.eig(S) # (I, 4), (I, 4, 4)
        q = eigv.transpose(-1,-2) # (I, 4, 4)
        q = q[:,0,:] # (I, 4)
        R_list = []
        for i in range(I):
            R_list.append(QuaternionQ(q[i].squeeze()).rotate()) # (I, 3, 3)
        R = torch.stack(R_list) # (I, 3, 3)
        eigl = eigl.float()
        R = R.to(X.device) ; X_cntr = X_cntr.to(X.device) ; Y_cntr = Y_cntr.to(X.device)
        t = Y_cntr.unsqueeze(0).expand(I,3) - torch.einsum("iab, ib -> ia", R, X_cntr) # (I, 3)
        rmsd_list = []
        rmsd = torch.sqrt((torch.max(torch.tensor(0.0),(X_norm + Y_norm - 2.0* eigl[:,0]))/L)) # (I)
        # print('rmsd\n', rmsd)
        # for i in range(I):
        #     diff = torch.max(torch.tensor(0.0),(X_norm + Y_norm - 2.0* eigl[i,0])) # (I)
        #     rmsd_element = torch.sqrt(diff/L) # (I)
        #     print('rmsd_element\n',rmsd_element)
        #     rmsd_list.append(rmsd_element)
        # rmsd = torch.stack(rmsd_list) # (I)
        return rmsd, R, t
    if len(interface_split)<2:
        # pred = pred.clone().detach()
        zero = torch.tensor(0.0)[None]
        print('output(zero) check', zero.shape)
        return zero # for reduction is 'mean'
    X, Y = input_to_ca(pred, true) # (I, L, 3, 3), (L, 3, 3)
    I, L, N, _ = X.shape
    # print('X.shape', X.shape)
    # print('Y.shape', Y.shape)
    MASK = mask[:,:,:3].squeeze() # (L, 3)
    MASK = MASK.unsqueeze(-1).expand(I,L,3,3) # (I, L, 3)
    MASK = MASK.to(pred.device)
    X = X * MASK # (I, L, 3, 3)
    # Y = Y.unsqueeze(0).expand(I,L,3,3) # (I, L, 3, 3)
    Y = Y * MASK # (I, L, 3, 3)
    Y = Y[0,:,:,:].squeeze() # (L, 3, 3)
    X = X.reshape(I,L*3,3) # (I, L*3, 3)
    Y = Y.reshape(L*3,3) # (L*3, 3)
    
    X_rec = X[:,:interface_split[0]*3,:] # (I, L_ab, 3)
    X_lig = X[:,interface_split[0]*3:,:] # (I, L_ag, 3)
    Y_rec = Y[:interface_split[0]*3,:] # (L_ab, 3)
    Y_lig = Y[interface_split[0]*3:,:] # (L_ag, 3)
    
    zero_rows_X = torch.all(X == 0, dim=-1) # (I, L, 3)
    zero_rows_X_rec = torch.all(X_rec == 0, dim=-1) # (I, L_ab)
    zero_rows_X_lig = torch.all(X_lig == 0, dim=-1) # (I, L_ag)
    zero_rows_Y = torch.all(Y == 0, dim=-1)
    zero_rows_Y_rec = torch.all(Y_rec == 0, dim=-1)
    zero_rows_Y_lig = torch.all(Y_lig == 0, dim=-1)
    non_zero_X_list = [] ;non_zero_X_rec_list = [] ; non_zero_X_lig_list = []
    for i in range(X.shape[0]):
        non_zero_X = X[i][~zero_rows_X[i]]
        non_zero_X_rec = X_rec[i][~zero_rows_X_rec[i]]
        non_zero_X_lig = X_lig[i][~zero_rows_X_lig[i]]

        non_zero_X_list.append(non_zero_X)
        non_zero_X_rec_list.append(non_zero_X_rec)
        non_zero_X_lig_list.append(non_zero_X_lig)

    non_zero_X = torch.stack(non_zero_X_list)
    non_zero_X_rec = torch.stack(non_zero_X_rec_list)
    non_zero_X_lig = torch.stack(non_zero_X_lig_list)

    non_zero_Y = Y[~zero_rows_Y]
    non_zero_Y_rec = Y_rec[~zero_rows_Y_rec]    
    non_zero_Y_lig = Y_lig[~zero_rows_Y_lig]

    X_rec_input = non_zero_X_rec.clone()
    Y_rec_input = non_zero_Y_rec.clone()
    # print('non_zero_X_rec.shape', non_zero_X_rec.shape)
    # for i in range(non_zero_X_rec.shape[1]):
    #     print(non_zero_X_rec[0,i])
    # print('non_zero_Y_rec.shape', non_zero_Y_rec.shape)
    # for i in range(non_zero_Y_rec.shape[0]):
    #     print(non_zero_Y_rec[i])
    receptor_rmsd, R_rec, t = kabsch_rotate(X_rec_input, Y_rec_input) # (I, 3, 3)
    # print('R_rec\n', R_rec)
    print('receptor rmsd', receptor_rmsd)
    ligand_rmsd, alignedXlig = getRMSDwithR(non_zero_X_lig, non_zero_Y_lig, R_rec, t) # (I)
    receptor_rmsd, alignedXrec = getRMSDwithR(non_zero_X_rec, non_zero_Y_rec, R_rec, t) # (I)
    total_rmsd, alignedX = getRMSDwithR(non_zero_X, non_zero_Y, R_rec, t) # (I)


    print('total_rmsd', total_rmsd)
    print('l_rmsd', ligand_rmsd)
    print('receptor rmsd', receptor_rmsd)
    if reduction == 'mean':
        ligand_rmsd = torch.mean(ligand_rmsd)[None]
        return ligand_rmsd
    elif reduction == 'none':
        return ligand_rmsd


@torch.cuda.amp.autocast(enabled=False)
def getLigandRMSD_svd(pred, true, mask, interface_split, reduction = 'mean', write_pdb_flag=False, L_s=None, item=None):
    """
    calculate the rotation matrix to rotated the pred to true

    Parameters
    ----------
    pred : tensor (I, B, L, N, 3) : first set of vectors
    true : tensor (B, L, N, 3) : second set of vectors
    mask : tensor (B,L,N) : mask for true
    interface_split : list : list of the number of residues in the interface

    Returns
    -------
    R : tensor (I, 3, 3) : rotation matrix
    """
    I, B, L, N, _ = pred.shape
    X = pred[:,:,:,:3,:] # (I,B, L, 3, 3)
    # X = X.unsqueeze(0) # for I in gpu
    # print(X.shape)
    Y = true[:,:,:3,:] # (B, L, 3, 3)
    # print(Y.shape)

    # print(interface_split)
    # print(sum(interface_split))

    MASK = mask[:,:,:3] # (B, L, 3)
    X = X * (MASK.unsqueeze(-1).expand(I,B,L,3,3)) # (I,B, L, 3, 3)
    Y = Y * MASK.unsqueeze(-1).expand(B,L,3,3) # (B,L, 3, 3)

    # print('before centering')
    # for l in range(L):
    #     print(X[0,l])
    X_rec = X[:,:,:interface_split[0],:] # (I,B, L_ab, 3, 3)
    X_lig = X[:,:,interface_split[0]:,:] # (I, B, L_ag, 3, 3)
    Y_rec = Y[:,:interface_split[0],:] # (B, L_ab, 3, 3)
    Y_lig = Y[:,interface_split[0]:,:] # (B, L_ag, 3,3)

    print('X_rec',X_rec.shape)

    # get center from the Ca atoms
    X_rec_center = X_rec[:,:,:,1].sum(dim=2)/interface_split[0]  # (I,B, 3)
    print(X_rec_center.shape)
    # print(X_rec_center)
    Y_rec_center = Y_rec[:,:,1].sum(dim=1)/interface_split[0] # (B,3)
    print(Y_rec_center.shape)


    X_trans = X - X_rec_center # (I,B, L, 3, 3)
    Y_trans = Y - Y_rec_center # (B,L, 3, 3)

    # print(mask.shape)
    MASK = mask[:,:,:3].squeeze() # (L, 3)
    print('MASK',MASK.shape)
    print('Y_trans',Y_trans.shape)
    X_trans = X_trans * MASK.unsqueeze(-1).expand(I,L,3,3) # (I, L, 3, 3)
    Y_trans = Y_trans * MASK.unsqueeze(-1).expand(L,3,3) # (L, 3, 3)

    # get rotational matrix
    X_bb = X_trans.reshape(I,L*3,3) # (I, L*3, 3)
    Y_bb = Y_trans.reshape(L*3,3) # (L*3, 3)

    H = torch.einsum("ila, lb -> iab", X_bb, Y_bb) # (I, 3, 3)
    H = H.float()
    U, S, V = torch.svd(H) # (I, 3, 3), (I, 3), (I, 3, 3)
    d = torch.sign(torch.det(torch.einsum("iab, ibc -> iac", V, U.transpose(-1,-2)))).reshape(I,1) # (I,1)
    S = torch.eye(3).unsqueeze(0).expand(I,3,3) # (I, 3, 3)
    S = S.to(X.device)
    S[:,2,2] = d
    R = V @ S @ U.transpose(-1,-2) # (I, 3, 3)
    R = R.transpose(-1,-2) # (I, 3, 3)
    # print('R\n',R)
    R = torch.tensor([[[-0.5194,  0.7072, -0.4797],[ 0.3399,  0.6860,  0.6434],[ 0.7841,  0.1711, -0.5966]]])
    # align the pred to true
    t = pred[:,:,:interface_split[0],1].sum(dim=2)/interface_split[0] # (I, 3)
    # aligned_pred, t = get_rotated_xyz(pred[0], R[0], t=t)
    aligned_pred = pred - t # (I, B, L, N, 3)
    aligned_pred = torch.einsum('iblna, iac -> iblnc',aligned_pred,R)
    true_t = true[:,:interface_split[0],1].sum(dim=1)/interface_split[0] # (B, 3)
    aligned_pred = aligned_pred + true[:,:interface_split[0],1].sum(dim=1)/interface_split[0]
    true = true[:,:,:14] ; mask = mask[:,:,:14]

    print(aligned_pred.shape)
    print(true.shape)
    aligned_pred = aligned_pred * mask.unsqueeze(-1)

    diff = true - aligned_pred
    print(diff.shape)

    tot_rmsd = torch.sqrt((diff**2).sum(dim=-1).mean())
    # diff_rec = true[:,:interface_split[0]] - aligned_pred[:,:interface_split[0]]
    diff_rec = diff[:,:,:interface_split[0]]
    rec_rmsd = torch.sqrt((diff_rec**2).sum(dim=-1).mean())

    # diff_lig = true[:,interface_split[0]:] - aligned_pred[:,interface_split[0]:]
    diff_lig = diff[:,:,interface_split[0]:]
    lig_rmsd = torch.sqrt((diff_lig**2).sum(dim=-1).mean())

    print('tot_rmsd',tot_rmsd)
    print('rec_rmsd',rec_rmsd)
    print('lig_rmsd',lig_rmsd)

    if write_pdb_flag:
        assert L_s is not None, "L_s must be provided"
        assert item is not None, "item must be provided"

        print('writing pdb')
        print('true[0].shape',true[0].shape)
        print('aligned_pred[0].shape',aligned_pred[0].shape)
        seq = torch.full((sum(interface_split),),0)
        write_pdb(seq, true[0],L_s=L_s,prefix=f'{item}_true',Bfacts=None)
        write_pdb(seq, aligned_pred[0][0],L_s=L_s,prefix=f'{item}_aligned_2',Bfacts=None)

    if reduction == 'mean':
        lig_rmsd = torch.mean(lig_rmsd)[None]
        return lig_rmsd
    elif reduction == 'none':
        return lig_rmsd

@torch.cuda.amp.autocast(enabled=False)    
def get_rotated_xyz(xyz, R, t= None):
    """ get rotated xyz

    Parameters
    ----------
    xyz : torch.tensor(I,B,L,N,3)
    R : torch.tensor(I,3,3)
    t : torch.tensor(I,B,3)
    """

    # get random translation t
    if t is None:
        t = torch.rand(3)
    # xyz = xyz.reshape(1, -1, 3)
    # xyz = torch.matmul(xyz, R)
    xyz = xyz - t.unsqueeze(-2).unsqueeze(-2)
    xyz = xyz.double() ; R= R.double() ; t = t.double()
    xyz = torch.einsum('iblna, iac -> iblnc',xyz,R)
    # print(xyz.shape)
    # xyz = xyz + t
    # xyz = xyz.reshape(1, -1, 14, 3)
    # print(xyz.shape)
    return xyz, t

@torch.cuda.amp.autocast(enabled=False)    
def get_ligandRMSD(rotated_xyz, true_xyz, mask, interface_split, reduction='mean',item=None, L_s=None):
    """
    calculate the rotation matrix to rotated the pred to true

    Parameters
    ----------
    pred : tensor (I, B, L, N, 3) : first set of vectors
    true : tensor (B, L, N, 3) : second set of vectors
    mask : tensor (B,L,N) : mask for true
    interface_split : list : list of the number of residues in the interface

    Returns
    -------
    R : tensor (I, 3, 3) : rotation matrix
    """
    print('interface_split',interface_split)
    if (len(interface_split) <2) or (torch.tensor([0]) in interface_split):
        print('return 0')
        return torch.tensor(0.0)
    
    I,B,L,N,_ = rotated_xyz.shape
    X = rotated_xyz[:,:,:,:3,:].clone()
    Y = true_xyz[:,:,:3,:].clone()
    if mask is not None:
        MASK = mask[:,:,:3]
        X = X * MASK.unsqueeze(0).unsqueeze(-1)
        Y = Y * MASK.unsqueeze(-1)

    X_rec = X[:,:,:interface_split[0]]
    X_lig = X[:,:,interface_split[0]:]
    Y_rec = Y[:,:interface_split[0]]
    Y_lig = Y[:,interface_split[0]:]

    X_rec_center = X_rec[:,:,:,1].sum(dim=2)/interface_split[0].to(X.device)
    Y_rec_center = Y_rec[:,:,1].sum(dim=1)/interface_split[0].to(X.device)

    # print('X_rec',X_rec.shape)
    # print('Y_rec',Y_rec.shape)
    # print('X_rec_center',X_rec_center.unsqueeze(-2).unsqueeze(-1).shape)
    # print('Y_rec_center',Y_rec_center.shape)

    X_trans = X_rec - X_rec_center.unsqueeze(-2).unsqueeze(-1)
    Y_trans = Y_rec - Y_rec_center

    X_bb = X_trans.reshape(I,-1,3).float()
    Y_bb = Y_trans.reshape(-1,3).float()


    H = torch.einsum('ila, lb -> iab', X_bb, Y_bb)
    H = H.float()
    U, S, V = torch.svd(H)
    d = torch.sign(torch.det(torch.einsum("iab, ibc -> iac", V, U.transpose(-1,-2)))).reshape(I,1)
    S = torch.eye(3).unsqueeze(0).repeat(I,1,1).to(X.device)
    S[:,2,2] = d.squeeze()
    R = V @ S @ U.transpose(-1,-2)
    R = R.transpose(-1,-2)

    t = rotated_xyz[:,:,:interface_split[0],1].sum(dim=2)/(interface_split[0].to(X.device)+1e-8)
    # print('t',t.shape)
    aligned_xyz, t = get_rotated_xyz(rotated_xyz, R, t)
    aligned_xyz = aligned_xyz + true_xyz[:,:interface_split[0],1].sum(dim=1)/interface_split[0].unsqueeze(0).unsqueeze(-2).unsqueeze(-2).to(X.device)
    # print('aligned_xyz',aligned_xyz.shape)

    tot_diff = true_xyz - aligned_xyz
    # print('tot_diff',tot_diff.shape)
    tot_diff = tot_diff * mask.unsqueeze(-1)
    # print('tot_diff',tot_diff.shape)
    lig_diff = tot_diff[:,:,interface_split[0]:]
    rec_diff = tot_diff[:,:,:interface_split[0]]

    # print(f"lig_diff has nan : {torch.isnan(lig_diff).any()}")
    # print(f"lig_diff.shape : {lig_diff.shape}")
    # print(f"torch.sqrt(lig_diff**2) : {torch.sqrt(lig_diff**2)}")

    tot_rmsd = torch.sqrt((tot_diff**2).sum(dim=(-1,-2,-3))/(tot_diff.shape[2])+1e-8)
    lig_rmsd = torch.sqrt((lig_diff**2).sum(dim=(-1,-2,-3))/(lig_diff.shape[2])+1e-8)
    rec_rmsd = torch.sqrt((rec_diff**2).sum(dim=(-1,-2,-3))/(rec_diff.shape[2])+1e-8)

    print(f'{item} total RMSD   : ', tot_rmsd.mean())
    print(f'{item} ligand RMSD  : ', lig_rmsd.mean())
    print(f'{item} receptor RMSD: ', rec_rmsd.mean())

    if item is not None:
        # write pdb
        # if L_s is None:
        #     L_s = [interface_split[0], interface_split[1]]
        size = (interface_split[0]+interface_split[1],)
        # print('L_s',L_s)
        if L_s.shape[0] == 1:
            L_s_list = [L_s[0,i] for i in range(L_s.shape[1])]
        else:
            L_s_list = [L_s[i] for i in range(L_s.shape[0])]

        seq = torch.full(size,0)
        # print('seq',seq.shape)
        # print('L_s',L_s_list)
        # print('true_xyz',true_xyz.shape)
        # write_pdb(seq, true_xyz[0], L_s=L_s_list,prefix=f"lrmsd_check_pdb_wt2/{item}_true.pdb", Bfacts=None)
        # write_pdb(seq, rotated_xyz[0,0], L_s=L_s_list,prefix=f"lrmsd_check_pdb_wt2/{item}_rotated.pdb", Bfacts=None)
        # write_pdb(seq, aligned_xyz[0,0], L_s=L_s_list,prefix=f"lrmsd_check_pdb_wt2/{item}_aligned.pdb", Bfacts=None)

        # rec_seq = torch.full((interface_split[0],),0)
        # write_pdb(rec_seq, true_xyz[0,:interface_split[0]], L_s=L_s_list[:2],prefix=f"lrmsd_check_pdb/{item}_true_rec.pdb", Bfacts=None)
        # write_pdb(rec_seq, rotated_xyz[0,0,:interface_split[0]], L_s=L_s_list[:2],prefix=f"lrmsd_check_pdb/{item}_rotated_rec.pdb", Bfacts=None)
        # write_pdb(rec_seq, aligned_xyz[0,0,:interface_split[0]], L_s=L_s_list[:2],prefix=f"lrmsd_check_pdb/{item}_aligned_rec.pdb", Bfacts=None)

    if reduction == 'mean':
        
        # return  lig_rmsd.mean()[None]
        return lig_rmsd.mean() # why 
    
    elif reduction == 'None':
        return lig_rmsd # changed