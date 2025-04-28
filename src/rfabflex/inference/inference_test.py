import sys
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import pickle
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from copy import deepcopy
from torch.utils import data
from contextlib import ExitStack
sys.path.append('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/scripts/training')
from train_RF2_rigid import EMA, count_parameters, LOAD_PARAM, Trainer
from data_loader_rigid2 import get_train_valid_set, DatasetComplex_antibody, loader_complex_antibody_kh, MSAFeaturize
from loss_jan30 import calc_c6d_loss, resolve_equiv_natives, get_t, FocalLoss
sys.path.append('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/src/rfabflex')

from common.util import *
from model.util_module import XYZConverter
from model.RoseTTAFoldModel import RoseTTAFoldModule
from common.kinematics import xyz_to_c6d, xyz_to_t2d, c6d_to_bins2
from opt_einsum import contract as einsum
USE_AMP = True

class Predictor:
    def __init__(self, model_name="RF2_apr23",interactive=False, model_param={}, loader_param={}, crop_size=256, maxcycle=4):
        self.model_name = model_name
        self.model_subname = "1.8.0"#"1.1.0"#"blur_lj"#"sc_lj"#"samechain"#"on_rigid" #"on_RF2"
        self.interactive = interactive
        self.model_param = model_param
        # print(f"model_param \n{model_param}")
        self.loader_param = loader_param
        self.valid_param = deepcopy(loader_param)
        self.crop_size = crop_size
        self.maxcycle = maxcycle

        self.l2a = long2alt
        self.aamask = allatom_mask
        
        self.xyz_converter = XYZConverter()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.activation = nn.Softmax(dim=1)

        # self._prepare_input = Trainer._prepare_input
        # self._get_model_input = Trainer._get_model_input

    
    
    def run_predict(self, world_size):
        # if "MASTER_ADDR" in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        # if "MASTER_PORT" in os.environ:
        os.environ["MASTER_PORT"] = "12783"
        if (
            not self.interactive
            and "SLURM_NTASKS" in os.environ
            and "SLURM_PROCID" in os.environ
        ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            print("Launched from SLURM",rank, world_size)
            self.inference(rank, world_size) #like train_model
        else:
            print("Launched from interactive")
            world_size = torch.cuda.device_count()
            mp.spawn(self.inference, args=(world_size,), nprocs=world_size, join=True)
    # def surface_to_epitope(self, surface_info):

    def do_inference(self, model, test_hl_ag_loader, gpu, epoch):
        #dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
        dstep = (20.0 - 2.0) / 36
        list_dist = [] ; list_pae = []
        # print('c6d[0].shape',c6d[0].shape)
        # dbins = np.linspace(2.0 + dstep, 20.5, 37,)
        dbins = torch.linspace(2.0 + dstep, 20.5, 37,)
        

        for k in range(64):
            pae_bin = 0.5 + 0.5 * k
            list_pae.append(pae_bin)
        with torch.no_grad():
            model.eval()
            pae_dict = {}
            disto_dict={}
            for inputs in test_hl_ag_loader:
                (
                    network_input,
                    xyz_prev,
                    mask_recycle,
                    true_crds, #[B, N_homo,L, N, 3]
                    mask_crds, #[B, L, N]
                    msa,
                    mask_msa,
                    unclamp,
                    negative,
                    L_s,
                    item,
                    cluster,
                    interface_split,
                    surface_info,
                ) = self._prepare_input(inputs, gpu)

                # if os.path.exists(f"/home/kkh517/submit_files/Project/inference_fullepi/inference_pdb/inference_pdbs_OnlyRigid_lj/{item[0]}/pae_dict.csv"):
                #     continue
                # epitope_info : surface_residue on antigen protein
                # for each residue in surface_info, define new epitope_info
                # put the index of surface_info where surface_info is 1 into the list
                pae_dict[str(item[0])] = {}
                index_list = torch.nonzero(surface_info[0]).squeeze(1).tolist()
                # print('index_list',index_list)
                for i_epi_res_1 in tqdm(index_list):
                
                    i_epi_res = i_epi_res_1 - int(interface_split[0]) 
                    if true_crds.shape[1] > 1:
                        antigen_crds = true_crds[:, 0,interface_split[0]: ,1,:]
                    else:
                        antigen_crds = true_crds[0][0][ interface_split[0]: ,1,:].unsqueeze(0)
                    antigen_crds = xyz_prev[:, interface_split[0]: ,1,:]
                    cond = torch.cdist(antigen_crds, antigen_crds) < 10 # [B, ag_len, ag_len] #  epitope_radius
                    # cond = torch.cdist(antigen_crds, antigen_crds) < 20 # epitope radius를 20까지 늘리고, epitope info를 나중에 재 정의
                    epitope_info = cond[0,i_epi_res].float().unsqueeze(0)
                    epitope_info = torch.cat((torch.zeros(1,interface_split[0]).to(epitope_info.device), epitope_info), dim=1).long()
                    # overlap = torch.logical_and(surface_info[0] == 1, epitope_info[0] == 1).sum()
                    # total_epi = (surface_info[0] == 1).sum()
                    # overlap_ratio = float(overlap) / float(total_epi) if total_epi > 0 else 0
                    
                # if True:
                #     i_epi_res = 999
                #     epitope_info = surface_info
                #     # breakpoint()
                #     show_proportion = 0.1
                #     true_indices = torch.where(epitope_info)[1]
                #     indices_to_convert = torch.randperm(len(true_indices))[
                #         :int(show_proportion * len(true_indices))
                #     ]
                #     indices_to_convert = true_indices[indices_to_convert]
                #     epitope_info[0][indices_to_convert] = 0
                #     write_pdb(msa[:,0,0][0], xyz_prev[0], L_s=L_s[0], Bfacts=epitope_info[0].bool(),prefix=f"templates_check/{item[0]}_templates_{str(i_epi_res+1)}th_epi") #ProjectName


                    # print('epitope_info',epitope_info)
                    count = 0
                    N_cycle = self.maxcycle
                    output_i = (None, None, None, xyz_prev, None, mask_recycle, epitope_info)
                    return_raw = False


                    for i_cycle in range(N_cycle):
                        with ExitStack() as stack:
                            stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                            stack.enter_context(model.no_sync())
                            # if i_cycle < N_cycle -1:
                            #     return_raw = True
                            # else:
                            #     return_raw = False
                            input_i = self._get_model_input(
                                network_input, output_i, i_cycle, return_raw = return_raw
                            )
                            output_i = model(**input_i)


                            # else:
                            (
                                logit_s, #(4, )
                                logit_aa_s,
                                logit_exp,
                                logit_epitope,
                                logit_pae,
                                p_bind,
                                pred_crds,
                                alphas,
                                pred_lddts,
                                msa_one,
                                pair,
                                state,
                                _,
                            ) = output_i
                            plddt_unbinned = lddt_unbin(pred_lddts)
                            _, final_crds = self.xyz_converter.compute_all_atom(
                                    msa[:, i_cycle, 0], pred_crds[-1], alphas[-1]
                                )
                            if i_cycle < N_cycle - 1:
                                
                                output_i = (msa_one, pair, state, pred_crds[-1], alphas[-1], None, epitope_info)
                                # if i_cycle % 4 == 3:
                                #     write_pdb(
                                #         msa[:, i_cycle, 0][0],
                                #         final_crds[0],
                                #         L_s[0],
                                #         Bfacts=plddt_unbinned[0],
                                #         prefix=f"{str(i_epi_res+1)}th_epi_{item[0]}_{i_cycle}",       
                                #     )
                                probab_pae = torch.nn.Softmax(dim=0)(torch.tensor(logit_pae[0]).float())
                                list_pae_tensor = torch.tensor(list_pae).float()
                                probab_pae = probab_pae.to(epitope_info.device)
                                list_pae_tensor = list_pae_tensor.to(epitope_info.device)
                                expected_pae = torch.einsum('ljk, l -> jk', probab_pae, list_pae_tensor)
                                same_chain = network_input["same_chain"] # TODO
                                same_chain_bool = ~same_chain.bool()
                                interface_pae = expected_pae * same_chain_bool[0].float()
                                interface_pae_mean = interface_pae.sum() / same_chain_bool[0].sum().float()
                                if interface_pae_mean < 10:
                                    cond = torch.cdist(final_crds[0, :interface_split[0], 1], final_crds[0, interface_split[0]:, 1]) < 10
                                    i, j = torch.where(cond)
                                    device = final_crds.device
                                    interface_split_device = interface_split[0].to(device)
                                    epitope_ifaces = j.to(device) + interface_split_device
                                    epitope_info = torch.zeros_like(epitope_info)
                                    epitope_info[0][epitope_ifaces] = 1
                                    # j = sorted(list(set(j)))
                                    print(f"epitope_info updated {j + 1}")

                                continue
                            else:
                                
                                # visusalize logit_s[0] and c6d[0]
                                write_pdb(
                                    msa[:, i_cycle,0][0],
                                    final_crds[0],
                                    L_s[0],
                                    Bfacts=plddt_unbinned[0],
                                    prefix=f"{str(i_epi_res+1)}th_epi_{item[0]}_{i_cycle}",
                                )
                                sys.stdout.write(f"\r{item[0]} inference done")
                                os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}", exist_ok=True)
                                os.system(f"mv {str(i_epi_res+1)}th_epi_{item[0]}_{i_cycle}.pdb inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}") #new_test
                                # os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_DockingSet_ES_reepi/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}", exist_ok=True)
                                # os.system(f"mv {str(i_epi_res+1)}th_epi_{item[0]}_{i_cycle}.pdb inference_pdb/{self.model_name}_{self.model_subname}_DockingSet_ES_reepi/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}") #docking set
                                # os.makedirs(f"inference_pdb/iitp/{item[0]}", exist_ok=True)
                                # os.system(f"mv {i_epi_res}th_epi_{item[0]}_{i_cycle}.pdb inference_pdb/iitp/{item[0]}/{i_epi_res}th_epi_{item[0]}.pdb")# iitp
                                count += 1

                            true_crds_2, mask_crds_2 = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                            # print('mask_crds',mask_crds.shape) # [1,L,27]

                            mask_2d = mask_crds_2[:, :, :3].bool().all(dim=-1)
                            mask_2d = mask_2d[:, :, None] * mask_2d[:, None, :]  # (B, L, L)
                            c6d = xyz_to_c6d(true_crds_2)
                            c6d_pred = xyz_to_c6d(pred_crds[-1])
                            same_chain = network_input["same_chain"] # TODO
                            c6d = c6d_to_bins2(c6d, network_input["same_chain"], negative=False) #, params={"DMIN":,"DMAX":,"DBINS":,"ABINS":,})
                            c6d_pred = c6d_to_bins2(c6d_pred, network_input["same_chain"], negative=False)
                            # c6d_true = torch.zeros_like(c6d, dtype=torch.long)
                            # for i in range(4):

                            #     c6d_true[..., i] = torch.bucketize(c6d[..., i], dbins, right=True)
                            c6d_true = dbins[c6d.long()]
                            c6d_pred = dbins[c6d_pred.long()]
                            pred = pred_crds
                            # true pae
                            
                            true = true_crds_2.unsqueeze(0)  # [1, B, L, n_atom, 3]
                            t_tilde_ij = get_t(true[:, :, :, 0], true[:, :, :, 1], true[:, :, :, 2], non_ideal=True)# (1,B,L,L,3)
                            t_ij = get_t(pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2])  # (I,B,L,L,3)

                            difference = torch.sqrt(torch.square(t_tilde_ij - t_ij).sum(dim=-1) + 1e-6)  # (I,B,L,L)
                            eij_label = difference[-1].clone().detach() # (L, L)                        
                            nbin = 64
                            bin_step = 0.5
                            pae_bins = torch.linspace(
                                bin_step,
                                bin_step * (nbin - 1),
                                nbin - 1,
                                dtype=logit_pae.dtype,
                                device=logit_pae.device,
                            )
                            true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
                            
                            
                            logit_gpu = logit_s[0].cpu().numpy()
                            logit_pae_gpu = logit_pae[0].cpu().numpy()
                            # print('true_pae_label.shape',true_pae_label.shape)
                            logit_pae_true = true_pae_label[0].cpu().numpy()
                            has_nan_here = np.isnan(logit_gpu).any()
                            has_nan_here_pae = np.isnan(logit_pae_gpu).any()
                            # print('has_nan_here',has_nan_here)
                            # print('has_nan_here_pae',has_nan_here_pae)
                            logit_gpu_max = logit_gpu - np.max(logit_gpu, axis=1, keepdims=True)
                            logit_pae_gpu_max = logit_pae_gpu - np.max(logit_pae_gpu, axis=0, keepdims=True)
                            # logit_pae_true
                            # logit_gpu_max = logit_gpu
                            # logit_pae_gpu_max = logit_pae_gpu

                            # Compute the softmax probabilities in a numerically stable way.
                            exp_logit_gpu = np.exp(logit_gpu_max)
                            exp_logit_gpu_pae = np.exp(logit_pae_gpu_max)
                            probab = torch.nn.Softmax(dim=1)(torch.tensor(logit_gpu).float()).float().detach().cpu().numpy()
                            probab_pae = torch.nn.Softmax(dim=0)(torch.tensor(logit_pae_gpu).float()).float().detach().cpu().numpy()
                            one_pae = np.sum(probab_pae, axis=0)
                            has_nan = np.isnan(probab).any()
                            distogram_contact_prob_12 = probab[0][:12].sum(axis=0)
                            expected_pae = np.einsum('ljk, l -> jk', probab_pae, np.array(list_pae))
                            interface_pae = expected_pae * (~same_chain.bool())[0].float().detach().cpu().numpy() * mask_2d[0].float().detach().cpu().numpy()
                            interface_pae_mean = interface_pae.sum() / (~same_chain.bool()*mask_2d.bool())[0].sum()
                            
                            plt.figure(figsize=(30,10))
                            plt.subplot(1,3,1)
                            # expected_dist = np.einsum('ljk, l -> jk', probab[0], np.array(list_dist))
                            
                            expected_dist = np.einsum('ljk, l -> jk', probab[0], dbins)
                            plt.imshow(expected_dist, interpolation='nearest', cmap='viridis_r')
                            plt.axhline(y=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.axvline(x=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.colorbar()
                            plt.title("logit_s based expected dist")
                            distogram_true = c6d_true[..., 0][0].float().detach().cpu().numpy()
                            plt.subplot(1,3,2)
                            plt.imshow(distogram_true, interpolation='nearest', cmap='viridis_r')
                            plt.axhline(y=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.axvline(x=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.colorbar()
                            plt.title("distogram_true")

                            plt.subplot(1,3,3)
                            distogram_pred = c6d_pred[..., 0][0].float().detach().cpu().numpy()
                            plt.imshow(distogram_pred, interpolation='nearest', cmap='viridis_r')
                            plt.axhline(y=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.axvline(x=int(torch.sum(same_chain[0,0])), color='r', linestyle='-')
                            plt.colorbar()
                            plt.title("distogram_pred")
                            


                            plt.savefig(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}.png")

                            plt.close()
                            
                            # write interface_pae_mean on plt.imshow
                            # plt.text(0, 0, f"interface_pae_mean: {interface_pae_mean:.2f}", fontsize=12, color='black')
                            # breakpoint()
                            # distogram_loss = calc_c6d_loss(logit_s, c6d, mask_2d)
                            # breakpoint()
                            # distogram_loss = calc_c6d_loss(logit_s, c6d, (~same_chain.bool())*mask_2d)
                            # breakpoint()
                            # breakpoint()
                            # epitope_mask = epitope_info[:,:,None] + epitope_info[:,None,:]
                            epitope_mask = torch.cdist(true_crds[0,0,:,1], true_crds[0,0,:,1])<10
                            distogram_mask = epitope_mask * (~same_chain.bool()) * mask_2d
                            distogram_loss = calc_c6d_loss(logit_s, c6d, distogram_mask)[0]
                            num_classes=37
                            class_counts = torch.bincount((c6d[..., 0]*distogram_mask).view(-1), minlength=num_classes)
                            weights = len((c6d[..., 0]*distogram_mask).view(-1))/(num_classes*class_counts.float()+1e-5)
                            criterion = FocalLoss(alpha=weights)
                            focal_loss = criterion(logit_s[0], c6d[..., 0])
                            # # breakpoint()
                            focal_loss = (distogram_mask.bool() * focal_loss).sum() / ((~distogram_mask.bool()).sum() + 1e-5)
                            # distogram_loss = distogram_loss.item()
                            focal_loss = focal_loss.item()
                            # breakpoint()
                            print(f'{str(i_epi_res+1)}th_epi_{item[0]} interface_pae_mean: {interface_pae_mean:.2f}')
                            print(f'{str(i_epi_res+1)}th_epi_{item[0]} distogram_loss: {distogram_loss.item()}')
                            pae_dict[str(item[0])][f"{str(i_epi_res+1)}th_epi"] = [interface_pae_mean, distogram_loss.item(), focal_loss]
                            # if logit_s is not None:
                            #     expected_dist = np.einsum('ljk, l -> jk', probab[0], dbins)
                            #     distogram_true = c6d_true[..., 0][0].float().cpu().numpy()
                            #     distogram_pred = c6d_pred[..., 0][0].float().cpu().numpy()
                            #     np.savez(
                            #         f"inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}/{str(i_epi_res+1)}th_epi_{item[0]}/logit_s_0.npz",
                            #         logit_s_0=logit_s[0].float().cpu().numpy(),
                            #         expected_dist=expected_dist,
                            #         distogram_true=distogram_true,
                            #         distogram_pred=distogram_pred
                            #     )
                            # breakpoint()
                            
                # breakpoint()
                os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}", exist_ok=True)
                # breakpoint()
                with open(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930_ES/{item[0]}/pae_dict.csv", 'w') as f: #new_test
                # os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_DockingSet_ES_reepi/{item[0]}", exist_ok=True)
                # with open(f"inference_pdb/{self.model_name}_{self.model_subname}_DockingSet_ES_reepi/{item[0]}/pae_dict.csv", 'w') as f:    # docking set
                    writer = csv.writer(f)
                    for key, value in pae_dict[str(item[0])].items():
                        writer.writerow([key, float(value[0]), float(value[1]), float(value[2])])
                            


    def load_model(self, ddp_model, model_name, rank):
        if model_name == "RF2_apr23":
            chk_fn = (f"/home/yubeen/rf_abag_templ/results/0919_test/models/weights/RF2_apr23.pt")
        else:
            chk_fn = f"/home/kkh517/submit_files/Project/halfblood/models_{self.model_subname}/RF2_apr23_best.pt"
            # chk_fn = "/home/kkh517/submit_files/Project/halfblood/models/RF2_apr23_best.pt"
            # chk_fn = f"/home/kkh517/submit_files/Project/halfblood/models_blur_lj/RF2_apr23_best.pt"
            # chk_fn = "/home/kkh517/submit_files/Project/halfblood/models_1.1.0/RF2_apr23_best.pt"

        print(f"loading model from {chk_fn}")
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        loaded_epoch = checkpoint["epoch"]
        print("loaded_epoch", loaded_epoch)
        # print(f"state_dict {checkpoint['model_state_dict'].keys()}")
        ddp_model.module.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        ddp_model.module.shadow.load_state_dict(checkpoint["model_state_dict"], strict=False)



    def inference(self, rank, world_size):
        print("running ddp on rank %d, world_size %d" % (rank, world_size))
        gpu = rank % torch.cuda.device_count()
        # print(f"world_size {world_size}, rank {rank}")
        init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(f"cuda:{gpu}")

        # move some global data to cuda device
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        self.xyz_converter = self.xyz_converter.to(gpu)

        model = EMA(RoseTTAFoldModule(**self.model_param).to(gpu), decay=0.99)
        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        self.load_model(ddp_model, self.model_name, gpu)

        if rank == 0:
            print('# of parameters:', count_parameters(model))

        (_, _, _, _,
          _, _, _, _, 
          _, _,_, _, hl_ag_test, _, _,) = get_train_valid_set(self.loader_param)
        hl_ag_test_2 = {k: v for k, v in hl_ag_test.items() if len(v) > 0}
        print(f"test set : {hl_ag_test_2.values()}")
    
        for cluster in hl_ag_test_2.keys():
            for item in hl_ag_test[cluster]:
                print(f"item {item}")
                if os.path.exists(f"/home/kkh517/submit_files/Project/epitope_sampler_halfblood/inference_pdb/halfblood_{self.model_subname}_After210930_ES/{item}/pae_dict.csv"): #new test
                # if os.path.exists(f"/home/kkh517/submit_files/Project/epitope_sampler_halfblood/inference_pdb/{self.model_subname}_After210930_ES")
                # if os.path.exists(f"/home/kkh517/submit_files/Project/epitope_sampler_halfblood/inference_pdb/{self.model_subname}_DockingSet_ES_/{item}/pae_dict.csv"): #docking set
                    print(f"skip {cluster}")
                    # pop the item from hl_ag_test
                    hl_ag_test.pop(cluster)
                    continue
                # if item in ['7z12_B_C_A']:
                #     print(f"skip {cluster}")
                    # pop the item from hl_ag_test
                    # hl_ag_test.pop(cluster)
                    # continue

                # # check if the template exists # new_test
                # if not os.path.exists(f"/home/yubeen/alphafold2.3_monomer/{item}/{item}_ag/ranked_0.pdb"):
                #     print(f"no antigen skip {cluster}")
                #     hl_ag_test.pop(cluster) # later
                    
        print('len(hl_ag_test)',len(hl_ag_test))
        self.n_hl_ag_test = (len(hl_ag_test.keys()) // world_size) * world_size
        test_hl_ag_set = DatasetComplex_antibody(
            list(hl_ag_test.keys())[:self.n_hl_ag_test],
            loader_complex_antibody_kh,
            hl_ag_test,
            self.valid_param,
            validation=True,
            unclamp=True
        )
        test_hl_ag_sampler = data.distributed.DistributedSampler(
            test_hl_ag_set, num_replicas=world_size, rank=rank
        )
        test_hl_ag_loader = data.DataLoader(
            test_hl_ag_set, sampler = test_hl_ag_sampler, **LOAD_PARAM
        )

        self.do_inference(ddp_model, test_hl_ag_loader, gpu, 0) # self.inference in train_RF2_rigid.py

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
            # random_index = random.randint(1, 5)
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
        assert len(sel[0]) >= xyz_t.shape[1]  # = L


        msa = msa.to(gpu, non_blocking=True)
        ins = ins.to(gpu, non_blocking=True)
        L_s = L_s.to(gpu, non_blocking=True)
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
        for i, msai in enumerate(msa): # len(sel[0]) = L
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
        # epitope_info = epitope_full_info
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
            mask_t_2d.float() * same_chain.float()[:, None]  # (B, T, L, L)
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
        input_i["validation"] = True
        return input_i
# if __name__ == "__main__":
if __name__ == '__main__':
    from arguments import get_args

    
    args, model_param, loader_param, _ = get_args()
    sys.stdout.write("Loader_param:\n, %s\n" % loader_param)
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mp.freeze_support()
    inference = Predictor(
        model_name=args.model_name,
        interactive=args.interactive,
        model_param=model_param,
        loader_param=loader_param,
        crop_size=args.crop,
        maxcycle=args.maxcycle,
    )
    inference.run_predict(torch.cuda.device_count())

