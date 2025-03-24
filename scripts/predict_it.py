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
from data_loader_rigid2 import *
from loss_jan30 import *
sys.path.append('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/src/rfabflex')
from collections import namedtuple, defaultdict, OrderedDict
import time
from common.util import *
from model.util_module import XYZConverter
from model.RoseTTAFoldModel import RoseTTAFoldModule
from common.kinematics import xyz_to_c6d, xyz_to_t2d, c6d_to_bins2
from opt_einsum import contract as einsum
USE_AMP = True

class Predictor:
    def __init__(self, model_name="RF_abag",interactive=False, model_param={}, loader_param={}, crop_size=256, 
                 heavy_a3m_tmp='', light_a3m_tmp='',antigen_a3m_tmp='',
                 antibody_template_pdb='', antigen_template_pdb ='',
                 epitope_idx='', item='', maxcycle=4, output_prefix = '',):
        self.model_name = model_name
        self.model_subname = "1.0.1"#sc_lj"#"samechain"#"on_rigid" #"on_RF2"
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

        self.heavy_a3m_tmp=heavy_a3m_tmp
        self.light_a3m_tmp=light_a3m_tmp
        self.antigen_a3m_tmp=antigen_a3m_tmp
        self.antibody_template_pdb=antibody_template_pdb
        self.antigen_template_pdb=antigen_template_pdb
        self.epitope_idx=epitope_idx
        self.item=item
        self.output_prefix = output_prefix
    
    def run_predict(self, world_size):
        # if "MASTER_ADDR" in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        # if "MASTER_PORT" in os.environ:
        os.environ["MASTER_PORT"] = "12749"
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

    def do_inference(self, model, gpu, inputs):
        with torch.no_grad():
            model.eval()
            pae_dict = {}
            # for inputs in test_hl_ag_loader:
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
            pae_dict[str(item)] = {}
            index_list = torch.nonzero(surface_info[0]).squeeze(1).tolist()
            # print('index_list',index_list)
            # for i_epi_res_1 in tqdm(index_list):
            
            #     i_epi_res = i_epi_res_1 - int(interface_split[0]) + 1
            #     if true_crds.shape[1] > 1:
            #         antigen_crds = true_crds[:, 0,interface_split[0]: ,1,:]
            #     else:
            #         antigen_crds = true_crds[0][0][ interface_split[0]: ,1,:].unsqueeze(0)
            #     antigen_crds = xyz_prev[:, interface_split[0]: ,1,:]
            #     cond = torch.cdist(antigen_crds, antigen_crds) < 10 # [B, ag_len, ag_len] #  epitope_radius
            #     epitope_info = cond[0,i_epi_res].float().unsqueeze(0)
            #     epitope_info = torch.cat((torch.zeros(1,interface_split[0]).to(epitope_info.device), epitope_info), dim=1).long()
                
            if True:
                i_epi_res = 999
                epitope_info = surface_info
                # os.makedirs(f"{self.output_prefix}/templates_check", exist_ok=True)
                # write_pdb(msa[:,0,0][0], xyz_prev[0], L_s=L_s[0], Bfacts=epitope_info[0].bool(),prefix=f"{self.output_prefix}/templates_check/{item}_templates_{i_epi_res}th_epi") #ProjectName



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
                
                        if i_cycle < N_cycle - 1:
                            output_i = (msa_one, pair, state, pred_crds[-1], alphas[-1], None, epitope_info)
                            _, final_crds = self.xyz_converter.compute_all_atom(
                                msa[:, i_cycle, 0], pred_crds[-1], alphas[-1]
                            )
                            # if i_cycle % 4 == 3:
                            #     write_pdb(
                            #         msa[:, i_cycle, 0][0],
                            #         final_crds[0],
                            #         L_s[0],
                            #         Bfacts=plddt_unbinned[0],
                            #         prefix=f"{self.output_prefix}/{i_epi_res}th_epi_{item}_{i_cycle}",       
                            #     )
                            continue
                        else:
                            
                            # visusalize logit_s[0] and c6d[0]
                            write_pdb(
                                msa[:, i_cycle,0][0],
                                final_crds[0],
                                L_s[0],
                                Bfacts=plddt_unbinned[0],
                                # prefix=f"{self.output_prefix}/{i_epi_res}th_epi_{item}_{i_cycle}",
                                prefix = f"{args.output_prefix}"
                            )
                            sys.stdout.write(f"\r{item} inference done")
                            # os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930/{item[0]}/{i_epi_res}th_epi_{item[0]}", exist_ok=True)
                            # os.system(f"mv {i_epi_res}th_epi_{item[0]}_{i_cycle}.pdb inference_pdb/{self.model_name}_{self.model_subname}_After210930/{item[0]}/{i_epi_res}th_epi_{item[0]}")
                            # os.makedirs(f"inference_pdb/iitp/{item[0]}", exist_ok=True)
                            # os.system(f"mv {i_epi_res}th_epi_{item[0]}_{i_cycle}.pdb inference_pdb/iitp/{item[0]}/{i_epi_res}th_epi_{item[0]}.pdb")# iitp
                            # os.system(f"mv ")
                            count += 1

                        # true_crds_2, mask_crds_2 = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                        # print('mask_crds',mask_crds.shape) # [1,L,27]

                        # mask_2d = mask_crds_2[:, :, :3].bool().all(dim=-1)
                        # mask_2d = mask_2d[:, :, None] * mask_2d[:, None, :]  # (B, L, L)
                        # c6d = xyz_to_c6d(true_crds_2)
                        same_chain = network_input["same_chain"] # TODO
                        # c6d = c6d_to_bins2(c6d, network_input["same_chain"], negative=False)
                        pred = pred_crds
                        # true pae
                        
                        # true = true_crds_2.unsqueeze(0)  # [1, B, L, n_atom, 3]
                        # t_tilde_ij = get_t(true[:, :, :, 0], true[:, :, :, 1], true[:, :, :, 2], non_ideal=True)# (1,B,L,L,3)
                        # t_ij = get_t(pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2])  # (I,B,L,L,3)

                        # difference = torch.sqrt(torch.square(t_tilde_ij - t_ij).sum(dim=-1) + 1e-6)  # (I,B,L,L)
                        # eij_label = difference[-1].clone().detach() # (L, L)                        
                        nbin = 64
                        bin_step = 0.5
                        pae_bins = torch.linspace(
                            bin_step,
                            bin_step * (nbin - 1),
                            nbin - 1,
                            dtype=logit_pae.dtype,
                            device=logit_pae.device,
                        )
                        # true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
                        
                        
                        list_dist = [] ; list_pae = []
                        # print('c6d[0].shape',c6d[0].shape)
                        # for j in range(36):
                        #     dist_bin = 2.25 + 0.5 * j
                        #     list_dist.append(dist_bin)
                        # list_dist.append(20.5)

                        for k in range(64):
                            pae_bin = 0.5 + 0.5 * k
                            list_pae.append(pae_bin)
                        # logit_gpu = logit_s[0].cpu().numpy()
                        logit_pae_gpu = logit_pae[0].cpu().numpy()
                        # print('true_pae_label.shape',true_pae_label.shape)
                        # logit_pae_true = true_pae_label[0].cpu().numpy()
                        # has_nan_here = np.isnan(logit_gpu).any()
                        # has_nan_here_pae = np.isnan(logit_pae_gpu).any()
                        # print('has_nan_here',has_nan_here)
                        # print('has_nan_here_pae',has_nan_here_pae)
                        # logit_gpu_max = logit_gpu - np.max(logit_gpu, axis=1, keepdims=True)
                        logit_pae_gpu_max = logit_pae_gpu - np.max(logit_pae_gpu, axis=0, keepdims=True)
                        # logit_pae_true
                        # logit_gpu_max = logit_gpu
                        # logit_pae_gpu_max = logit_pae_gpu

                        # Compute the softmax probabilities in a numerically stable way.
                        # exp_logit_gpu = np.exp(logit_gpu_max)
                        # exp_logit_gpu_pae = np.exp(logit_pae_gpu_max)
                        # probab = torch.nn.Softmax(dim=1)(torch.tensor(logit_gpu).float()).float().detach().cpu().numpy()
                        # print('exp_logit_gpu.shape',exp_logit_gpu.shape)
                        # print('exp_logit_gpu_pae.shape',exp_logit_gpu_pae.shape)
                        probab_pae = torch.nn.Softmax(dim=0)(torch.tensor(logit_pae_gpu).float()).float().detach().cpu().numpy()
                        # probab_true_pae = torch.nn.Softmax(dim=0)(torch.tensor(logit_pae_true).float()).float().detach().cpu().numpy()
                        
                        # probab_pae = nn.Softmax(dim=1)(torch.tensor(logit_pae_gpu)).float().detach().cpu().numpy()
                        # print('exp_logit_gpu_pae.shape',exp_logit_gpu_pae.shape)
                        # print('probab_pae.shape',probab_pae.shape)
                        # prob check
                        # one_pae = np.sum(probab_pae, axis=0)
                        expected_pae = np.einsum('ljk, l -> jk', probab_pae, np.array(list_pae))
                        # expected_true_pae = np.einsum('ljk, l -> jk', probab_true_pae, np.array(list_pae))
                        # pae = np.sum(probab_pae, dim=0).detach().cpu().numpy()
                        # print('pae.shape',pae.shape)
                        # plt.imshow(expected_pae, cmap="jet", interpolation="nearest")
                        # plt.colorbar()
                        # plt.title(f"pae[pred]")
                        # print('mask_t_2d.shape',network_input["mask_t"].shape)
                        interface_pae = expected_pae * (~same_chain.bool())[0].float().detach().cpu().numpy() #* mask_2d[0].float().detach().cpu().numpy()
                        
                        # calculate the mean of pae in interface region
                        interface_pae_mean = interface_pae.sum() / (~same_chain.bool())[0].sum()
                        # write interface_pae_mean on plt.imshow
                        # plt.text(0, 0, f"interface_pae_mean: {interface_pae_mean:.2f}", fontsize=12, color='black')
                        print(f'{item} interface_pae_mean: {interface_pae_mean:.2f}')
                        # pae_dict[str(item)][f"{i_epi_res}th_epi"] = interface_pae_mean
                        # /home/kkh517/submit_files/Project/epitope_sampler_halfblood/antibody_meeting_inputs/Trastuzumab_Abmpnn_output/100th_AbMPNN_Trastuzumab_seq_pae.csv
                        pae_dict[str(item)][self.output_prefix.split('/')[-1]] = interface_pae_mean

            # os.makedirs(f"inference_pdb/{self.model_name}_{self.model_subname}_After210930/{item[0]}", exist_ok=True)
            # with open(f"{self.output_prefix}/{str(i_epi_res+1)}_{item}_pae_dict.csv", 'w') as f:
            output_dir = '/'.join(self.output_prefix.split('/')[:-1])+'/output'
            os.makedirs(output_dir, exist_ok=True)
            file_path = f"{output_dir}/pae.csv"
            write_mode = 'a' if os.path.exists(file_path) else 'w'
            with open(file_path, write_mode) as f:
                writer = csv.writer(f)
                for key, value in pae_dict[str(item)].items():
                    writer.writerow([key, float(value)])
                        


    def load_model(self, ddp_model, model_name, rank):
        if model_name == "RF2_apr23":
            chk_fn = (f"/home/yubeen/rf_abag_templ/results/0919_test/models/weights/RF2_apr23.pt")
        else:
            chk_fn = f"/home/kkh517/submit_files/Project/halfblood/models_{self.model_subname}/RF2_apr23_best.pt"
            # chk_fn = f"/home/kkh517/submit_files/Project/halfblood/models_/RF2_apr23_best.pt"
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
        inputs = self.make_inputs()
        # do_inference 
        self.do_inference(ddp_model, gpu, inputs) # self.inference in train_RF2_rigid.py

        destroy_process_group()
    def make_inputs(self):
        heavy_a3m_tmp = self.heavy_a3m_tmp
        light_a3m_tmp = self.light_a3m_tmp
        antigen_a3m_tmp = self.antigen_a3m_tmp
        antibody_template_pdb = self.antibody_template_pdb
        antigen_template_pdb = self.antigen_template_pdb
        epitope_idx= self.epitope_idx
        item = self.item

        ######## starts from here
        # if heavy msa exists
        

        a3m = defaultdict(list)
        template_params = defaultdict(list)
        pdb_id, hchain, lchain, ag_chain = item.split('_')
        if hchain!= '#':
            # for i, _ in enumerate(hchain):
            a3m['H'].append(get_msa(heavy_a3m_tmp, item))
                # template_params['H'].append()
        if lchain!= '#':
            # for i, _ in enumerate(lchain):
            a3m['L'].append(get_msa(light_a3m_tmp, item))
        assert ag_chain != '#', "ag_chain should not be blanck"
        # for i, _ in ag_chain:
        a3m['AG'].append(get_msa(antigen_a3m_tmp,item)) # only single chain antigen
        # print(f"a3m {a3m}")
        a3m_in = []
        length_dict = defaultdict(int)
        # homo_list = {'H': [[i] for i in range(len(heavy_a3m_tmp))], 'L':[[i] for i in range(len(light_a3m_tmp))], 'AG':[[i] for i in range(len(antigen_a3m_tmp))]}
        homo_list= {'H':[[0]], 'L':[[0]], 'AG':[[0]]}
        for k, v in homo_list.items():  # [[0, 1]] or [[0], [1]]
            count = 0
            for idx in v:
                for idx2 in idx:
                    count += 1
            length_dict[k] = count #total length

        for key in ['H', 'L', 'AG']:
            if key in homo_list:
                for value in homo_list[key]:
                    # print(f'value {value} {a3m[key]}')
                    a3m_in.append(a3m[key][value[0]])
            homo_list_final = []

        h_length = length_dict['H']
        hl_length = length_dict['H'] + length_dict['L']
        ag_length = length_dict['AG']

        for k, v in homo_list.items():
            if k == 'H':
                homo_list_final.extend(v)
            elif k == 'L':
                for i in v:
                    homo_list_final.append([j + h_length for j in i])
            else:
                for i in v:
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
        chain_properties.extend([2] * ag_length)
        print(f"chain_properties {chain_properties}")
        #featurize_antibody_complex a3m = a3m_in, homo_list = homo_list_final
        # msa, ins def
        a3m = merge_a3m_antibody(a3m_in, homo_list_final)
        print(f"homo_list {homo_list_final}")
        msa = a3m['msa'].long() ; ins = a3m['ins'].long() ; N_final_dict = a3m['N_final_dict']
        if len(msa) >5:
            msas, inss = [], []
            msas.append(msa[None, 0])
            # logger.info('msa zero', msa[None, 0].shape)
            inss.append(ins[None, 0])
            # logger.info('ins zero', ins[None, 0].shape)
            for i in np.arange(len(N_final_dict)):
                sN, eN = N_final_dict[i][0], N_final_dict[i][1]
                if eN - sN > 50:
                    # logger.info('msa shape', i, msa[sN:eN].shape)
                    # logger.info('ins shape', i, ins[sN:eN].shape)
                    msa_del, ins_del = MSABlockDeletion(msa[sN:eN], ins[sN:eN])
                    # logger.info('msa del', i, msa_del.shape)
                    # logger.info('ins del', i, ins_del.shape)
                    msas.append(msa_del)
                    inss.append(ins_del)
                else:
                    msas.append(msa[sN:eN])
                    inss.append(ins[sN:eN])
            msa = torch.cat(msas, dim=0)
            ins = torch.cat(inss, dim=0)
        # print(f"msa {msa.shape}")
        sample = len(msa)
        if sample > 3000:
            msa_sel = torch.randperm(sample-1)[:2999]
            msa = torch.cat((msa[:1, :], msa[1:, :][msa_sel]), dim=0)
            ins = torch.cat((ins[:1, :], ins[1:, :][msa_sel]), dim=0)
        # print(f"msa after samle {msa.shape}")
        # xyz_prev, mask_prev def
        tplts = {'H': [antibody_template_pdb], 'L':None, 'AG': [antigen_template_pdb] }
        print(f"tplts {tplts}")
        L_s = a3m['L_s']
        # xyz_t, f1d_t, mask_t = CustomTemplFeaturize_kh(item, tplts, L_s)
        # breakpoint()
        xyz_t, f1d_t, mask_t = TemplFeaturize_Tlqkf(item, tplts, L_s)
        
        


        xyz_prev = xyz_t[0].clone()
        mask_prev = mask_t[0].clone()
        # idx def
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
        #chain_idx def
        chain_idx = torch.zeros(sum(L_s)).long()
        for k, (i, j) in enumerate(zip(L_start, L_end)):
            chain_idx[i:j] = chain_properties[k]
        # interface_split def
        ab_length = 0 ; ab_length2 = 0
        if (np.array(chain_properties) > 1).sum() == 0:
            ab_ch_count = 1 
        else:
            ab_ch_count = (np.array(chain_properties) < 2).sum()

        for i in range(ab_ch_count):
            ab_length += L_s[i]

        ag_length = sum(L_s) - ab_length 
        interface_split = [ab_length, ag_length]

        ## same_chain def
        same_chain = torch.zeros((sum(L_s), sum(L_s))).long()
        same_chain[0:ab_length, 0:ab_length] = 1
        total_length = sum(L_s)
        same_chain[ab_length: total_length, ab_length:total_length] = 1
        # epitope_info def
        epitope_info = torch.zeros(total_length)
        epitope_ifaces = torch.Tensor([int(i) + ab_length -1  for i in epitope_idx.split(',')]).long()
        print(f"epitope_idx {epitope_idx}")
        print(f"epitope_ifaces {epitope_ifaces}")
        epitope_info[epitope_ifaces] = 1

        if epitope_info.sum() != 0:
            seq = msa[0].long() ; atoms = xyz_t[0] # pdb for all atoms
            epi_idx = epitope_info.nonzero().squeeze()
            epi_t = torch.full((xyz_t.shape[1],),False)
            epi_t[epi_idx] = True
            # print('epi_t',epi_t.shape)
            xyz_cen = xyz_t - xyz_t[:,:,1].mean(dim=1)[:,None,None,:]
            epitope_xyz = xyz_cen[0][epi_idx]
            # print('epitope_xyz',epitope_xyz.shape)
            epitope_center = epitope_xyz[:,1].mean(dim=0)
            xyz_t[:,:interface_split[0]] += epitope_center
            # os.makedirs(f"templates_check", exist_ok=True)
            # write_pdb(seq, atoms, L_s=L_s, Bfacts=epi_t,prefix=f"{self.output_prefix}/templates_{item}") #ProjectName
        # def sel
        sel = torch.arange(sum(L_s))
        L_s= torch.Tensor(L_s)
        idx_pdb = idx.long()#.to(self.device, non_blocking=True)
        xyz_t = xyz_t.float()#.to(self.device, non_blocking=True)
        t1d = f1d_t.float()#.to(self.device, non_blocking=True)
        xyz_prev = xyz_prev.float()#.to(self.device, non_blocking=True)
        ## loader end -> needs to make it into batch
        inputs = (item, item, sel, torch.tensor(L_s), msa, ins, None, None, idx.long(), xyz_t.float(), f1d_t.float(), mask_t, xyz_prev.float(), mask_prev, same_chain, chain_idx, False, False, None, epitope_info, False)
        return inputs        
    
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
        new_inputs = []
        for i, input in enumerate(inputs):
            # print(f"{i}th input {type(input)}")
            if type(input) == torch.Tensor:
                
                input = input.unsqueeze(0) #batchwize
                
                # print(f"{i}th input {input.shape}")
            # else:
            #     input = torch.tensor([input])
            new_inputs.append(input)
        [   cluster,
            item,
            sel,
            L_s,
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
        ] = new_inputs
            

        # (
        #     # seq,
        #     # msa,
        #     # msa_masked,
        #     # msa_full,
        #     # mask_msa,
        #     cluster,
        #     item,
        #     sel,
        #     L_s,
        #     # params,
        #     msa,
        #     ins,
        #     true_crds,
        #     mask_crds,
        #     idx_pdb,
        #     xyz_t,
        #     t1d,
        #     mask_t,
        #     xyz_prev,
        #     mask_prev,
        #     same_chain,
        #     chain_idx,
        #     unclamp,
        #     negative,
        #     interface_split,
        #     epi_full,
        #     validation,
        # ) = inputs
        input_dict = {}
        for i, name in enumerate(['cluster','item','sel','L_s','msa','ins','true_crds','mask_crds','idx_pdb','xyz_t','t1d','mask_t','xyz_prev','mask_prev','same_chain','chain_idx','unclamp','negative','interface_split','epi_full','validation']):
            input_dict[name] = str(inputs[i])

        with open('/home/kkh517/submit_files/Project/epitope_sampler_halfblood/input_inference.json','w', encoding='utf-8') as json_file:
            json.dump(input_dict, json_file)
        # assert len(sel[0]) >= xyz_t.shape[1]  # = L


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
            # true_crds_list.append(true_crds[i, :, sel[i]])
            # mask_crds_list.append(mask_crds[i, :, sel[i]])
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
        # true_crds = torch.stack(true_crds_list)
        # mask_crds = torch.stack(mask_crds_list)
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
        # true_crds = true_crds.to(gpu, non_blocking=True)  # (B, N_homo, L, 27, 3)
        # mask_crds = mask_crds.to(gpu, non_blocking=True)  # (B, N_homo, L, 27)
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
    from arguments import get_args_inference

    
    args, model_param, loader_param, _ = get_args_inference()
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
        heavy_a3m_tmp = args.heavy_a3m_tmp,
        light_a3m_tmp = args.light_a3m_tmp,
        antigen_a3m_tmp = args.antigen_a3m_tmp,
        antibody_template_pdb= args.antibody_template_pdb,
        antigen_template_pdb=args.antigen_template_pdb,
        epitope_idx = args.epitope_idx,
        item = args.item,
        output_prefix = args.output_prefix,
    )
    inference.run_predict(torch.cuda.device_count())

