import random 
import torch
import numpy as np
from typing import Union, Mapping, Literal
from typing_extensions import Self
from pathlib import Path, PosixPath
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
import pickle


@dataclass
class Batch:
    atom_pos: torch.Tensor
    atom_mask: torch.BoolTensor
    res_type: torch.LongTensor
    seq_idx: torch.LongTensor
    res_mask: torch.BoolTensor
    rigid: Rigid
    name: Union[str, list]
    L_s: list[int, int, int]

    def __post_init__(self):
        *rest_dim, L, N = self.atom_mask.shape
        assert self.atom_pos.shape == (*rest_dim, L, N, 3)
        assert self.res_type.shape == (*rest_dim, L)
        assert self.seq_idx.shape == (*rest_dim, L)
        assert self.res_mask.shape == (*rest_dim, L)
        assert self.rigid.shape == (*rest_dim, L)

        if isinstance(self.name, list):
            assert np.array(self.name).shape == (*rest_dim,)
        if len(rest_dim) > 1:
            raise ValueError("Batch should have only one dimension")
        
    def get_res_name(self) -> np.ndarray:
        mapping = np.vectorize(lambda x: chemical.PROTEIN_RES_NAMES_THREE[x])
        return mapping(self.res_type.cpu())
    
    @property
    def trans(self) -> torch.Tensor:
        return self.rigid.get_trans()
    
    @property
    def rot_mats(self) -> torch.Tensor:
        return self.rigid.get_rots().get_rot_mats()
    
    @property
    def device(self) -> torch.device:
        return self.atom_pos.device
    
    @property
    def shape(self) -> tuple[int]:
        return self.atom_mask.shape
    
    @classmethod
    def collate_fn(cls, batch_list: list[Self]) -> Self:
        def _auto_collate(data_list: list):
            """
            Automatically collates a list of data elements into a batch.

            Args:
                data_list (list): A list of data elements to be collated. All elements in the list
                                  must be of the same type.

            Returns:
                Union[torch.Tensor, Rigid, None, list]: The collated batch, which can be:
                    - A torch.Tensor if the elements in the list are tensors.
                    - A Rigid object if the elements in the list are of type Rigid.
                    - None if the elements in the list are None.
                    - The original list if the elements do not match the above types.

            Raises:
                AssertionError: If the elements in the list are not of the same type.

            Notes:
                - For torch.Tensor elements, the `auto_tensor_collate` function is used to collate them.
                - For Rigid elements, the function creates a new Rigid object using the rotation matrices
                  and translations extracted from the elements.
            """
            assert all(isinstance(data_list[0], type(data)) for data in data_list)
            if isinstance(data_list[0], torch.Tensor):
                return auto_tensor_collate(data_list)
            elif isinstance(data_list[0], Rigid):
                return create_rigid(
                    auto_tensor_collate(
                        [data.get_rots().get_rot_mats() for data in data_list]
                    ),
                    auto_tensor_collate([data.get_trans() for data in data_list]),
                )
            elif data_list[0] is None:
                return None
            else:
                return data_list

        for batch in batch_list:
            assert batch.shape == batch_list[0].shape
        return cls(
            **{
                k: _auto_collate([getattr(batch, k) for batch in batch_list])
                for k in asdict(batch_list[0])
            }
        )
    
    def to(self, device: Union[str, torch.device]):
        def _to(data):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, Rigid):
                return rigid_to_device(data, device)
            else:
                return data
        return self.__class__(
            **{k: _to(v) for k,v in asdict(self).items()}
        )
    
    def duplicate(self, num:int) -> Self:
        def _duplicate(data, offset:Literal[0,1]):
            num_rest = len(data.shape) - offset
            if isinstance(data, torch.Tensor):
                return data.tile(num, *[1]*num_rest)
            elif isinstance(data, Rigid):
                return create_rigid(
                    data.get_rots().get_rot_mats().tile(num, *[1]*num_rest, 1, 1),
                    data.get_trans().tile(num, *[1]*num_rest, 1),
                )
            elif data is None:
                return None
            else:
                if offset == 0:
                    return [data]*num
                return data * num
        if len(self.shape) == 2:
            offset = 0
        elif len(self.shape) == 3:
            offset = 1
        else:
            raise ValueError("Batch shape should have only one dimension")
        return self.__class__(
            **{k: _duplicate(v, offset) for k,v in asdict(self).items()}
        )
    

class AbData(torch.utils.data.Dataset):
    """
    """

    crop_idx_cache: Mapping[str, tuple[list, list]] = {}
    
    def __init__(
            self, 
            input_dir_path: Union[str, PosixPath],
            cluster_data: list[list[str]],
            crop_chain: bool = True,
            crop_length: int = 256,
    ):
        self.input_dir_path = input_dir_path
        self.cluster_data = cluster_data
        self.crop_chain = crop_chain
        self.crop_length = crop_length
    
    def create_dataloader(self, batch_size: int = 1):
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=Batch.collate_fn,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    @staticmethod
    def from_input_data(
        input_data:Mapping[str, Mapping[str, np.ndarray]],
        crop_chain: bool = False,
        crop_length: int = 256,
    ) -> Batch:
        """
        """
        input_name = input_data["input_name"]
        _, heavy, light, antigen = input_name.split('_')

        heavy_seq = input_data['var_seq'][heavy]
        light_seq = input_data['var_seq'][light]
        antigen_seq = input_data['var_seq'][antigen]
        sequence = heavy_seq + light_seq + antigen_seq
        res_names = [i for i in sequence]
        heavy_str = input_data['var_str'][heavy]
        light_str = input_data['var_str'][light]
        antigen_str = input_data['var_str'][antigen]
        complex_str = [heavy_str, light_str, antigen_str]
        positions = np.concatenate(complex_str)

        seq_len = len(res_names)
        seq_idx = torch.arange(seq_len)
        L_s = [len(heavy_seq), len(light_seq), len(antigen_seq)]

        # TODO : crop here
        
        atom_pos = torch.from_numpy(positions).float()
        atom_mask = ~torch.isnan(atom_pos).any(-1)

        atom_CoM = torch.nanmean(atom_pos[:, chemical.ATOM_CENTER_IDX], 0) #
        atom_pos = atom_pos - atom_CoM

        #get res type
        PROTEIN_RES_NAMES = tuple(chemical.restype_1to3.keys())
        mapping = lambda x: PROTEIN_RES_NAMES.index(x)
        res_type = np.vectorize(mapping)(res_names)
        res_type = torch.from_numpy(res_type)

        # get frames
        res_mask, rigid = all_atom_to_frames(res_type, atom_pos, atom_mask)
        rot_mats = rigid.get_rots().get_rot_mats()
        trans = rigid.get_trans()
        rot_mats[~res_mask] = torch.eye(3)
        trans[~res_mask] = 0
        rigid=create_rigid(rot_mats,trans)

        return Batch(
            atom_pos=atom_pos,
            atom_mask=atom_mask,
            res_type=res_type,
            seq_idx=seq_idx,
            res_mask=res_mask,
            rigid=rigid,
            name=input_name,
            L_s=L_s,
        )
    
    @staticmethod
    def from_input_path(
        input_path: Union[str, PosixPath],
        crop_chain: bool = False,
        crop_length: int = 256,
    ) -> Batch:
        """
        """
        with open(input_path, 'rb') as f:
            input_data = pickle.load(f)
        input_data['input_name'] = input_path.stem
        return AbData.from_input_data(input_data, crop_chain, crop_length)
    
    def __len__(self):
        return len(self.cluster_data)
    
    def __getitem__(self, idx:int) -> Batch:
        members = self.cluster_data[idx]
        pdb_name = random.choice(members) if self.crop_chain else members[0]
        return AbData.from_input_path(
            Path(self.input_dir_path) / f"{pdb_name}.pkl",
            self.crop_chain,
            self.crop_length,
            )
