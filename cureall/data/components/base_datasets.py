# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import List

import numpy as np
import torch
from unicore.data import BaseWrapperDataset


class FromNumpyDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return torch.from_numpy(self.dataset[idx]).to(torch.float32)


class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, extra_keys: List[str] = ["target"], conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.conf_size = conf_size
        self.extra_keys = extra_keys
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16) 
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        smi = self.dataset[smi_idx]["smiles"]
        base_dict = {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "smiles": smi,
        }
        for key in self.extra_keys:
            base_dict.update({key: self.dataset[smi_idx].get(key, None)})
            base_dict[key] = self.dataset[smi_idx].get(key, None)

        return base_dict

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
