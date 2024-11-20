import os
from functools import lru_cache
from typing import Any, Dict

import torch
import torch.utils
import torch.utils.data
from transformers import PreTrainedTokenizer
from unicore.data import (
    AppendTokenDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RawArrayDataset,
    RightPadDataset,
    RightPadDataset2D,
)
from .atom_type_dataset import AtomTypeDataset
from .coord_pad_dataset import RightPadDatasetCoord
from .cropping_dataset import CroppingDataset
from .distance_dataset import DistanceDataset, EdgeTypeDataset
from .normalize_dataset import NormalizeDataset
from .conformer_sample_dataset import ConformerSampleDataset
from .key_dataset import KeyDataset
from .remove_hydrogen_dataset import RemoveHydrogenDataset
from .lmdb_dataset import LMDBDataset

from .base_datasets import FromNumpyDataset, TTADataset
from .uce_datasets import UCEBaseDataset
from .unimol_tokenizer import UniMolTokenizer


class UniMolTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 512,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        assert len(raw_data) < self.max_seq_len and len(raw_data) > 0
        return torch.tensor(self.tokenizer.encode(raw_data.tolist()), dtype=torch.long)


class UniMolDatasetCollectionBase:
    def __init__(self, configs, split: str = "train"):
        self.configs = configs
        self.seed = configs["seed"]
        self.split = split
        self.tokenizer = UniMolTokenizer()
        self.prepare_dataset()

    def PrependAndAppend(self, dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    def prepare_dataset(self):
        dataset = LMDBDataset(os.path.join(self.configs["data_dir"], self.split + ".lmdb"))
        smi_dataset = KeyDataset(dataset, "smiles")
        if self.split == self.configs.train_split:
            smi_dataset = KeyDataset(dataset, "smiles")
            sampled_dataset = ConformerSampleDataset(dataset, self.seed, "atoms", "coordinates")
            dataset = AtomTypeDataset(dataset, sampled_dataset)
        else:
            dataset = TTADataset(dataset, self.seed, "atoms", "coordinates", conf_size=self.configs["conf_size"])
            dataset = AtomTypeDataset(dataset, dataset)
            smi_dataset = KeyDataset(dataset, "smiles")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.configs["remove_hydrogen"],
            self.configs["remove_polar_hydrogen"],
        )
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.configs["max_atoms"])
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = UniMolTokenizeDataset(src_dataset, self.tokenizer, self.configs["max_seq_len"])
        self.src_dataset = self.PrependAndAppend(src_dataset, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        self.edge_type = EdgeTypeDataset(self.src_dataset, len(self.tokenizer))

        coord_dataset = KeyDataset(dataset, "coordinates")
        coord_dataset = FromNumpyDataset(coord_dataset)
        self.coord_dataset = self.PrependAndAppend(coord_dataset, 0.0, 0.0)
        self.distance_dataset = DistanceDataset(self.coord_dataset)
        self.smi_dataset = RawArrayDataset(smi_dataset)

    def get_datasets_union(self):
        return (
            self.src_dataset,
            self.coord_dataset,
            self.distance_dataset,
            self.edge_type,
            self.smi_dataset,
        )

    def get_datasets_dict(self):
        return {
            "src": self.src_dataset,
            "coord": self.coord_dataset,
            "distance": self.distance_dataset,
            "edge_type": self.edge_type,
        }

    def get_datasets_nested(self):
        return NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        self.src_dataset,
                        pad_idx=self.tokenizer.pad_token_id,
                    ),
                    "src_coord": RightPadDatasetCoord(
                        self.coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        self.distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        self.edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(self.smi_dataset),
            }
        )


class UniMolDatasetCollectionWithLabels(UniMolDatasetCollectionBase):
    """
    Dataset collection for UniMol fine-tuning only, plus labels.
    """

    def __init__(self, configs, split: str = "train"):
        super().__init__(configs, split)
        self.configs = configs
        self.seed = configs["seed"]
        self.split = split
        self.tokenizer = UniMolTokenizer()
        self.prepare_dataset()

    def prepare_dataset(self):
        dataset = LMDBDataset(os.path.join(self.configs["data_dir"], self.split + ".lmdb"))
        if self.split == self.configs.train_split:
            smi_dataset = KeyDataset(dataset, "smiles")
            cell_embedding_dataset = KeyDataset(dataset, "cell_embedding")
            self.cell_embedding_dataset = FromNumpyDataset(cell_embedding_dataset)

            target_dataset = KeyDataset(dataset, "target")
            control_dataset = KeyDataset(dataset, "control")

            self.target_dataset = FromNumpyDataset(target_dataset)
            self.control_dataset = FromNumpyDataset(control_dataset)

            sampled_dataset = ConformerSampleDataset(dataset, self.seed, "atoms", "coordinates")
            dataset = AtomTypeDataset(dataset, sampled_dataset)

        else:
            # get multiple conformers
            dataset = TTADataset(
                dataset,
                self.seed,
                "atoms",
                "coordinates",
                conf_size=self.configs["conf_size"],
                extra_keys=["target", "control", "cell_embedding"],
            )
            dataset = AtomTypeDataset(dataset, dataset)
            smi_dataset = KeyDataset(dataset, "smiles")

            # make sure the length of all datasets are the same
            cell_embedding_dataset = KeyDataset(dataset, "cell_embedding")
            self.cell_embedding_dataset = FromNumpyDataset(cell_embedding_dataset)
            target_dataset = KeyDataset(dataset, "target")
            control_dataset = KeyDataset(dataset, "control")
            self.target_dataset = FromNumpyDataset(target_dataset)
            self.control_dataset = FromNumpyDataset(control_dataset)

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.configs["remove_hydrogen"],
            self.configs["remove_polar_hydrogen"],
        )
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.configs["max_atoms"])
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")

        src_dataset = UniMolTokenizeDataset(src_dataset, self.tokenizer, self.configs["max_seq_len"])
        self.src_dataset = self.PrependAndAppend(src_dataset, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        self.edge_type = EdgeTypeDataset(self.src_dataset, len(self.tokenizer))

        coord_dataset = KeyDataset(dataset, "coordinates")
        coord_dataset = FromNumpyDataset(coord_dataset)
        self.coord_dataset = self.PrependAndAppend(coord_dataset, 0.0, 0.0)
        self.distance_dataset = DistanceDataset(self.coord_dataset)
        self.smi_dataset = RawArrayDataset(smi_dataset)

    def get_datasets_nested(self):
        return NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        self.src_dataset,
                        pad_idx=self.tokenizer.pad_token_id,
                    ),
                    "src_coord": RightPadDatasetCoord(
                        self.coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        self.distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        self.edge_type,
                        pad_idx=0,
                    ),
                    "src_cell_embedding": self.cell_embedding_dataset,
                },
                "label": {
                    "target": self.target_dataset,
                    "control": self.control_dataset,
                },
                "smi_name": RawArrayDataset(self.smi_dataset),
            }
        )


class DatasetCollectionWithLabels(UniMolDatasetCollectionBase):
    """
    Dataset collection for cureall models, plus labels.
    """

    def __init__(self, configs, split: str = "train"):
        super().__init__(configs, split)
        self.configs = configs
        self.seed = configs["seed"]
        self.split = split
        self.tokenizer = UniMolTokenizer()
        self.prepare_dataset()

    def prepare_dataset(self):
        dataset = LMDBDataset(os.path.join(self.configs["data_dir"], self.split + ".lmdb"))

        if self.split == self.configs.train_split:
            smi_dataset = KeyDataset(dataset, "smiles")

            # put extra items here, the ConformerSampleDataset will drop those keys
            target_dataset = KeyDataset(dataset, "target")
            control_dataset = KeyDataset(dataset, "control")
            self.uce_dataset = UCEBaseDataset(dataset, uce_padding_length=1536)

            self.target_dataset = FromNumpyDataset(target_dataset)
            self.control_dataset = FromNumpyDataset(control_dataset)

            sampled_dataset = ConformerSampleDataset(dataset, self.seed, "atoms", "coordinates")
            dataset = AtomTypeDataset(dataset, sampled_dataset)

        else:
            # get multiple conformers
            dataset = TTADataset(
                dataset,
                self.seed,
                "atoms",
                "coordinates",
                conf_size=self.configs["conf_size"],
                extra_keys=["target", "control", "cell_batch"],
            )

            dataset = AtomTypeDataset(dataset, dataset)
            smi_dataset = KeyDataset(dataset, "smiles")

            # make sure the length of all datasets are the same
            target_dataset = KeyDataset(dataset, "target")
            control_dataset = KeyDataset(dataset, "control")
            self.target_dataset = FromNumpyDataset(target_dataset)
            self.control_dataset = FromNumpyDataset(control_dataset)
            self.uce_dataset = UCEBaseDataset(dataset, uce_padding_length=1536)

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.configs["remove_hydrogen"],
            self.configs["remove_polar_hydrogen"],
        )
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.configs["max_atoms"])
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")

        src_dataset = UniMolTokenizeDataset(src_dataset, self.tokenizer, self.configs["max_seq_len"])
        self.src_dataset = self.PrependAndAppend(src_dataset, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        self.edge_type = EdgeTypeDataset(self.src_dataset, len(self.tokenizer))

        coord_dataset = KeyDataset(dataset, "coordinates")
        coord_dataset = FromNumpyDataset(coord_dataset)
        self.coord_dataset = self.PrependAndAppend(coord_dataset, 0.0, 0.0)
        self.distance_dataset = DistanceDataset(self.coord_dataset)
        self.smi_dataset = RawArrayDataset(smi_dataset)

    def get_datasets_nested(self):
        return NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        self.src_dataset,
                        pad_idx=self.tokenizer.pad_token_id,
                    ),
                    "src_coord": RightPadDatasetCoord(
                        self.coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        self.distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        self.edge_type,
                        pad_idx=0,
                    ),
                    "uce_batches": self.uce_dataset,
                },
                "label": {
                    "target": self.target_dataset,
                    "control": self.control_dataset,
                },
                "smi_name": RawArrayDataset(self.smi_dataset),
            }
        )
