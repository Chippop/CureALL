# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from cureall.models.components import UCETransformerModel

logger = logging.getLogger(__name__)


class UCEConfig(PretrainedConfig):
    model_type = "uce"

    def __init__(
        self,
        n_layers: int = 33,
        n_heads: int = 20,
        embed_size: int = 1280,
        ffn_size: int = 5120,
        token_dim: int = 5120,
        dropout: float = 0.1,
        output_dim: int = 1280,
        pretrained_model_name_or_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_size = embed_size
        self.ffn_size = ffn_size
        self.token_dim = token_dim
        self.dropout = dropout
        self.output_dim = output_dim
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    @classmethod
    def from_dict(cls, json_object):
        return UCEConfig(**json_object)

    def to_dict(self):
        output = super().to_dict()
        return output


class UCEModel(PreTrainedModel):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__(configs)
        self.configs = configs
        self.model = UCETransformerModel(
            n_layers=configs.n_layers,
            n_heads=configs.n_heads,
            d_model=configs.embed_size,
            ffn_size=configs.ffn_size,
            token_dim=configs.token_dim,
            dropout=configs.dropout,
            output_dim=configs.output_dim,
        )
        self.init_modules()

    def init_modules(self):
        empty_pe = torch.zeros(145469, 5120)
        empty_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
        checkpoint = torch.load(self.configs.pretrained_model_name_or_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)

    def forward(self, batch_seq: torch.Tensor, mask: torch.Tensor, **kwargs):
        batch_seq = batch_seq.permute(1, 0)
        batch_seq = self.model.pe_embedding(batch_seq.long())
        batch_seq = nn.functional.normalize(batch_seq, dim=2)
        _, embedding = self.model(batch_seq, mask)

        return embedding

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()
