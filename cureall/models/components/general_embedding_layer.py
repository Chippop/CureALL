import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralEmbedding(torch.nn.Module):
    """
    General layer for embedding.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1, *kwargs):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.out_proj(x)
