# ruff: noqa: B008
import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_linear = nn.Linear(in_dim, hidden_dim)
        self.k_linear = nn.Linear(in_dim, hidden_dim)
        self.v_linear = nn.Linear(in_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, query):
        b_size, q_len, v_len = query.size(0), query.size(1), value.size(1)
        q = self.q_linear(query)
        k = self.k_linear(value)
        v = self.v_linear(value)
        q = q.view(b_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b_size, v_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b_size, v_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        qk = F.softmax(torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim), dim=-1)
        x = torch.matmul(qk, v).permute(0, 2, 1, 3).reshape(b_size, q_len, -1)
        x = self.out_linear(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)
