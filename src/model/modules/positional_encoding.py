from pathlib import Path, PosixPath

import numpy as np
import polars as pl
import torch
from sklearn.metrics.pairwise import haversine_distances
from torch import nn


class VerticalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 60, learnable: bool = False):
        super().__init__()
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0, hidden_dim, 2).float() * 2 / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class HorizontalEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_path: PosixPath = Path("../data/input"),
        from_coord: bool = False,
    ):
        super().__init__()
        self.add_path = input_path / "additional"
        self.from_coord = from_coord
        self.grid_nunq = 384

        if from_coord:
            self.embedding = self.load_grid_embedding()
            self.fc = nn.Linear(self.grid_nunq, hidden_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.embedding = nn.Embedding(self.grid_nunq, hidden_dim)

    def load_grid_embedding(self):
        grid_feat = pl.read_parquet(self.add_path / "grid_feat.parquet")
        lat_lon = grid_feat.sort("grid_id").select("lat", "lon").to_numpy()
        lat_lon = np.radians(lat_lon)
        distance = haversine_distances(lat_lon)
        embedding = distance / distance.max()
        return torch.tensor(embedding, dtype=torch.float32)

    def forward(self, x, g_id):
        if self.from_coord:
            emb = self.embedding[g_id]
            emb = self.fc(emb)
            emb = self.bn(emb)
        else:
            emb = self.embedding(g_id)
        x = x + emb.unsqueeze(1)
        return x


class VHPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 60, learnable: bool = False):
        super().__init__()
        self.v_enc = VerticalEncoding(hidden_dim, max_length, learnable=learnable)
        self.h_enc = HorizontalEncoding(hidden_dim, from_coord=False)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, g_id):
        x = self.v_enc(x)
        x = self.h_enc(x, g_id)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return x
