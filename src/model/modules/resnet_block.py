import torch
from torch import nn

from src.torch.models.modules import SEBlock1D


class InceptionModule(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv_2_1 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv_2_2 = nn.Conv1d(out_dim // 4, out_dim // 4, kernel_size=3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv_3_2 = nn.Conv1d(out_dim // 4, out_dim // 4, kernel_size=5, stride=1, padding=2)
        self.conv_4_1 = nn.Conv1d(in_dim, out_dim // 4, kernel_size=1, stride=1, padding=0)
        self.conv_4_2 = nn.Conv1d(out_dim // 4, out_dim // 4, kernel_size=7, stride=1, padding=3)
        self.bn_1 = nn.BatchNorm1d(out_dim)
        self.conv_5 = nn.Conv1d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2_1(x)
        x_2 = self.conv_2_2(x_2)
        x_3 = self.conv_3_1(x)
        x_3 = self.conv_3_2(x_3)
        x_4 = self.conv_4_1(x)
        x_4 = self.conv_4_2(x_4)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x = self.bn_1(x)
        x = self.conv_5(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = nn.ELU(), kernel_size: int = 5, dropout: float = 0.0, inception: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_skip = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        if inception:
            self.conv_1 = InceptionModule(in_dim, out_dim)
            self.conv_2 = InceptionModule(out_dim, out_dim)
        else:
            self.conv_1 = nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
            self.conv_2 = nn.Conv1d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            )
        self.se_block = SEBlock1D(channel=out_dim)
        self.bn_1 = nn.BatchNorm1d(out_dim)
        self.bn_2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = activation

    def forward(self, x):
        x_s = x if self.in_dim == self.out_dim else self.conv_skip(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.se_block(x)
        x = self.bn_2(x)
        x = x + x_s
        x = self.act(x)
        x = self.dropout(x)
        return x
