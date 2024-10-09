import torch
from torch import nn

from src.model.models.base import BaseModel


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        dropout: float = 0.00,
        residual: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.residual = residual
        if self.residual and in_channel != out_channel:
            self.res_conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

        self.conv1 = nn.Conv1d(
            in_channel, out_channel, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.conv2 = nn.Conv1d(
            out_channel, out_channel, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        x_s = x
        if self.residual and self.in_channel != self.out_channel:
            x_s = self.res_conv(x_s)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn1(x)
        if self.residual:
            x = x + x_s
        x = self.relu(x)
        # x = self.dropout(x)
        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        dropout: float = 0.00,
        residual: bool = False,
        no_down: bool = False,
    ):
        super().__init__()
        self.no_down = no_down
        # self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channel, out_channel, kernel_size, dropout, residual)

    def forward(self, x):
        if not self.no_down:
            x = self.pooling(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        dropout: float = 0.00,
        residual: bool = False,
        no_up: bool = False,
    ):
        super().__init__()
        self.no_up = no_up
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.double_conv = DoubleConv(in_channel, out_channel, kernel_size, dropout, residual)
        self.batch_norm = nn.BatchNorm1d(in_channel)

    def forward(self, x1, x2):
        if not self.no_up:
            x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.batch_norm(x)
        x = self.double_conv(x)
        return x


class UNet1D(BaseModel):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        max_hidden_channel: int = 512,
        kernel_size: int = 3,
        dropout: float = 0.00,
        residual: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channel, max_hidden_channel // 8, kernel_size=1, padding=0),
            nn.BatchNorm1d(max_hidden_channel // 8),
            nn.CELU(alpha=1.0),
        )
        self.dconv = DoubleConv(
            max_hidden_channel // 8, max_hidden_channel // 4, kernel_size, dropout, residual
        )
        self.down1 = Down(
            max_hidden_channel // 4, max_hidden_channel // 2, kernel_size, dropout, residual
        )
        self.down2 = Down(
            max_hidden_channel // 2, max_hidden_channel, kernel_size, dropout, residual
        )
        self.down3 = Down(
            max_hidden_channel, max_hidden_channel, kernel_size, dropout, residual, no_down=True
        )
        self.down4 = Down(
            max_hidden_channel, max_hidden_channel, kernel_size, dropout, residual, no_down=True
        )
        self.up1 = Up(
            max_hidden_channel * 2, max_hidden_channel, kernel_size, dropout, residual, no_up=True
        )
        self.up2 = Up(
            max_hidden_channel * 2, max_hidden_channel, kernel_size, dropout, residual, no_up=True
        )
        self.up3 = Up(
            max_hidden_channel + max_hidden_channel // 2,
            max_hidden_channel // 2,
            kernel_size,
            dropout,
            residual,
        )
        self.up4 = Up(
            max_hidden_channel // 2 + max_hidden_channel // 4,
            max_hidden_channel // 4,
            kernel_size,
            dropout,
            residual,
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(max_hidden_channel // 4, 32, kernel_size=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, out_channel, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = super().forward(x)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        # Encoder
        x1 = self.dconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        # Decode
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return x
