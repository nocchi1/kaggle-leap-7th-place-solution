from torch import nn

from src.model.models.base import BaseModel
from src.model.modules.positional_encoding import VerticalEncoding
from src.model.modules.resnet_block import ResNetBlock


class LEAPConv1D(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        block_num: int = 15,
        kernel_size: int = 5,
        multi_task: bool = False,
    ):
        super().__init__()
        if multi_task:
            out_dim = out_dim * 3

        activation = nn.ELU()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            activation,
        )
        self.pe = VerticalEncoding(hidden_dim, learnable=True)
        self.conv_blocks = nn.ModuleList()
        for i in range(block_num):
            if i >= (block_num // 2) and i < (block_num - 1):
                self.conv_blocks.append(
                    ResNetBlock(
                        hidden_dim,
                        hidden_dim,
                        activation=activation,
                        kernel_size=kernel_size,
                        inception=False,
                    )
                )
            else:
                self.conv_blocks.append(
                    ResNetBlock(
                        hidden_dim,
                        hidden_dim,
                        activation=activation,
                        kernel_size=kernel_size,
                        inception=True,
                    )
                )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = super().forward(x)
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pe(x)
        x = x.transpose(1, 2)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        out = self.head(x)
        return out.transpose(1, 2)


class GridPredConv1D(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        block_num: int = 15,
        kernel_size: int = 5,
    ):
        super().__init__()
        activation = nn.ELU()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            activation,
        )
        self.pe = VerticalEncoding(hidden_dim, learnable=True)
        self.conv_blocks = nn.ModuleList()
        for _ in range(block_num):
            self.conv_blocks.append(
                ResNetBlock(
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    kernel_size=kernel_size,
                    inception=False,
                )
            )
        self.conv_shrink = nn.Sequential(
            nn.Conv1d(hidden_dim, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            activation,
        )
        self.flatten = nn.Flatten()
        self.head = nn.Linear(60 * 32, out_dim)

    def forward(self, x):
        # x = super().forward(x) # The model can achieve good performance even without this feature
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pe(x)
        x = x.transpose(1, 2)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.conv_shrink(x)
        x = self.flatten(x)
        out = self.head(x)
        return out
