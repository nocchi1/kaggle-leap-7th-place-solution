from torch import nn

from src.model.models.base import BaseModel
from src.model.modules.conv_extractor import ConvExtractor
from src.model.modules.squeezeformer_block import ResidualConnectionModule, SqueezeformerBlock


class LEAPSqueezeformer(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        block_num: int = 10,
        kernel_size: int = 5,
        multi_task: bool = False,
    ):
        super().__init__()
        if multi_task:
            out_dim = out_dim * 3

        self.conv_extractor = ConvExtractor(in_dim, hidden_dim)
        # self.pe = VerticalEncoding(hidden_dim, learnable=True)

        self.squeeze_blocks = nn.ModuleList()
        dropout_p = 0.0
        for idx in range(block_num):
            if idx < (block_num // 2 - 1) or idx == (block_num - 1):
                self.squeeze_blocks.append(
                    SqueezeformerBlock(
                        encoder_dim=hidden_dim,
                        feed_forward_dropout_p=dropout_p,
                        attention_dropout_p=dropout_p,
                        conv_dropout_p=dropout_p,
                        conv_kernel_size=kernel_size,
                        half_step_residual=False,
                    )
                )
            else:
                self.squeeze_blocks.append(
                    ResidualConnectionModule(
                        module=SqueezeformerBlock(
                            encoder_dim=hidden_dim,
                            feed_forward_dropout_p=dropout_p,
                            attention_dropout_p=dropout_p,
                            conv_dropout_p=dropout_p,
                            conv_kernel_size=kernel_size,
                            half_step_residual=False,
                        )
                    )
                )

        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.LayerNorm([60, 64]), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, x):
        x = super().forward(x)
        x = self.conv_extractor(x)
        # x = self.pe(x)
        for layer in self.squeeze_blocks:
            x = layer(x)
        x = self.head(x)
        return x
