import torch
from omegaconf import DictConfig
from torch import nn

from src.utils.constant import VERTICAL_TARGET_COLS


class LEAPLoss(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        if config.loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif config.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif config.loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.main_weight = 0.80

    def forward(self, y_pred, y_true):
        bs = y_pred.size(0)
        if y_true.size(-1) == self.config.out_dim * 3:  # When performing multi-task learning
            loss = self.loss_fn(y_pred, y_true)
            loss_m_v = loss[:, :, : len(VERTICAL_TARGET_COLS)].reshape(bs, -1)
            loss_m_s = loss[:, :, len(VERTICAL_TARGET_COLS) : self.config.out_dim].mean(dim=1)
            loss_s_v = torch.cat(
                [
                    loss[:, :, self.config.out_dim : self.config.out_dim + len(VERTICAL_TARGET_COLS)].reshape(bs, -1),
                    loss[
                        :,
                        :,
                        self.config.out_dim * 2 : self.config.out_dim * 2 + len(VERTICAL_TARGET_COLS),
                    ].reshape(bs, -1),
                ],
                dim=1,
            )
            loss_s_s = torch.cat(
                [
                    loss[
                        :,
                        :,
                        self.config.out_dim + len(VERTICAL_TARGET_COLS) : self.config.out_dim * 2,
                    ].mean(dim=1),
                    loss[:, :, self.config.out_dim * 2 + len(VERTICAL_TARGET_COLS) :].mean(dim=1),
                ],
                dim=1,
            )
            loss = torch.cat([loss_m_v, loss_m_s], dim=-1).mean() * self.main_weight + torch.cat(
                [loss_s_v, loss_s_s], dim=-1
            ).mean() * (1 - self.main_weight)

        elif y_true.size(-1) == self.config.out_dim:
            y_pred = y_pred[:, :, : self.config.out_dim]
            loss = self.loss_fn(y_pred, y_true)
            loss_v = loss[:, :, : len(VERTICAL_TARGET_COLS)].reshape(bs, -1)
            loss_s = loss[:, :, len(VERTICAL_TARGET_COLS) :].mean(dim=1)
            loss = torch.cat([loss_v, loss_s], dim=-1).mean()

        else:
            raise ValueError("Mismatched y_true and y_pred size")

        return loss
