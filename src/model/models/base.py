import torch
import torch.nn.functional as F
from torch import nn

from src.utils.constant import ADDITIONAL_VERTICAL_INPUT_COLS, VERTICAL_INPUT_COLS


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_vertical = len(VERTICAL_INPUT_COLS) + len(ADDITIONAL_VERTICAL_INPUT_COLS)

    def forward(self, x):
        v_x = x[:, :, : self.n_vertical]
        diff_feat = self.calc_diff_feats(v_x)
        x = torch.cat([x, diff_feat], dim=-1)
        return x

    def calc_lag_features(self, x):
        feat = []
        # Calculate lag features for both forward and backward directions
        for t in [1, 2, 3, 4, 5]:
            x_f = torch.cat(
                [torch.zeros(x.size(0), t, x.size(2), device=x.device), x[:, : (60 - t), :]], dim=1
            )
            feat.append(x_f)
            x_b = torch.cat(
                [x[:, t:, :], torch.zeros(x.size(0), t, x.size(2), device=x.device)], dim=1
            )
            feat.append(x_b)
        feat = torch.cat(feat, dim=-1)
        return feat

    def calc_diff_feats(self, x):
        # Forward direction diff
        forward_diff = torch.diff(x, dim=1)
        forward_diff = torch.cat([torch.zeros_like(x[:, 0, :]).unsqueeze(1), forward_diff], dim=1)
        # Forward direction second diff
        forward_diff2 = torch.diff(forward_diff, dim=1)
        forward_diff2 = torch.cat([torch.zeros_like(x[:, 0, :]).unsqueeze(1), forward_diff2], dim=1)
        # Backward direction diff
        backward_diff = torch.diff(x.flip(1), dim=1).flip(1)
        backward_diff = torch.cat([backward_diff, torch.zeros_like(x[:, 0, :]).unsqueeze(1)], dim=1)
        # Backward direction second diff
        backward_diff2 = torch.diff(backward_diff.flip(1), dim=1).flip(1)
        backward_diff2 = torch.cat(
            [backward_diff2, torch.zeros_like(x[:, 0, :]).unsqueeze(1)], dim=1
        )
        feat = torch.cat([forward_diff, forward_diff2, backward_diff, backward_diff2], dim=-1)
        return feat

    def calc_moving_feats(self, x):
        feat = []
        x = x.transpose(1, 2)  # (batch, hidden, sequence)
        # Moving statistics
        for w in [3, 5, 7, 15, 29]:
            # Mean, max, min
            feat.append(F.avg_pool1d(x, w, stride=1, padding=w // 2))
            feat.append(F.max_pool1d(x, w, stride=1, padding=w // 2))
            feat.append(-1 * F.max_pool1d(-1 * x, w, stride=1, padding=w // 2))
            # Standard deviation
            x_mean = F.avg_pool1d(x, w, stride=1, padding=w // 2)
            x_diff = (x - x_mean) ** 2
            x_std = F.avg_pool1d(x_diff, w, stride=1, padding=w // 2).sqrt()
            x_std = torch.where(torch.isinf(x_std) | torch.isnan(x_std), 0, x_std)
            feat.append(x_std)

        # Global statistics
        feat.append(x.mean(dim=2, keepdim=True).repeat(1, 1, x.size(2)))
        feat.append(x.max(dim=2, keepdim=True).values.repeat(1, 1, x.size(2)))
        feat.append(x.min(dim=2, keepdim=True).values.repeat(1, 1, x.size(2)))
        x_std = x.std(dim=2, keepdim=True).repeat(1, 1, x.size(2))
        x_std = torch.where(torch.isinf(x_std) | torch.isnan(x_std), 0, x_std)
        feat.append(x_std)

        feat = torch.cat(feat, dim=1)
        feat = feat.transpose(1, 2)
        return feat
