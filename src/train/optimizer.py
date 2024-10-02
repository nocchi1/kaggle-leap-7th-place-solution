from typing import Tuple

from torch import nn
from torch.optim import Adam, AdamW


def get_optimizer(
    model: nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
):
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        raise ValueError(f"Invalid optimizer_type: {optimizer_type}")
    return optimizer
