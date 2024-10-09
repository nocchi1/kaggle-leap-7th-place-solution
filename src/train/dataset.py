from pathlib import PosixPath

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class LEAPDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        ids: np.ndarray | None = None,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        if config.run_mode == "hf":
            h5_path = config.add_path / "huggingface" / "hoge.h5"
            self.h5_file = h5py.File(h5_path, "r")
            self.X = self.h5_file["X"]
            self.y = self.h5_file["y"]
        else:
            self.ids = ids
            self.X = X
            self.y = y
        self.y_dtype = torch.float if config.task_type == "main" else torch.long

    def __getitem__(self, idx: int):
        data = [torch.tensor(self.X[idx], dtype=torch.float)]
        if self.y is not None:
            data.append(torch.tensor(self.y[idx], dtype=self.y_dtype))
        return data

    def __len__(self):
        return len(self.X)

    # TODO: テストする
    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()


def get_dataloader(
    config: DictConfig,
    ids: np.ndarray | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    is_train: bool = True,
) -> DataLoader:
    dataset = LEAPDataset(config, ids, X, y)
    if is_train:
        data_loader = DataLoader(
            dataset, batch_size=config.train_batch, shuffle=True, pin_memory=True, drop_last=True
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=config.eval_batch, shuffle=False, pin_memory=True, drop_last=False
        )
    return data_loader
