import gc
import random
from typing import Literal

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
        hf_read_type: Literal["npy", "hdf5"] = "npy",
    ):
        self.y_dtype = torch.float if config.task_type == "main" else torch.long
        self.out_dim = config.out_dim * 3 if config.multi_task else config.out_dim
        self.hf_path = config.add_path / "huggingface"

        if X is not None or y is not None:
            self.ids = ids
            self.X = X
            self.y = y
        elif config.run_mode == "hf" and hf_read_type == "npy":
            self.batch_file_size = config.batch_file_size
            self.all_yms = self.get_all_yms()
            self.update()
        # HDF5の読み込みが遅い
        elif config.run_mode == "hf" and hf_read_type == "hdf5":
            self.h5_file = h5py.File(self.hf_path / "hf_data.h5", "r")
            self.X = self.h5_file["X"]
            self.y = self.h5_file["y"]

    def __getitem__(self, idx: int):
        data = [torch.tensor(self.X[idx], dtype=torch.float)]
        if self.y is not None:
            data.append(torch.tensor(self.y[idx, :, : self.out_dim], dtype=self.y_dtype))
        return data

    def __len__(self):
        return len(self.X)

    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()

    def update(self):
        extract_yms = random.sample(self.all_yms, min(self.batch_file_size, len(self.all_yms)))
        self.all_yms = [ym for ym in self.all_yms if ym not in extract_yms]
        if len(self.all_yms) == 0:
            self.all_yms = self.get_all_yms()

        X, y = [], []
        for ym in extract_yms:
            X.append(np.load(self.hf_path / f"X_{ym}.npy"))
            y.append(np.load(self.hf_path / f"y_{ym}.npy")[:, :, : self.out_dim])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        self.X, self.y = X, y
        del X, y
        gc.collect()

    def get_all_yms(self):
        npy_files = list(self.hf_path.glob("X_*.npy"))
        all_yms = [file.stem.split("_")[1] for file in npy_files]
        return all_yms


def get_dataloader(
    config: DictConfig,
    ids: np.ndarray | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    hf_read_type: Literal["npy", "hdf5"] = "npy",
    is_train: bool = True,
) -> DataLoader:
    dataset = LEAPDataset(config, ids, X, y, hf_read_type)
    if is_train:
        data_loader = DataLoader(
            dataset, batch_size=config.train_batch, shuffle=True, pin_memory=True, drop_last=True
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=config.eval_batch, shuffle=False, pin_memory=True, drop_last=False
        )
    return data_loader
