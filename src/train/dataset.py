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
        y_mt: np.ndarray | None = None,
        y_dtype: torch.dtype = torch.float,
        load_from_hdf5: bool = False,
        hdf5_path: PosixPath | None = None,
    ):
        # HFデータを使用するときはHDF5からデータを読み込む
        if load_from_hdf5:
            self.h5_file = h5py.File(hdf5_path, "r")
            self.X = self.h5_file["X"]
            self.y = self.h5_file["y"]
            if config.multi_task:
                self.y_mt = self.h5_file["y_mt"]
        else:
            self.ids = ids  # 結果を保存するときに参照するために保持
            self.X = X
            self.y = y
            self.y_mt = y_mt
        self.y_dtype = y_dtype

    def __getitem__(self, idx: int):
        data = [torch.tensor(self.X[idx], dtype=torch.float)]
        if self.y is not None:
            data.append(torch.tensor(self.y[idx], dtype=self.y_dtype))
        if self.y_mt is not None:
            data.append(torch.tensor(self.y_mt[idx], dtype=self.y_dtype))
        return data

    def __len__(self):
        return len(self.X)


def get_dataloader(
    config: DictConfig,
    ids: np.ndarray,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    y_mt: np.ndarray | None = None,
    is_train: bool = True,
) -> DataLoader:
    y_dtype = torch.float if config.task_type == "main" else torch.long
    load_from_hdf5 = True if config.run_mode == "hf" else False
    hdf5_path = config.add_path / "huggingface"
    dataset = LEAPDataset(config, ids, X, y, y_mt, y_dtype, load_from_hdf5, hdf5_path)

    if is_train:
        data_loader = DataLoader(
            dataset, batch_size=config.train_batch, shuffle=True, pin_memory=True, drop_last=True
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=config.eval_batch, shuffle=False, pin_memory=True, drop_last=False
        )
    return data_loader
