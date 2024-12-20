import os
import random
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_csv_to_parquet(file_path: Path, delete_csv: bool = False):
    parent_path = file_path.parent
    file_name = file_path.stem
    df = pl.read_csv(file_path)
    df.write_parquet(parent_path / f"{file_name}.parquet")
    if delete_csv:
        file_path.unlink()


def clean_message(message: str):
    return re.sub(r"\s+", " ", message).strip()
