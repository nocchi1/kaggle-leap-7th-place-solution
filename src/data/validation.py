from pathlib import PosixPath
from typing import Literal

import numpy as np
import polars as pl
from omegaconf import DictConfig


def split_validation(config: DictConfig, train_df: pl.DataFrame) -> pl.DataFrame:
    split_method = "shared" if config.shared_valid else "timeseries"
    train_df = split_holdout(
        train_df,
        split_method=split_method,
        valid_ratio=config.valid_ratio,
        valid_dir=config.input_path,
    )
    return train_df


def split_holdout(
    train_df: pl.DataFrame,
    split_method: Literal["random", "timeseries", "shared"],
    valid_ratio: float | None = None,
    valid_dir: PosixPath | None = None,
):
    if split_method == "random":
        train_df = train_df.with_columns(
            fold=pl.when(pl.Series(np.random.rand(len(train_df))) < valid_ratio)
            .then(0)
            .otherwise(1)
        )
    elif split_method == "timeseries":
        time_ids = sorted(train_df["time_id"].unique())
        valid_start = int(len(time_ids) * (1 - valid_ratio))
        valid_ids = time_ids[valid_start:]
        train_df = train_df.with_columns(
            fold=(
                pl.when(pl.col("time_id").is_in(valid_ids))
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .cast(pl.Int8)
            )
        )
    elif split_method == "shared":  # Use valid_df shared within the team
        valid_path = valid_dir / "18_pp.parquet"
        if valid_path.exists():
            valid_df = pl.read_parquet(valid_path)
        else:
            # Match the data types of valid_df with train_df
            valid_df = pl.read_parquet(valid_dir / "18.parquet")
            exprs = []
            for col in train_df.columns:
                if col in valid_df.columns:
                    exprs.append(pl.col(col).cast(train_df[col].dtype))
            valid_df = valid_df.with_columns(exprs)
            valid_df.write_parquet(valid_dir / "18_pp.parquet")

        valid_df = valid_df.with_columns(fold=pl.lit(0).cast(pl.Int8))
        train_df = train_df.with_columns(fold=pl.lit(1).cast(pl.Int8))
        # grid_id, time_id do not exist in valid
        train_df = pl.concat([train_df, valid_df], how="diagonal")
    else:
        raise ValueError(f"Invalid split_method: {split_method}")
    return train_df


def split_cross_validation(train_df: pl.DataFrame, n_splits: int) -> pl.DataFrame:
    time_ids = sorted(train_df["time_id"].unique())
    n = len(time_ids)
    bs = n // n_splits + 1
    train_df = train_df.with_columns(fold=pl.lit(-1).cast(pl.Int8))
    for fold in range(n_splits):
        fold_ids = time_ids[fold * bs : (fold + 1) * bs]
        train_df = train_df.with_columns(
            pl.when((pl.col("fold") == -1) & (pl.col("time_id").is_in(fold_ids)))
            .then(fold)
            .otherwise(pl.col("fold"))
            .alias("fold")
        )
    return train_df
