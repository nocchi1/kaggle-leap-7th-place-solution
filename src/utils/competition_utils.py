from pathlib import PosixPath

import polars as pl
from omegaconf import DictConfig

from src.utils.constant import (
    ADDITIONAL_VERTICAL_INPUT_COLS,
    GRID_SCALER_INPUT_COLS,
    PP_TARGET_COLS,
    SCALER_INPUT_COLS,
    SCALER_TARGET_COLS,
    VERTICAL_INPUT_COLS,
    VERTICAL_TARGET_COLS,
    ZERO_WEIGHT_TARGET_COLS,
)


def shrink_memory(train_df: pl.DataFrame, test_df: pl.DataFrame, min_th: float = 1e-37) -> tuple[pl.DataFrame, pl.DataFrame]:
    sample_id_expr = pl.col("sample_id").map_elements(lambda x: int(x.split("_")[1]), return_dtype=pl.Int32)
    train_exprs, test_exprs = [sample_id_expr], [sample_id_expr]
    for col in [col for col in train_df.columns if col != "sample_id"]:
        abs_min_val = train_df[col].abs().min()
        if abs_min_val > min_th: # Convert to FP32 only for values without the risk of underflow
            train_exprs.append(pl.col(col).cast(pl.Float32))
            if col in test_df.columns:
                test_exprs.append(pl.col(col).cast(pl.Float32))
    train_df = train_df.with_columns(train_exprs)
    test_df = test_df.with_columns(test_exprs)
    return train_df, test_df


def get_sub_factor(input_path: PosixPath, old: bool = False) -> dict[str, float]:
    suffix = "_old" if old else ""
    sample_df = pl.read_parquet(input_path / f"sample_submission{suffix}.parquet", n_rows=1)
    factor_dict = dict(zip(sample_df.columns[1:], sample_df.to_numpy()[0][1:]))
    return factor_dict


def multiply_old_factor(input_path: PosixPath, train_df: pl.DataFrame) -> pl.DataFrame:
    factor_dict = get_sub_factor(input_path, old=True)
    exprs = []
    for col, factor in factor_dict.items():
        exprs.append(pl.col(col) * factor)
    train_df = train_df.with_columns(exprs)
    return train_df


def get_io_columns(config: DictConfig) -> tuple[list[str], list[str]]:
    input_cols = []
    for col in VERTICAL_INPUT_COLS:
        input_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in ADDITIONAL_VERTICAL_INPUT_COLS:
        input_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in SCALER_INPUT_COLS:
        input_cols.append(col)
    if config.task_type == 'main' and config.use_grid_feat:
        for col in GRID_SCALER_INPUT_COLS:
            input_cols.append(col)

    target_cols = []
    for col in VERTICAL_TARGET_COLS:
        target_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in SCALER_TARGET_COLS:
        target_cols.append(col)
    # Remove columns where weight=0 and those applied with pp
    target_cols = [col for col in target_cols if col not in ZERO_WEIGHT_TARGET_COLS + PP_TARGET_COLS]
    return input_cols, target_cols
