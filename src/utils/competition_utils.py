from pathlib import PosixPath
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import r2_score

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


def shrink_memory(
    apply_df: pl.DataFrame, refer_df: pl.DataFrame | None = None, min_th: float = 1e-37
) -> pl.DataFrame:
    if "sample_id" in apply_df.columns and apply_df["sample_id"].dtype == pl.Utf8:
        apply_df = apply_df.with_columns(
            sample_id=pl.col("sample_id").map_elements(
                lambda x: int(x.split("_")[1]), return_dtype=pl.Int32
            )
        )

    num_cols = [col for col in apply_df.columns if apply_df[col].dtype == pl.Float64]
    exprs = []
    if refer_df is None:
        for col in num_cols:
            abs_min_val = apply_df[col].abs().min()
            if abs_min_val > min_th:
                exprs.append(pl.col(col).cast(pl.Float32))
    else:
        for col in num_cols:
            if col in refer_df.columns:
                dtype = refer_df[col].dtype
                exprs.append(pl.col(col).cast(dtype))
    apply_df = apply_df.with_columns(exprs)
    return apply_df


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
    if config.task_type == "main" and config.use_grid_feat:
        for col in GRID_SCALER_INPUT_COLS:
            input_cols.append(col)

    target_cols = []
    for col in VERTICAL_TARGET_COLS:
        target_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in SCALER_TARGET_COLS:
        target_cols.append(col)
    # Remove columns where weight=0 and those applied with pp
    target_cols = [
        col for col in target_cols if col not in ZERO_WEIGHT_TARGET_COLS + PP_TARGET_COLS
    ]
    return input_cols, target_cols


def clipping_input(
    train_df: pl.DataFrame | None,
    test_df: pl.DataFrame,
    input_cols: list[str],
    clip_dict: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    if train_df is None and clip_dict is None:
        raise ValueError("train_df or clip_dict is required.")

    exprs = []
    clip_dict_ = {} if clip_dict is None else clip_dict
    for col in input_cols:
        if train_df is not None:
            min_val, max_val = train_df[col].min(), train_df[col].max()
            clip_dict_[col] = (min_val, max_val)
        else:
            min_val, max_val = clip_dict[col]
        exprs.append(pl.col(col).clip(min_val, max_val).alias(col))
    test_df = test_df.with_columns(exprs)
    return test_df, clip_dict_


def remove_duplicate_records(target_df: pl.DataFrame, refer_df: pl.DataFrame) -> pl.DataFrame:
    """
    Function to detect and remove duplicate rows between DataFrames with different data types (e.g., Float32 - Float64)

    args:
        target_df: DataFrame in which duplicate rows will be removed, based on refer_df
        refer_df: DataFrame used as a reference to check for duplicates
    """
    use_cols = (
        [f"state_t_{i}" for i in range(60)]
        + [f"state_v_{i}" for i in range(60)]
        + [f"state_u_{i}" for i in range(60)]
    )
    # Convert to integers for duplicate detection
    target_df = target_df.with_columns(
        [(pl.col(col) * 10000).cast(pl.Int32).alias(f"{col}_int") for col in use_cols]
    )
    target_df = target_df.with_columns(target_flag=pl.lit(1))
    refer_df = refer_df.with_columns(
        [(pl.col(col) * 10000).cast(pl.Int32).alias(f"{col}_int") for col in use_cols]
    )
    refer_df = refer_df.with_columns(target_flag=pl.lit(0))

    target_df = pl.concat([target_df, refer_df], how="diagonal")
    use_int_cols = [f"{col}_int" for col in use_cols]
    target_df = target_df.unique(subset=use_int_cols, keep="none")
    target_df = target_df.filter(pl.col("target_flag") == 1)
    target_df = target_df.drop(use_int_cols + ["target_flag"])
    return target_df


def evaluate_metric(
    y_pred: np.ndarray, y_true: np.ndarray, eval_idx: list[int] | None = None
) -> float | tuple[float, list[float]]:
    target_num = 368
    total_score = 0
    indiv_scores = []

    for i in range(y_pred.shape[1]):
        if i not in eval_idx:
            total_score += 1
            indiv_scores.append(1)
        else:
            score = r2_score(y_true[:, i], y_pred[:, i], force_finite=True)
            total_score += score
            indiv_scores.append(score)

    eval_num = len(indiv_scores)
    # y_pred内に存在しないカラムは1として計算する -> sub_factorが0のカラム, 後処理を適用するカラム
    if target_num - eval_num > 0:
        total_score = (total_score * eval_num + (target_num - eval_num)) / target_num
    return total_score, indiv_scores
