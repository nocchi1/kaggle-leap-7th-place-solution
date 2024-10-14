import itertools
from typing import Literal

import numpy as np
import polars as pl

from src.utils.competition_utils import get_io_columns
from src.utils.constant import (
    ADDITIONAL_VERTICAL_INPUT_COLS,
    GRID_SCALER_INPUT_COLS,
    SCALER_INPUT_COLS,
    SCALER_TARGET_COLS,
    VERTICAL_INPUT_COLS,
    VERTICAL_TARGET_COLS,
)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.input_cols, self.target_cols = get_io_columns(config)

    def scaling(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        train_df = self._input_scaling(train_df, self.config.input_scale_method, compute_stats=True)
        test_df = self._input_scaling(test_df, self.config.input_scale_method, compute_stats=False)
        if self.config.task_type == "main":
            train_df = self._target_scaling(train_df, self.config.target_scale_method, compute_stats=True)

        train_df = train_df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        test_df = test_df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return train_df, test_df

    def _input_scaling(
        self,
        df: pl.DataFrame,
        scale_method: str = "standard",
        min_denominator: float = 1e-8,
        compute_stats: bool = True,
    ):
        exprs = []
        if compute_stats:
            x_numerators, x_denominators = [], []
            for col in self.input_cols:
                if scale_method == "standard":
                    numerator = df[col].mean()
                    denominator = df[col].std()
                elif scale_method == "minmax":
                    numerator = df[col].min()
                    denominator = df[col].max() - df[col].min()
                elif scale_method == "robust":
                    numerator = df[col].median()
                    denominator = df[col].quantile(0.75) - df[col].quantile(0.25)
                # Ref: https://github.com/leap-stc/ClimSim/blob/671db93cac4df30715628e4976f9b9f17a9b4ec6/climsim_utils/data_utils.py#L808
                elif scale_method == "host":
                    numerator = df[col].mean()
                    denominator = df[col].max() - df[col].min()

                denominator = np.maximum(denominator, min_denominator)
                exprs.append(((pl.col(col) - numerator) / denominator).alias(col))
                x_numerators.append(numerator)
                x_denominators.append(denominator)

            x_numerators = np.array(x_numerators).reshape(1, -1)
            x_denominators = np.array(x_denominators).reshape(1, -1)
            np.save(self.config.output_path / f"x_numerators_{scale_method}.npy", x_numerators)
            np.save(self.config.output_path / f"x_denominators_{scale_method}.npy", x_denominators)
        else:
            x_numerators = np.load(self.config.output_path / f"x_numerators_{scale_method}.npy")
            x_denominators = np.load(self.config.output_path / f"x_denominators_{scale_method}.npy")
            for i, col in enumerate(self.input_cols):
                exprs.append(((pl.col(col) - x_numerators[0, i]) / x_denominators[0, i]).alias(col))

        df = df.with_columns(exprs)
        return df

    def _target_scaling(
        self,
        df: pl.DataFrame,
        scale_method: str = "standard_y2",
        min_denominator: float = 1e-8,
        compute_stats: bool = True,
    ):
        exprs = []
        if compute_stats:
            y_numerators, y_denominators = [], []
            for col in self.target_cols:
                if scale_method == "standard":
                    numerator = df[col].mean()
                    denominator = df[col].std()
                # Ref: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484#2811881
                elif scale_method == "standard_y2":
                    numerator = df[col].mean()
                    denominator = np.sqrt((df[col] * df[col]).mean())
                elif scale_method == "minmax":
                    numerator = df[col].min()
                    denominator = df[col].max() - df[col].min()
                elif scale_method == "robust":
                    numerator = df[col].median()
                    denominator = df[col].quantile(0.75) - df[col].quantile(0.25)

                denominator = np.maximum(denominator, min_denominator)
                exprs.append(((pl.col(col) - numerator) / denominator).alias(col))
                y_numerators.append(numerator)
                y_denominators.append(denominator)

            y_numerators = np.array(y_numerators).reshape(1, -1)
            y_denominators = np.array(y_denominators).reshape(1, -1)
            np.save(self.config.output_path / f"y_numerators_{scale_method}.npy", y_numerators)
            np.save(self.config.output_path / f"y_denominators_{scale_method}.npy", y_denominators)
        else:
            y_numerators = np.load(self.config.output_path / f"y_numerators_{scale_method}.npy")
            y_denominators = np.load(self.config.output_path / f"y_denominators_{scale_method}.npy")
            for i, col in enumerate(self.target_cols):
                exprs.append(((pl.col(col) - y_numerators[0, i]) / y_denominators[0, i]).alias(col))

        df = df.with_columns(exprs)
        return df

    def convert_numpy_array(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> dict[str, np.ndarray]:
        if self.config.task_type == "main" and self.config.multi_task:
            train_df = self._get_forward_and_back_target(train_df, shift_steps=1)

        X_train = self._convert_input_array(train_df, self.config.input_shape)
        X_valid = self._convert_input_array(valid_df, self.config.input_shape)
        X_test = self._convert_input_array(test_df, self.config.input_shape)

        if self.config.task_type == "main":
            y_train = self._convert_target_array(train_df, self.config.target_shape)
            y_valid = self._convert_target_array(valid_df, self.config.target_shape)
        elif self.config.task_type == "grid_pred":
            y_train = train_df["grid_id"].to_numpy()
            y_valid = valid_df["grid_id"].to_numpy()

        if self.config.task_type == "main" and self.config.multi_task:
            # get the target of the forward and backward steps
            y_train = np.concatenate(
                [
                    y_train,
                    self._convert_target_array(train_df, self.config.target_shape, suffix="_lag"),
                    self._convert_target_array(train_df, self.config.target_shape, suffix="_lead"),
                ],
                axis=-1,
            )

        return {
            "train_ids": train_df["sample_id"].to_numpy(),
            "valid_ids": valid_df["sample_id"].to_numpy(),
            "test_ids": test_df["sample_id"].to_numpy(),
            "X_train": X_train,
            "X_valid": X_valid,
            "X_test": X_test,
            "y_train": y_train,
            "y_valid": y_valid,
        }

    def _convert_input_array(self, df: pl.DataFrame, input_shape: Literal["2dim", "3dim"]) -> np.ndarray:
        if input_shape == "2dim":
            X_array = df.select(self.input_cols).to_numpy()
        elif input_shape == "3dim":
            X_array = []
            for col in VERTICAL_INPUT_COLS + ADDITIONAL_VERTICAL_INPUT_COLS:
                feat_array = df.select([f"{col}_{i}" for i in range(60)]).to_numpy()
                X_array.append(feat_array)

            for col in SCALER_INPUT_COLS:
                feat_array = np.repeat(df[col].to_numpy().reshape(-1, 1), repeats=60, axis=-1)
                X_array.append(feat_array)

            if self.config.task_type == "main" and self.config.use_grid_feat:
                for col in GRID_SCALER_INPUT_COLS:
                    feat_array = np.repeat(df[col].to_numpy().reshape(-1, 1), repeats=60, axis=-1)
                    X_array.append(feat_array)

            X_array = np.stack(X_array, axis=-1)

        return X_array

    def _convert_target_array(
        self, df: pl.DataFrame, target_shape: Literal["2dim", "3dim"], suffix: str = ""
    ) -> np.ndarray:
        if target_shape == "2dim":
            target_cols = [f"{col}{suffix}" for col in self.target_cols]
            y_array = df.select(target_cols).to_numpy()
        elif target_shape == "3dim":
            # Include columns excluded from prediction to keep the vertical structure when holding targets as a sequence
            y_array = []
            for col in VERTICAL_TARGET_COLS:
                target_array = df.select([f"{col}_{i}{suffix}" for i in range(60)]).to_numpy()
                y_array.append(target_array)

            for col in SCALER_TARGET_COLS:
                target_array = np.repeat(df[f"{col}{suffix}"].to_numpy().reshape(-1, 1), repeats=60, axis=-1)
                y_array.append(target_array)
            y_array = np.stack(y_array, axis=-1)
        return y_array

    def _get_forward_and_back_target(self, df: pl.DataFrame, shift_steps: int = 1) -> pl.DataFrame:
        target_cols = list(itertools.chain(*[[f"{col}_{i}" for i in range(60)] for col in VERTICAL_TARGET_COLS]))
        target_cols += SCALER_TARGET_COLS
        df = df.sort("sample_id")
        df = df.with_columns(
            [pl.col(col).shift(shift_steps).over("grid_id").alias(f"{col}_lag") for col in target_cols]
            + [pl.col(col).shift(-shift_steps).over("grid_id").alias(f"{col}_lead") for col in target_cols]
        )
        df = df.filter(pl.all_horizontal(pl.col("*").is_not_null()))
        return df
