from typing import Literal

import polars as pl
import xarray as xr
from omegaconf import DictConfig

from src.utils import convert_csv_to_parquet, multiply_old_factor, shrink_memory


class DataProvider:
    def __init__(self, config: DictConfig):
        self.config = config

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        if not (self.config.input_path / "train_shrinked.parquet").exists():
            if not (self.config.input_path / "train.parquet").exists():
                convert_csv_to_parquet(self.config.input_path / "train.csv", delete_csv=True)
            if not (self.config.input_path / "test.parquet").exists():
                convert_csv_to_parquet(self.config.input_path / "test.csv", delete_csv=True)
            if not (self.config.input_path / "sample_submission.parquet").exists():
                convert_csv_to_parquet(self.config.input_path / "sample_submission.csv", delete_csv=True)
            if not (self.config.input_path / "sample_submission_old.parquet").exists():
                convert_csv_to_parquet(self.config.input_path / "sample_submission_old.csv", delete_csv=True)

            train_df = pl.read_parquet(self.config.input_path / "train.parquet")
            test_df = pl.read_parquet(self.config.input_path / "test.parquet")
            train_df, test_df = shrink_memory(train_df, test_df)
            train_df.write_parquet(self.config.input_path / "train_shrinked.parquet")
            test_df.write_parquet(self.config.input_path / "test_shrinked.parquet")
            (self.config.input_path / "train.parquet").unlink()
            (self.config.input_path / "test.parquet").unlink()
        else:
            train_df = pl.read_parquet(self.config.input_path / "train_shrinked.parquet")
            test_df = pl.read_parquet(self.config.input_path / "test_shrinked.parquet")

        train_df = train_df.with_columns(time_id=pl.col("sample_id") // 384)
        train_df = self._downsample(train_df, self.config.run_mode)

        if self.task_type == "main" and self.config.use_grid_info:
            train_df, test_df = self._merge_grid_info(train_df, test_df)
        if self.task_type == "grid_pred":
            train_df = train_df.with_columns(grid_id=(pl.col("sample_id") % 384).cast(pl.Int64))

        train_df = self.split_validation(self.config, train_df)
        if self.mul_old_factor:
            train_df = multiply_old_factor(train_df, self.config.input_path)
        train_df = train_df.drop(["time_id"])
        return train_df, test_df

    def _downsample(self, train_df: pl.DataFrame, run_mode: Literal["full", "dev", "debug"]):
        time_ids = sorted(train_df["time_id"].unique())
        num_ids = 100 if run_mode == "debug" else 9000 if run_mode == "dev" else len(time_ids)
        use_ids = time_ids[-num_ids:]
        train_df = train_df.filter(pl.col("time_id").is_in(use_ids))
        return train_df

    def _merge_grid_info(self, train_df: pl.DataFrame, test_df: pl.DataFrame):
        train_df = train_df.with_columns(grid_id=(pl.col("sample_id") % 384).cast(pl.Int64))
        test_grid_path = self.config.input_path / "additional" / "test_grid_id.parquet"
        if test_grid_path.exists():
            test_grid_id = pl.read_parquet(test_grid_path)
        else:
            raise FileNotFoundError(f"{test_grid_path} is not found. You need to predict the test grid.")
        test_df = test_df.join(test_grid_id, on="sample_id", how="left")
        grid_feat = self._load_grid_info(use_cols=["lat", "lon"])
        train_df = train_df.join(grid_feat, on=["grid_id"], how="left")
        test_df = test_df.join(grid_feat, on=["grid_id"], how="left")
        return train_df, test_df

    def _load_grid_info(self, use_cols: list[str] | None = None):
        feat_path = self.config.input_path / "additional" / "grid_feat.parquet"
        if feat_path.exists():
            grid_feat = pl.read_parquet(feat_path)
        else:
            grid_info_path = self.config.input_path / "additional" / "ClimSim_low-res_grid-info.nc"
            grid_info = xr.open_dataset(grid_info_path)
            grid_info = pl.from_pandas(grid_info.to_dataframe().reset_index())
            grid_info = grid_info.rename({"ncol": "grid_id"})
            grid_feat = grid_info.group_by("grid_id").agg([pl.col(col).mean() for col in grid_info.columns if col not in ["grid_id", "time"]])
            grid_feat.write_parquet(feat_path)

        if use_cols is not None:
            grid_feat = grid_feat.select(["grid_id"] + use_cols)
        return grid_feat
