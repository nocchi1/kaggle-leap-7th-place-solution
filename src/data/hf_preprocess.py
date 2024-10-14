import gc
import pickle
from typing import Literal

import h5py
import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.data.feature_engineering import FeatureEngineering
from src.data.preprocess import Preprocessor
from src.utils.competition_utils import (
    clipping_input,
    get_io_columns,
    multiply_old_factor,
    shrink_memory,
)


class HFPreprocessor:
    def __init__(self, config):
        self.config = config
        self.hf_files = list((self.config.add_path / "huggingface").glob("*.parquet"))
        if len(self.hf_files) == 0:
            raise FileNotFoundError("You need to download the additional data from HF.")

        self.input_cols, self.target_cols = get_io_columns(config)
        self.fer = FeatureEngineering(config)
        self.ppr = Preprocessor(config)
        self.input_clip_dict = pickle.load(open(self.config.output_path / "input_clip_dict.pkl", "rb"))
        # Year-months used for shared_valid
        self.valid_ym = [
            "0008-07",
            "0008-08",
            "0008-09",
            "0008-10",
            "0008-11",
            "0008-12",
            "0009-01",
        ]

    def shrink_file_size(self):
        shrink_num = len([file for file in self.hf_files if "_shrinked" in file.stem])
        if len(self.hf_files) > 0 and shrink_num == 0:
            refer_df = pl.read_parquet(self.config.input_path / "train_shrinked.parquet", n_rows=10)
            for file in tqdm(self.hf_files):
                df = pl.read_parquet(file)
                df = shrink_memory(df, refer_df)
                df.write_parquet(self.config.add_path / "huggingface" / f"{file.stem}_shrinked.parquet")
                file.unlink()
            self.hf_files = list((self.config.add_path / "huggingface").glob("*.parquet"))

    def convert_numpy_array(self, save_method: Literal["npy", "hdf5"] = "npy", unlink_parquet: bool = True):
        output_path = self.config.add_path / "huggingface"
        self.hf_files = sorted(self.hf_files, key=lambda x: x.stem)
        for i, file in enumerate(tqdm(self.hf_files)):
            ym = file.stem.replace("_shrinked", "")
            # Don't use ym after valid_ym
            if ym in self.valid_ym:
                continue

            df = pl.read_parquet(file)
            df = df.with_columns(grid_id=(pl.col("sample_id") % 384), time_id=pl.col("sample_id") // 384)
            if self.config.mul_old_factor:
                df = multiply_old_factor(self.config.input_path, df)

            df = self.fer.feature_engineering(df)
            df = self.ppr._input_scaling(df, self.config.input_scale_method, compute_stats=False)
            df = self.ppr._target_scaling(df, self.config.target_scale_method, compute_stats=False)
            df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))

            df, _ = clipping_input(
                refer_df=None,
                target_df=df,
                input_cols=self.input_cols,
                clip_dict=self.input_clip_dict,
            )
            # Create targets for multi-task, even if multi_task=True is not set
            df = self.ppr._get_forward_and_back_target(df, shift_steps=7)
            X_train = self.ppr._convert_input_array(df, self.config.input_shape)
            y_train = np.concatenate(
                [
                    self.ppr._convert_target_array(df, self.config.target_shape),
                    self.ppr._convert_target_array(df, self.config.target_shape, suffix="_lag"),
                    self.ppr._convert_target_array(df, self.config.target_shape, suffix="_lead"),
                ],
                axis=-1,
            )

            if save_method == "npy":
                np.save(output_path / f"X_{ym}.npy", X_train)
                np.save(output_path / f"y_{ym}.npy", y_train)
            elif save_method == "hdf5":
                if i == 0:
                    with h5py.File(output_path / "hf_data.h5", "w") as f:
                        f.create_dataset("X", data=X_train, maxshape=(None, *X_train.shape[1:]), chunks=True)
                        f.create_dataset("y", data=y_train, maxshape=(None, *y_train.shape[1:]), chunks=True)
                else:
                    with h5py.File(output_path / "hf_data.h5", "a") as f:
                        f["X"].resize((f["X"].shape[0] + X_train.shape[0]), axis=0)
                        f["X"][-X_train.shape[0] :] = X_train
                        f["y"].resize((f["y"].shape[0] + y_train.shape[0]), axis=0)
                        f["y"][-y_train.shape[0] :] = y_train
            else:
                raise ValueError(f"save_method should be 'npy' or 'hdf5'. Got {save_method}.")

            del X_train, y_train
            gc.collect()
            if unlink_parquet:
                file.unlink()
