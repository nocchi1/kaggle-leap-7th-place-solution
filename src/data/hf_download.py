import os
import pickle
import shutil
from pathlib import Path

import huggingface_hub
import numpy as np
import polars as pl
import xarray as xr
from tqdm.auto import tqdm

from src.utils.constant import (
    SCALER_INPUT_COLS,
    SCALER_TARGET_COLS,
    VERTICAL_INPUT_COLS,
    VERTICAL_TARGET_COLS,
)


class HFDataLoader:
    def __init__(self, input_dir: Path):
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        self.repo_id = "LEAP/ClimSim_low-res"
        self.input_dir = input_dir
        self.add_dir = input_dir / "additional"
        self.output_dir = self.add_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "huggingface").mkdir(exist_ok=True, parents=True)
        self.input_vars = VERTICAL_INPUT_COLS + SCALER_INPUT_COLS
        self.target_vars = VERTICAL_TARGET_COLS + SCALER_TARGET_COLS
        self.train_columns = self.get_column_name()
        self.dir_patterns = self.get_dir_patterns()
        self.grid_info = xr.open_dataset(self.add_dir / "ClimSim_low-res_grid-info.nc")
        self.input_feature_num = 558  # lat,lonを追加 + cam_in_SNOWHICEを除去
        self.target_feature_num = 368
        self.output_normalize = False
        self.output_scale = xr.open_dataset(self.add_dir / "output_scale.nc")

    def download(self):
        error_pattern = []
        for pattern in tqdm(self.dir_patterns):
            print(f"Downloading HF:{pattern}")
            allow_patterns = f"train/{pattern}/*.nc"
            try:
                self.download_from_hf(allow_patterns=allow_patterns)
                raw_files = list(self.output_dir.glob(f"train/{pattern}/E3SM-MMF.mli.*.nc"))
                raw_files = sorted(raw_files, key=lambda x: x.name)
                generator = self.get_generator(raw_files)
                df = self.get_dataframe(generator)
                df.write_parquet(self.output_dir / "huggingface" / f"{pattern}.parquet")
                shutil.rmtree(self.output_dir / f"train/{pattern}")
                shutil.rmtree(self.output_dir / f".cache/huggingface/download/train/{pattern}")
            except Exception:
                error_pattern.append(pattern)
        if len(error_pattern) > 0:
            pickle.dump(error_pattern, open(self.output_dir / "error_pattern.pkl", "wb"))
        shutil.rmtree(self.output_dir / "train")
        shutil.rmtree(self.output_dir / ".cache")

    def download_from_hf(self, allow_patterns: str):
        huggingface_hub.snapshot_download(
            repo_id=self.repo_id,
            allow_patterns=allow_patterns,
            cache_dir=None,
            local_dir=self.output_dir,
            repo_type="dataset",
            force_download=True,
            etag_timeout=3600,
        )

    def get_dataframe(self, generator):
        numpy_input, numpy_target = [], []
        for data_input, data_target in generator:
            numpy_input.append(data_input)
            numpy_target.append(data_target)
        numpy_input = np.concatenate(numpy_input, axis=0)
        numpy_target = np.concatenate(numpy_target, axis=0)
        train_array = np.concatenate([numpy_input, numpy_target], axis=-1)
        train_df = pl.DataFrame(train_array, schema=self.train_columns)
        train_df = train_df.with_row_index(name="sample_id")
        return train_df

    def get_generator(self, raw_files: list[Path]):
        return self.generate(raw_files)

    def generate(self, raw_files: list[Path]):
        for file in raw_files:
            data_input = self.get_input(file)
            data_target = self.get_target(file)
            if self.output_normalize:
                data_target = data_target * self.output_scale
            # data_input = data_input.drop(['lat','lon'])
            data_input = data_input.stack({"batch": {"ncol"}})
            data_input = data_input.to_stacked_array("mlvar", sample_dims=["batch"], name="mli")
            data_target = data_target.stack({"batch": {"ncol"}})
            data_target = data_target.to_stacked_array("mlvar", sample_dims=["batch"], name="mlo")
            data_input = data_input.values
            data_target = data_target.values
            assert data_input.shape[-1] == self.input_feature_num
            assert data_target.shape[-1] == self.target_feature_num
            if data_input.dtype != np.float64:
                data_input = data_input.astype(np.float64)
            if data_target.dtype != np.float64:
                data_target = data_target.astype(np.float64)
            yield data_input, data_target

    def get_dir_patterns(self):
        dir_patterns = []
        for year in range(1, 10):
            for month in range(1, 13):
                if year == 1 and month == 1:
                    continue
                if year == 9 and month > 1:
                    break
                dir_name = f"000{year}-{str(month).zfill(2)}"
                dir_patterns.append(dir_name)
        return dir_patterns

    def get_column_name(self):
        train_columns = []
        for col in VERTICAL_INPUT_COLS:
            for i in range(60):
                train_columns.append(col + "_" + str(i))
        for col in SCALER_INPUT_COLS:
            train_columns.append(col)
        train_columns += ["lat", "lon"]
        for col in VERTICAL_TARGET_COLS:
            for i in range(60):
                train_columns.append(col + "_" + str(i))
        for col in SCALER_TARGET_COLS:
            train_columns.append(col)
        return train_columns

    def get_xrdata(self, file_path: Path, file_vars: list[str] | None = None):
        data = xr.open_dataset(file_path, engine="netcdf4")
        if file_vars is not None:
            data = data[file_vars]
        data = data.merge(self.grid_info[["lat", "lon"]])
        data = data.where((data["lat"] > -999) * (data["lat"] < 999), drop=True)
        data = data.where((data["lon"] > -999) * (data["lon"] < 999), drop=True)
        return data

    def get_input(self, file_path: Path):
        return self.get_xrdata(file_path, self.input_vars)

    def get_target(self, file_path: Path):
        data_input = self.get_input(file_path)
        data_target = self.get_xrdata(file_path.with_name(file_path.name.replace(".mli.", ".mlo.")))
        data_target["ptend_t"] = (data_target["state_t"] - data_input["state_t"]) / 1200
        data_target["ptend_q0001"] = (data_target["state_q0001"] - data_input["state_q0001"]) / 1200
        data_target["ptend_q0002"] = (data_target["state_q0002"] - data_input["state_q0002"]) / 1200
        data_target["ptend_q0003"] = (data_target["state_q0003"] - data_input["state_q0003"]) / 1200
        data_target["ptend_u"] = (data_target["state_u"] - data_input["state_u"]) / 1200
        data_target["ptend_v"] = (data_target["state_v"] - data_input["state_v"]) / 1200
        data_target = data_target[self.target_vars]
        return data_target


if __name__ == "__main__":
    input_dir = Path("data/input")
    loader = HFDataLoader(input_dir)
    loader.download()
