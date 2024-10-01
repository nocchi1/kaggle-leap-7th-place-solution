from pathlib import PosixPath

import numpy as np
import polars as pl
import xarray as xr
from omegaconf import DictConfig


class FeatureEngineering:
    def __init__(self, config: DictConfig):
        hybi_path = config.input_path / "additional" / "hybi.npy"
        if hybi_path.exists():
            self.hybi = np.load(hybi_path)
        else:
            grid_info_path = config.input_path / "additional" / "ClimSim_low-res_grid-info.nc"
            grid_info = xr.open_dataset(grid_info_path)
            grid_info = pl.from_pandas(grid_info.to_dataframe().reset_index())
            self.hybi = grid_info.unique(subset=['ilev', 'hybi'], maintain_order=True)['hybi'].to_numpy()
            np.save(hybi_path, self.hybi)

        self.use_latlon = config.use_grid_feat

    def feature_engineering(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.use_latlon:
            df = self.convert_coordinate(df)
        df = self.get_ice_rate_feat(df)
        df = self.get_weather_feat(df)
        return df

    def convert_coordinate(self, df: pl.DataFrame):
        df = df.with_columns(
            lat_sin=((pl.col("lat") / 90) * 2 * np.pi).sin(),
            lat_cos=((pl.col("lat") / 90) * 2 * np.pi).cos(),
            lon_sin=((pl.col("lon") / 360) * 2 * np.pi).sin(),
            lon_cos=((pl.col("lon") / 360) * 2 * np.pi).cos(),
        )
        return df

    def get_ice_rate_feat(self, df: pl.DataFrame):
        Tf, Ts = 273.15, 258.15
        # Ice ratio inside the cloud
        exprs = []
        for i in range(60):
            exprs.append((pl.col(f"state_q0003_{i}") / (pl.col(f"state_q0002_{i}") + pl.col(f"state_q0003_{i}"))).alias(f"state_ice_rate_{i}"))
        df = df.with_columns(exprs)

        # Theoretical ice ratio inside the cloud
        exprs = []
        for i in range(60):
            exprs.append((pl.col(f"state_ice_rate_{i}") - ((Tf - pl.col(f"state_t_{i}")) / (Tf - Ts)).clip(0, 1)).alias(f"state_ice_rate_diff_{i}"))
        df = df.with_columns(exprs)

        exprs = []
        for i in range(60):
            exprs.extend(
                [
                    pl.when(
                        (pl.col(f"state_ice_rate_{i}").is_null()) | (pl.col(f"state_ice_rate_{i}").is_nan()) | (pl.col(f"state_ice_rate_{i}").is_infinite())
                    )
                    .then(pl.lit(0))
                    .otherwise(pl.col(f"state_ice_rate_{i}"))
                    .alias(f"state_ice_rate_{i}"),
                    pl.when(
                        (pl.col(f"state_ice_rate_diff_{i}").is_null())
                        | (pl.col(f"state_ice_rate_diff_{i}").is_nan())
                        | (pl.col(f"state_ice_rate_diff_{i}").is_infinite())
                    )
                    .then(pl.lit(0))
                    .otherwise(pl.col(f"state_ice_rate_diff_{i}"))
                    .alias(f"state_ice_rate_diff_{i}"),
                ]
            )
        df = df.with_columns(exprs)
        return df

    def get_weather_feat(self, df: pl.DataFrame):
        # Pressure difference
        dp_val = self.hybi * df["state_ps"].to_numpy().reshape(-1, 1)
        dp_val = np.diff(dp_val, axis=1)

        # Relative humidity
        t_val = df.select([f"state_t_{i}" for i in range(60)]).to_numpy()
        q1_val = df.select([f"state_q0001_{i}" for i in range(60)]).to_numpy()
        esat_val = 6.112 * np.exp((17.67 * (t_val - 273.16)) / (t_val - 29.65))  # Saturation vapor pressure (simplified)
        rh_val = dp_val / esat_val * q1_val

        # Vapor pressure
        vp_val = (q1_val * dp_val) / (0.622 + q1_val * (1 - 0.622))

        df = pl.concat(
            [
                df,
                pl.DataFrame(dp_val, schema=[f"state_dp_{i}" for i in range(60)]),
                pl.DataFrame(rh_val, schema=[f"state_rh_{i}" for i in range(60)]),
                pl.DataFrame(vp_val, schema=[f"state_vp_{i}" for i in range(60)]),
            ],
            how="horizontal",
        )
        return df
