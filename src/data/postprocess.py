from typing import Literal

import loguru
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from src.utils.competition_utils import get_io_columns, get_sub_factor


class PostProcessor:
    def __init__(self, config: DictConfig, logger: loguru._Logger, additional: bool = True):
        self.config = config
        self.logger = logger
        self.additional = additional

        _, self.target_cols = get_io_columns(config)
        self.old_factor_dict = get_sub_factor(config.input_path, old=True)
        self.sub_cols = pl.read_parquet(
            config.input_path / "sample_submission.parquet", n_rows=1
        ).columns

        self.pp_x_cols = [f"state_q0002_{i}" for i in range(12, 27)]
        self.pp_y_cols = [f"ptend_q0002_{i}" for i in range(12, 27)]
        self.valid_pp_df = pl.read_parquet(
            config.input_path / "18_shrinked.parquet", columns=["sample_id"] + self.pp_x_cols
        )
        self.test_pp_df = pl.read_parquet(
            config.input_path / "test_shrinked.parquet", columns=["sample_id"] + self.pp_x_cols
        )

        add_pp_y_cols = [f"ptend_q0002_{i}" for i in range(60)] + [
            f"ptend_q0003_{i}" for i in range(60)
        ]
        self.add_pp_y_cols = [col for col in add_pp_y_cols if col in self.target_cols]
        self.add_pp_x_cols = [col.replace("ptend", "state") for col in self.add_pp_y_cols]
        self.add_valid_pp_df = pl.read_parquet(
            config.input_path / "18_shrinked.parquet", columns=self.sub_cols + self.add_pp_x_cols
        )
        self.add_test_pp_df = pl.read_parquet(
            config.input_path / "test_shrinked.parquet", columns=["sample_id"] + self.add_pp_x_cols
        )
        self.th_dict = {}

    def postprocess(self, oof_df: pl.DataFrame, sub_df: pl.DataFrame):
        oof_df = self.complement_columns(oof_df)
        oof_df = self.reverse_sub_factor(oof_df)
        oof_df = self.replace_postprocess(oof_df, "oof")

        sub_df = self.complement_columns(sub_df)
        sub_df = self.reverse_sub_factor(sub_df)
        sub_df = self.replace_postprocess(sub_df, "sub")

        if self.additional:
            oof_df = self.additional_postprocess(oof_df, "oof")
            sub_df = self.additional_postprocess(sub_df, "sub")

        oof_df = self.create_oof_df(oof_df)
        sub_df = self.create_sub_df(sub_df)
        return oof_df, sub_df

    def complement_columns(self, pred_df: pl.DataFrame):
        lack_cols = list(set(self.sub_cols) - set(pred_df.columns))
        for col in lack_cols:
            pred_df = pred_df.with_columns([pl.lit(0).alias(col)])
        return pred_df

    def reverse_sub_factor(self, pred_df: pl.DataFrame):
        if self.config.mul_old_factor:
            exprs = []
            for col in self.target_cols:
                if self.old_factor_dict[col] != 0:
                    exprs.append((pl.col(col) / self.old_factor_dict[col]).alias(col))

            pred_df = pred_df.with_columns(exprs)
        return pred_df

    def replace_postprocess(self, pred_df: pl.DataFrame, pred_type: Literal["oof", "sub"]):
        pp_df = self.valid_pp_df if pred_type == "oof" else self.test_pp_df
        pred_df = pred_df.join(pp_df, on=["sample_id"], how="left")

        exprs = []
        for x_col, y_col in zip(self.pp_x_cols, self.pp_y_cols):
            exprs.append((-1 * pl.col(x_col) / 1200).alias(y_col))
        pred_df = pred_df.with_columns(exprs)
        pred_df = pred_df.drop(self.pp_x_cols)
        return pred_df

    def additional_postprocess(self, pred_df: pl.DataFrame, pred_type: Literal["oof", "sub"]):
        pp_df = self.add_valid_pp_df if pred_type == "oof" else self.add_test_pp_df
        pred_df = pred_df.join(pp_df, on=["sample_id"], how="left", suffix="_gt")
        exprs = []
        for x_col, y_col in zip(self.add_pp_x_cols, self.add_pp_y_cols):
            exprs.append((pl.col(x_col) + pl.col(y_col) * 1200).alias(f"{x_col}_next"))
        pred_df = pred_df.with_columns(exprs)

        if pred_type == "oof":
            self.tuning_threshold(pred_df)

        assert len(self.th_dict) > 0  # oofから実行する必要がある
        exprs = []
        for y_col, (best_th, _) in self.th_dict.items():
            x_col = y_col.replace("ptend", "state")
            exprs.append(
                pl.when(pl.col(f"{x_col}_next") < best_th)
                .then(-1 * pl.col(x_col) / 1200)
                .otherwise(pl.col(y_col))
                .alias(y_col)
            )
        pred_df = pred_df.with_columns(exprs)

        if pred_type == "oof":
            scores = []
            for col in self.target_cols:
                score = r2_score(pred_df[f"{col}_gt"].to_numpy(), pred_df[col].to_numpy())
                scores.append(score)
            total_score = (np.sum(scores) + (368 - len(scores))) / 368
            self.logger.info(f"After Additional Postprocess: {total_score:.5f}")

        drop_cols = (
            self.add_pp_x_cols
            + [f"{col}_next" for col in self.add_pp_x_cols]
            + [col for col in pred_df.columns if "_gt" in col]
        )
        pred_df = pred_df.drop(drop_cols)
        return pred_df

    def tuning_threshold(self, pred_df: pl.DataFrame):
        iterations = tqdm(
            zip(self.add_pp_x_cols, self.add_pp_y_cols),
            total=len(self.add_pp_x_cols),
            desc="Additional PP Tuning...",
        )
        for x_col, y_col in iterations:
            best_score = r2_score(pred_df[f"{y_col}_gt"].to_numpy(), pred_df[y_col].to_numpy())
            best_th = None
            for base_th in [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
                for corr in range(1, 10):
                    if base_th == 0 and corr >= 2:
                        break

                    th = base_th * corr
                    preds = pred_df.select(
                        pl.when(pl.col(f"{x_col}_next") < th)
                        .then(-1 * pl.col(x_col) / 1200)
                        .otherwise(pl.col(y_col))
                    ).to_numpy()

                    truths = pred_df[f"{y_col}_gt"].to_numpy()
                    score = r2_score(truths, preds)
                    if score > best_score:
                        best_score = score
                        best_th = th

            if best_th is not None:
                self.th_dict[y_col] = (best_th, best_score)

    def create_oof_df(self, oof_df: pl.DataFrame):
        oof_df = oof_df.select(self.sub_cols)
        oof_df.write_parquet(self.config.oof_path / "oof_pp.parquet")
        return oof_df

    def create_sub_df(self, sub_df: pl.DataFrame):
        sub_df = sub_df.with_columns(
            sample_id=pl.concat_str([pl.lit("test_"), pl.col("sample_id")])
        )
        sub_df = sub_df.select(self.sub_cols)
        sub_df.write_csv(self.config.output_path / "submission_pp.csv")
        return sub_df
