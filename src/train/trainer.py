import pickle
from collections import defaultdict
from typing import Dict, List, Literal, Tuple

import loguru
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.train import ComponentFactory
from src.train.train_utils import AverageMeter
from src.utils import clean_message
from src.utils.competition_utils import evaluate_metric, get_io_columns, get_sub_factor
from src.utils.constant import (
    PP_TARGET_COLS,
    SCALER_TARGET_COLS,
    TARGET_MIN_MAX,
    VERTICAL_TARGET_COLS,
)


class Trainer:
    def __init__(self, config: DictConfig, logger: loguru._Logger, save_suffix: str = ""):
        self.config = config
        self.eval_step = config.eval_step[config.run_mode]
        self.logger = logger
        self.save_suffix = save_suffix
        self.detail_pbar = True

        self.model = ComponentFactory.get_model(config)
        self.model = self.model.to(config.device)
        n_device = torch.cuda.device_count()
        if n_device > 1:
            self.model = nn.DataParallel(self.model)
        self.loss_fn = ComponentFactory.get_loss(config)
        self.train_loss = AverageMeter()
        self.valid_loss = AverageMeter()

        _, self.target_cols = get_io_columns(config)
        self.model_target_cols = self.get_model_target_cols()
        self.factor_dict = get_sub_factor(config.input_path, old=False)
        self.old_factor_dict = get_sub_factor(config.input_path, old=True)

        self.y_numerators = np.load(
            config.output_path / f"y_numerators_{config.target_scale_method}.npy"
        )
        self.y_denominators = np.load(
            config.output_path / f"y_denominators_{config.target_scale_method}.npy"
        )
        self.target_min_max = [TARGET_MIN_MAX[col] for col in self.target_cols]

        self.valid_ids = None
        self.test_ids = None
        self.valid_pp_df = None
        self.test_pp_df = None
        self.pp_run = True
        self.pp_y_cols = PP_TARGET_COLS
        self.pp_x_cols = [col.replace("ptend", "state") for col in self.pp_y_cols]

        self.best_score_dict = defaultdict(lambda: (-1, -np.inf))

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        colwise_mode: bool = True,
        retrain: bool = False,
        retrain_weight_name: str = "",
        retrain_best_score: float = -np.inf,
        eval_only: bool = False,
    ):
        if eval_only:
            self.best_score_dict = pickle.load(
                open(self.config.output_path / f"best_score_dict{self.save_suffix}.pkl", "rb")
            )
            eval_method = "colwise" if colwise_mode else "single"
            score, cw_score, preds, _ = self.valid_evaluate(
                valid_loader, current_epoch=-1, eval_count=-1, eval_method=eval_method
            )
            self.save_oof_df(self.valid_ids, preds)
            return score, cw_score, -1

        self.optimizer = ComponentFactory.get_optimizer(self.config, self.model)
        self.scheduler = ComponentFactory.get_scheduler(
            self.config, self.optimizer, steps_per_epoch=len(train_loader)
        )
        global_step = 0
        eval_count = 0
        best_score = -np.inf

        if retrain:
            self.best_score_dict = pickle.load(
                open(self.config.output_path / f"best_score_dict{self.save_suffix}.pkl", "rb")
            )
            self.model.load_state_dict(
                torch.load(self.config.output_path / f"{retrain_weight_name}.pth")
            )
            weight_numbers = [
                int(file.stem.split("_")[-1].replace("eval", ""))
                for file in list(self.config.output_path.glob(f"model{self.save_suffix}_eval*.pth"))
            ]
            eval_count = sorted(weight_numbers)[-1] + 1
            best_score = retrain_best_score

        # 学習ループの開始
        for epoch in tqdm(range(self.config.epochs)):
            self.model.train()
            self.train_loss.reset()

            iterations = (
                tqdm(train_loader, total=len(train_loader)) if self.detail_pbar else train_loader
            )
            for data in iterations:
                _, loss = self.forward_step(data, calc_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.train_loss.update(loss.item(), n=data[0].size(0))
                global_step += 1

                if global_step % self.eval_step == 0:
                    score, _, preds, update_num = self.valid_evaluate(
                        valid_loader,
                        current_epoch=epoch,
                        eval_count=eval_count,
                        eval_method="single",
                    )
                    if colwise_mode and update_num > 0:
                        torch.save(
                            self.model.state_dict(),
                            self.config.output_path
                            / f"model{self.save_suffix}_eval{eval_count}.pth",
                        )

                    if score > best_score:
                        best_score = score
                        best_preds = preds
                        best_epochs = epoch
                        torch.save(
                            self.model.state_dict(),
                            self.config.output_path / f"model{self.save_suffix}_best.pth",
                        )

                    eval_count += 1
                    self.model.train()

            message = f"""
                [Train] :
                    Epoch={epoch},
                    Loss={self.train_loss.avg:.5f},
                    LR={self.optimizer.param_groups[0]["lr"]:.5e}
            """
            self.logger.info(clean_message(message))

        if colwise_mode:
            self.remove_unuse_weights()
            best_score, best_cw_score, best_preds, _ = self.valid_evaluate(
                valid_loader, current_epoch=-1, eval_count=-1, eval_method="colwise"
            )

        self.save_oof_df(self.valid_ids, best_preds)
        return best_score, best_cw_score, best_epochs

    def valid_evaluate(
        self,
        valid_loader: DataLoader,
        current_epoch: int,
        eval_count: int,
        eval_method: Literal["single", "colwise"] = "single",
    ):
        if self.valid_ids is None:
            self.valid_ids = valid_loader.dataset.ids

        if eval_method == "single":
            load_best_weight = True if eval_count == -1 else False
            preds = self.inference_loop(
                valid_loader, mode="valid", load_best_weight=load_best_weight
            )
        elif eval_method == "colwise":
            preds = self.inference_loop_colwise(valid_loader, "valid", self.best_score_dict)

        labels = valid_loader.dataset.y
        if self.config.target_shape == "3dim":
            labels = self.convert_target_3dim_to_2dim(labels)
        preds = self.restore_pred(preds)
        labels = self.restore_pred(labels)

        if self.pp_run and self.valid_pp_df is None:
            self.load_postprocess_input("valid")
        if self.pp_run:
            preds = self.postprocess(preds, run_type="valid")
        if self.config.out_clip:
            preds = self.clipping_pred(preds)

        eval_idx = [
            i for i, col in enumerate(self.target_cols) if self.factor_dict[col] != 0
        ]  # factor_dictの値が0のものは自動でR2=1になるようにする
        score, indiv_scores = evaluate_metric(preds, labels, eval_idx=eval_idx)
        cw_score, update_num = self.update_best_score(indiv_scores, eval_count)

        message = f"""
            [Valid] :
                Epoch={current_epoch},
                Loss={self.valid_loss.avg:.5f},
                Score={score:.5f},
                Best Col-Wise Score={cw_score:.5f}
        """
        self.logger.info(clean_message(message))
        return score, cw_score, preds, update_num

    def test_predict(
        self, test_loader: DataLoader, eval_method: Literal["single", "colwise"] = "single"
    ):
        if self.test_ids is None:
            self.test_ids = test_loader.dataset.ids

        if eval_method == "single":
            preds = self.inference_loop(test_loader, mode="test", load_best_weight=True)
        elif eval_method == "colwise":
            self.best_score_dict = pickle.load(
                open(self.config.output_path / f"best_score_dict{self.save_suffix}.pkl", "rb")
            )
            preds = self.inference_loop_colwise(test_loader, "test", self.best_score_dict)

        preds = self.restore_pred(preds)
        if self.pp_run and self.test_pp_df is None:
            self.load_postprocess_input("test")
        if self.pp_run:
            preds = self.postprocess(preds, run_type="test")
        if self.config.out_clip:
            preds = self.clipping_pred(preds)

        pred_df = pl.DataFrame(preds, schema=self.target_cols)
        pred_df = pred_df.with_columns(sample_id=pl.Series(self.test_ids))
        return pred_df

    def inference_loop(
        self,
        eval_loader: DataLoader,
        mode: Literal["valid", "test"],
        load_best_weight: bool = False,
    ):
        self.model.eval()
        if mode == "valid":
            self.valid_loss.reset()

        # テストデータを推論するときはbest_weightを読み込む
        if load_best_weight:
            self.model.load_state_dict(
                torch.load(self.config.output_path / f"model{self.save_suffix}_best.pth")
            )

        preds = []
        with torch.no_grad():
            iterations = (
                tqdm(eval_loader, total=len(eval_loader)) if self.detail_pbar else eval_loader
            )
            for data in iterations:
                if mode == "valid":
                    out, loss = self.forward_step(data, calc_loss=True)
                    self.valid_loss.update(loss.item(), n=data[0].size(0))
                elif mode == "test":
                    out, _ = self.forward_step(data, calc_loss=False)
                preds.append(out.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds

    def inference_loop_colwise(
        self,
        test_loader: DataLoader,
        mode: Literal["valid", "test"],
        best_score_dict: dict[str, tuple[int, float]],
    ):
        self.model.eval()
        if mode == "valid":
            self.valid_loss.reset()

        selected_counts = list(set([eval_count for eval_count, _ in best_score_dict.values()]))
        all_preds = np.zeros((len(test_loader.dataset), len(self.target_cols)))
        for eval_count in tqdm(selected_counts):
            self.model.load_state_dict(
                torch.load(
                    self.config.output_path / f"model{self.save_suffix}_eval{eval_count}.pth"
                )
            )
            preds = []
            with torch.no_grad():
                iterations = (
                    tqdm(test_loader, total=len(test_loader)) if self.detail_pbar else test_loader
                )
                for data in iterations:
                    if mode == "valid":
                        out, loss = self.forward_step(data, calc_loss=True)
                        self.valid_loss.update(loss.item(), n=data[0].size(0))
                    elif mode == "test":
                        out, _ = self.forward_step(data, calc_loss=False)
                    preds.append(out.detach().cpu().numpy())
            preds = np.concatenate(preds, axis=0)

            target_cols = [
                col for col, (count, _) in best_score_dict.items() if count == eval_count
            ]
            for col in target_cols:
                idx = self.target_cols.index(col)
                all_preds[:, idx] = preds[:, idx]
        return all_preds

    def update_best_score(self, indiv_scores: list[float], eval_count: int):
        update_num = 0
        for col, score in zip(self.target_cols, indiv_scores):
            if score > self.best_score_dict[col][1] and eval_count != -1:
                self.best_score_dict[col] = (eval_count, score)
                update_num += 1

        best_cw_score = (
            np.sum([score for _, score in self.best_score_dict.values()])
            + (368 - len(self.target_cols))
        ) / 368
        if update_num > 0 and eval_count != -1:
            pickle.dump(
                dict(self.best_score_dict),
                open(self.config.output_path / f"best_score_dict{self.save_suffix}.pkl", "wb"),
            )
        return best_cw_score, update_num

    def remove_unuse_weights(self):
        selected_counts = set([v[0] for v in self.best_score_dict.values()])
        weight_paths = list(self.config.output_path.glob(f"model{self.save_suffix}_eval*.pth"))
        for path in weight_paths:
            eval_count = int(path.stem.split("_")[-1].replace("eval", ""))
            if eval_count not in selected_counts:
                path.unlink()

    def forward_step(self, data: torch.Tensor, calc_loss: bool = True):
        if calc_loss:
            x, y = data
            x, y = x.to(self.config.device), y.to(self.config.device)
            out = self.model(x)
            loss = self.loss_fn(out, y)
        else:
            x = data[0]
            x = x.to(self.config.device)
            out = self.model(x)
            loss = None

        if self.config.multi_task:
            out = out[:, :, : self.config.out_dim]

        if self.config.target_shape == "3dim":
            out = self.convert_target_3dim_to_2dim(out)
        return out, loss

    def convert_target_3dim_to_2dim(
        self, y: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        y_v = y[:, :, : len(VERTICAL_TARGET_COLS)]
        y_s = y[:, :, len(VERTICAL_TARGET_COLS) :]
        if isinstance(y, np.ndarray):
            y_v = np.transpose(y_v, (0, 2, 1)).reshape(y.shape[0], -1)
            y_s = y_s.mean(axis=1)
            y = np.concatenate([y_v, y_s], axis=-1)
        elif isinstance(y, torch.Tensor):
            y_v = y_v.permute(0, 2, 1).reshape(y.size(0), -1)
            y_s = y_s.mean(dim=1)
            y = torch.cat([y_v, y_s], dim=-1)
        y = self.alignment_target_idx(y)
        return y

    def alignment_target_idx(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        target_colsとモデルの出力の順番を合わせる
        """
        align_order = [self.model_target_cols.index(col) for col in self.target_cols]
        assert len(y.shape) == 2
        y = y[:, align_order]
        return y

    def get_model_target_cols(self):
        model_target_cols = []
        for col in VERTICAL_TARGET_COLS:
            model_target_cols.extend([f"{col}_{i}" for i in range(60)])
        for col in SCALER_TARGET_COLS:
            model_target_cols.append(col)
        return model_target_cols

    def restore_pred(self, preds: np.ndarray):
        return preds * self.y_denominators + self.y_numerators

    def clipping_pred(self, preds: np.ndarray):
        for i in range(preds.shape[1]):
            preds[:, i] = np.clip(preds[:, i], self.target_min_max[i][0], self.target_min_max[i][1])
        return preds

    def save_oof_df(self, sample_ids: np.ndarray, preds: np.ndarray):
        oof_df = pl.DataFrame(preds, schema=self.target_cols)
        oof_df = oof_df.with_columns(sample_id=pl.Series(sample_ids))
        oof_df.write_parquet(self.config.oof_path / f"oof{self.save_suffix}.parquet")

    def postprocess(self, preds: np.ndarray, run_type: Literal["valid", "test"]):
        pp_x = self.valid_pp_df if run_type == "valid" else self.test_pp_df
        for x_col, y_col in zip(self.pp_x_cols, self.pp_y_cols):
            if y_col in self.target_cols:
                idx = self.target_cols.index(y_col)
                old_factor = self.old_factor_dict[y_col] if self.config.mul_old_factor else 1
                preds[:, idx] = (-1 * pp_x[x_col].to_numpy() / 1200) * old_factor
        return preds

    def load_postprocess_input(self, data_type: Literal["valid", "test"]):
        if data_type == "valid":
            valid_path = (
                self.config.input_path / "18_shrinked.parquet"
                if self.config.shared_valid
                else self.config.input_path / "train_shrinked.parquet"
            )
            self.valid_pp_df = (
                pl.scan_parquet(valid_path)
                .select(["sample_id"] + self.pp_x_cols)
                .filter(pl.col("sample_id").is_in(self.valid_ids))
                .collect()
            )
            id_df = pl.DataFrame({"sample_id": self.valid_ids})
            self.valid_pp_df = id_df.join(self.valid_pp_df, on="sample_id", how="left")

        elif data_type == "test":
            self.test_pp_df = pl.read_parquet(
                self.config.input_path / "test_shrinked.parquet",
                columns=["sample_id"] + self.pp_x_cols,
            )
            id_df = pl.DataFrame({"sample_id": self.test_ids})
            self.test_pp_df = id_df.join(self.test_pp_df, on="sample_id", how="left")
