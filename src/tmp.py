class Trainer:
    def __init__(
        self,
        config,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        save_suffix: str = "",
        logger: loguru._Logger | None = None,
        detail_pbar: bool = True,
    ):
        self.config = config
        self.run_mode = config.run_mode
        self.device = config.device
        self.input_path = config.input_path
        self.output_path = config.output_path
        self.oof_path = config.oof_path

        self.target_cols = config.target_cols
        self.mul_old_factor = config.mul_old_factor
        self.target_min_max = [TARGET_MIN_MAX[col] for col in config.target_cols]
        self.factor_dict = get_sub_factor(config.input_path, old=False)
        self.old_factor_dict = get_sub_factor(config.input_path, old=True)
        self.out_clip = config.out_clip

        self.model = model
        self.model.to(self.device)
        self.loss_fn = loss_fn
        self.train_loss = AverageMeter()
        self.valid_loss = AverageMeter()
        self.save_suffix = save_suffix
        self.logger = logger
        self.detail_pbar = detail_pbar

        self.hf_ym_list = []
        self.valid_ids = None
        self.test_ids = None
        self.seq_target_cols = None
        self.pp_x_cols = [f"state_q0002_{i}" for i in range(12, 27)]
        self.pp_y_cols = [f"ptend_q0002_{i}" for i in range(12, 27)]
        self.pp_run = len(set(self.target_cols) & set(self.pp_y_cols)) > 0
        self.valid_pp_x = None
        self.test_pp_x = None
        self.best_score_dict = {}

    def init_optimizer_and_scheduler(self, step_per_epoch: int):
        optimizer = get_optimizer(
            self.model,
            method=self.config.optimizer_method,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        if self.config.scheduler_method == "linear":
            total_steps = self.config.epochs * step_per_epoch
            scheduler_args = {
                "start_factor": self.config.linear_start_factor,
                "end_factor": self.config.linear_end_factor,
                "total_iters": total_steps * self.config.linear_end_step_ratio,
            }
        elif self.config.scheduler_method == "multistep":
            scheduler_args = {
                "milestones": self.config.multi_milestones,
                "gamma": self.config.multi_gamma,
            }
        elif self.config.scheduler_method == "cosine":
            T_0 = self.config.cosine_t0_epoch * step_per_epoch
            scheduler_args = {
                "T_0": T_0,
                "T_mult": self.config.cosine_t_mult,
                "eta_min": self.config.cosine_min_lr,
                "warmup_steps": self.config.cosine_warmup_steps,
                "gamma": self.config.cosine_gamma,
            }
        scheduler = get_scheduler(
            optimizer, method=self.config.scheduler_method, scheduler_args=scheduler_args
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_train_loader_from_hf(self, files_per_epoch: int = 5):
        npy_path = self.config.input_path / "huggingface" / "npy"
        if len(self.hf_ym_list) == 0:
            npy_files = list(npy_path.glob("X_*.npy"))
            self.hf_ym_list = [file.stem.split("_")[1] for file in npy_files]
        ym_extract = random.sample(self.hf_ym_list, min(files_per_epoch, len(self.hf_ym_list)))
        self.hf_ym_list = [ym for ym in self.hf_ym_list if ym not in ym_extract]
        X, y = [], []
        for ym in ym_extract:
            X.append(np.load(npy_path / f"X_{ym}.npy"))
            y.append(np.load(npy_path / f"y_{ym}.npy"))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        train_loader = get_dataloader(self.config, sample_ids=None, X=X, y=y, is_train=True)
        del X, y
        gc.collect()
        return train_loader

    def train(
        self,
        train_loader: DataLoader | None,
        valid_loader: DataLoader,
        colwise_best_weight: bool = False,
        eval_only: bool = False,
        eval_colwise: bool = False,
        retrain: bool = False,
        retrain_weight_name: str | None = None,
        retrain_best_score: float | None = None,
        retrain_eval_count: int | None = None,  # 前回の最終eval_countを指定する
    ) -> Tuple[float, int]:
        if eval_only:
            self.best_score_dict = pickle.load(
                open(self.output_path / f"best_score_dict{self.save_suffix}.pkl", "rb")
            )
            score, _, preds = self.valid_evaluate(
                valid_loader, -1, -1, eval_colwise, load_best_weight=True
            )
            self.save_oof_df(preds, self.valid_ids, self.target_cols)
            return score, -1
        if retrain:
            self.best_score_dict = pickle.load(
                open(self.output_path / f"best_score_dict{self.save_suffix}.pkl", "rb")
            )
            self.model.load_state_dict(torch.load(self.output_path / f"{retrain_weight_name}.pth"))

        global_step = 0
        eval_count = 0 if retrain_eval_count is None else retrain_eval_count + 1
        best_score = -np.inf if retrain_best_score is None else retrain_best_score
        step_per_epoch = len(train_loader) if self.run_mode != "hf" else self.config.eval_step
        self.init_optimizer_and_scheduler(step_per_epoch=step_per_epoch)
        for epoch in tqdm(range(self.config.epochs)):
            self.model.train()
            self.train_loss.reset()
            if self.run_mode == "hf":
                del train_loader
                gc.collect()
                train_loader = self.get_train_loader_from_hf()

            iterations = (
                tqdm(train_loader, total=len(train_loader)) if self.detail_pbar else train_loader
            )
            for batched in iterations:
                _, loss = self.forward_step(batched, calc_loss=True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.train_loss.update(loss.item(), n=batched[0].size(0))
                global_step += 1

                if global_step % self.config.eval_step == 0:
                    score, _, preds = self.valid_evaluate(valid_loader, epoch, eval_count)
                    if colwise_best_weight:
                        torch.save(
                            self.model.state_dict(),
                            self.output_path / f"model{self.save_suffix}_eval{eval_count}.pth",
                        )
                    if score > best_score:
                        best_score = score
                        best_preds = preds
                        best_epochs = epoch
                        torch.save(
                            self.model.state_dict(),
                            self.output_path / f"model{self.save_suffix}_best.pth",
                        )
                    eval_count += 1
                    self.model.train()

            message = f"""
                [Train] :
                    Epoch={epoch},
                    Loss={self.train_loss.avg:.7f},
                    LR={self.optimizer.param_groups[0]["lr"]:.4e}
            """
            self.logger.info(clean_message(message))

        pickle.dump(
            self.best_score_dict,
            open(self.output_path / f"best_score_dict{self.save_suffix}.pkl", "wb"),
        )
        if colwise_best_weight:
            best_score, _, best_preds = self.valid_evaluate(
                valid_loader, -1, -1, eval_colwise=True, load_best_weight=True
            )
            self.remove_unuse_weights()

        self.save_oof_df(best_preds, self.valid_ids, self.target_cols)
        return best_score, best_epochs  # 全体スコアが最高の時のEpoch

    def valid_evaluate(
        self,
        valid_loader: DataLoader,
        current_epoch: int,
        eval_count: int,
        eval_colwise: bool = False,
        load_best_weight: bool = False,
    ):
        if self.valid_ids is None:
            self.valid_ids = valid_loader.dataset.sample_ids

        if eval_colwise:
            preds = self.inference_loop_colwise(valid_loader, "valid", self.best_score_dict)
        else:
            preds = self.inference_loop(valid_loader, "valid", load_best_weight)

        labels = valid_loader.dataset.y
        if self.config.target_shape == "3dim":
            labels = self.convert_target_3dim_to_2dim(labels)
        labels = self.restore_pred(labels)

        if self.pp_run and self.valid_pp_x is None:
            self.load_input_for_postprocess("valid")
        if self.pp_run:
            preds = self.postprocess(preds, run_type="valid")
        if self.out_clip:
            preds = self.clipping_pred(preds)

        eval_idx = [
            i for i, col in enumerate(self.target_cols) if self.factor_dict[col] != 0
        ]  # factor_dictの値が0のものは自動でR2=1になるようにする
        score, indiv_score = evaluate_metric(preds, labels, individual=True, eval_idx=eval_idx)
        save_dict = (
            False if load_best_weight else True
        )  # 通常の学習ループの時のみbest_score_dictの保存を行う
        colwise_score = self.update_best_score(indiv_score, eval_count, save_dict=save_dict)
        message = f"""
            [Valid] :
                Epoch={current_epoch},
                Loss={self.valid_loss.avg:.7f},
                Score={score:.5f},
                Best Col-Wise Score={colwise_score:.5f}
        """
        self.logger.info(clean_message(message))
        return score, colwise_score, preds

    def forward_step(self, batched: torch.Tensor, calc_loss: bool = True):
        if calc_loss:
            x, y = batched
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            if self.config.target_shape == "3dim":
                out = self.convert_target_3dim_to_2dim(out)
                y = self.convert_target_3dim_to_2dim(y)
            loss = self.loss_fn(out, y)
            return out, loss
        else:
            x = batched
            x = x.to(self.device)
            out = self.model(x)
            if self.config.target_shape == "3dim":
                out = self.convert_target_3dim_to_2dim(out)
            return out, None

    def inference_loop(
        self,
        test_loader: DataLoader,
        mode: Literal["valid", "test"],
        load_best_weight: bool = False,
    ):
        self.model.eval()
        if mode == "valid":
            self.valid_loss.reset()
        if load_best_weight:
            self.model.load_state_dict(
                torch.load(self.output_path / f"model{self.save_suffix}_best.pth")
            )

        preds = []
        with torch.no_grad():
            iterations = (
                tqdm(test_loader, total=len(test_loader)) if self.detail_pbar else test_loader
            )
            for batched in iterations:
                if mode == "valid":
                    out, loss = self.forward_step(batched, calc_loss=True)
                    self.valid_loss.update(loss.item(), n=batched[0].size(0))
                elif mode == "test":
                    out, _ = self.forward_step(batched, calc_loss=False)
                preds.append(out.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        preds = self.restore_pred(preds)
        return preds

    def inference_loop_colwise(
        self,
        test_loader: DataLoader,
        mode: Literal["valid", "test"],
        best_score_dict: Dict[str, Tuple[int, float]],
    ):
        self.model.eval()
        if mode == "valid":
            self.valid_loss.reset()

        use_evals = list(set([eval_count for _, (eval_count, _) in best_score_dict.items()]))
        final_preds = np.zeros((len(test_loader.dataset), len(self.target_cols)))
        for eval_count in tqdm(use_evals, desc="Inference Col-Wise Weight"):
            self.model.load_state_dict(
                torch.load(self.output_path / f"model{self.save_suffix}_eval{eval_count}.pth")
            )
            preds = []
            with torch.no_grad():
                iterations = (
                    tqdm(test_loader, total=len(test_loader)) if self.detail_pbar else test_loader
                )
                for batched in iterations:
                    if mode == "valid":
                        out, loss = self.forward_step(batched, calc_loss=True)
                        self.valid_loss.update(loss.item(), n=batched[0].size(0))
                    elif mode == "test":
                        out, _ = self.forward_step(batched, calc_loss=False)
                    preds.append(out.detach().cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            preds = self.restore_pred(preds)

            target_cols = [
                col for col, (count, _) in best_score_dict.items() if count == eval_count
            ]
            for col in target_cols:
                idx = self.target_cols.index(col)
                final_preds[:, idx] = preds[:, idx]
        return final_preds

    def update_best_score(self, indiv_score: List[float], eval_count: int, save_dict: bool):
        for col, score in zip(self.target_cols, indiv_score):
            if col not in self.best_score_dict or score > self.best_score_dict[col][1]:
                self.best_score_dict[col] = (eval_count, score)
        best_colwise_score = (
            np.sum([score for _, score in self.best_score_dict.values()])
            + (368 - len(self.target_cols))
        ) / 368
        if save_dict:
            pickle.dump(
                self.best_score_dict,
                open(self.output_path / f"best_score_dict{self.save_suffix}.pkl", "wb"),
            )
        return best_colwise_score

    def remove_unuse_weights(self):
        use_eval_counts = set([v[0] for v in self.best_score_dict.values()])
        weight_paths = list(self.output_path.glob(f"model{self.save_suffix}_eval*.pth"))
        for path in weight_paths:
            eval_count = int(path.stem.split("_")[-1].replace("eval", ""))
            if eval_count not in use_eval_counts:
                path.unlink()

    def convert_target_3dim_to_2dim(
        self, y: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        y_v = y[:, :, : len(VERTICAL_TARGET_COLS)]
        y_s = y[:, :, len(VERTICAL_TARGET_COLS) :]
        if type(y) == np.ndarray:
            y_v = np.transpose(y_v, (0, 2, 1)).reshape(y.shape[0], -1)
            y_s = y_s.mean(axis=1)
            y = np.concatenate([y_v, y_s], axis=-1)
        else:
            y_v = y_v.permute(0, 2, 1).reshape(y.size(0), -1)
            y_s = y_s.mean(dim=1)
            y = torch.cat([y_v, y_s], dim=-1)
        y = self.alignment_target_idx(y)
        return y

    def alignment_target_idx(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.seq_target_cols is None:
            seq_target_cols = []
            for col in VERTICAL_TARGET_COLS:
                seq_target_cols.extend([f"{col}_{i}" for i in range(60)])
            for col in SCALER_TARGET_COLS:
                seq_target_cols.append(col)
            self.seq_target_cols = seq_target_cols
        align_order = [self.seq_target_cols.index(col) for col in self.target_cols]
        assert len(y.shape) == 2
        y = y[:, align_order]
        return y

    def restore_pred(self, preds: np.ndarray):
        return preds * self.config.y_denominators + self.config.y_numerators

    def clipping_pred(self, preds: np.ndarray):
        for i in range(preds.shape[1]):
            preds[:, i] = np.clip(preds[:, i], self.target_min_max[i][0], self.target_min_max[i][1])
        return preds

    def postprocess(self, preds: np.ndarray, run_type: Literal["valid", "test"]):
        pp_x = self.valid_pp_x if run_type == "valid" else self.test_pp_x
        for y_col, x_col in zip(self.pp_y_cols, self.pp_x_cols):
            if y_col in self.target_cols:
                idx = self.target_cols.index(y_col)
                old_factor = self.old_factor_dict[y_col] if self.mul_old_factor else 1
                preds[:, idx] = (-1 * pp_x[x_col].to_numpy() / 1200) * old_factor
        return preds

    def load_input_for_postprocess(self, data_type: Literal["valid", "test"]):
        if data_type == "valid":
            self.valid_pp_x = (
                pl.scan_parquet(self.input_path / "train_pp.parquet")
                .select(["sample_id"] + self.pp_x_cols)
                .filter(pl.col("sample_id").is_in(self.valid_ids))
                .collect()
            )
            id_df = pl.DataFrame({"sample_id": self.valid_ids})
            self.valid_pp_x = id_df.join(self.valid_pp_x, on="sample_id", how="left")

        elif data_type == "test":
            self.test_pp_x = pl.read_parquet(
                Config.input_path / "test_pp.parquet", columns=["sample_id"] + self.pp_x_cols
            )
            id_df = pl.DataFrame({"sample_id": self.test_ids})
            self.test_pp_x = id_df.join(self.test_pp_x, on="sample_id", how="left")

    def save_oof_df(self, preds: np.ndarray, sample_ids: np.ndarray, target_cols: List[str]):
        oof_df = pl.DataFrame(preds, schema=target_cols)
        oof_df = oof_df.with_columns(sample_id=pl.Series(sample_ids))
        oof_df.write_parquet(self.oof_path / f"oof{self.save_suffix}.parquet")
