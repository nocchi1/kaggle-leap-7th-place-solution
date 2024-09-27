def evaluate_metric(y_pred: np.ndarray, y_true: np.ndarray, individual: bool = False, eval_idx: List[int] | None = None) -> float | Tuple[float, List[float]]:
    total_target_num = 368
    if eval_idx is not None:
        y_pred, y_true = y_pred[:, eval_idx], y_true[:, eval_idx]
    score = r2_score(y_true, y_pred, force_finite=True)
    # y_pred内に存在しないカラムは1として計算する (sub_factorが0のカラム, 後処理を適用するカラム)
    if total_target_num - y_pred.shape[1] > 0:
        score = (score * y_pred.shape[1] + (total_target_num - y_pred.shape[1])) / total_target_num
    if individual:
        indiv_score = [r2_score(y_true[:, i], y_pred[:, i], force_finite=True) for i in range(y_true.shape[1])]
        return score, indiv_score
    return score


def get_io_columns(task_type: Literal["main", "grid_pred"]) -> Tuple[List[str], List[str]]:
    input_cols = []
    for col in VERTICAL_INPUT_COLS:
        input_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in SCALER_INPUT_COLS:
        input_cols.append(col)
    if task_type == "grid_pred":
        input_cols = [col for col in input_cols if col not in ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]]
    target_cols = []
    for col in VERTICAL_TARGET_COLS:
        target_cols.extend([f"{col}_{i}" for i in range(60)])
    for col in SCALER_TARGET_COLS:
        target_cols.append(col)
    target_cols = [col for col in target_cols if col not in ZERO_WEIGHT_TARGET_COLS + PP_TARGET_COLS]
    return input_cols, target_cols


def get_sub_factor(input_path: PosixPath, old: bool = False) -> Dict[str, float]:
    suffix = "_old" if old else ""
    sample_df = pl.read_parquet(input_path / f"sample_submission{suffix}.parquet", n_rows=1)
    factor_dict = dict(zip(sample_df.columns[1:], sample_df.to_numpy()[0][1:]))
    return factor_dict


def multiply_old_factor(train_df: pl.DataFrame, factor_path: PosixPath) -> pl.DataFrame:
    factor_dict = get_sub_factor(factor_path, old=True)
    exprs = []
    for col, factor in factor_dict.items():
        exprs.append(pl.col(col) * factor)
    train_df = train_df.with_columns(exprs)
    return train_df