import numpy as np
from sklearn.metrics import r2_score


def evaluate_metric(y_pred: np.ndarray, y_true: np.ndarray, individual: bool = False, eval_idx: list[int] | None = None) -> float | tuple[float, list[float]]:
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
