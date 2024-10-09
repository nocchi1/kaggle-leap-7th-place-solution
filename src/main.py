import gc
from pathlib import Path, PosixPath

import polars as pl

from src.data import DataProvider, FeatureEngineering, Preprocessor
from src.utils import TimeUtil, get_config, get_logger, seed_everything
from src.utils.competition_utils import clipping_input

config = get_config(exp, config_dir=Path("../config"))
logger = get_logger(config.output_path)
logger.info(
    f"exp: {exp} | run_mode={config.run_mode}, multi_task={config.multi_task}, loss_type={config.loss_type}"
)

seed_everything(config.seed)

with TimeUtil.timer("Data Loading..."):
    dpr = DataProvider(config)
    train_df, test_df = dpr.load_data()

with TimeUtil.timer("Feature Engineering..."):
    fer = FeatureEngineering(config)
    train_df = fer.feature_engineering(train_df)
    test_df = fer.feature_engineering(test_df)

with TimeUtil.timer("Scaling Features..."):
    ppr = Preprocessor(config)
    train_df, test_df = ppr.scaling(train_df, test_df)
    input_cols, target_cols = ppr.input_cols, ppr.target_cols
    if config.task_type == "grid_pred":
        train_df = train_df.drop(target_cols)

    valid_df = train_df.filter(pl.col("fold") == 0)
    train_df = train_df.filter(pl.col("fold") != 0)
    valid_df, input_clip_dict = clipping_input(train_df, valid_df, input_cols)
    test_df, _ = clipping_input(None, test_df, input_cols, input_clip_dict)

with TimeUtil.timer("Converting to arrays for NN..."):
    array_data = ppr.convert_numpy_array(train_df, valid_df, test_df)
    del train_df, valid_df, test_df
    gc.collect()
