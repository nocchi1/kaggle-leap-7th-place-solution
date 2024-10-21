# ruff: noqa: PLR0915
import argparse
import gc
import pickle
from pathlib import Path

import polars as pl

from src.data import DataProvider, FeatureEngineering, HFPreprocessor, PostProcessor, Preprocessor
from src.train import GridPredTrainer, Trainer, get_dataloader
from src.utils import TimeUtil, get_config, get_logger, seed_everything
from src.utils.competition_utils import clipping_input


def main():
    # Setup
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_name", help="config file name", type=str, required=True)
    args = parser.parse_args()

    config = get_config(args.config_name, config_dir=Path("./config"))
    logger = get_logger(config.output_path)
    logger.info(f"Start exp={config.exp}, run_mode={config.run_mode}")
    seed_everything(config.seed)

    # Preprocess
    with TimeUtil.timer("Data Loading..."):
        dpr = DataProvider(config)
        train_df, test_df = dpr.load_data()

    with TimeUtil.timer("Feature Engineering..."):
        fer = FeatureEngineering(config)
        train_df = fer.feature_engineering(train_df)
        test_df = fer.feature_engineering(test_df)

    with TimeUtil.timer("Scaling and Clipping Features..."):
        ppr = Preprocessor(config)
        train_df, test_df = ppr.scaling(train_df, test_df)

        valid_df = train_df.filter(pl.col("fold") == 0)
        train_df = train_df.filter(pl.col("fold") != 0)
        valid_df, input_clip_dict = clipping_input(train_df, valid_df, ppr.input_cols)
        test_df, _ = clipping_input(None, test_df, ppr.input_cols, input_clip_dict)
        pickle.dump(input_clip_dict, open(config.output_path / "input_clip_dict.pkl", "wb"))

    with TimeUtil.timer("Converting to arrays for NN..."):
        array_data = ppr.convert_numpy_array(train_df, valid_df, test_df)
        del train_df, valid_df, test_df
        gc.collect()

    if config.run_mode == "hf":
        with TimeUtil.timer("HF Data Preprocessing..."):
            del array_data["train_ids"], array_data["X_train"], array_data["y_train"]
            gc.collect()

            hf_ppr = HFPreprocessor(config)
            hf_ppr.shrink_file_size()
            hf_ppr.convert_numpy_array(unlink_parquet=True)

    with TimeUtil.timer("Creating Torch DataLoader..."):
        if config.run_mode == "hf":
            train_loader = get_dataloader(config, hf_read_type="npy", is_train=True)
        else:
            train_loader = get_dataloader(
                config,
                array_data["train_ids"],
                array_data["X_train"],
                array_data["y_train"],
                is_train=True,
            )
        valid_loader = get_dataloader(
            config,
            array_data["valid_ids"],
            array_data["X_valid"],
            array_data["y_valid"],
            is_train=False,
        )
        test_loader = get_dataloader(config, array_data["test_ids"], array_data["X_test"], is_train=False)
        del array_data
        gc.collect()

    # Training
    if config.task_type == "main":
        # Main Task Training
        trainer = Trainer(config, logger)
        best_score, best_cw_score, best_epochs = trainer.train(
            train_loader,
            valid_loader,
            colwise_mode=True,
        )
        logger.info(
            f"First Training Results: best_score={best_score}, best_cw_score={best_cw_score}, best_epochs={best_epochs}"
        )

        # Additional Training
        config.loss_type = config.add_loss_type
        config.epochs = config.add_epochs
        config.lr = config.add_lr
        config.first_cycle_epochs = config.add_first_cycle_epochs

        trained_weights = sorted(
            config.output_path.glob("model_eval*.pth"),
            key=lambda x: int(x.stem.split("_")[-1].replace("eval", "")),
        )

        trainer = Trainer(config, logger)
        best_score, best_cw_score, best_epochs = trainer.train(
            train_loader,
            valid_loader,
            colwise_mode=True,
            retrain=True,
            retrain_weight_name=trained_weights[-1].stem,
            retrain_best_score=best_score,
        )
        logger.info(
            f"Additional Training Results: best_score={best_score}, best_cw_score={best_cw_score}, best_epochs={best_epochs}"
        )

        # Inference
        pred_df = trainer.test_predict(test_loader, eval_method="single")

        # PostProcess
        oof_df = pl.read_parquet(config.oof_path / "oof.parquet")
        por = PostProcessor(config, logger)
        oof_df, sub_df = por.postprocess(oof_df, pred_df)
        logger.info(f"OOF: {oof_df.shape}, Submission: {sub_df.shape}")

    elif config.task_type == "grid_pred":
        # Grid Prediction Task Training (This task is not executed)
        trainer = GridPredTrainer(config, logger)
        best_score, best_epochs = trainer.train(train_loader, valid_loader)
        logger.info(f"Grid Pred Training Results: best_score={best_score}, best_epochs={best_epochs}")

        # Inference
        pred_df = trainer.test_predict(test_loader)
        test_grid_id = pl.read_parquet(config.output_path / "submission.parquet")
        test_grid_id.write_parquet(config.add_path / "test_grid_id.parquet")


if __name__ == "__main__":
    main()
