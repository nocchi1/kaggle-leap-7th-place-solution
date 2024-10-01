from tqdm.auto import tqdm

from src.data.validation import split_validation
from src.utils import convert_csv_to_parquet, multiply_old_factor, shrink_memory


class HFPreprocessor:
    def __init__(self, config):
        self.config = config
        self.hf_files = list((self.config.add_path / "huggingface").glob("*.parquet"))

        self.input_cols, self.target_cols = get_io_columns(config)
        self.fer = FeatureEngineering(config)
        self.ppr = Preprocessor(config)
        self.input_clip_dict = pickle.load(open(self.config.output_path / "input_clip_dict.pkl", "rb"))

        # shared_validをサンプリングしているyear-month
        self.valid_ym = ["0008-07", "0008-08", "0008-09", "0008-10", "0008-11", "0008-12", "0009-01"]

    def shrink_file_size(self):
        shrink_num = len([file for file in self.hf_files if "_shrinked" in file.stem])
        if len(self.hf_files) > 0 and shrink_num == 0:
            refer_df = pl.read_parquet(self.config.input_path / "train_shrinked.parquet", n_rows=100)
            for file in tqdm(self.hf_files):
                df = pl.read_parquet(file)
                df = self.shrink_memory(df, refer_df)
                df.write_parquet(self.input_path / "huggingface" / f"{file.stem}_shrinked.parquet")
                file.unlink()
            self.hf_files = list((self.input_path / "huggingface").glob("*.parquet"))

    def convert_numpy_array(self, unlink_parquet: bool = True):
        npy_path = self.config.add_path / "huggingface" / "npy"
        npy_path.mkdir(exist_ok=True, parents=True)
        self.hf_files = sorted(self.hf_files, key=lambda x: x.stem)
        for file in tqdm(self.hf_files):
            ym = file.stem.replace("_shrinked", "")
            if (npy_path / "X_{ym}.npy").exists():
                continue
            # valid_ym以降のymは使用しない
            if ym in self.valid_ym:
                continue

            df = pl.read_parquet(file)
            if self.config.mul_old_factor:
                df = multiply_old_factor(self.config.input_path, df)

            df = self.fer.feature_engineering(df)
            df = self.ppr._input_scaling(df, self.config.input_scale_method, compute_stats=False)
            df = self.ppr._target_scaling(df, self.config.target_scale_method, compute_stats=False)
            df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))

            df, _ = clipping_input(None, df, self.input_cols, self.input_clip_dict)
            if self.config.multi_task:
                df = self.ppr._get_forward_and_back_target(df)

            X_train = self.ppr._convert_input_array(df, self.config.input_shape)
            y_train = self.ppr._convert_target_array(df, self.config.target_shape)
            np.save(npy_path / f"X_{ym}.npy", X_train)
            np.save(npy_path / f"y_{ym}.npy", y_train)

            if self.config.multi_task:
                y_train_mt = np.concatenate(
                    [
                        self.ppr._convert_target_array(df, self.config.target_shape, suffix="_lag"),
                        self.ppr._convert_target_array(df, self.config.target_shape, suffix="_lead"),
                    ],
                    axis=-1,
                )
                np.save(npy_path / f"y_mt_{ym}.npy", y_train_mt)
                del y_train_mt
                gc.collect()

            del df, X_train, y_train
            gc.collect()
            if unlink_parquet:
                file.unlink()
