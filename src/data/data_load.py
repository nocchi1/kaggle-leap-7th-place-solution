


class CompetitionDataLoader:
    def __init__(self, config):
        self.config = config
        self.input_path = config.input_path
        self.task_type = config.task_type
        self.run_mode = config.run_mode
        self.mul_old_factor = config.mul_old_factor
        self.split_method = (
            'shared' if self.config.shared_valid else
            'fix' if self.run_mode in ['hf', 'full'] else
            'time' if self.run_mode == 'dev' else
            'random'
        )
    
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if not (self.input_path / 'train_pp.parquet').exists():
            if not (self.input_path / 'train.parquet').exists():
                self.convert_csv_to_parquet('train')
            if not (self.input_path / 'test.parquet').exists():
                self.convert_csv_to_parquet('test')
            if not (self.input_path / 'sample_submission.parquet').exists():
                self.convert_csv_to_parquet('sample_submission')
            if not (self.input_path / 'sample_submission_old.parquet').exists():
                self.convert_csv_to_parquet('sample_submission_old')
                
            train_df = pl.read_parquet(self.input_path / 'train.parquet')
            test_df = pl.read_parquet(self.input_path / 'test.parquet')
            train_df, test_df = self.shrink_memory(train_df, test_df)
            
            train_df.write_parquet(self.input_path / 'train_pp.parquet')
            test_df.write_parquet(self.input_path / 'test_pp.parquet')
            (self.input_path / 'train.parquet').unlink()
            (self.input_path / 'test.parquet').unlink()
        else:
            train_df = pl.read_parquet(self.input_path / 'train_pp.parquet')
            test_df = pl.read_parquet(self.input_path / 'test_pp.parquet')
            
        train_df = train_df.with_columns(time_id = pl.col('sample_id') // 384)
        train_df = self.downsample(train_df)
        # train_df, test_df = self.merge_grid_info(train_df, test_df)
        train_df = self.split_validation(train_df)
        if self.mul_old_factor:
            train_df = multiply_old_factor(train_df, self.input_path)
        train_df = train_df.drop(['time_id'])
        return train_df, test_df
    
    def convert_csv_to_parquet(self, file_name: str):
        train_df = pl.read_csv(self.input_path / f'{file_name}.csv')
        train_df.write_parquet(self.input_path / f'{file_name}.parquet')
        (self.input_path / f'{file_name}.csv').unlink()
        
    def shrink_memory(self, train_df: pl.DataFrame, test_df: pl.DataFrame):
        train_df = train_df.with_columns(
            sample_id = pl.col('sample_id').map_elements(lambda x: int(x.split('_')[1]), return_dtype=pl.Int32)
        )
        test_df = test_df.with_columns(
            sample_id = pl.col('sample_id').map_elements(lambda x: int(x.split('_')[1]), return_dtype=pl.Int32)
        )
        train_df, test_df = self.shrink_float(train_df, test_df)
        return train_df, test_df

    def shrink_float(self, train_df: pl.DataFrame, test_df: pl.DataFrame):
        train_exprs = []
        test_exprs = []
        feat_cols = [col for col in train_df.columns if col != 'sample_id']
        for col in feat_cols:
            abs_min_val = train_df[col].abs().min()
            if abs_min_val > 1e-37: # アンダーフローのリスクがないもののみをFP32に変換
                train_exprs.append(pl.col(col).cast(pl.Float32))
                if col in test_df.columns:
                    test_exprs.append(pl.col(col).cast(pl.Float32))
        train_df = train_df.with_columns(train_exprs)
        test_df = test_df.with_columns(test_exprs)
        return train_df, test_df
    
    def downsample(self, train_df: pl.DataFrame):
        time_ids = sorted(train_df['time_id'].unique())
        start_idx = 26200 if self.run_mode == 'debug' else 18000 if self.run_mode == 'dev' else 0
        use_ids = time_ids[start_idx:]
        train_df = train_df.filter(pl.col('time_id').is_in(use_ids))
        return train_df
    
    def split_validation(self, train_df: pl.DataFrame):
        if self.split_method == 'random':
            _, valid_idx = train_test_split(
                np.arange(len(train_df)), 
                test_size=self.config.valid_ratio, 
                shuffle=True,
                random_state=self.config.seed
            )
            train_df = train_df.with_row_index('dummy_idx')
            train_df = train_df.with_columns(
                fold = (
                    pl.when(pl.col('dummy_idx').is_in(valid_idx))
                    .then(pl.lit(0)).otherwise(pl.lit(1))
                    .cast(pl.Int8)
                )
            )
            train_df = train_df.drop('dummy_idx')
            
        elif self.split_method == 'time':
            time_ids = sorted(train_df['time_id'].unique())
            valid_start = int(len(time_ids) * (1 - self.config.valid_ratio))
            valid_ids = time_ids[valid_start:]
            train_df = train_df.with_columns(
                fold = (
                    pl.when(pl.col('time_id').is_in(valid_ids))
                    .then(pl.lit(0)).otherwise(pl.lit(1))
                    .cast(pl.Int8)
                )
            )
            
        elif self.split_method == 'fix':
            valid_id_path = self.input_path / 'valid_id.parquet'
            if valid_id_path.exists():
                valid_id_df = pl.read_parquet(self.input_path / 'valid_id.parquet')
            else:
                train_df = train_df.with_columns(
                    fold = (
                        pl.when(pl.col('time_id') >= 23652) # 時系列の末尾1Mサンプル切り取ったものをvalidとする
                        .then(pl.lit(0)).otherwise(pl.lit(1))
                        .cast(pl.Int8)
                    )
                )
                valid_id_df = train_df.select('sample_id', 'fold')
                valid_id_df.write_parquet(valid_id_path)
            train_df = train_df.join(valid_id_df, on='sample_id', how='left')
            
        elif self.split_method == 'shared':
            # チーム内で共有したvalidを使用する
            valid_pp_path = self.input_path / '18_pp.parquet'
            if valid_pp_path.exists():
                valid_df = pl.read_parquet(valid_pp_path)
            else:
                valid_df = pl.read_parquet(self.input_path / '18.parquet')
                valid_df = self.pp_shared_validation(train_df, valid_df)
                valid_df.write_parquet(self.input_path / '18_pp.parquet')
            valid_df = valid_df.with_columns(fold = pl.lit(0).cast(pl.Int8))
            train_df = train_df.with_columns(fold = pl.lit(1).cast(pl.Int8))
            train_df = pl.concat([train_df, valid_df], how='diagonal') # validにはgrid_id, time_idが存在しない
            
        return train_df
    
    def pp_shared_validation(self, train_df: pl.DataFrame, valid_df: pl.DataFrame):
        exprs = []
        for col in train_df.columns:
            if col in valid_df.columns:
                exprs.append(pl.col(col).cast(train_df[col].dtype))
        valid_df = valid_df.with_columns(exprs)
        
        # HFデータを参照して緯度経度を付与する
        hf_files = [f'0008-{str(i).zfill(2)}' for i in range(2, 13)] + ['0009-01']
        hf_df = []
        for file in hf_files:
            hf_df.append(
                pl.scan_parquet(self.input_path / 'huggingface' / f'{file}_pp.parquet')
                .with_columns(exprs)
                .collect()
            )
        hf_df = pl.concat(hf_df)
        
        # 以下のカラムの組み合わせからレコードを一意に特定
        check_cols = (
            [f'state_t_{i}' for i in range(60)] + 
            [f'state_v_{i}' for i in range(60)] +
            [f'state_u_{i}' for i in range(60)]
        )
        valid_df = valid_df.with_columns(
            (pl.col(col) * 1000).cast(pl.Int32).alias(f'{col}_int')
            for col in check_cols
        )
        hf_df = hf_df.with_columns(
            (pl.col(col) * 1000).cast(pl.Int32).alias(f'{col}_int')
            for col in check_cols
        )
        check_int_cols = [f'{col}_int' for col in check_cols]
        hf_df = hf_df.select(check_int_cols + ['lat', 'lon'])
        valid_df = valid_df.join(hf_df, on=check_int_cols, how='left')
        valid_df = valid_df.drop(check_int_cols).select([col for col in train_df.columns if col in valid_df])
        del hf_df; gc.collect()
        return valid_df