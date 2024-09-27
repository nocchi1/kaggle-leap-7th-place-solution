#!/bin/bash

. .venv/bin/activate

# Download Kaggle Competition Data
if [ "$1" = "true" ]; then
  kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f train.csv -p data/input
else
  # This dataset is a downsampled version (1M samples).
  kaggle datasets download ryotak12/leap-train-downsampling -p data/input
fi
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test.csv -p data/input
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test_old.csv -p data/input
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f sample_submission.csv -p data/input
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f sample_submission_old.csv -p data/input

# Download Necessary Data for Host Repo



# 指定する引数
# - コンペ学習データを全てダウンロードするか
# - HFデータをダウンロードするか
