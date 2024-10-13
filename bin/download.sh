#!/bin/bash

. .venv/bin/activate

full_train_download=${1:-"true"}
hf_download=${2:-"true"}

# Download Kaggle Competition Data
if [ "$full_train_download" = "true" ]; then
  kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f train.csv -p data/input
else
  # This dataset is a downsampled version (1M samples).
  kaggle datasets download ryotak12/leap-train-downsampling -p data/input
fi
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test.csv -p data/input
# kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test_old.csv -p data/input
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f sample_submission.csv -p data/input
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f sample_submission_old.csv -p data/input

# Install unzip if not installed
if ! command -v unzip &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y unzip
fi

# Unzip the downloaded files
for file in data/input/*.zip; do
    if [ -f "$file" ]; then
        unzip -o "$file" -d data/input &> /dev/null
        rm "$file"
    fi
done

# Download Necessary Data for Host Repo
curl -L -o data/input/additional/ClimSim_low-res_grid-info.nc https://github.com/leap-stc/ClimSim/raw/main/grid_info/ClimSim_low-res_grid-info.nc
curl -L -o data/input/additional/output_scale.nc https://github.com/leap-stc/ClimSim/raw/main/preprocessing/normalizations/outputs/output_scale.nc

# Download Shared Validation Data - This data is sampled at a 1/7 interval over the period from 0008-02 to 0009-01
kaggle datasets download ryotak12/leap-shared-validation -f 18.parquet -p data/input

# Download HF Additional Data
if [ "$hf_download" = "true" ]; then
    python src/data/hf_download.py
fi
