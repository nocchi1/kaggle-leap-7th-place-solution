# Frequently Used Commands

[Kaggle API Documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md)

```bash
# Download Competition Datasets -> Run download.sh to download all necessary data
kaggle competitions download leap-atmospheric-physics-ai-climsim -p data_path

# Submit to Competition
kaggle competitions submit leap-atmospheric-physics-ai-climsim -f file_path -m message

# Download Datasets
kaggle datasets download zillow/zecon -p data_path --unzip

# Upload Datasets
python src/kaggle/upload.py -p data_path -t dataset_title # --public
