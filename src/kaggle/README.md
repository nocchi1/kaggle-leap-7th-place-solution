# Frequently Used Commands

- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md)

```bash
# Download Competition Datasets
python -m download -c leap-atmospheric-physics-ai-climsim -p data_path -u

# Upload Datasets
python -m upload -p data_path -t dataset_title # --public

# Download Datasets
kaggle datasets download zillow/zecon -p data_path --unzip

# Submit to Competition
kaggle competitions submit leap-atmospheric-physics-ai-climsim -f file_path -m message