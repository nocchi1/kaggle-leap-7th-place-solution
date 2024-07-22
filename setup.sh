#!/bin/bash

# 環境構築
rye pin 3.10
rye sync
. venv/bin/activate

# data/ ディレクトリ作成
cd data
mkdir input input/additional input/huggingface output oof