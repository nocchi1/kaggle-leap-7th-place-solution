#!/bin/bash

# ryeの設定と仮想環境の構築
curl -sSf https://rye-up.com/get | bash
rye pin $(cat .python-version)
rye sync
