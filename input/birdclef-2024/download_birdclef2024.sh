#!/bin/bash
pip install kaggle
export KAGGLE_CONFIG_DIR=$1
kaggle competitions download -c birdclef-2024 -p input/birdclef-2024
unzip input/birdclef-2024/birdclef-2024.zip -d input/birdclef-2024
rm input/birdclef-2024/birdclef-2024.zip