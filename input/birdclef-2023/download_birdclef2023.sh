#!/bin/bash
pip install kaggle
export KAGGLE_CONFIG_DIR=$1
kaggle competitions download -c birdclef-2023 -p input/birdclef-2023
unzip input/birdclef-2023/birdclef-2023.zip -d input/birdclef-2023
rm input/birdclef-2023/birdclef-2023.zip
