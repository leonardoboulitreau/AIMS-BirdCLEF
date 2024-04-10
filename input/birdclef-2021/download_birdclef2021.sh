#!/bin/bash
pip install kaggle
export KAGGLE_CONFIG_DIR=$1
kaggle competitions download -c birdclef-2021 -p input/birdclef-2021
unzip input/birdclef-2021/birdclef-2021.zip -d input/birdclef-2021
rm input/birdclef-2021/birdclef-2021.zip
