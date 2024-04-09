#!/bin/bash
pip install kaggle
export KAGGLE_CONFIG_DIR=$1
kaggle competitions download -c birdclef-2022 -p input/birdclef-2022
unzip input/birdclef-2022/birdclef-2022.zip -d input/birdclef-2022
rm input/birdclef-2022/birdclef-2022.zip
