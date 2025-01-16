#! /bin/bash

set -e

for model in \
    discogs-maest-5s-pw-129e \
    discogs-maest-10s-fs-129e \
    discogs-maest-10s-dw-75e \
    discogs-maest-10s-pw-129e \
    discogs-maest-20s-pw-129e \
    discogs-maest-30s-pw-129e \
    discogs-maest-30s-pw-73e-ts \
    discogs-maest-30s-pw-129e-519l
do
    echo processing "$model"

    git clone git@hf.co:mtg-upf/"$model" models_in/

    cd models_in/"$model"

    git remote set-url origin https://palonso:${HF_TOKEN}@huggingface.co/mtg-upf/"$model"
    git lfs pull
done
