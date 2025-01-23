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

    optimum-cli export onnx \
        --model models_in/"$model" \
        --task audio-classification \
        models_out/ 

    mv models_out/model.onnx models_out/"$model".onnx
    rm models_out/config.json
done
