set -e

dir="keras-onnx-tensorflow-converter/src/"

timestamps_in_5=316
timestamps_in_10=626
timestamps_in_20=1256
timestamps_in_30=1876


n_nodes=24

for model in \
    discogs-maest-5s-pw-129e \
    # discogs-maest-10s-fs-129e \
    # discogs-maest-10s-dw-75e \
    # discogs-maest-10s-pw-129e \
    # discogs-maest-20s-pw-129e \
    # discogs-maest-30s-pw-129e \
    # discogs-maest-30s-pw-73e-ts \
    # discogs-maest-30s-pw-129e-519l
do

    echo "extracting model head (n last layers)"

    python ${dir}extract_model_head.py \
        "models_out/${model}.multi_out.onnx" \
        "classifiers_out/genre_discogs400-${model}.onnx" \
        ${n_nodes}


    echo "converting opset"

    python ${dir}modify_onnx_opset.py \
        "classifiers_out/genre_discogs400-${model}.onnx" \
        "classifiers_out/genre_discogs400-${model}.onnx" \
        --opset 18 \
        --ir-version 9 \
        --model-version 7


    echo "fixing model name"

    name_in=embeddings
    python ${dir}change_interface_names.py \
        -f \
        "classifiers_out/genre_discogs400-${model}.onnx" \
        "classifiers_out/genre_discogs400-${model}.onnx" \
        -i ${name_in} 


    echo "converting to Tensorflow"

    onnx2tf \
        -i "classifiers_out/genre_discogs400-${model}.onnx" \
        -kat ${name_in} \
        -cotof \
        -coion \
        -otfv1pb
    
    mv  saved_model/genre_discogs400-${model}_float32.pb classifiers_out/genre_discogs400-${model}.pb

done

echo "done!"
