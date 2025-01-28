set -e


dir="keras-onnx-tensorflow-converter/src/"

timestamps_in_5=281
timestamps_in_10=560
timestamps_in_20=1127
timestamps_in_30=1685

n_nodes=24

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

    if [[ $model = *5s* ]]; then
        timestamps=${timestamps_in_5}

    elif [[ $model = *10s* ]]; then
        timestamps=${timestamps_in_10}

    elif [[ $model = *20s* ]]; then
        timestamps=${timestamps_in_20}

    elif [[ $model = *30s* ]]; then
        timestamps=${timestamps_in_30}
    fi


    if [[ $model = *519l ]]; then
        classes=519
    else
        classes=400
    fi

    echo "extracting model head (n last layers)"
    python ${dir}extract_model_head.py \
        "models_out/${model}.onnx" \
        "classifiers_out/genre_discogs${classes}-${model}.onnx" \
        ${n_nodes} \
        "layer_11_embeddings" \
        --input-shape "batch_size" ${timestamps} 768 \
        --output-node-names "logits" "activations"


    echo "fixing model name"
    name_in=embeddings
    python ${dir}change_interface_names.py \
        -f \
        "classifiers_out/genre_discogs${classes}-${model}.onnx" \
        "classifiers_out/genre_discogs${classes}-${model}.onnx" \
        -i ${name_in} 


    echo "converting to TensorFlow"
    onnx2tf \
        -i "classifiers_out/genre_discogs${classes}-${model}.onnx" \
        -kat ${name_in} \
        -cotof \
        -coion \
        -otfv1pb
    
    mv  saved_model/genre_discogs${classes}-${model}_float32.pb classifiers_out/genre_discogs${classes}-${model}.pb

done

echo "done!"
