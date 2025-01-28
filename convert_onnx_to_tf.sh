set -e

dir="keras-onnx-tensorflow-converter/src/"
name_in=melspectrogram
melspectrogram_bands=96

timestamps_in_5=316
timestamps_in_10=626
timestamps_in_20=1256
timestamps_in_30=1876

# timestamps_out_30=1685

embeddings=768

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
    cp models_out/${model}.onnx models_out/${model}.tmp.onnx

    echo "Fixing model name"
    python ${dir}change_interface_names.py \
	-f \
        models_out/${model}.tmp.onnx \
        models_out/${model}.tmp.onnx \
	-i ${name_in} \
	-o logits

    # select embedding duration
    if [[ $model = *5s* ]]; then
        timestamps_in=${timestamps_in_5}

    elif [[ $model = *10s* ]]; then
        timestamps_in=${timestamps_in_10}

    elif [[ $model = *20s* ]]; then
        timestamps_in=${timestamps_in_20}

    elif [[ $model = *30s* ]]; then
        timestamps_in=${timestamps_in_30}
    fi

    # select num of labels
    if [[ $model = *519l* ]]; then
        classes=519

    else 
        classes=400
    fi

    echo "This models produces" ${timestamps_in} "timestamps" 

    echo "Adding output nodes"
    for layer in {0..11}
    do
        # apply left zero-pad so that layer has always 2 digits:
        n_layer=$(printf "%02d\n" $layer)
        python ${dir}add_output_node.py \
            models_out/${model}.tmp.onnx \
            models_out/${model}.tmp.onnx \
            /audio_spectrogram_transformer/encoder/layer.${layer}/output/Add \
            layer_${n_layer}_embeddings \
            --output-shape "batch_size" "n" ${embeddings}
    done


    echo "Adding sigmoid node"
    # Add a Sigmoid output node since it was not included in the Pytorch model.
    python ${dir}add_output_node.py \
        models_out/${model}.tmp.onnx \
        models_out/${model}.tmp.onnx \
	/classifier/dense/Gemm \
        activations \
        --node-type Sigmoid \
        --output-shape "batch_size" ${classes} \


    onnx2tf \
        -i models_out/${model}.tmp.onnx \
        -ois ${name_in}:1,${timestamps_in},${melspectrogram_bands} \
        -kat ${name_in} \
        -cotof \
        -coion \
        -otfv1pb

        # --param_replacement_file param_replacement_file.json \
        # --not_use_onnxsim \

    cp saved_model/${model}.tmp_float32.pb models_out/${model}.pb

    mv models_out/${model}.tmp.onnx models_out/${model}.onnx

    echo "Done with ${model}"
done
