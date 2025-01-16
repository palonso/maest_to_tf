set -e

dir="keras-onnx-tensorflow-converter/src/"
name_in=serving_default_melspectrogram


for model in discogs-maest-30s-pw-129e-519l
do
    cp models_out/${model}.onnx models_out/${model}.tmp.onnx

    echo "Fixing model name"
    python ${dir}change_interface_names.py \
	-f \
        models_out/${model}.tmp.onnx \
        models_out/${model}.tmp.onnx \
	-i ${name_in} \
	-o logits

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
            --output-shape "batch_size","timestamps",768
    done

    echo "Adding sigmoid node"
    # Add a Sigmoid output node since it was not included in the Pytorch model.
    python ${dir}add_output_node.py \
        models_out/${model}.tmp.onnx \
        models_out/${model}.tmp.onnx \
	/classifier/dense/Gemm \
        activations \
        --node-type Sigmoid \
        --output-shape "batch_size",519 \

    onnx2tf \
        -i models_out/${model}.tmp.onnx \
        -ois ${name_in}:"-1",1876,96 \
        -kat ${name_in} \
        -cotof \
        -coion \
        -otfv1pb

    echo "Done with ${model}"
done
