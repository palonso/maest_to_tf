from time import time
import numpy as np
import torch
import onnxruntime as ort
from essentia.standard import TensorflowPredict
from essentia import Pool
from transformers import ASTForAudioClassification


# Data prep
np.random.seed(23)

for model_stem in [
    "discogs-maest-5s-pw-129e",
    "discogs-maest-10s-fs-129e",
    "discogs-maest-10s-dw-75e",
    "discogs-maest-10s-pw-129e",
    "discogs-maest-20s-pw-129e",
    "discogs-maest-30s-pw-129e",
    "discogs-maest-30s-pw-73e-ts",
    "discogs-maest-30s-pw-129e-519l",
]:

    print(f"Processing {model_stem}")

    data_len = 0
    if "5s" in model_stem:
        data_len = 316
    elif "10s" in model_stem:
        data_len = 626
    elif "20s" in model_stem:
        data_len = 1256
    elif "30s" in model_stem:
        data_len = 1876

    data = np.random.rand(1, 1, data_len, 96).astype("float32")


    model_ort = f"models_out/{model_stem}.onnx"
    session = ort.InferenceSession(model_ort)

    # Prepare inputs and outputs
    inputs = {"melspectrogram": data.squeeze(1)}
    output_names = [f"layer_{i:02d}_embeddings" for i in range(0, 12)]
    output_names.append("logits")

    st = time()
    outputs = session.run(output_names, inputs)
    eta = time() - st
    print(f"ONNX realtime ETA: {eta:.2f}")
    outputs_onnx = np.stack([h for h in outputs[:-1]]).squeeze()
    logits_onnx = outputs[-1].squeeze()

    model_tf = f"models_in/{model_stem}/"

    model_trans = ASTForAudioClassification.from_pretrained(model_tf, output_hidden_states=True)
    model_trans.eval()

    data_pt = torch.Tensor(data).squeeze(1)

    # Trans infer
    st = time()
    out_trans = model_trans(data_pt)

    eta = time() - st
    print(f"Transformers ETA: {eta:.2f}")


    outputs_tf = np.stack([h.detach().numpy().squeeze() for h in out_trans.hidden_states])
    logits_tf = out_trans.logits.detach().numpy().squeeze()
    print(f"Transformers output shapes: {outputs_tf.shape}")

    # Essentia infer
    inputs = ["melspectrogram"]

    # compare intermediate states
    outputs = [f"PartitionedCall/Identity_{i}" for i in range(1, 13)]
    outputs.append("PartitionedCall/Identity")

    model_es = f"models_out/{model_stem}.pb"
    model = TensorflowPredict(graphFilename=model_es, inputs=inputs, outputs=outputs)

    pool = Pool()
    pool.set(inputs[0], data)

    # Ess infer
    st = time()
    pool_out = model(pool)

    eta = time() - st
    print(f"Essentia ETA: {eta:.2f}")

    outputs_es = np.stack([pool_out[o] for o in outputs[:-1]]).squeeze()
    logits_es = pool_out[outputs[-1]].squeeze()
    print(f"Essentia output shapes: {outputs_es.shape}")

    print(f"Comparing logits")
    np.testing.assert_allclose(logits_es, logits_tf, rtol=1e-03, atol=1e-03)
    print("Essentia OK!")

    np.testing.assert_allclose(logits_onnx, logits_tf, rtol=1e-03, atol=1e-03)
    print("ONNX OK!")

    for layer in range(12):
        print(f"Processing layer {layer}")
        np.testing.assert_allclose(outputs_es[layer], outputs_tf[layer + 1], rtol=1e-03, atol=1e-03)
        print("Essentia OK!")

        np.testing.assert_allclose(outputs_onnx[layer], outputs_tf[layer + 1], rtol=1e-03, atol=1e-03)
        print("ONNX OK!")
