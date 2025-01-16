# Convert MAEST models to Tensorflow
Scripts to convert MAEST models to Tensorflow


# Steps 
1. Clone this repo and init submodules
```bash
git clone --recurse-submodules <URL>
```

2. Create a conda env for more control (optional)
```bash
conda create -y --name m_conv python=3.10 && conda activate m_conv
```

3. Install tooling
```bash
 pip install -r requirements.txt
```

4. Install [GitHub LFS]() to download HF models

5. Setup and env var with your [HuggingFace token]()
```bash
HF_TOKEN=<HF_TOKEN>
```

6. Download the MAEST weights 
```
./download_models.sh
```

7. Convert models to ONNX with [Optimum]()
```bash
./convert_torch_to_onnx.sh
```

8. [Prepare](# Model preparation) and convert models to TensorFlow frozen models with [onnx2tf]()
```bash
./convert_onnx_to_tf.sh
```

# Model preparation

