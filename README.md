```bash
conda activate system
pip install torch==2.5.0 torchvision

pip install hdi1 --no-build-isolation

pip uninstall torch torchvision -y
pip install torch==2.5.0 torchvision
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

python -m hdi1 "A cat holding a sign that says 'hello world'"

python -m hdi1 "A cat holding a sign that says 'hello world'" -m fast

edit share = True
vim /environment/miniconda3/envs/system/lib/python*/site-packages/hdi1/web.py

python -m hdi1.web
featurize port export 7860

可ui 调用

古装提示词：
这是一张高质量的艺术照片，一位留着黑色长发的年轻女子，穿着白色的中国传统旗袍，站在模糊的雪地背景下。灯光柔和自然，突显出她平静的表情。该图像使用浅景深，聚焦在背景模糊的物体上。构图遵循三分法，女人的脸偏离中心。这张照片可能是用单反相机拍摄的，可能是佳能EOS 5D Mark IV，光圈设置为f/2.8，快门速度为1/200，ISO 400。其美学品质极高，展现出优雅的简洁与宁静的氛围。

```

# HiDream-I1 4Bit Quantized Model

This repository is a fork of `HiDream-I1` quantized to 4 bits, allowing the full model to run in less than 16GB of VRAM. 

The original repository can be found [here](https://github.com/HiDream-ai/HiDream-I1).

> `HiDream-I1` is a new open-source image generative foundation model with 17B parameters that achieves state-of-the-art image generation quality within seconds.

![HiDream-I1 Demo](assets/demo.jpg)

![image](https://github.com/user-attachments/assets/d4715fb9-efe1-40c3-bd4e-dfd626492eea)

## Models

We offer both the full version and distilled models. The parameter size are the same, so they require the same amount of GPU memory to run. However, the distilled models are faster because of reduced number of inference steps.

| Name            | Min VRAM | Steps | HuggingFace                                                                                                                  |
|-----------------|----------|-------|------------------------------------------------------------------------------------------------------------------------------|
| HiDream-I1-Full | 16 GB    | 50    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Full) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Full-nf4) |
| HiDream-I1-Dev  | 16 GB    | 28    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Dev-nf4)   |
| HiDream-I1-Fast | 16 GB    | 16    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Fast-nf4) |

## Hardware Requirements

- GPU Architecture: NVIDIA `>= Ampere` (e.g. A100, H100, A40, RTX 3090, RTX 4090)
- GPU RAM: `>= 16 GB`
- CPU RAM: `>= 16 GB`

## Quick Start

Simply run:

```
pip install hdi1 --no-build-isolation
```

> [!NOTE]
> It's recommended that you start a new python environment for this package to avoid dependency conflicts.  
> To do that, you can use `conda create -n hdi1 python=3.12` and then `conda activate hdi1`.  
> Or you can use `python3 -m venv venv` and then `source venv/bin/activate` on Linux or `venv\Scripts\activate` on Windows.

### Command Line Interface

Then you can run the module to generate images:

``` python 
python -m hdi1 "A cat holding a sign that says 'hello world'"

# or you can specify the model
python -m hdi1 "A cat holding a sign that says 'hello world'" -m fast
```

> [!NOTE]
> The inference script will try to automatically download `meta-llama/Llama-3.1-8B-Instruct` model files. You need to [agree to the license of the Llama model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on your HuggingFace account and login using `huggingface-cli login` in order to use the automatic downloader.

### Web Dashboard

We also provide a web dashboard for interactive image generation. You can start it by running:

``` python
python -m hdi1.web
```

![Screenshot 2025-04-08 200120](https://github.com/user-attachments/assets/0c464033-5619-489d-b9de-fef5a7119cfc)

## License

The code in this repository and the HiDream-I1 models are licensed under [MIT License](./LICENSE).
