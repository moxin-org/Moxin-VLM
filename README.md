# Moxin VLM

[![arXiv](https://img.shields.io/badge/arXiv-2412.06845-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2412.06845v4)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Usage**](#usage) | [**Pretrained Models**](#pretrained-models)

---

## Installation

```bash

git clone https://github.com/Bobchenyx/prismatic-vlm-moxin.git
cd prismatic-vlm-moxin

conda create -n moxin-vlm python=3.10 -y
conda activate moxin-vlm

pip install torch==2.4.1 torchvision==0.19.1
pip install transformers==4.44.0

pip install -e .

# Install Flash Attention 2 
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install flash-attn==2.6.3 --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

## Usage

Once installed, loading and running inference with pretrained models is easy:

```python
import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (local path) 
model_path = "" # PATH TO LOCAL MOXIN VLM
vlm = load(model_id)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "What is going on in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)
```

For a complete terminal-based CLI for interacting with our VLMs, check out [scripts/generate.py](scripts/generate.py). 
```bash
python scripts/generate.py --model_path runs/prism-moxin-dinosiglip-224px+7b+stage-finetune+x7

```

## Pretrained Models

Please find our Pretrained Models on our huggingface page.

---
**Explicit Notes on Model Licensing & Commercial Use**: 


## Acknowledgments

This project is based on [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) by [TRI-ML](https://github.com/TRI-ML). 
Special thanks to the original contributors for their excellent work.

## Citation 
If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/html/2412.06845v2):

```bibtex
@inproceedings{karamcheti2024prismatic,
  title = {Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models},
  author = {Siddharth Karamcheti and Suraj Nair and Ashwin Balakrishna and Percy Liang and Thomas Kollar and Dorsa Sadigh},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2024},
}
```

