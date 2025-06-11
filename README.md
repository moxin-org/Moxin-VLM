# Moxin VLM

[![arXiv](https://img.shields.io/badge/arXiv-2412.06845-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2412.06845v4)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Usage**](#usage) | [**Pretrained Models**](#pretrained-models)

---

## Installation

```bash

git clone https://github.com/moxin-org/Moxin-VLM.git
cd Moxin-VLM

conda create -n moxin-vlm python=3.10 -y
conda activate moxin-vlm

pip install torch==2.4.1 torchvision==0.19.1
pip install transformers==4.46.0 peft==0.15.2

pip install -e .

# Install Flash Attention 2 
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install flash-attn==2.6.3 --no-build-isolation

```

If you run into any problems during the installation process, please file a GitHub Issue.

## Usage

For a complete terminal-based CLI for interacting with our VLMs, check out [scripts/generate.py](scripts/generate.py). 
```bash
python scripts/generate.py --model_path moxin-org/Moxin-7B-VLM

```

## Pretrained Models

Please find our Pretrained Models on our huggingface page: [moxin-org/Moxin-7B-VLM](https://huggingface.co/moxin-org/Moxin-7B-VLM).

We've also attach scripts to download and run our model locally.

```bash
python scripts/snapshot_download.py

python scripts/generate.py --model_path <path_to_local_dir>
```

---

## Acknowledgments

This project is based on [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) by [TRI-ML](https://github.com/TRI-ML). 

Special thanks to the original contributors for their excellent work.

## Citation 
If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2412.06845v4):

```bibtex
@article{zhao2024fully,
  title={Fully Open Source Moxin-7B Technical Report},
  author={Zhao, Pu and Shen, Xuan and Kong, Zhenglun and Shen, Yixin and Chang, Sung-En and Rupprecht, Timothy and Lu, Lei and Nan, Enfu and Yang, Changdi and He, Yumei and others},
  journal={arXiv preprint arXiv:2412.06845},
  year={2024}
}
```

