"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = ""  # HF for Moxin-VLM 


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> PrismaticVLM:
    # assert os.path.isdir(model_id_or_path), f"Model path `{model_id_or_path}` must be a valid local directory"
    # overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")
    
    # # Get paths for `config.json` and pretrained checkpoint
    # config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
    # assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
    # assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"

    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if not isinstance(model_id_or_path, str) or '/' not in model_id_or_path:
            raise ValueError(f"Invalid model_id format: `{model_id_or_path}`. Expected format: 'username/model-name'")
    
        try:
            from huggingface_hub import repo_exists, repo_info
            if not repo_exists(model_id_or_path):
                raise ValueError(f"Repository `{model_id_or_path}` does not exist on Hugging Face Hub")
            
            # 检查必需文件是否存在
            info = repo_info(model_id_or_path)
            available_files = [f.rfilename for f in info.siblings]
            
            required_files = ["config.json", "checkpoints/latest-checkpoint.pt"]
            missing_files = [f for f in required_files if f not in available_files]
            
            if missing_files:
                raise ValueError(f"Missing required files in repository `{model_id_or_path}`: {missing_files}")
                
        except Exception as e:
            if "Missing required files" in str(e) or "does not exist" in str(e):
                raise e
            else:
                raise ConnectionError(f"Failed to verify repository. Check your internet connection and HF Hub access: {e}")
        
        overwatch.info(f"Downloading `{model_id_or_path} from HF Hub")
        config_json = hf_hub_download(repo_id=model_id_or_path, filename=f"config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=model_id_or_path, filename=f"checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
    )

    return vlm
