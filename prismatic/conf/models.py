"""
models.py

Draccus Dataclass Definition for a ModelConfig object, with various registered subclasses for each model family and
variant thereof. A given model variant configures the following attributes:
    - Pretrained Visual Representation (e.g., OpenAI CLIP ViT-L/14) + Pretrained LLM Backbone (e.g., LLaMa-2 7B)
    - VLM Configuration + Parameters (e.g., MLP Projector, Image Preprocessing, etc.)
    - [Optional] Stage 1 (`align`) Optimization Hyperparameters
    - Stage 2 (`finetune`) Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    model_id: str                                           # Unique Model ID that fully specifies a given variant
    arch_specifier: str                                     # Architecture specifier string (e.g., "gelu-mlp")

    # Pretrained Backbones
    vision_backbone_id: str                                 # Pretrained Visual Featurizer (from TIMM) to load
    llm_backbone_id: str                                    # Pretrained LLM (from HF Transformers) to load

    # Backbone Parameters
    image_resize_strategy: str                              # Resizing strategy in < crop | letterbox | corner-pad >
    llm_max_length: int                                     # Maximum context length for LLM (can be < than max!)

    # === Multi-Stage Optimization Hyperparameters ===
    # By default, we assume an AdamW optimizer with FSDP (Gradient Sharding or Full Sharding depending on stage)

    # Align Stage Optimization Parameters
    align_epochs: int                                       # Epochs to Run (in case `max_steps` is not specified)
    align_max_steps: Optional[int]                          # [Optional] Max Gradient Steps (overrides epochs)
    align_global_batch_size: int                            # Global Batch Size (divided across processes)
    align_per_device_batch_size: int                        # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    align_learning_rate: float                              # Peak Learning Rate (lr_scheduler sets warmup/decay)
    align_weight_decay: float                               # Weight Decay for AdamW Optimizer
    align_max_grad_norm: float                              # Max Grad Norm (for global gradient clipping)
    align_lr_scheduler_type: str                            # LR Scheduler (default: "linear-warmup+cosine-decay")
    align_warmup_ratio: float                               # Fraction of total steps to warmup

    align_train_strategy: str                               # Align Train Strategy (default: "fsdp-shard-grad-op")

    # Finetune Stage Optimization Parameters
    finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    finetune_max_steps: Optional[int]                       # [Optional] Max Gradient Steps (overrides epochs)
    finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    finetune_warmup_ratio: float                            # Fraction of total steps to warmup

    finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True

    # Enable Traditional Mixed Precision Training via Torch Native AMP (`autocast`)

    # I changed the following line from True to False to use fp16 instead of bfloat16
    enable_mixed_precision_training: bool = True            # Whether to enable mixed precision training
    # Did not change this line
    reduce_in_full_precision: bool = False                  # Whether to run gradient reduction in FP32

    # fmt: on


# === LLaVa v1.5 Reproduction - Fully Specified Configurations ===
@dataclass
class LLaVa_v15_Reproduction_7B(ModelConfig):
    model_id: str = "reproduction-llava-v15+7b"
    arch_specifier: str = "gelu-mlp"

    vision_backbone_id: str = "clip-vit-l-336px"
    llm_backbone_id: str = "vicuna-v15-7b"

    image_resize_strategy: str = "letterbox"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_max_steps: Optional[int] = None
    align_global_batch_size: int = 8
    align_per_device_batch_size: int = 8

    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03

    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 1
    finetune_max_steps: Optional[int] = None
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 16

    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03

    finetune_train_strategy: str = "fsdp-full-shard"


@dataclass
class LLaVa_v15_Reproduction_13B(LLaVa_v15_Reproduction_7B):
    model_id: str = "reproduction-llava-v15+13b"
    llm_backbone_id: str = "vicuna-v15-13b"


# === Section 4.1 :: Optimization Procedure ===


# Section 4.1A :: 🚀 --> Necessity of Multi-Stage Training
@dataclass
class Exp_7B_One_Stage(LLaVa_v15_Reproduction_7B):
    model_id: str = "one-stage+7b"
    arch_specifier: str = "no-align+gelu-mlp"


# ~ LLM Backbones
@dataclass
class Ext_Exp_7B_Moxin(Exp_7B_One_Stage):
    model_id: str = "moxin+7b"
    llm_backbone_id: str = "moxin-7b-pure"


# Moxin LLM Backbone here
@dataclass
class Prism_Moxin_7B_DINOSigLIP_224px(Exp_7B_One_Stage):
    model_id: str = "prism-moxin-dinosiglip-224px+7b"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    llm_backbone_id: str = "moxin-7b-pure"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    finetune_epochs: int = 2

# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # LLM Backbone 
    EXT_EXP_MOXIN_7B = Ext_Exp_7B_Moxin

    # Moxin
    PRISM_MOXIN_DINOSIGLIP_224PX_7B = Prism_Moxin_7B_DINOSigLIP_224px

    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)
