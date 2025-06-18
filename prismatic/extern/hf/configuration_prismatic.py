"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.

"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        llm_backbone_id: str = "moxin-7b-pure",
        arch_specifier: str = "no-align+fused-gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "resize-naive",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # Vision backbone configuration
        self.use_fused_vision_backbone = True if use_fused_vision_backbone is None else use_fused_vision_backbone
        self.timm_model_ids = ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"]
        self.timm_override_act_layers = [None, None]
        self.image_sizes = [224, 224]
        self.image_resize_strategy = image_resize_strategy

        # LLM backbone configuration
        self.hf_llm_id = "moxin-org/Moxin-7B-LLM"
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING["mistral"](**text_config)
            if text_config is not None
            else CONFIG_MAPPING["mistral"]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)