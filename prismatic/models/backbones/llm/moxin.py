"""
moxin.py

Class definition for all LLMs derived from MistralForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder, MoxinInstructPromptBuilder

# Registry =>> Support Moxin Models (from HF Transformers)
# fmt: off
MOXIN_MODELS = {
    # === Base Moxin ===
    # "moxin-7b-pure": {
    #     "llm_family": "moxin", "llm_cls": MistralForCausalLM, "hf_hub_path": "moxin-org/moxin-llm-7b"
    # },
    "moxin-7b-pure": {
        "llm_family": "moxin", "llm_cls": MistralForCausalLM, "hf_hub_path": "moxin-org/moxin-instruct-7b"
    },

    # === Moxin Instruct ===
    # "moxin-7b-instruct": {
    #     "llm_family": "mistral", "llm_cls": MistralForCausalLM, "hf_hub_path": "moxin-org/moxin-instruct-7b"
    # }
}
# fmt: on


class MoxinLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **MOXIN_MODELS[llm_backbone_id],
        )

        # [Special Case] Mistral PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.endswith("-instruct"):
            return MoxinInstructPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return MistralDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
