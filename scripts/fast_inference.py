"""
fast_inference.py

Run with an HF-exported Prismatic model path.
"""

import time

from prismatic.extern.hf.configuration_prismatic import PrismaticConfig
from prismatic.extern.hf.modeling_prismatic import PrismaticForConditionalGeneration
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor, PrismaticImageProcessor

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer

from transformers.utils import logging
logging.set_verbosity_error()

print("[*] Using direct Prismatic classes (bypassing AutoClasses)")

# === Verification Arguments ===
MODEL_PATH = "Moxin-7B-VLM-hf"
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
)

SAMPLE_PROMPTS_FOR_GENERATION = [
    "In: What is sitting in the coffee?\nOut:",
    "In: What's the name of the food on the plate?\nOut:",
    "In: caption.\nOut:",
    "In: How many beignets?\nOut:",
    "In: Can you give me a lyrical description of the scene?\nOut:",
]

@torch.inference_mode()
def verify_prismatic() -> None:
    print(f"[*] Verifying PrismaticForConditionalGeneration using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[*] Using device: {device}") 

    # Load each component separately
    print("[*] Instantiating Image Processor, Tokenizer and Processor")
    
    # Directly load image processor and tokenizer
    image_processor = PrismaticImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Manually create processor
    processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    print("[*] Loading VLM in BF16 with Flash-Attention Enabled")
    vlm = PrismaticForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # Iterate over Sample Prompts =>> Generate
    image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw).convert("RGB")
    # num_tokens, total_time = 0, 0.0

    print("[*] Iterating over Sample Prompts\n===\n")
    for idx, prompt in enumerate(SAMPLE_PROMPTS_FOR_GENERATION):
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # Run Inference
        gen_ids = None
        for _ in range(5):
            start_time = time.time()
            gen_ids = vlm.generate(**inputs, do_sample=False, min_length=1, max_new_tokens=512)
            # total_time += time.time() - start_time

            gen_ids = gen_ids[0, inputs.input_ids.shape[1] :]
            # num_tokens += len(gen_ids)

        # ===
        gen_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"[{idx + 1}] Input Prompt => {prompt}\n    Generated    => {gen_text}\n")

    # Compute Tokens / Second
    # print(f"[*] Generated Tokens per Second = {num_tokens / total_time} w/ {num_tokens = } and {total_time = }")


if __name__ == "__main__":
    verify_prismatic()