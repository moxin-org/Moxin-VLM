# !pip install huggingface_hub hf_transfer
import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Starting download...")

snapshot_download(
    repo_id = "moxin-org/moxin-vlm-7b",
    local_dir = "moxin-vlm-7b",
)

print("Download finished.")