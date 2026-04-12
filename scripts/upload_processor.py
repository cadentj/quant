#!/usr/bin/env python3
"""Upload the processor files from the base model to an existing HF repo."""

from transformers import AutoProcessor
from huggingface_hub import HfApi
import tempfile

BASE_MODEL = "google/gemma-4-E2B-it"
BASE_REVISION = "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
TARGET_REPO = "kh4dien/gemma4-e2b-it-gptq-baseline-v2"

processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

with tempfile.TemporaryDirectory() as tmp:
    processor.save_pretrained(tmp)
    HfApi().upload_folder(
        folder_path=tmp,
        repo_id=TARGET_REPO,
        repo_type="model",
        commit_message="Add processor files (preprocessor_config.json etc.)",
    )

print(f"Uploaded processor to https://huggingface.co/{TARGET_REPO}")
