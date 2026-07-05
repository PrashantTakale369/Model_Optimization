"""
Push to Hugging Face Hub — Qwen3.5-4B-INT8
============================================
Uploads the quantized model to AxisQuant/Qwen3.5-4B-INT8.

Usage:
    huggingface-cli login          # once — saves token to ~/.cache/huggingface
    python push_to_hub.py
    python push_to_hub.py --model ./qwen3.5-4b-int8 --repo AxisQuant/Qwen3.5-4B-INT8
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import OUTPUT_DIR, HF_REPO_ID


def push(local_path: str, repo_id: str, private: bool = False) -> None:
    print(f"Loading model from {local_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        device_map="cpu",           # keep on CPU for upload; no GPU needed
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Pushing tokenizer to {repo_id} ...", flush=True)
    tokenizer.push_to_hub(repo_id, private=private)

    print(f"Pushing model to {repo_id} ...", flush=True)
    model.push_to_hub(
        repo_id,
        private=private,
        safe_serialization=True,   # save as safetensors, not pickle
    )
    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=OUTPUT_DIR,
                        help=f"Local path to quantized model (default: {OUTPUT_DIR})")
    parser.add_argument("--repo",    default=HF_REPO_ID,
                        help=f"HuggingFace repo id (default: {HF_REPO_ID})")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo")
    args = parser.parse_args()

    push(args.model, args.repo, args.private)


if __name__ == "__main__":
    main()
