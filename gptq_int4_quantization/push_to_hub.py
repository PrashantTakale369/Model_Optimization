"""
Push to Hugging Face Hub — Qwen3.6-27B GPTQ INT4
==================================================
Uploads the quantized model to AxisQuant/Qwen3.6-27b-gptq-int4.

Usage:
    huggingface-cli login          # once — saves token to ~/.cache/huggingface
    python push_to_hub.py
    python push_to_hub.py --model ./qwen3.6-27b-gptq-int4 --repo AxisQuant/Qwen3.6-27b-gptq-int4
"""

import argparse
from transformers import AutoTokenizer
from gptqmodel import GPTQModel  # type: ignore

from config import OUTPUT_DIR, HF_REPO_ID


def push(local_path: str, repo_id: str, private: bool = False) -> None:
    print(f"Loading tokenizer from {local_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    print(f"Loading GPTQ model from {local_path} ...", flush=True)
    model = GPTQModel.from_quantized(
        local_path,
        trust_remote_code=True,
        device_map="cpu",   # no GPU needed for upload
    )

    print(f"Pushing tokenizer to {repo_id} ...", flush=True)
    tokenizer.push_to_hub(repo_id, private=private)

    print(f"Pushing model to {repo_id} ...", flush=True)
    model.push_to_hub(
        repo_id,
        private=private,
        safe_serialization=True,
    )
    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=OUTPUT_DIR,
                        help=f"Local path to GPTQ quantized model (default: {OUTPUT_DIR})")
    parser.add_argument("--repo",    default=HF_REPO_ID,
                        help=f"HuggingFace repo id (default: {HF_REPO_ID})")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo")
    args = parser.parse_args()

    push(args.model, args.repo, args.private)


if __name__ == "__main__":
    main()
