"""
Evaluation — Qwen3.5-4B-INT8
================================
Runs two checks against the quantized model:

  1. Perplexity on WikiText-2 (lower is better; we target ≤ +1.5% vs BF16 baseline)
  2. GSM8K accuracy (grade-school math; we target ≥ 85%)

Why these two?
  Perplexity is a fast, broad quality signal that catches weight corruption or
  large precision loss without needing to run a full benchmark suite.
  GSM8K tests multi-step reasoning — the capability most sensitive to INT8 rounding
  in the MLP layers, making it a good canary for quality degradation.
"""

import argparse
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import OUTPUT_DIR, PPL_STRIDE, PPL_MAX_LEN, GSM8K_SAMPLES

WIKITEXT_SPLIT = "test"


def load_model(model_path: str):
    print(f"Loading model from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


# ── Perplexity ───────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, stride: int = PPL_STRIDE, max_len: int = PPL_MAX_LEN) -> float:
    """
    Sliding-window perplexity on WikiText-2 test set.
    Uses a stride so every token gets a full left-context up to max_len.
    """
    print("\nComputing WikiText-2 perplexity ...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=WIKITEXT_SPLIT)
    text    = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    nlls, count = [], 0
    device = next(model.parameters()).device

    for begin in tqdm(range(0, input_ids.size(1) - 1, stride), desc="perplexity"):
        end        = min(begin + max_len, input_ids.size(1))
        target_len = end - (begin + stride) if begin > 0 else end - begin
        chunk      = input_ids[:, begin:end].to(device)

        with torch.no_grad():
            out = model(chunk, labels=chunk)
        # only count loss over the target_len tokens (not the context window)
        nll = out.loss * target_len
        nlls.append(nll.item())
        count += target_len

        if end == input_ids.size(1):
            break

    ppl = math.exp(sum(nlls) / count)
    print(f"  WikiText-2 perplexity: {ppl:.4f}")
    return ppl


# ── GSM8K ────────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Pull the final numeric answer after '####'."""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    # fallback: last number in the string
    import re
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def compute_gsm8k_accuracy(model, tokenizer, n_samples: int = GSM8K_SAMPLES) -> float:
    print(f"\nComputing GSM8K accuracy ({n_samples} samples) ...", flush=True)
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    device  = next(model.parameters()).device
    correct = 0

    for item in tqdm(dataset, desc="gsm8k"):
        prompt = (
            "Solve the math problem step by step.\n\n"
            f"Problem: {item['question']}\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = extract_answer(generated)
        gold = extract_answer(item["answer"])
        if pred == gold:
            correct += 1

    acc = correct / len(dataset)
    print(f"  GSM8K accuracy: {acc:.4f}  ({correct}/{len(dataset)})")
    return acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OUTPUT_DIR,
                        help=f"Path to quantized model (default: {OUTPUT_DIR})")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip perplexity (faster run)")
    parser.add_argument("--skip-gsm8k", action="store_true",
                        help="Skip GSM8K (faster run)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    results = {}
    if not args.skip_ppl:
        results["perplexity_wikitext2"] = compute_perplexity(model, tokenizer)
    if not args.skip_gsm8k:
        results["gsm8k_accuracy"] = compute_gsm8k_accuracy(model, tokenizer)

    print("\n── Evaluation Summary ──────────────────")
    for k, v in results.items():
        print(f"  {k:35s} {v:.4f}")

    print("\nTargets:")
    print("  perplexity_wikitext2   <= baseline * 1.015  (~+1.5% max)")
    print("  gsm8k_accuracy         >= 0.85")


if __name__ == "__main__":
    main()
