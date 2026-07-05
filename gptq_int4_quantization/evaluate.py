"""
Evaluation — Qwen3.6-27B GPTQ INT4
=====================================
Runs four benchmark checks against the quantized model:

  1. ARC-Challenge  — science reasoning multiple choice
  2. GSM8K          — grade-school math
  3. MMLU-Redux     — broad knowledge (57 subjects)
  4. HumanEval      — Python code generation (pass@1)

Why these four?
  Together they cover reasoning (ARC), math (GSM8K), knowledge breadth (MMLU),
  and code generation (HumanEval) — the four capability axes where 4-bit
  quantization most commonly causes regression. Verified BF16 baselines are
  published in the model card so any delta is immediately visible.

BF16 baselines (from model card):
  ARC-Challenge : 64.08%
  GSM8K         : 96.82%
  MMLU-Redux    : 88.42%
  HumanEval     : 77.44%
"""

import argparse
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import (
    OUTPUT_DIR,
    ARC_SAMPLES,
    GSM8K_SAMPLES,
    MMLU_SAMPLES,
    HUMANEVAL_SAMPLES,
    BASELINES,
)


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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device  = next(model.parameters()).device
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# ── ARC-Challenge ────────────────────────────────────────────────────────────

def evaluate_arc(model, tokenizer) -> float:
    print("\nEvaluating ARC-Challenge ...", flush=True)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    correct = 0
    for item in tqdm(ds, desc="ARC"):
        choices = item["choices"]
        labels  = choices["label"]
        texts   = choices["text"]
        prompt  = (
            f"Question: {item['question']}\n"
            + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            + "\nAnswer:"
        )
        pred = generate(model, tokenizer, prompt, max_new_tokens=4).strip()
        gold = item["answerKey"]
        if pred.upper().startswith(gold.upper()):
            correct += 1
    acc = correct / len(ds)
    print(f"  ARC-Challenge: {acc:.4f}  ({correct}/{len(ds)})")
    return acc


# ── GSM8K ────────────────────────────────────────────────────────────────────

def extract_number(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def evaluate_gsm8k(model, tokenizer, n: int = GSM8K_SAMPLES) -> float:
    print(f"\nEvaluating GSM8K ({n} samples) ...", flush=True)
    ds = load_dataset("gsm8k", "main", split="test").select(range(n))
    correct = 0
    for item in tqdm(ds, desc="GSM8K"):
        prompt = f"Solve step by step.\n\nProblem: {item['question']}\nAnswer:"
        pred   = extract_number(generate(model, tokenizer, prompt))
        gold   = extract_number(item["answer"])
        if pred == gold:
            correct += 1
    acc = correct / len(ds)
    print(f"  GSM8K accuracy: {acc:.4f}  ({correct}/{len(ds)})")
    return acc


# ── MMLU-Redux ────────────────────────────────────────────────────────────────

def evaluate_mmlu(model, tokenizer, n: int = MMLU_SAMPLES) -> float:
    print(f"\nEvaluating MMLU-Redux ({n} samples) ...", flush=True)
    ds = load_dataset("edinburgh-dawg/mmlu-redux", split="test").select(range(n))
    choices_labels = ["A", "B", "C", "D"]
    correct = 0
    for item in tqdm(ds, desc="MMLU"):
        opts   = item["choices"]
        prompt = (
            f"Question: {item['question']}\n"
            + "\n".join(f"{l}. {t}" for l, t in zip(choices_labels, opts))
            + "\nAnswer:"
        )
        pred = generate(model, tokenizer, prompt, max_new_tokens=4).strip().upper()
        gold = choices_labels[item["answer"]]
        if pred.startswith(gold):
            correct += 1
    acc = correct / len(ds)
    print(f"  MMLU-Redux accuracy: {acc:.4f}  ({correct}/{len(ds)})")
    return acc


# ── HumanEval ─────────────────────────────────────────────────────────────────

def evaluate_humaneval(model, tokenizer) -> float:
    """
    pass@1: generate one solution per problem, check if it passes all unit tests.
    Uses the execeval approach — runs generated code in a subprocess sandbox.
    Requires: pip install human-eval
    """
    print("\nEvaluating HumanEval (pass@1) ...", flush=True)
    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        print("  Skipped — install with: pip install human-eval")
        return 0.0

    import tempfile, json, os
    problems   = read_problems()
    samples    = []
    for task_id, problem in tqdm(problems.items(), desc="HumanEval"):
        prompt   = problem["prompt"]
        solution = generate(model, tokenizer, prompt, max_new_tokens=512)
        samples.append({"task_id": task_id, "completion": solution})

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        tmp = f.name

    results = evaluate_functional_correctness(tmp)
    os.unlink(tmp)
    acc = results["pass@1"]
    print(f"  HumanEval pass@1: {acc:.4f}")
    return acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=OUTPUT_DIR,
                        help=f"Path to quantized model (default: {OUTPUT_DIR})")
    parser.add_argument("--tasks", nargs="+",
                        choices=["arc", "gsm8k", "mmlu", "humaneval"],
                        default=["arc", "gsm8k", "mmlu", "humaneval"],
                        help="Which benchmarks to run")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    results = {}

    if "arc"       in args.tasks: results["arc_challenge"]  = evaluate_arc(model, tokenizer)
    if "gsm8k"     in args.tasks: results["gsm8k"]          = evaluate_gsm8k(model, tokenizer)
    if "mmlu"      in args.tasks: results["mmlu_redux"]      = evaluate_mmlu(model, tokenizer)
    if "humaneval" in args.tasks: results["humaneval_pass1"] = evaluate_humaneval(model, tokenizer)

    print("\n── Evaluation Summary ──────────────────────────────")
    for k, v in results.items():
        base  = BASELINES.get(k)
        delta = f"  Δ {(v - base)*100:+.2f}%" if base else ""
        print(f"  {k:25s}  {v:.4f}  (BF16 baseline: {base:.4f}{delta})")


if __name__ == "__main__":
    main()
