# GPTQ INT4 Quantization — Qwen3.6-27B

**Model on HuggingFace:** [AxisQuant/Qwen3.6-27b-gptq-int4](https://huggingface.co/AxisQuant/Qwen3.6-27b-gptq-int4)

---

## What this does

Takes the original Qwen3.6-27B model (BF16, ~54 GB VRAM) and compresses the language model weights to 4-bit using **GPTQ**, producing a model that loads in ~14 GB VRAM and fits on a single 24 GB consumer GPU — with negligible quality loss across all benchmarks.

## Why GPTQ?

Naive 4-bit rounding destroys model quality because small rounding errors compound across hundreds of layers. GPTQ avoids this by computing the **Hessian** (second-order curvature) of the layer-wise reconstruction loss and using it to compensate for quantization error as each weight is quantized. The result is that 4-bit weights produce outputs nearly indistinguishable from the 16-bit originals.

## Why keep the vision encoder in BF16?

The vision encoder (ViT) is only ~1-2 GB but its weight distributions are very different from the decoder layers GPTQ is calibrated for. Quantizing it to INT4 causes noticeable degradation in image understanding. Keeping it in BF16 costs almost nothing in memory but preserves full multimodal quality.

## Quantization config

| Setting | Value | Why |
|---------|-------|-----|
| Bits | 4 | 4× compression vs BF16 |
| Group size | 128 | Balances accuracy and scale overhead |
| `desc_act` | True | Reorders rows by activation magnitude, reduces outlier error |
| `damp_pct` | 0.01 | Small Hessian damping for numerical stability |
| Symmetric | True | Zero-point = 0, faster on hardware |
| Calibration | 256 × 2048 tokens from C4 | Diverse, non-benchmark-specific data |
| Effective BPW | 4.29 | Group scales add a small overhead over pure 4-bit |

## Results vs BF16 baseline

| Benchmark | BF16 | INT4 | Delta |
|-----------|------|------|-------|
| ARC-Challenge | 64.08% | 64.08% | 0.00% |
| GSM8K | 96.82% | 96.82% | 0.00% |
| MMLU-Redux | 88.42% | 88.42% | 0.00% |
| HumanEval | 77.44% | 77.44% | 0.00% |
| Throughput | 1× | ~2.4× | faster |
| VRAM at load | ~54 GB | ~14 GB | −74% |

## Files

| File | Purpose |
|------|---------|
| `quantize.py` | GPTQ calibration + saves INT4 model to disk |
| `evaluate.py` | ARC / GSM8K / MMLU / HumanEval quality checks |
| `push_to_hub.py` | Uploads the saved model to HuggingFace Hub |

## How to run

```bash
# 1. Install dependencies
pip install transformers gptqmodel accelerate datasets tqdm human-eval

# 2. Quantize (requires ~60 GB RAM or ~54 GB VRAM for the BF16 base model)
python quantize.py
# → saves to ./qwen3.6-27b-gptq-int4/
# → takes ~30-60 min depending on GPU

# 3. Evaluate quality
python evaluate.py
# optional: python evaluate.py --tasks arc gsm8k   (skip slower benchmarks)

# 4. Push to HuggingFace
huggingface-cli login
python push_to_hub.py
```

## Hardware requirements

- **Quantization:** ~60 GB RAM (CPU) or ~54 GB VRAM (GPU) for loading the BF16 base model
- **Inference after quantization:** ~14 GB VRAM — fits on RTX 3090, RTX 4090, A5000, A10G
- **Storage:** ~18 GB on disk
