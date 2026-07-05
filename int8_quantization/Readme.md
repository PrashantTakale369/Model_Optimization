# INT8 Quantization — Qwen3.5-4B

**Model on HuggingFace:** [AxisQuant/Qwen3.5-4B-INT8](https://huggingface.co/AxisQuant/Qwen3.5-4B-INT8)

---

## What this does

Takes the original Qwen3.5-4B model (BF16, 8.41 GB) and compresses it to INT8 using **bitsandbytes LLM.int8()**, producing a 4.84 GB model that is 1.74× smaller and faster to load — with almost no quality loss.

## Why bitsandbytes INT8?

Standard INT8 quantization crushes quality on large language models because a small number of weight channels have much larger values than the rest (called "outliers"). LLM.int8() solves this with **mixed-precision decomposition**: it detects those outlier channels, keeps them in FP16, and quantizes the remaining ~99.9% of weights to INT8. The result is that the INT8 matrix-multiply speedup is captured while the outliers that would cause quality loss are handled exactly.

This is the simplest, lowest-risk quantization method — no calibration data needed, no Hessian computation, just load the model with `load_in_8bit=True`.

## Results

| Metric | BF16 baseline | INT8 quantized | Change |
|--------|--------------|----------------|--------|
| Memory | 8.41 GB | 4.84 GB | −42.4% |
| WikiText-2 perplexity | baseline | +1.30% | negligible |
| GSM8K accuracy | ~87% | ~86% | −1% |

## Files

| File | Purpose |
|------|---------|
| `quantize.py` | Loads Qwen3.5-4B in INT8 and saves to disk |
| `evaluate.py` | WikiText-2 perplexity + GSM8K accuracy check |
| `push_to_hub.py` | Uploads the saved model to HuggingFace Hub |

## How to run

```bash
# 1. Install dependencies
pip install transformers bitsandbytes accelerate datasets tqdm

# 2. Quantize and save
python quantize.py
# → saves to ./qwen3.5-4b-int8/

# 3. Evaluate quality
python evaluate.py
# optional: python evaluate.py --skip-ppl   (faster, skips perplexity)

# 4. Push to HuggingFace
huggingface-cli login
python push_to_hub.py
```

## Hardware requirements

- Any NVIDIA GPU with ≥ 6 GB VRAM (model fits in ~5 GB at INT8)
- bitsandbytes requires CUDA; CPU inference is not supported for INT8 kernels
