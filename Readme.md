# Model Optimization :  Quantization 

Post-training quantization experiments on large language models, with results published on HuggingFace under the **AxisQuant** organization.

Each folder is a self-contained quantization pipeline: quantize → evaluate → push to Hub.

---

## Models

### 1. Qwen3.5-4B — INT8 (bitsandbytes)

**HuggingFace:** [AxisQuant/Qwen3.5-4B-INT8](https://huggingface.co/AxisQuant/Qwen3.5-4B-INT8)  
**Folder:** `int8_quantization/`

Compresses Qwen3.5-4B from BF16 (8.41 GB) to INT8 (4.84 GB) using bitsandbytes LLM.int8() mixed-precision decomposition. No calibration data needed. Memory reduced by 42.4% with negligible quality loss (WikiText-2 perplexity +1.3%, GSM8K ~86%).

**Best for:** Quick deployment, limited VRAM, or as a baseline before trying GPTQ.

---

### 2. Qwen3.6-27B — GPTQ INT4

**HuggingFace:** [AxisQuant/Qwen3.6-27b-gptq-int4](https://huggingface.co/AxisQuant/Qwen3.6-27b-gptq-int4)  
**Folder:** `gptq_int4_quantization/`

Compresses Qwen3.6-27B from BF16 (~54 GB VRAM) to GPTQ INT4 (~14 GB VRAM) using Hessian-aware second-order quantization. Calibrated on 256 samples of C4 English. Vision encoder kept in BF16. ~2.4× faster inference. All four benchmarks show negligible degradation vs the BF16 baseline.

**Best for:** Running a 27B model on a single 24 GB consumer GPU (RTX 3090/4090, A5000, A10G).

---

## Repository structure

```
Model_Optimization/
│
├── int8_quantization/
│   ├── quantize.py        # Load Qwen3.5-4B in INT8 and save
│   ├── evaluate.py        # WikiText-2 perplexity + GSM8K accuracy
│   ├── push_to_hub.py     # Upload to AxisQuant/Qwen3.5-4B-INT8
│   └── Readme.md
│
├── gptq_int4_quantization/
│   ├── quantize.py        # GPTQ calibration + save INT4 model
│   ├── evaluate.py        # ARC / GSM8K / MMLU-Redux / HumanEval
│   ├── push_to_hub.py     # Upload to AxisQuant/Qwen3.6-27b-gptq-int4
│   └── Readme.md
│
└── Readme.md              # This file
```

---

## Quantization methods compared

| Method | Bits | Calibration data | Quality loss | VRAM reduction | Complexity |
|--------|------|-----------------|--------------|----------------|------------|
| bitsandbytes INT8 | 8 | None | Very low (~1%) | ~50% | Trivial |
| GPTQ INT4 | 4 | ~256 samples | Very low (<1%) | ~75% | Moderate |
| AWQ INT4 | 4 | ~128 samples | Very low (<1%) | ~75% | Moderate |

Both models in this repo use post-training quantization (PTQ) — no retraining or gradient computation required.

---

## Quick start

```bash
# INT8 (Qwen3.5-4B)
cd int8_quantization
pip install transformers bitsandbytes accelerate datasets tqdm
python quantize.py

# GPTQ INT4 (Qwen3.6-27B)
cd gptq_int4_quantization
pip install transformers gptqmodel accelerate datasets tqdm
python quantize.py
```

See each folder's `Readme.md` for full instructions including evaluation and Hub upload.
