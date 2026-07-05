"""
Configuration — Qwen3.5-4B INT8 quantization pipeline.
Edit this file to change paths, model IDs, or eval settings.

"""

# ── Model ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_DIR    = "./qwen3.5-4b-int8"
HF_REPO_ID    = "AxisQuant/Qwen3.5-4B-INT8"

# ── Quantization ─────────────────────────────────────────────────────────────

LOAD_IN_8BIT          = True
LLM_INT8_THRESHOLD    = 6.0    # outlier detection cutoff; default 6.0 works well
LLM_INT8_HAS_FP16_WEIGHT = False

# ── Evaluation ───────────────────────────────────────────────────────────────

# Perplexity (WikiText-2)
PPL_STRIDE  = 512
PPL_MAX_LEN = 2048

# GSM8K — set to 1319 for the full test split
GSM8K_SAMPLES = 200
