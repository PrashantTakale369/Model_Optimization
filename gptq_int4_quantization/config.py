"""
Configuration — Qwen3.6-27B GPTQ INT4 quantization pipeline.
Edit this file to change paths, model IDs, quant settings, or eval settings.
"""

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen3.6-27B"
OUTPUT_DIR    = "./qwen3.6-27b-gptq-int4"
HF_REPO_ID    = "AxisQuant/Qwen3.6-27b-gptq-int4"

# ── Quantization ─────────────────────────────────────────────────────────────
BITS       = 4
GROUP_SIZE = 128
DESC_ACT   = True    # reorder rows by activation magnitude → lower outlier error
DAMP_PCT   = 0.01    # Hessian damping for numerical stability
SYM        = True    # symmetric quant (zero-point = 0), faster on hardware

# ── Calibration data ─────────────────────────────────────────────────────────
CALIB_DATASET = "allenai/c4"
CALIB_SPLIT   = "en"
NUM_CALIB     = 256    # number of calibration sequences
MAX_SEQ_LEN   = 2048   # tokens per sequence

# ── Evaluation ───────────────────────────────────────────────────────────────
# Set to None to use the full split
ARC_SAMPLES       = None   # full ARC-Challenge test = 1172
GSM8K_SAMPLES     = 300
MMLU_SAMPLES      = 500
HUMANEVAL_SAMPLES = None   # full HumanEval = 164

# BF16 baselines from the published model card (used to compute Δ in evaluate.py)
BASELINES = {
    "arc_challenge":   0.6408,
    "gsm8k":           0.9682,
    "mmlu_redux":      0.8842,
    "humaneval_pass1": 0.7744,
}
