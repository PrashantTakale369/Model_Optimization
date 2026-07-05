"""
GPTQ INT4 Quantization — Qwen3.6-27B

"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore

from config import (
    BASE_MODEL_ID,
    OUTPUT_DIR,
    BITS,
    GROUP_SIZE,
    DESC_ACT,
    DAMP_PCT,
    SYM,
    CALIB_DATASET,
    CALIB_SPLIT,
    NUM_CALIB,
    MAX_SEQ_LEN,
)


def build_quant_config() -> QuantizeConfig:
    """
    group_size=128: balances accuracy vs overhead (128 weights share one scale).
    desc_act=True:  reorders rows by activation magnitude before quantization,
                    which reduces reconstruction error on outlier-heavy layers.
    damp_pct=0.01:  small Hessian damping for numerical stability.
    sym=True:       symmetric quantization (zero-point = 0) — faster on hardware.
    """
    return QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=DESC_ACT,
        damp_pct=DAMP_PCT,
        sym=SYM,
    )


def load_calibration_data(tokenizer, n: int = NUM_CALIB, seq_len: int = MAX_SEQ_LEN):
    """
    C4 (Colossal Clean Crawled Corpus) is a diverse, deduplicated web text corpus.
    Using it rather than task-specific data avoids overfitting the quantization
    to any particular benchmark domain.
    """
    print(f"Loading calibration data ({n} samples from C4) ...", flush=True)
    ds = load_dataset(CALIB_DATASET, CALIB_SPLIT, split="train", streaming=True)
    samples = []
    for row in ds:
        enc = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
        )
        if enc.input_ids.shape[1] == seq_len:
            samples.append(enc.input_ids)
        if len(samples) >= n:
            break
    print(f"  Collected {len(samples)} calibration sequences of length {seq_len}", flush=True)
    return samples


def main():
    quant_config = build_quant_config()

    print(f"Loading tokenizer from {BASE_MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    calib_data = load_calibration_data(tokenizer)

    print(f"Loading model for GPTQ quantization (this needs ~60 GB RAM / 54 GB VRAM) ...", flush=True)
    model = GPTQModel.from_pretrained(
        BASE_MODEL_ID,
        quant_config=quant_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # non-quantized layers (vision) stay BF16
    )

    print("Running GPTQ calibration (sequential layer-wise, ~30-60 min) ...", flush=True)
    model.quantize(
        calib_data,
        batch_size=1,
    )

    print(f"Saving INT4 model to {OUTPUT_DIR} ...", flush=True)
    model.save_quantized(OUTPUT_DIR, use_safetensors=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nDone. Run evaluate.py to verify quality, then push_to_hub.py to upload.")


if __name__ == "__main__":
    main()
