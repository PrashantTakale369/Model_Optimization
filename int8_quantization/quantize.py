"""
INT8 Quantization — Qwen3.5-4B
================================
Method : bitsandbytes LLM.int8()
Base   : Qwen/Qwen3.5-4B  (BF16, ~8.41 GB)
Output : AxisQuant/Qwen3.5-4B-INT8  (~4.84 GB, 1.74x smaller)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    BASE_MODEL_ID,
    OUTPUT_DIR,
    LOAD_IN_8BIT,
    LLM_INT8_THRESHOLD,
    LLM_INT8_HAS_FP16_WEIGHT,
)


def build_quantization_config() -> BitsAndBytesConfig:
    """
    load_in_8bit activates bitsandbytes LLM.int8() decomposition.
    llm_int8_threshold controls the outlier detection cutoff (default 6.0 is
    well-tuned for most models; lower = more outliers kept in FP16).
    """
    return BitsAndBytesConfig(
        load_in_8bit=LOAD_IN_8BIT,
        llm_int8_threshold=LLM_INT8_THRESHOLD,
        llm_int8_has_fp16_weight=LLM_INT8_HAS_FP16_WEIGHT,
    )


def load_model_and_tokenizer(model_id: str, quant_config: BitsAndBytesConfig):
    print(f"Loading tokenizer from {model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    print(f"Loading model in INT8 (this moves weights to GPU) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",          # spreads across available GPUs / CPU offload
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # non-quantized layers stay in BF16
    )
    return model, tokenizer


def report_memory(model) -> None:
    """Print per-dtype parameter counts to confirm INT8 conversion."""
    dtype_counts: dict = {}
    for _, p in model.named_parameters():
        key = str(p.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + p.numel()

    total = sum(dtype_counts.values())
    print("\nParameter dtype breakdown:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype:15s}  {count:>12,}  ({100*count/total:.1f}%)")
    print(f"  {'TOTAL':15s}  {total:>12,}")


def save_model(model, tokenizer, output_dir: str) -> None:
    print(f"\nSaving quantized model to {output_dir} ...", flush=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved.", flush=True)


def main():
    quant_config = build_quantization_config()
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, quant_config)

    report_memory(model)

    save_model(model, tokenizer, OUTPUT_DIR)
    print(f"\nDone. Run evaluate.py to verify quality, then push_to_hub.py to upload.")


if __name__ == "__main__":
    main()
