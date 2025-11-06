# scripts/train_sft.py

import argparse
import os
import json
from typing import Dict, Any, List

import yaml
import mlflow
from datasets import Dataset, load_dataset
from transformers import TrainingArguments  # kept for type parity in TRL, not used directly
from trl import SFTTrainer, SFTConfig

from scripts.utils import (
    get_tokenizer,
    build_bnb_config,
    get_lora_config,
    load_base_model,
)

# ------------------------------
# Helpers
# ------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config as a dict (UTF-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def force_numbers_in_sft(cfg: Dict[str, Any]) -> None:
    """Cast numeric fields under sft: to correct types (prevents optimizer type errors)."""
    sft = cfg.get("sft", {})
    num_int_keys = [
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "max_seq_len",
    ]
    num_float_keys = [
        "learning_rate",
        "num_train_epochs",
        "weight_decay",
        "warmup_ratio",
    ]
    for k in num_int_keys:
        if k in sft and sft[k] is not None:
            sft[k] = int(sft[k])
    for k in num_float_keys:
        if k in sft and sft[k] is not None:
            sft[k] = float(sft[k])

def load_jsonl_dataset(path: str) -> Dataset:
    """
    Robust JSONL loader:
    - First try datasets.load_dataset("json", ...)
    - If that fails (encoding/format quirks), fallback to manual JSONL parse.
    """
    try:
        ds = load_dataset("json", data_files=path)["train"]
        return ds
    except Exception:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Bad JSON at line {i} in {path}: {e}")
        if not rows:
            raise ValueError(f"No valid records found in {path}.")
        return Dataset.from_list(rows)

def format_example(ex: Dict[str, Any]) -> Dict[str, str]:
    """Prompt template for SFT."""
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()

    if inp:
        prompt = f"### Instruction\n{instr}\n\n### Input\n{inp}\n\n### Response\n"
    else:
        prompt = f"### Instruction\n{instr}\n\n### Response\n"
    return {"text": prompt + out}

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if "sft" not in cfg:
        raise KeyError("Config missing 'sft' section.")
    force_numbers_in_sft(cfg)

    # MLflow
    mlflow.set_tracking_uri(cfg.get("tracking", {}).get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment("sft")
    mlflow.start_run()
    mlflow.log_params({
        "base_model_id": cfg.get("base_model_id"),
        "dataset_path": cfg["sft"].get("dataset_path"),
        "max_seq_len": cfg["sft"].get("max_seq_len"),
        "lr": cfg["sft"].get("learning_rate"),
        "epochs": cfg["sft"].get("num_train_epochs"),
    })

    # Tokenizer & Model
    tokenizer = get_tokenizer(cfg.get("tokenizer_id"), cfg["base_model_id"])
    bnb = build_bnb_config(cfg.get("quantization"))
    model = load_base_model(cfg["base_model_id"], quantization_config=bnb)

    # LoRA PEFT
    lora_cfg = get_lora_config(model, cfg["lora"])

    # Dataset (robust JSONL)
    ds = load_jsonl_dataset(cfg["sft"]["dataset_path"])

    # Map to "text" and drop empties
    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: bool(ex["text"].strip()))

    # Heuristic: if dataset is tiny, disable packing to avoid "no packed sequence" error
    pack_threshold = 32  # tune as desired
    enable_packing = len(ds) >= pack_threshold

    # If tiny dataset and very large seq length, lower a bit so short samples still produce tokens
    max_seq_len = int(cfg["sft"]["max_seq_len"])
    if not enable_packing and max_seq_len > 256:
        max_seq_len = 256

    # Build SFTConfig (modern TRL API; avoids deprecation warnings)
    sft_cfg = SFTConfig(
        output_dir=cfg["sft"].get("adapter_out", "outputs/sft_adapter"),
        per_device_train_batch_size=int(cfg["sft"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["sft"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["sft"]["learning_rate"]),
        num_train_epochs=float(cfg["sft"]["num_train_epochs"]),
        weight_decay=float(cfg["sft"]["weight_decay"]),
        warmup_ratio=float(cfg["sft"]["warmup_ratio"]),
        logging_steps=int(cfg["sft"]["logging_steps"]),
        save_steps=int(cfg["sft"]["save_steps"]),
        fp16=False,
        bf16=True if bnb or os.getenv("BF16", "1") == "1" else False,
        report_to=[],  # keep local; MLflow handled manually above
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=enable_packing,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=sft_cfg,
    )

    trainer.train()
    out_dir = cfg["sft"].get("adapter_out", "outputs/sft_adapter")
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="sft_adapter")
    mlflow.end_run()
    print(f"SFT complete. Adapter saved to: {out_dir}")

if __name__ == "__main__":
    main()
