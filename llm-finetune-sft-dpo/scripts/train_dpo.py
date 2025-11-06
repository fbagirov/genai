# scripts/train_dpo.py
import argparse
import os
import json
from typing import Dict, Any, List

import yaml
import mlflow
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, get_peft_model

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

def force_numbers_in_dpo(cfg: Dict[str, Any]) -> None:
    """Cast numeric fields under dpo: to correct types (prevents optimizer type errors)."""
    dpo = cfg.get("dpo", {})
    num_int = ["per_device_train_batch_size", "gradient_accumulation_steps", "logging_steps", "save_steps",
               "max_length", "max_prompt_length"]
    num_float = ["learning_rate", "num_train_epochs", "beta"]
    for k in num_int:
        if k in dpo and dpo[k] is not None:
            dpo[k] = int(dpo[k])
    for k in num_float:
        if k in dpo and dpo[k] is not None:
            dpo[k] = float(dpo[k])

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

def normalize_dpo_columns(ds: Dataset) -> Dataset:
    """
    Ensure dataset has columns: prompt, chosen, rejected (as strings).
    If your file used alternate names, map them here.
    """
    cols = set(ds.column_names)

    # Common alternate names (edit as needed)
    mapping_candidates = [
        {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        {"prompt": "instruction", "chosen": "chosen", "rejected": "rejected"},
        {"prompt": "question", "chosen": "chosen", "rejected": "rejected"},
    ]
    picked = None
    for cand in mapping_candidates:
        if all(cand[k] in cols for k in ("prompt", "chosen", "rejected")):
            picked = cand
            break

    if picked is None:
        missing = [k for k in ("prompt", "chosen", "rejected") if k not in cols]
        raise ValueError(f"DPO dataset missing required columns: {missing}. Present: {sorted(cols)}")

    # If names already match, just cast to str and return
    if picked["prompt"] == "prompt" and picked["chosen"] == "chosen" and picked["rejected"] == "rejected":
        return ds.map(lambda ex: {
            "prompt": str(ex["prompt"]),
            "chosen": str(ex["chosen"]),
            "rejected": str(ex["rejected"]),
        })

    # Otherwise rename by projecting
    def project(ex):
        return {
            "prompt": str(ex[picked["prompt"]]),
            "chosen": str(ex[picked["chosen"]]),
            "rejected": str(ex[picked["rejected"]]),
        }
    ds2 = ds.map(project, remove_columns=ds.column_names)
    return ds2

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if "dpo" not in cfg:
        raise KeyError("Config missing 'dpo' section.")
    force_numbers_in_dpo(cfg)

    # MLflow
    mlflow.set_tracking_uri(cfg.get("tracking", {}).get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment("dpo")
    mlflow.start_run()
    mlflow.log_params({
        "base_model_id": cfg.get("base_model_id"),
        "dataset_path": cfg["dpo"].get("dataset_path"),
        "beta": cfg["dpo"].get("beta"),
        "epochs": cfg["dpo"].get("num_train_epochs"),
        "lr": cfg["dpo"].get("learning_rate"),
        "init_from_sft": cfg["dpo"].get("init_from_sft", True),
    })

    # Tokenizer & Models
    tokenizer = get_tokenizer(cfg.get("tokenizer_id"), cfg["base_model_id"])
    bnb = build_bnb_config(cfg.get("quantization"))

    policy = load_base_model(cfg["base_model_id"], quantization_config=bnb)
    ref_model = load_base_model(cfg["base_model_id"], quantization_config=bnb)

    # LoRA on policy (trainable)
    lora_cfg = get_lora_config(policy, cfg["lora"])
    policy = get_peft_model(policy, lora_cfg)

    # Optionally initialize from SFT adapter
    if cfg["dpo"].get("init_from_sft", True):
        sft_dir = cfg["sft"].get("adapter_out", "outputs/sft_adapter")
        print(f"Loading SFT adapter from: {sft_dir}")
        policy = PeftModel.from_pretrained(policy, sft_dir, is_trainable=True)

    # Dataset: prompt, chosen, rejected
    ds = load_jsonl_dataset(cfg["dpo"]["dataset_path"])
    ds = normalize_dpo_columns(ds)

    # Lengths (use config if present; otherwise derive from SFT max_seq_len)
    max_length = int(cfg["dpo"].get("max_length", cfg["sft"].get("max_seq_len", 512)))
    max_prompt_length = int(cfg["dpo"].get("max_prompt_length", min(256, max_length // 2)))

    # DPOConfig (replaces using raw TrainingArguments fields)
    dpo_cfg = DPOConfig(
        output_dir=cfg["dpo"].get("adapter_out", "outputs/dpo_adapter"),
        per_device_train_batch_size=int(cfg["dpo"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["dpo"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["dpo"]["learning_rate"]),
        num_train_epochs=float(cfg["dpo"]["num_train_epochs"]),
        logging_steps=int(cfg["dpo"]["logging_steps"]),
        save_steps=int(cfg["dpo"]["save_steps"]),
        bf16=True if bnb or os.getenv("BF16", "1") == "1" else False,
        report_to=[],
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    # IMPORTANT: new TRL API — do NOT pass prompt_text_column/chosen_text_column/rejected_text_column
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        beta=float(cfg["dpo"]["beta"]),
        args=dpo_cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    out_dir = cfg["dpo"].get("adapter_out", "outputs/dpo_adapter")
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="dpo_adapter")
    mlflow.end_run()
    print(f"DPO complete. Adapter saved to: {out_dir}")

if __name__ == "__main__":
    main()
