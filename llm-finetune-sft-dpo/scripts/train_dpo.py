# scripts/train_dpo.py
import argparse, os, yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional

from datasets import load_dataset
import mlflow
from peft import PeftModel, get_peft_model
from trl import DPOTrainer, DPOConfig

from scripts.utils import (
    get_tokenizer,
    build_bnb_config,
    get_lora_config,
    load_base_model,
)

@dataclass
class Config:
    base_model_id: str
    tokenizer_id: Optional[str]
    output_dir: str
    quantization: Dict[str, Any]
    lora: Dict[str, Any]
    tracking: Dict[str, Any]
    dpo: Dict[str, Any]
    sft: Dict[str, Any]
    serve: Optional[Dict[str, Any]] = None  # optional in your YAML

def read_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = read_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------- MLflow ----------
    mlflow.set_tracking_uri(cfg.tracking.get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment("dpo")
    mlflow.start_run()
    mlflow.log_params(
        {
            "base_model_id": cfg.base_model_id,
            "dataset_path": cfg.dpo["dataset_path"],
            "beta": cfg.dpo["beta"],
            "epochs": cfg.dpo["num_train_epochs"],
            "lr": cfg.dpo["learning_rate"],
            "init_from_sft": cfg.dpo.get("init_from_sft", True),
        }
    )

    # ---------- Tokenizer & base models ----------
    tokenizer = get_tokenizer(cfg.tokenizer_id, cfg.base_model_id)
    bnb = build_bnb_config(cfg.quantization)

    # Policy and reference start from the same base
    policy = load_base_model(cfg.base_model_id, quantization_config=bnb)
    ref_model = load_base_model(cfg.base_model_id, quantization_config=bnb)

    # Apply LoRA to the policy (trainable)
    lora_cfg = get_lora_config(policy, cfg.lora)
    policy = get_peft_model(policy, lora_cfg)

    # Optionally initialize policy from previously trained SFT adapter
    if cfg.dpo.get("init_from_sft", True):
        sft_dir = cfg.sft.get("adapter_out", "outputs/sft_adapter")
        print(f"Loading SFT adapter from: {sft_dir}")
        policy = PeftModel.from_pretrained(policy, sft_dir, is_trainable=True)

    # ---------- Dataset ----------
    # Expects JSONL with keys: prompt, chosen, rejected
    ds = load_dataset("json", data_files=cfg.dpo["dataset_path"])["train"]

    # ---------- DPO config (new TRL API) ----------
    use_bf16 = True if (bnb or os.getenv("BF16", "1") == "1") else False
    max_len = int(cfg.sft.get("max_seq_len", 512))

    dpo_cfg = DPOConfig(
        output_dir=cfg.dpo.get("adapter_out", "outputs/dpo_adapter"),
        per_device_train_batch_size=cfg.dpo["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.dpo["gradient_accumulation_steps"],
        learning_rate=cfg.dpo["learning_rate"],
        num_train_epochs=cfg.dpo["num_train_epochs"],
        logging_steps=cfg.dpo["logging_steps"],
        save_steps=cfg.dpo["save_steps"],
        bf16=use_bf16,
        report_to=[],  # disable HF tracking; MLflow used above

        # moved here in recent TRL
        max_length=max_len,
        max_prompt_length=min(256, max_len),
        max_target_length=256,
    )

    # ---------- Trainer ----------
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        beta=cfg.dpo["beta"],
        args=dpo_cfg,
        train_dataset=ds,      # columns: prompt, chosen, rejected
        tokenizer=tokenizer,
        # don't pass prompt_text_column/chosen_text_column/rejected_text_column on new TRL
    )

    # ---------- Train ----------
    trainer.train()

    # ---------- Save adapter & tokenizer ----------
    out_dir = cfg.dpo.get("adapter_out", "outputs/dpo_adapter")
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="dpo_adapter")
    mlflow.end_run()
    print(f"DPO complete. Adapter saved to: {out_dir}")

if __name__ == "__main__":
    main()
