
import argparse, os, yaml
from dataclasses import dataclass
from typing import Dict, Any
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer
import mlflow
from peft import PeftModel, LoraConfig, get_peft_model
from scripts.utils import get_tokenizer, build_bnb_config, get_lora_config, load_base_model

@dataclass
class Config:
    base_model_id: str
    tokenizer_id: str | None
    output_dir: str
    quantization: Dict[str, Any]
    lora: Dict[str, Any]
    tracking: Dict[str, Any]
    dpo: Dict[str, Any]
    sft: Dict[str, Any]

def read_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = read_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg.tracking.get("mlflow_tracking_uri","mlruns"))
    mlflow.set_experiment("dpo")
    mlflow.start_run()
    mlflow.log_params({
        "base_model_id": cfg.base_model_id,
        "dataset_path": cfg.dpo["dataset_path"],
        "beta": cfg.dpo["beta"],
        "epochs": cfg.dpo["num_train_epochs"],
        "lr": cfg.dpo["learning_rate"],
        "init_from_sft": cfg.dpo.get("init_from_sft", True),
    })

    # Tokenizer
    tokenizer = get_tokenizer(cfg.tokenizer_id, cfg.base_model_id)

    # Base / policy model + ref model
    bnb = build_bnb_config(cfg.quantization)
    policy = load_base_model(cfg.base_model_id, quantization_config=bnb)
    ref_model = load_base_model(cfg.base_model_id, quantization_config=bnb)

    # Apply LoRA to policy (and reference if desired)
    lora_cfg = get_lora_config(policy, cfg.lora)
    policy = get_peft_model(policy, lora_cfg)

    if cfg.dpo.get("init_from_sft", True):
        sft_dir = cfg.sft.get("adapter_out","outputs/sft_adapter")
        print(f"Loading SFT adapter from: {sft_dir}")
        policy = PeftModel.from_pretrained(policy, sft_dir, is_trainable=True)

    # Dataset: expects fields prompt, chosen, rejected
    ds = load_dataset("json", data_files=cfg.dpo["dataset_path"])["train"]

    args_tr = TrainingArguments(
        output_dir=cfg.dpo.get("adapter_out","outputs/dpo_adapter"),
        per_device_train_batch_size=cfg.dpo["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.dpo["gradient_accumulation_steps"],
        learning_rate=cfg.dpo["learning_rate"],
        num_train_epochs=cfg.dpo["num_train_epochs"],
        logging_steps=cfg.dpo["logging_steps"],
        save_steps=cfg.dpo["save_steps"],
        bf16=True if bnb or os.getenv("BF16","1")=="1" else False,
        report_to=["none"],
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        beta=cfg.dpo["beta"],
        args=args_tr,
        train_dataset=ds,
        tokenizer=tokenizer,
        # dataset columns: 'prompt', 'chosen', 'rejected'
        prompt_text_column="prompt",
        chosen_text_column="chosen",
        rejected_text_column="rejected",
    )

    trainer.train()
    out_dir = cfg.dpo.get("adapter_out","outputs/dpo_adapter")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="dpo_adapter")
    mlflow.end_run()
    print(f"DPO complete. Adapter saved to: {out_dir}")

if __name__ == "__main__":
    main()
