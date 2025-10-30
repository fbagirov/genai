
import argparse, os, json, yaml
from dataclasses import dataclass
from typing import Dict, Any
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import mlflow
from scripts.utils import get_tokenizer, build_bnb_config, get_lora_config, load_base_model

@dataclass
class Config:
    base_model_id: str
    tokenizer_id: str | None
    output_dir: str
    quantization: Dict[str, Any]
    lora: Dict[str, Any]
    tracking: Dict[str, Any]
    sft: Dict[str, Any]

def read_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def format_example(ex):
    instr = ex.get("instruction","").strip()
    inp = ex.get("input","").strip()
    out = ex.get("output","").strip()
    if inp:
        prompt = f"### Instruction\n{instr}\n\n### Input\n{inp}\n\n### Response\n"
    else:
        prompt = f"### Instruction\n{instr}\n\n### Response\n"
    return {"text": prompt + out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = read_config(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Tracking
    mlflow.set_tracking_uri(cfg.tracking.get("mlflow_tracking_uri","mlruns"))
    mlflow.set_experiment("sft")
    mlflow.start_run()
    mlflow.log_params({
        "base_model_id": cfg.base_model_id,
        "dataset_path": cfg.sft["dataset_path"],
        "max_seq_len": cfg.sft["max_seq_len"],
        "lr": cfg.sft["learning_rate"],
        "epochs": cfg.sft["num_train_epochs"],
    })

    # Tokenizer & Model
    tokenizer = get_tokenizer(cfg.tokenizer_id, cfg.base_model_id)
    bnb = build_bnb_config(cfg.quantization)
    model = load_base_model(cfg.base_model_id, quantization_config=bnb)

    # LoRA PEFT
    lora_cfg = get_lora_config(model, cfg.lora)
    # SFTTrainer will wrap with PEFT automatically if peft_config is passed

    # Dataset
    ds = load_dataset("json", data_files=cfg.sft["dataset_path"])["train"]
    ds = ds.map(format_example, remove_columns=ds.column_names)

    args_tr = TrainingArguments(
        output_dir=cfg.sft.get("adapter_out","outputs/sft_adapter"),
        per_device_train_batch_size=cfg.sft["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.sft["gradient_accumulation_steps"],
        learning_rate=cfg.sft["learning_rate"],
        num_train_epochs=cfg.sft["num_train_epochs"],
        weight_decay=cfg.sft["weight_decay"],
        warmup_ratio=cfg.sft["warmup_ratio"],
        logging_steps=cfg.sft["logging_steps"],
        save_steps=cfg.sft["save_steps"],
        fp16=False,
        bf16=True if bnb or os.getenv("BF16","1")=="1" else False,
        dataloader_pin_memory=False,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        packing=True,
        max_seq_length=cfg.sft["max_seq_len"],
        peft_config=lora_cfg,
        args=args_tr,
        dataset_text_field="text",
    )

    trainer.train()
    out_dir = cfg.sft.get("adapter_out","outputs/sft_adapter")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="sft_adapter")
    mlflow.end_run()
    print(f"SFT complete. Adapter saved to: {out_dir}")

if __name__ == "__main__":
    main()
