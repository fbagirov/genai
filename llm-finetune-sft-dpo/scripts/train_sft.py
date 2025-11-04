def main():
    import argparse, os, yaml, torch, mlflow
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from scripts.utils import get_tokenizer, build_bnb_config, get_lora_config, load_base_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    # ---- read config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- tracking
    mlflow.set_tracking_uri(cfg.get("tracking", {}).get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment("sft")
    mlflow.start_run()
    mlflow.log_params({
        "base_model_id": cfg["base_model_id"],
        "dataset_path": cfg["sft"]["dataset_path"],
        "max_seq_len": cfg["sft"]["max_seq_len"],
        "lr": cfg["sft"]["learning_rate"],
        "epochs": cfg["sft"]["num_train_epochs"],
    })

    # ---- tokenizer & model
    tokenizer = get_tokenizer(cfg.get("tokenizer_id"), cfg["base_model_id"])
    bnb = build_bnb_config(cfg.get("quantization"))
    model = load_base_model(cfg["base_model_id"], quantization_config=bnb)

    # ---- lora
    lora_cfg = get_lora_config(model, cfg["lora"])

    # ---- dataset -> 'text' field expected by SFT
    ds = load_dataset("json", data_files=cfg["sft"]["dataset_path"])["train"]

    def format_example(ex):
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if inp:
            prompt = f"### Instruction\n{instr}\n\n### Input\n{inp}\n\n### Response\n"
        else:
            prompt = f"### Instruction\n{instr}\n\n### Response\n"
        return {"text": prompt + out}

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # ---- precision flags (avoid bf16 on non-Ampere or CPU)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+

    # ---- SFT config (new TRL API)
    sft_args = SFTConfig(
        output_dir=cfg["sft"].get("adapter_out", "outputs/sft_adapter"),
        per_device_train_batch_size=cfg["sft"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["sft"]["gradient_accumulation_steps"],
        learning_rate=cfg["sft"]["learning_rate"],
        num_train_epochs=cfg["sft"]["num_train_epochs"],
        weight_decay=cfg["sft"]["weight_decay"],
        warmup_ratio=cfg["sft"]["warmup_ratio"],
        logging_steps=cfg["sft"]["logging_steps"],
        save_steps=cfg["sft"]["save_steps"],
        dataset_text_field="text",
        max_seq_length=cfg["sft"]["max_seq_len"],
        packing=False,                 # omit or set appropriately
        fp16=False,
        bf16=False,
        report_to=[],  # if you're using SFTConfig / new TRL           # keep MLflow separate
        )

    # ---- trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,           # TRL 0.9.x expects tokenizer=
        train_dataset=ds,
        peft_config=lora_cfg,
        args=sft_args,
    )

    trainer.train()
    out_dir = cfg["sft"].get("adapter_out", "outputs/sft_adapter")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    mlflow.log_params({"adapter_out": out_dir})
    mlflow.log_artifacts(out_dir, artifact_path="sft_adapter")
    mlflow.end_run()
    print(f"SFT complete. Adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
