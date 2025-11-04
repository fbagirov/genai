
.PHONY: sft dpo serve eval deid export clean

sft:
	python scripts/train_sft.py --config configs/config.yaml

dpo:
	python scripts/train_dpo.py --config configs/config.yaml

serve:
	uvicorn serve.fastapi_server:app --reload --port 8000

eval:
	python eval/run_lmeval.py --model_path outputs/dpo_adapter --task hellaswag --limit 50

deid:
	python scripts/deid_presidio.py --in data/sft/sample_sft.jsonl --out data/sft/sample_sft.deid.jsonl

export:
	python scripts/export_lora.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter outputs/dpo_adapter --out_dir outputs/merged_full_model

clean:
	rm -rf outputs mlruns
