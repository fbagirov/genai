# llm-finetune-sft-dpo

**Goal:** Fine-tune a small instruction model with **Supervised Fine-Tuning (SFT)** and compare against **preference optimization** (**DPO**). Uses **LoRA** adapters (parameter-efficient), optional **4/8-bit quantization**, local **MLflow** tracking, and a simple **FastAPI** server that loads the LoRA at inference.

> **Default base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (works on CPU albeit slow). You can swap for `gpt2` for a super-light demo or any 7B+ model if you have GPU VRAM.


## What is Fine Tuning? How is it different from prompting? 

The base models (Llama, GPT, etc.) are trained on some text - books, articles, Wikipedia, Reddit, etc. When the model is trained, it starts with random weights assigned to words and patterns. During traing it sees millions of examples that are similar to each other (for example, for "How are ___?" the most fit word is "you"). Each time it predicts wrong, the algorithm shifts the weights slightly so the next time it is closer to the correct answer. After substantial training, those shifts accumulate into trillions of fine-tuned numbers - the based model weights. 

**Prompting** - When you want to steer the model by wording the prompt differently (zero-shot, few-shot, chain-of-thought, etc.), the base model weights do not change. Also, for the same task, you'll have to prompt the model each time (best for ad-hoc tasks)

**Fine-tuning** - When you fine-tune, you train a new model checkpoints so that it learns patterns from many examples (and store the changes after the fine-tuning in LoRA adapters). The model's weights in this case will change. In this case, you can teach the model once and it will produce the result you need by default (best for scalable, domain-specific generation)


## Features
- **SFT** (Supervised Fine-Tuning/teaches what to generate) on a small curated instruction dataset  
- **DPO** (Direct Preference Optimization/refines which version is better) on paired comparisons (`chosen` vs `rejected`)  
- **LoRA** adapters via `peft` (keeps base weights frozen by default)  
- Optional **4-bit**/8-bit loading (`bitsandbytes`) to reduce VRAM  
- **MLflow** for runs, params, and artifacts (local `./mlruns/`)  
- **FastAPI** server that loads the LoRA at inference (or merged weights)  
- Minimal **eval** hook via `lm-eval-harness` (optional)  
- **De-identification** pre-processing with Presidio (optional, for data governance)  

---

## Quickstart

### Environment
```bash
python -m venv .venv 
source .venv/bin/activate # alternatively .venv/Script/Activate.ps1 (for Windows PowerShell)
pip install --upgrade pip
pip install -r requirements.txt
# (Optional, GPU/NVIDIA) quantization + speedups
pip install -r requirements.gpu.txt
# (Optional) PII de-identification
pip install -r requirements.extra.txt
```

If bitsandbytes doesn’t install (CPU-only or macOS), set quantization.load_in_4bit: false in the config.

### Configure

Edit configs/config.yaml to pick base model, LoRA params, training hyperparams


### De-identify your dataset (optional)

```bash
python scripts/deid_presidio.py \
  --in data/sft/sample_sft.jsonl \
  --out data/sft/sample_sft.deid.jsonl

```

### Supervised Fine-Tuning (SFT)

```bash
python -m scripts.train_sft --config configs/config.yaml
```
Artifacts: outputs/sft_adapter/ (LoRA weights), tokenizer, and MLflow run in mlruns/.

### Direct Prefrence Optimization (DPO)

```bash
python -m scripts.train_dpo --config configs/config.yaml
```

### Serve for inference (FastAPI)

```bash
# Uses base model + LoRA adapter at runtime
uvicorn serve.fastapi_server:app --reload --port 8000

# Then query it (Powershell)
Invoke-RestMethod -Uri "http://localhost:8000/v1/generate" `
  -Method POST -ContentType "application/json" `
  -Body '{"prompt":"Write a two-sentence project summary.","max_new_tokens":128}'

```

### Merge LoRA → full model (optional)

```bash
python scripts/export_lora.py \
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter outputs/dpo_adapter \
  --out_dir outputs/merged_full_model

```
Merged weights are large. Prefer adapters for sharing

### LM Eval Harness (optional)

```bash
# Example: tiny subset for a smoke test
python eval/run_lmeval.py --model_path outputs/dpo_adapter --task hellaswag --limit 50
```


### Test

Covers config file validity (tests/test_config_loads.py), import and structure of main modules, environment sanity for MLflow and FastAPI

```bash
pytest -q
```

## Fine tuning using your own data

**1. Prepare your training data.**

Convert your inputs and outputs (a prompt and a resulting document, respectively) into the SFT format:

{"instruction": "Generate a marketing story for the client",
 "input": "context or bullet points you’d normally provide",
 "output": "the full polished document that worked best"}

 Or, if you only have a final documents, you can use SFT format like: 

 {"instruction": "Write a document similar in tone and structure to the successful marketing story.",
 "input": "",
 "output": "<paste the full text of one successful story>"}

Place all examples (dozens or hundreds into data/sft/mydata.jsonl)

**2. Train SFT**

Update configs/configmyaml

```bash
sft:
  dataset_path: data/sft/mydata.jsonl
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 2
```

Then run

```bash
python -m scripts.train_sft --config configs/config.yaml
```
...which will result in your "new style" adapter (located in outputs/stf_adapter/ )

**3. Add Preference Data (optional)**

Create a DPO dataset if you can rank outputs by effectiveness ("document A got better engagement than document B"):

```bash
{"prompt": "Write an onboarding guide", 
 "chosen": "Version that led to higher retention", 
 "rejected": "Version that performed worse"}
```

Then train

```bash
python -m scripts.train_dpo --config configs/config.yaml
```

...which will result in an adapter that prefers your best patterns (located in outputs/dpo_adapter)

**4. Serve and generate**

Start the inference server: 

```bash
uvicorn serve.fastapi_server:app --reload --port 8000
```

You can query it

```bash
Invoke-RestMethod -Uri "http://localhost:8000/v1/generate" `
  -Method POST -ContentType "application/json" `
  -Body '{"prompt":"Create a new marketing story for client X in the standard style that will result in highest engagement","max_new_tokens":512}'

```

**5. Iterate**

- Inspect outputs - adjust datasets (add better examples, remove weak ones)
- Re-train SFT, optionally re-run DPO
- Track experiments in MLflow (mlflow ui --port 5000)

Over time, the model becomes your domain-specific and will be producing documents alighned with your most desirable outcomes. 



## Architecture 

                ┌───────────────────────────────┐
                │        Data Sources            │
                │  SFT: instruction → output     │
                │  DPO: prompt, chosen, rejected │
                └──────────────┬────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │     Pre-processing         │
                 │  (optional Presidio scrub) │
                 └─────────────┬──────────────┘
                               │
        ┌──────────────────────▼─────────────────────────┐
        │   Training Scripts (SFT / DPO, LoRA adapters)  │
        │ - PEFT + TRL + Transformers                    │
        │ - MLflow logging for runs/artifacts            │
        └──────────────────────┬─────────────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │       Outputs              │
                 │ - LoRA adapters (SFT, DPO) │
                 │ - Checkpoints, tokenizer   │
                 └─────────────┬──────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │    Inference Server        │
                 │ (FastAPI + HuggingFace)    │
                 └─────────────┬──────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │       REST Clients         │
                 │  (curl, Postman, UI apps)  │
                 └────────────────────────────┘


## Flow Diagram

           ┌────────────┐
           │ config.yaml│
           └─────┬──────┘
                 │
                 ▼
        ┌───────────────────┐
        │  train_sft.py     │
        │  (SFT training)   │
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  train_dpo.py     │
        │  (DPO fine-tune)  │
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  MLflow tracking  │
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  FastAPI server   │
        │  (load LoRA)      │
        └────────┬──────────┘
                 │
                 ▼
        ┌───────────────────┐
        │  Client request   │
        │  (curl/UI)        │
        └───────────────────┘

## Config Toggles

| Section                          | Key                                        | Description                                 |
| -------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **base_model_id**                | e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HF model for fine-tuning                    |
| **quantization.load_in_4bit**    | true/false                                 | Enables 4-bit quantization (saves VRAM)     |
| **lora.r / alpha / dropout**     | int/float                                  | LoRA rank, scaling, and dropout hyperparams |
| **lora.target_modules**          | list or `auto`                             | Which attention layers to adapt             |
| **sft.dataset_path**             | path                                       | Path to JSONL dataset for SFT               |
| **sft.num_train_epochs**         | int                                        | Number of fine-tuning epochs                |
| **dpo.dataset_path**             | path                                       | JSONL dataset for DPO preference tuning     |
| **serve.adapter_path**           | path                                       | Which LoRA adapter to load for inference    |
| **tracking.mlflow_tracking_uri** | path                                       | Local or remote MLflow tracking store       |


## Limits

| Category         | Constraint                                            | Workaround                                      |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------- |
| **Hardware**     | CPU-only training is slow; large models need GPU VRAM | Use small models like `TinyLlama` or `gpt2`     |
| **Quantization** | `bitsandbytes` unavailable on macOS/CPU               | Disable `quantization.load_in_4bit`             |
| **Dataset size** | Sample data only (few records)                        | Replace with your domain-specific JSONL         |
| **Evaluation**   | LM Eval harness is minimal                            | Expand evaluation tasks in `eval/run_lmeval.py` |
| **Privacy**      | No built-in anonymization beyond Presidio             | Run `deid_presidio.py` before training          |
| **Merge size**   | Full merged model is large                            | Keep LoRA adapters instead of merging           |

## License

MIT (see License)