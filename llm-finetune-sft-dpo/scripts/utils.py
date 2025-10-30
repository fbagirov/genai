
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import re

def get_tokenizer(tokenizer_id: str | None, base_model_id: str):
    tid = tokenizer_id or base_model_id
    tok = AutoTokenizer.from_pretrained(tid, use_fast=True)
    if tok.pad_token is None:
        # set pad to eos for causal LM
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    return tok

def build_bnb_config(qcfg: dict | None):
    if not qcfg or not qcfg.get("load_in_4bit", False):
        return None
    compute_dtype = getattr(torch, str(qcfg.get("bnb_4bit_compute_dtype", "bfloat16")))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
    )

def infer_target_modules(model) -> List[str]:
    # Heuristics: LLaMA-like vs GPT2-like
    names = [n for n, _ in model.named_modules()]
    joined = " ".join(names).lower()
    if re.search(r"q_proj|k_proj|v_proj|o_proj", joined):
        return ["q_proj","k_proj","v_proj","o_proj"]
    if re.search(r"gate_proj|up_proj|down_proj", joined):
        return ["gate_proj","up_proj","down_proj"]
    # GPT2 style
    if re.search(r"c_attn|c_proj", joined):
        return ["c_attn","c_proj"]
    # fallback: common attention projections
    return ["q_proj","v_proj"]

def get_lora_config(model, lcfg: dict):
    target = lcfg.get("target_modules", "auto")
    if target == "auto":
        target = infer_target_modules(model)
    return LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("alpha", 32)),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM",
    )

def load_base_model(base_model_id: str, quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return model
