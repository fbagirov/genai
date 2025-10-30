
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import torch
import threading

BASE_MODEL = os.getenv("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "outputs/dpo_adapter")

app = FastAPI()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print(f"Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "base_model": BASE_MODEL, "adapter": ADAPTER_PATH}

@app.post("/v1/generate")
def generate(req: GenRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=True,
        top_p=req.top_p,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    # collect full text
    output_text = "".join(list(streamer))
    return {"output": output_text}
