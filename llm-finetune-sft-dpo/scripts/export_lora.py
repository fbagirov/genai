
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)
    print(f"Merged model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
