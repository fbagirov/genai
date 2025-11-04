
import argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to adapter (LoRA) or merged model")
    ap.add_argument("--task", default="hellaswag")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except Exception as e:
        print("lm-eval not installed or incompatible. Install pinned version from requirements.txt")
        sys.exit(1)

    # Simple HF model wrapper; if an adapter dir is provided, we rely on env var to load PEFT at init.
    os.environ.setdefault("HF_EVAL_MODEL_ID", args.model_path)
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_path},dtype=float32",
        tasks=[args.task],
        limit=args.limit,
    )
    print(results)

if __name__ == "__main__":
    main()
