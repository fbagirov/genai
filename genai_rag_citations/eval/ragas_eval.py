import argparse
import json
from typing import List, Dict


import pandas as pd
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate


from rag_pipeline import read_config
from retriever.hybrid import HybridRetriever

def load_eval_dataset(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    cfg = read_config(args.config)
    retriever = HybridRetriever(cfg)

    records = load_eval_dataset(args.dataset_path)

    questions = [r["question"] for r in records]
    ground_truths = [r["ground_truth"] for r in records]

    contexts = []
    for r in records:
        hits, _ = retriever.retrieve(r["question"], cfg.retrieval["k_dense"], cfg.retrieval["k_bm25"], cfg.retrieval["weight_dense"], cfg.retrieval["weight_bm25"], args.k)
        contexts.append([h["text"] for h in hits])

    # In a full pipeline you'd also generate answers with the LLM.
    # For demo, we set answers empty and focus on retriever metrics (precision/recall) and faithfulness when text exists.
    answers = ["" for _ in questions]

    data = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        data,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    print(result)

if __name__ == "__main__":
    main()