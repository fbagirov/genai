import argparse
import json
import yaml
import chromadb
from sentence_transformers import SentenceTransformer

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rows = read_jsonl(cfg["data"]["dataset_path"])
    success_outcomes = set(cfg["data"].get("success_outcomes", ["won"]))
    only_success = bool(cfg["vectorstore"].get("metadata_filter", {}).get("only_success", True))
    if only_success:
        rows = [r for r in rows if r.get("outcome") in success_outcomes]

    persist_dir = cfg["vectorstore"]["persist_dir"]
    collection_name = cfg["vectorstore"]["collection"]
    model_name = cfg.get("embeddings", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    embedder = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)

    ids, docs, metas = [], [], []
    for r in rows:
        ids.append(r["id"])
        docs.append(r["body"])
        metas.append({
            "subject": r.get("subject",""),
            "industry": r.get("industry",""),
            "persona": r.get("persona",""),
            "product": r.get("product",""),
            "tone": r.get("tone",""),
            "created_at": r.get("created_at",""),
        })

    print(f"Indexing {len(ids)} emails to {persist_dir}/{collection_name} ...")
    embeddings = embedder.encode(
        [f"{m['industry']} {m['persona']} {m['product']} {m['tone']} | {m['subject']} {d}" for m, d in zip(metas, docs)],
        show_progress_bar=True
    ).tolist()

    batch = 64
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i:i+batch],
            documents=docs[i:i+batch],
            metadatas=metas[i:i+batch],
            embeddings=embeddings[i:i+batch],
        )
    print("Done.")

if __name__ == "__main__":
    main()
