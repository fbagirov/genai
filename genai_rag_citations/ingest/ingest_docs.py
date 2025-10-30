import argparse
import os
import glob
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any


import yaml
from pydantic import BaseModel
from pypdf import PdfReader
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class Config(BaseModel):
    embedding: Dict[str, Any]
    paths: Dict[str, str]

def read_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def load_text_from_file(fp: str) -> List[Dict[str, Any]]:
    chunks = []
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(fp)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {"source_path": fp, "page": i + 1, "doc_id": os.path.basename(fp)}
                    })
    else:
        with open(fp, "r", errors="ignore", encoding="utf-8") as f:
            text = f.read()
        # naive splitter
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            chunks.append({
                "text": p,
                "metadata": {"source_path": fp, "page": i + 1, "doc_id": os.path.basename(fp)}
                })
    return chunks




@dataclass
class Stores:
    chroma: Any
    bm25: Any



def build_dense_store(cfg: Config, docs: List[Dict[str, Any]]):
    os.makedirs(cfg.paths["persist_dir"], exist_ok=True)
    client = PersistentClient(path=cfg.paths["persist_dir"])

    # Embeddings
    if cfg.embedding["provider"] == "sentence_transformers":
        device = cfg.embedding.get("device", "cpu")
        model_name = cfg.embedding["model"]
        sbert = SentenceTransformer(model_name, device=None if device == "auto" else device)


        def _embed(batch):
            return sbert.encode(batch, normalize_embeddings=True).tolist()


    elif cfg.embedding["provider"] == "openai":
        from openai import OpenAI
        client_ = OpenAI()
        model_name = cfg.embedding["model"]

        def _embed(batch):
            resp = client_.embeddings.create(model=model_name, input=batch)
            return [d.embedding for d in resp.data]
    else:
        raise ValueError("Unknown embedding provider")


    collection = client.get_or_create_collection(name="docs")

    ids = []    
    texts = []
    metadatas = []
    for i, d in enumerate(docs):
        ids.append(f"doc_{i}")
        texts.append(d["text"])
        metadatas.append(d["metadata"])


    embeddings = _embed(texts)
    # Upsert in small batches to avoid memory spikes
    batch = 128
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


    return client    

def build_bm25_index(cfg: Config, docs: List[Dict[str, Any]]):
    tokenized = [d["text"].split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "raw_docs": docs,
    }
    with open(cfg.paths["bm25_cache"], "wb") as f:
        pickle.dump(payload, f)
    return bm25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--persist_dir", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()


    cfg = read_config(args.config)
    if args.persist_dir:
        cfg.paths["persist_dir"] = args.persist_dir

    files = []
    for ext in ("*.pdf", "*.md", "*.txt"):
        files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    assert files, f"No input files found in {args.input_dir}"

    docs = []
    for fp in files:
        docs.extend(load_text_from_file(fp))

    build_dense_store(cfg, docs)
    build_bm25_index(cfg, docs)
    print(f"Ingested {len(docs)} chunks from {len(files)} files.")

if __name__ == "__main__":
    main()