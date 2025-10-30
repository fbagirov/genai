# retriever/hybrid.py
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # optional

class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = PersistentClient(path=cfg.paths["persist_dir"])
        self.collection = self.client.get_or_create_collection("docs")

        # Dense embeddings
        if cfg.embedding["provider"] == "sentence_transformers":
            device = cfg.embedding.get("device", "cpu")
            model_name = cfg.embedding["model"]
            self.sbert = SentenceTransformer(
                model_name, device=None if device == "auto" else device
            )
            self._dense = lambda q: self.sbert.encode(
                [q], normalize_embeddings=True
            ).tolist()[0]
        else:
            assert OpenAI is not None, "openai provider requires openai package"
            self._openai = OpenAI()
            model_name = cfg.embedding["model"]
            self._dense = lambda q: self._openai.embeddings.create(
                model=model_name, input=[q]
            ).data[0].embedding

        # BM25 cache
        with open(cfg.paths["bm25_cache"], "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.raw_docs = payload["raw_docs"]

        # Optional reranker
        self.reranker = None
        if cfg.reranker.get("enabled", False):
            try:
                from FlagEmbedding import FlagReranker
                self.reranker = FlagReranker(cfg.reranker["model"], use_fp16=True)
            except Exception as e:
                print("[warn] Reranker requested but not available:", e)

    def dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_emb = self._dense(query)
        # NOTE: 'ids' no longer allowed in include (Chroma API change)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        items = []
        for i in range(len(res["documents"][0])):
            items.append(
                {
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - res["distances"][0][i],  # similarity
                }
            )
        return items

    def bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        toks = query.split()
        scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:k]
        items = []
        for i in idxs:
            d = self.raw_docs[i]
            items.append(
                {
                    "id": f"bm25_{i}",
                    "text": d["text"],
                    "metadata": d["metadata"],
                    "score": float(scores[i]),
                }
            )
        return items

    def fuse(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        w_dense: float,
        w_bm25: float,
        topk: int,
    ) -> List[Dict[str, Any]]:
        def normalize(xs):
            if not xs:
                return []
            vals = np.array([x["score"] for x in xs], dtype=float)
            if vals.max() - vals.min() < 1e-9:
                norm = np.ones_like(vals)
            else:
                norm = (vals - vals.min()) / (vals.max() - vals.min())
            for i, x in enumerate(xs):
                x["norm_score"] = float(norm[i])
            return xs

        dense = normalize(dense)
        sparse = normalize(sparse)

        merged: Dict[str, Dict[str, Any]] = {}
        for x in dense:
            merged[x["id"]] = x | {"fused": w_dense * x["norm_score"]}
        for x in sparse:
            merged[x["id"]] = merged.get(x["id"], {**x, "fused": 0.0})
            merged[x["id"]]["fused"] += w_bm25 * x["norm_score"]

        items = list(merged.values())
        items.sort(key=lambda z: z["fused"], reverse=True)
        items = items[:topk]

        # Optional reranking
        if self.reranker:
            passages = [it["text"] for it in items]
            scores = self.reranker.compute_score([["query", p] for p in passages])
            for it, s in zip(items, scores):
                it["fused"] = float(s)
            items.sort(key=lambda z: z["fused"], reverse=True)
        return items

    def retrieve(
        self,
        query: str,
        k_dense: int,
        k_bm25: int,
        w_dense: float,
        w_bm25: float,
        topk: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        dense = self.dense_search(query, k=k_dense)
        sparse = self.bm25_search(query, k=k_bm25)
        fused = self.fuse(dense, sparse, w_dense, w_bm25, topk)

        # Gentle prior for repo-purpose questions
        ql = (query or "").lower()
        if any(w in ql for w in ("repo", "repository", "implement", "purpose", "readme", "about")):
            for it in fused:
                doc = ((it.get("metadata") or {}).get("doc_id") or "").lower()
                if "about" in doc or "readme" in doc:
                    it["fused"] = float(it.get("fused", 0.0)) + 0.25
            fused.sort(key=lambda z: z.get("fused", 0.0), reverse=True)

        conf = float(np.mean([float(x.get("fused", 0.0)) for x in fused])) if fused else 0.0
        return fused, conf
