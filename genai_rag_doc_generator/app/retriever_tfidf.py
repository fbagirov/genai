import json
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, data_path: str):
        self.docs = []
        self.meta = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["outcome"] == "won":
                    self.docs.append(row["email"])
                    self.meta.append(row)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = scores.argsort()[-k:][::-1]
        return [self.docs[i] for i in top_idx]

# from typing import List, Dict, Any
# import chromadb
# from sentence_transformers import SentenceTransformer

# class Retriever:
#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.persist_dir = cfg["vectorstore"]["persist_dir"]
#         self.collection_name = cfg["vectorstore"]["collection"]
#         model_name = cfg.get("embeddings", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
#         self.embedder = SentenceTransformer(model_name)

#         self.client = chromadb.PersistentClient(path=self.persist_dir)
#         self.collection = self.client.get_or_create_collection(self.collection_name)

#     def _query_text(self, industry: str, persona: str, product: str, value_prop: str, tone: str) -> str:
#         return f"industry={industry}; persona={persona}; product={product}; value_prop={value_prop}; tone={tone}"

#     def retrieve_success_examples(self, industry: str, persona: str, product: str, value_prop: str, tone: str, k: int = 5) -> List[Dict[str, Any]]:
#         q = self._query_text(industry, persona, product, value_prop, tone)
#         q_emb = self.embedder.encode([q]).tolist()[0]

#         results = self.collection.query(query_embeddings=[q_emb], n_results=k)

#         examples: List[Dict[str, Any]] = []
#         ids = results.get("ids", [[]])[0]
#         metas = results.get("metadatas", [[]])[0]
#         docs = results.get("documents", [[]])[0]
#         dists = results.get("distances", [[]])[0]

#         min_sim = float(self.cfg.get("retrieval", {}).get("min_similarity", 0.0))

#         for _id, meta, doc, dist in zip(ids, metas, docs, dists):
#             sim = 1.0 - float(dist) if dist is not None else 0.0
#             if sim < min_sim:
#                 continue
#             item = {
#                 "id": _id,
#                 "similarity": sim,
#                 "subject": (meta or {}).get("subject", ""),
#                 "body": doc,
#                 "industry": (meta or {}).get("industry", ""),
#                 "persona": (meta or {}).get("persona", ""),
#                 "product": (meta or {}).get("product", ""),
#                 "tone": (meta or {}).get("tone", ""),
#                 "created_at": (meta or {}).get("created_at", ""),
#             }
#             examples.append(item)
#         return examples
