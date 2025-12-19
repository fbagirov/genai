import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

class SuccessScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dataset_path = cfg["data"]["dataset_path"]
        self.success_outcomes = set(cfg["data"].get("success_outcomes", ["won"]))
        self._vec = None
        self._clf = None
        self._train()

    def _train(self):
        rows = _read_jsonl(self.dataset_path)
        texts, y = [], []
        for r in rows:
            texts.append(f"{r.get('subject','')}\n{r.get('body','')}")
            y.append(1 if r.get("outcome") in self.success_outcomes else 0)

        self._vec = TfidfVectorizer(ngram_range=(1,2), max_features=6000)
        X = self._vec.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=200)
        self._clf.fit(X, np.array(y))

    def predict_success_probability(self, subject: str, body: str) -> float:
        text = f"{subject}\n{body}"
        X = self._vec.transform([text])
        p = float(self._clf.predict_proba(X)[0, 1])
        return round(p, 3)
