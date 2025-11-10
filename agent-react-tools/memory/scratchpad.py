
import json, time, os

class Scratchpad:
    def __init__(self, run_id: str, dirpath: str = "runs"):
        self.run_id = run_id
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)
        self.path = os.path.join(self.dirpath, f"{run_id}.jsonl")

    def log(self, kind: str, text: str):
        rec = {"ts": time.time(), "kind": kind, "text": text}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
