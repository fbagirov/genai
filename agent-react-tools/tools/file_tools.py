
import os

def file_read(path: str) -> str:
    if not os.path.exists(path):
        return f"[file_read] not found: {path}"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def file_write(payload: str) -> str:
    # format: "<path> :: <content>"
    if "::" not in payload:
        return "[file_write] expected 'path :: content'"
    path, content = [p.strip() for p in payload.split("::", 1)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"[file_write] wrote {len(content)} chars to {path}"
