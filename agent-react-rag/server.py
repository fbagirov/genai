"""Flask HTTP server for EC2 deployment.

Exposes the same RAG functionality as lambda_app.py but as a plain HTTP server.
Lazy-initialises the model on the first /query or /ingest call so the process
starts quickly even on a slow t2.micro.

Endpoints:
  GET  /health        — liveness probe, no model load
  POST /query         — { "question": "..." }                  → { "answer": "..." }
  POST /ingest        — { "pdf_base64": "...", "filename": "..." }
                                                                → { "message": "..." }
"""

import base64
import os
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/query")
def query():
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "missing 'question' field"}), 400

    from src.main import answer_from_input
    result = answer_from_input({"question": question})
    answer = getattr(result, "content", None) or str(result)
    return jsonify({"answer": answer})


@app.post("/ingest")
def ingest():
    data = request.get_json(silent=True) or {}
    pdf_b64 = data.get("pdf_base64")
    filename = data.get("filename", "upload.pdf")

    if not pdf_b64:
        return jsonify({"error": "missing 'pdf_base64' field"}), 400

    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception:
        return jsonify({"error": "invalid base64 in 'pdf_base64'"}), 400

    safe_name = Path(filename).name or "upload.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    tmp_path = Path("/tmp") / safe_name
    tmp_path.write_bytes(pdf_bytes)

    try:
        from src.main import add_pdf_to_store
        chunks = add_pdf_to_store(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return jsonify({"message": f"Ingested {chunks} chunks from {safe_name}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
