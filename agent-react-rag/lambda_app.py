import base64
import json
import os
from pathlib import Path
from typing import Any, Dict


def _get_rag():
    try:
        from src import main as rag_main
    except Exception as e:
        raise ImportError("RAG module not available: %s" % e)
    return rag_main


def lambda_handler(event: Dict, context: Any) -> Dict:
    """AWS Lambda handler for API Gateway (proxy integration).

    Routes:
      POST /query  — { "question": "..." }         → { "answer": "..." }
      POST /ingest — { "pdf_base64": "...", "filename": "doc.pdf" }
                     or a raw PDF body with isBase64Encoded: true
                   → { "message": "Ingested N chunks from doc.pdf" }
    """
    try:
        path = (event.get("path") or "/").rstrip("/") or "/"
        method = (event.get("httpMethod") or "POST").upper()

        if path == "/ingest" and method == "POST":
            return _handle_ingest(event)
        else:
            return _handle_query(event)

    except ImportError as exc:
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}
    except Exception as exc:
        return {"statusCode": 500, "body": json.dumps({"error": str(exc)})}


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def _handle_query(event: Dict) -> Dict:
    body = _parse_body(event)
    if isinstance(body, str):
        question = body
    elif isinstance(body, dict):
        question = body.get("question")
    else:
        question = None

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "missing 'question' field"})}

    rag = _get_rag()
    result = rag.answer_from_input({"question": question})
    answer = getattr(result, "content", None) or str(result)
    return {"statusCode": 200, "body": json.dumps({"answer": answer})}


# ---------------------------------------------------------------------------
# PDF ingest handler
# ---------------------------------------------------------------------------

def _handle_ingest(event: Dict) -> Dict:
    """Accept a PDF as either:
    - A raw binary body with ``isBase64Encoded: true`` (API Gateway binary upload)
    - A JSON body with a ``pdf_base64`` field (base64-encoded string)
    """
    is_b64_body = event.get("isBase64Encoded", False)
    raw_body = event.get("body") or ""

    if is_b64_body and raw_body:
        pdf_bytes = base64.b64decode(raw_body)
        filename = _safe_filename(
            (event.get("queryStringParameters") or {}).get("filename", "upload.pdf")
        )
    else:
        body = _parse_body(event)
        if not isinstance(body, dict):
            return {"statusCode": 400, "body": json.dumps({"error": "expected JSON body with 'pdf_base64'"})}

        pdf_b64 = body.get("pdf_base64")
        if not pdf_b64:
            return {"statusCode": 400, "body": json.dumps({"error": "missing 'pdf_base64' field"})}

        try:
            pdf_bytes = base64.b64decode(pdf_b64)
        except Exception:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid base64 in 'pdf_base64'"})}

        filename = _safe_filename(body.get("filename", "upload.pdf"))

    tmp_path = Path("/tmp") / filename
    tmp_path.write_bytes(pdf_bytes)

    try:
        rag = _get_rag()
        chunks = rag.add_pdf_to_store(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "statusCode": 200,
        "body": json.dumps({"message": f"Ingested {chunks} chunks from {filename}"}),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_body(event: Dict):
    """Return the decoded event body as a dict or string."""
    body = event.get("body") or ""
    if isinstance(body, str):
        try:
            return json.loads(body) if body else {}
        except Exception:
            return body
    return body


def _safe_filename(name: str) -> str:
    """Strip path separators so callers cannot write outside /tmp."""
    safe = Path(name).name or "upload.pdf"
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe
