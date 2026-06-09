import json
import os
from typing import Any, Dict


def _call_rag(question: str) -> Any:
    """Try to call the project's RAG function if available.

    Returns a serializable result or raises an ImportError if the project
    entrypoint isn't available in the container.
    """
    try:
        # Import lazily so the container can still build if heavy deps are missing
        from src import main as rag_main
    except Exception as e:
        raise ImportError("RAG module not available: %s" % e)

    # Expect the repo to expose `answer_from_input(input_obj)`
    if not hasattr(rag_main, "answer_from_input"):
        raise AttributeError("`answer_from_input` not found in src.main")

    return rag_main.answer_from_input({"question": question})


def lambda_handler(event: Dict, context: Any) -> Dict:
    """AWS Lambda handler compatible with API Gateway (proxy) invocations.

    Expects a JSON body with a `question` string. Returns JSON with `answer`.
    """
    try:
        if isinstance(event, dict) and "body" in event:
            body = event.get("body") or ""
        else:
            body = event or ""

        if isinstance(body, str):
            try:
                body = json.loads(body) if body else {}
            except Exception:
                body = {"question": body}

        question = None
        if isinstance(body, dict):
            question = body.get("question")

        if not question:
            return {"statusCode": 400, "body": json.dumps({"error": "missing question"})}

        result = _call_rag(question)

        # If result is an object with `content` attribute (LLM libs), try to extract it
        if hasattr(result, "content"):
            answer = getattr(result, "content")
        else:
            answer = result

        return {"statusCode": 200, "body": json.dumps({"answer": str(answer)})}

    except ImportError as ie:
        return {"statusCode": 500, "body": json.dumps({"error": str(ie)})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
