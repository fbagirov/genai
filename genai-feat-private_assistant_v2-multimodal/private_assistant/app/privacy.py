from typing import Dict, Any
from .settings import get_settings

# Optional: Presidio for local PII scrub
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _presidio_ok = True
except Exception:
    _presidio_ok = False

_cfg = get_settings()
_priv = _cfg.get("privacy", {"log_requests": False, "scrub_pii": False})
_log_requests = bool(_priv.get("log_requests", False))
_scrub = bool(_priv.get("scrub_pii", False)) and _presidio_ok

if _scrub and _presidio_ok:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

def maybe_scrub(text: str) -> str:
    if not _scrub:
        return text
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text

def guard_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Never log by default
    if not _log_requests:
        return payload
    # If you really want logs, add a redacted print here (kept disabled)
    return payload
