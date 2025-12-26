# private_assistant/attachments/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes

from pypdf import PdfReader
import docx

from PIL import Image
import pytesseract


@dataclass
class AttachmentContext:
    source: str
    content: str


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for p in reader.pages:
        chunks.append(p.extract_text() or "")
    return "\n\n".join(chunks).strip()


def _read_docx(path: Path) -> str:
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text]
    return "\n".join(paras).strip()


def _ocr_image(path: Path, *, lang: str = "eng", tesseract_cmd: str = "") -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    img = Image.open(str(path)).convert("RGB")
    return (pytesseract.image_to_string(img, lang=lang) or "").strip()


def build_attachment_context(paths: List[Path], cfg: Dict[str, Any]) -> List[AttachmentContext]:
    att_cfg = cfg.get("attachments", {})
    enabled = bool(att_cfg.get("enabled", True))
    if not enabled:
        return []

    allowed_ext = set([e.lower() for e in att_cfg.get("allowed_ext", [])]) or {
        ".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"
    }
    max_file_mb = float(att_cfg.get("max_file_mb", 10))
    max_total_chars = int(att_cfg.get("max_total_text_chars", 25000))

    ocr_enabled = bool(att_cfg.get("ocr_enabled", True))
    ocr_lang = att_cfg.get("ocr_lang", "eng")
    tesseract_cmd = att_cfg.get("tesseract_cmd", "")

    out: List[AttachmentContext] = []
    total = 0

    for p in paths:
        ext = p.suffix.lower()
        if ext not in allowed_ext:
            continue

        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > max_file_mb:
            out.append(AttachmentContext(source=p.name, content=f"[Skipped: too large ({size_mb:.1f} MB)]"))
            continue

        try:
            if ext in [".txt", ".md"]:
                text = _safe_read_text(p)
            elif ext == ".pdf":
                text = _read_pdf(p)
            elif ext == ".docx":
                text = _read_docx(p)
            elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                if not ocr_enabled:
                    text = "[Image provided. OCR disabled in config.]"
                else:
                    text = _ocr_image(p, lang=ocr_lang, tesseract_cmd=tesseract_cmd)
                    if not text:
                        text = "[OCR produced no text. Image may be low-res or not contain text.]"
            else:
                continue

        except Exception as e:
            text = f"[Failed to process: {type(e).__name__}: {e}]"

        # Cap total injection size
        if total >= max_total_chars:
            break
        if len(text) + total > max_total_chars:
            text = text[: max_total_chars - total] + "\n...[truncated]"
        total += len(text)

        out.append(AttachmentContext(source=p.name, content=text))

    return out


def format_context_block(items: List[AttachmentContext]) -> str:
    if not items:
        return ""

    parts = ["[ATTACHMENTS CONTEXT - extracted locally]\n"]
    for i, it in enumerate(items, 1):
        parts.append(f"--- Attachment {i}: {it.source} ---\n{it.content}\n")
    return "\n".join(parts).strip()
