from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import docx


@dataclass
class LoadedText:
    source_name: str
    text: str


def load_text_file(path: Path) -> LoadedText:
    return LoadedText(source_name=path.name, text=path.read_text(encoding="utf-8", errors="ignore"))


def load_pdf(path: Path) -> LoadedText:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append(page.extract_text() or "")
    return LoadedText(source_name=path.name, text="\n\n".join(pages).strip())


def load_docx(path: Path) -> LoadedText:
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text]
    return LoadedText(source_name=path.name, text="\n".join(paras).strip())
