from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
import pytesseract


@dataclass
class OcrResult:
    source_name: str
    text: str


def ocr_image(path: Path, lang: str = "eng", tesseract_cmd: str = "") -> OcrResult:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    img = Image.open(str(path)).convert("RGB")
    text = pytesseract.image_to_string(img, lang=lang) or ""
    return OcrResult(source_name=path.name, text=text.strip())
