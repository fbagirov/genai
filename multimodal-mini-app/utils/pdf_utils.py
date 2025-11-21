
from typing import List
from PIL import Image
from pdf2image import convert_from_bytes

def pdf_to_images(pdf_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    except Exception:
        return []
