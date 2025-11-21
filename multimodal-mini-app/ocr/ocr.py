
from PIL import Image
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

_paddle = None

def _get_paddle():
    global _paddle
    if _paddle is None and PaddleOCR is not None:
        _paddle = PaddleOCR(use_angle_cls=True, lang='en')
    return _paddle

def run_ocr(img: Image.Image) -> str:
    img = img.convert("RGB")
    if pytesseract is not None:
        try:
            txt = pytesseract.image_to_string(img)
            if txt and txt.strip():
                return txt.strip()
        except Exception:
            pass
    o = _get_paddle()
    if o is None:
        return "[OCR engines unavailable. Install Tesseract or PaddleOCR.]"
    arr = np.array(img)
    res = o.ocr(arr, cls=True)
    lines = []
    for page in res:
        for line in page:
            lines.append(line[1][0])
    return "\n".join(lines).strip() if lines else "[No text detected]"
