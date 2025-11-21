
# multimodal-mini-app (Image + Text)

**Goal:** Caption images, run OCR, and answer questions about images or PDFs.  
**Stack:** Toggle between **LLaVA-v1.6**, **Florence-2**, or **OpenAI GPT‑4o‑mini** for vision; OCR via **Tesseract** (primary) with **PaddleOCR** fallback. UI built with **Gradio** (drag-and-drop).

> **Privacy:** The app does **not retain** uploaded images or PDFs. Files are processed in-memory for the current session only. Disable Gradio analytics with `GRADIO_ANALYTICS_ENABLED=False`.

## Features
- Caption (short/detailed), OCR, and VQA for images or PDF pages
- Provider toggle: OpenAI / LLaVA / Florence‑2
- PDF rendering to images (first page or all pages)
- Latency per image/page
- Eval script for small hand-labeled set

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**OCR external dependency:**  
- Tesseract: macOS `brew install tesseract`; Ubuntu `sudo apt-get install tesseract-ocr`; Windows installer on GitHub (UB-Mannheim).  
- PaddleOCR is installed via pip and used as fallback.

**Provider setup:**  
- OpenAI: `export OPENAI_API_KEY=sk-...` (Windows: `$env:OPENAI_API_KEY="sk-..."`)  
- LLaVA/Florence: ensure HF weights are available; GPU recommended.

Run the app:
```bash
python app.py
```

## Eval (toy)
```bash
python eval/eval.py --dataset eval/sample_eval.jsonl --provider openai --task caption
```

## Edge cases
- Low-res scans → increase DPI when rendering PDFs
- Skewed/rotated pages → PaddleOCR with angle_cls
- Handwriting → limited OCR accuracy
- Large PDFs → process first page or small batch
