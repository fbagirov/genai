
# multimodal-mini-app (Image + Text)

This app is a multi-modal mini-assistant that can:
- Caption images (what’s in this photo?)
- Run OCR (extract text from screenshots / scans / PDFs)
- Perform Visual Question Answering (VQA) (answer a question about the image/PDF)

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
- Tesseract: macOS `brew install tesseract`; Ubuntu `sudo apt-get install tesseract-ocr`; Windows `pip install tesseract`  
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

## Architecture 

- app.py – Gradio UI + request routing:
    - Handles file upload (image or PDF)
    - Normalizes input (single image vs PDF pages)
    - Calls the appropriate vision model and/or OCR backend
    - Formats a combined answer + raw extracted text
- Vision backends
    - OpenAI (e.g. gpt-4o-mini) or other vision-capable model
    - Generates captions or answers based on image pixels or rendered PDF pages
- OCR backends
    - Tesseract (default) or PaddleOCR as a fallback
    - Used when task = "OCR" or when you want text to feed into the LLM
- PDF handling
    - Uses a PDF renderer (e.g. pdf2image) to convert pages into images
    - Can process the first page only or multiple pages, depending on settings
- Privacy
    - Files are processed in-memory for the current request
    - No long-term retention in the app itself (no local DB / logging of file contents)


## Flow process

    A[User uploads file\n(image or PDF)] --> B[Gradio input\nfile component]
    B --> C{File type?}
    C -->|Image| D[Load image\n(PIL)]
    C -->|PDF| E[Render PDF pages\n(pdf2image)]
    E --> F{Multi-page?}
    F -->|No| G[Use first page only]
    F -->|Yes| H[Select subset of pages]

    D --> I{Task?}
    G --> I

    I -->|Caption| J[Call vision model\n(e.g. GPT-4o-mini)]
    I -->|OCR| K[Run OCR\n(Tesseract / PaddleOCR)]
    I -->|VQA| L[Call vision model\nwith user question]

    J --> M[Format caption\n(+ optional OCR text)]
    K --> M
    L --> M

    M --> N[Return result to Gradio UI]


## Config toggles

.env: 

```
# Vision provider: openai | llava | florence (depending on what you've wired)
VISION_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # or another vision-capable model

# OCR backend: tesseract | paddle
OCR_BACKEND=tesseract

# Privacy (if implemented)
DELETE_UPLOADS_ON_COMPLETE=true

```

- VISION_PROVIDER
    - openai – calls OpenAI’s vision model (needs OPENAI_API_KEY).
    - llava / florence – local or HF models (heavier but more private).
- OPENAI_MODEL
    - Smaller/cheaper models → faster, slightly lower quality.
    - Larger models → better understanding, slower / more expensive.
- OCR_BACKEND
    - tesseract – widely available, good enough for many scans.
    - paddle – often better on complex layouts, but heavier to install.

## UI Parameters (for Gradio)

In the interface, you’ll see:
- Task selector
    - Caption – generate a description of the image/PDF page.
    - OCR – extract text only (no reasoning).
    - VQA – answer a user-supplied question about the image/PDF.
- Multi-page toggle (for PDFs)
    - Off – only the first page is processed (faster).
    - On – multiple pages processed (slower but more context).
- Question box (for VQA)
    - Natural language question: “What is the total amount due?”
- Temperature
    - Lower (0.0–0.2) → more deterministic, factual answers.
    - Higher (0.5–0.7) → more creative, but more variability.
- Max tokens / max length
    - Limits how long the answer can be.
    - Lower values → faster responses; higher → more detailed explanations.
- Caption detail level
    - Short vs detailed captions; may be passed in the prompt template.


## Edge cases
- Low-res scans → increase DPI when rendering PDFs
- Skewed/rotated pages → PaddleOCR with angle_cls
- Handwriting → limited OCR accuracy
- Large PDFs → process first page or small batch

## Limitations

- Model limits
    - Vision models have context/window limits; very large PDFs may need:
    - Only selected pages processed, or
    - Summaries per page instead of whole-document reasoning.
- OCR quality
    - Low-resolution scans, rotated pages, handwriting, or heavy noise can cause poor OCR.
    - Very complex layouts (tables, multi-column PDFs) may lose structure.
- PDF size
    - Many pages or high-resolution pages can:
    - Increase latency,
    - Consume more memory,
    - Hit provider limits (for image or token size).
- Latency
    - VQA + multi-page PDFs + large models = slow.
    - For snappy demos, prefer:
    - Single page,
    - Smaller models,
    - Lower max_tokens.
- Privacy
    - The app itself does not persist uploads, but:
    - Cloud providers (OpenAI) see the image/PDF content unless you use a local model.
    - For strict privacy, use local vision models (LLaVA/Florence) and local OCR only.