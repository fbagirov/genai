
import os, io, time
from PIL import Image
import gradio as gr

from models.vision import VisionModel
from ocr.ocr import run_ocr
from utils.pdf_utils import pdf_to_images

DEFAULT_PROVIDER = os.getenv("VISION_PROVIDER","openai")
MAX_SIDE = 1536

def load_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    if max(w,h) > MAX_SIDE:
        sc = MAX_SIDE / max(w,h)
        img = img.resize((int(w*sc), int(h*sc)))
    return img

def caption_image(img, provider, detail, temperature, max_tokens):
    vm = VisionModel(provider=provider, temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    text = vm.caption(img, detail=detail)
    return text, time.time()-t0

def vqa_image(img, question, provider, temperature, max_tokens):
    vm = VisionModel(provider=provider, temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    text = vm.vqa(img, question=question)
    return text, time.time()-t0

def handle_pdf(pdf_bytes, multi_page):
    pages = pdf_to_images(pdf_bytes, dpi=220)
    return pages if multi_page else pages[:1]

def build_app():
    with gr.Blocks(title="Multi-Modal Mini-App") as demo:
        gr.Markdown("# 🖼️📄 Multi-Modal Mini-App — Caption, OCR, and VQA")
        gr.Markdown("> Privacy: files processed in-memory; no retention.")

        with gr.Row():
            provider = gr.Dropdown(choices=["openai","llava","florence"], value=DEFAULT_PROVIDER, label="Provider")
            task = gr.Dropdown(choices=["Caption","OCR","VQA"], value="Caption", label="Task")
            multi_page = gr.Checkbox(False, label="PDF: process all pages")
        with gr.Row():
            file_input = gr.File(label="Drop image (PNG/JPG) or PDF", file_types=["image","pdf"])
        with gr.Row():
            question = gr.Textbox(label="Question (for VQA)", placeholder="e.g., What is the main object?")
        with gr.Accordion("Advanced", open=False):
            temperature = gr.Slider(0.0, 1.2, value=0.2, step=0.1, label="Temperature (OpenAI)")
            max_tokens = gr.Slider(32, 1024, value=256, step=16, label="Max tokens (OpenAI)")
            caption_detail = gr.Radio(choices=["short","detailed"], value="short", label="Caption detail")

        output_text = gr.Textbox(label="Output")
        latency = gr.Number(label="Latency (sec)", precision=3)

        def run(file, provider, task, multi_page, question, temperature, max_tokens, caption_detail):
            if file is None:
                return "Please upload an image or PDF.", 0.0
            name = (getattr(file, "name", "upload") or "").lower()
            data = file.read() if hasattr(file, "read") else file
            is_pdf = name.endswith(".pdf")
            pages = handle_pdf(data, multi_page) if is_pdf else [load_image(data)]
            if not pages:
                return "Could not render any pages.", 0.0

            t0 = time.time()
            outs = []
            for img in pages:
                if task == "Caption":
                    text, _ = caption_image(img, provider, caption_detail, temperature, max_tokens)
                elif task == "OCR":
                    text = run_ocr(img)
                else:
                    q = (question or "What is in this image?").strip()
                    text, _ = vqa_image(img, q, provider, temperature, max_tokens)
                outs.append(text.strip())
            return ("\n\n---\n\n".join(outs)), time.time()-t0

        file_input.change(run, [file_input, provider, task, multi_page, question, temperature, max_tokens, caption_detail], [output_text, latency])
        task.change(lambda t: ("", 0.0), None, [output_text, latency])
        provider.change(lambda t: ("", 0.0), None, [output_text, latency])
        gr.Markdown("Set `OPENAI_API_KEY` for OpenAI; ensure HF models are available for local providers.")
    return demo

if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED","False")
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7861)
