
import os, io, base64
from typing import Optional
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
except Exception:
    AutoProcessor = AutoModelForCausalLM = AutoModelForVision2Seq = None
    torch = None

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class VisionModel:
    def __init__(self, provider: str = "openai", temperature: float = 0.2, max_tokens: int = 256):
        self.provider = provider
        self.temperature = float(temperature or 0.2)
        self.max_tokens = int(max_tokens or 256)
        self.llava_model_id = os.getenv("LLAVA_MODEL_ID","llava-hf/llava-1.6-mistral-7b-hf")
        self.florence_model_id = os.getenv("FLORENCE_MODEL_ID","microsoft/Florence-2-large")
        self.client = None
        if self.provider == "openai" and OpenAI is not None:
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None
        self._hf_model = None
        self._hf_proc = None

    def caption(self, img: Image.Image, detail: str = "short") -> str:
        prompt = "Provide a short caption." if detail == "short" else "Provide a detailed caption."
        return self._call(img, prompt, task="caption")

    def vqa(self, img: Image.Image, question: str) -> str:
        return self._call(img, f"Answer concisely: {question}", task="vqa")

    def _call(self, img: Image.Image, prompt: str, task: str):
        if self.provider == "openai":
            return self._openai(img, prompt)
        if self.provider == "llava":
            return self._llava(img, prompt)
        if self.provider == "florence":
            return self._florence(img, prompt, task=task)
        return "[Unknown provider]"

    def _openai(self, img: Image.Image, prompt: str) -> str:
        if self.client is None:
            return "[OpenAI not available or API key missing.]"
        b64 = pil_to_b64(img)
        try:
            resp = self.client.chat.completions.create(
                model=os.getenv("OPENAI_VISION_MODEL","gpt-4o-mini"),
                messages=[{"role":"user","content":[
                    {"type":"text","text": prompt},
                    {"type":"image_url","image_url":{"url": f"data:image/png;base64,{b64}"}}]}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI error] {e}"

    def _ensure_llava(self):
        if self._hf_model is not None and self._hf_proc is not None:
            return
        if AutoProcessor is None:
            raise RuntimeError("Transformers not installed.")
        self._hf_proc = AutoProcessor.from_pretrained(self.llava_model_id, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(self.llava_model_id, trust_remote_code=True, device_map="auto")

    def _llava(self, img: Image.Image, prompt: str) -> str:
        try:
            self._ensure_llava()
            proc, model = self._hf_proc, self._hf_model
            inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=self.max_tokens)
            out = proc.batch_decode(gen, skip_special_tokens=True)[0]
            return out.strip()
        except Exception as e:
            return f"[LLaVA error] {e}"

    def _ensure_florence(self):
        if self._hf_model is not None and self._hf_proc is not None:
            return
        if AutoProcessor is None:
            raise RuntimeError("Transformers not installed.")
        self._hf_proc = AutoProcessor.from_pretrained(self.florence_model_id, trust_remote_code=True)
        self._hf_model = AutoModelForVision2Seq.from_pretrained(self.florence_model_id, trust_remote_code=True, device_map="auto")

    def _florence(self, img: Image.Image, prompt: str, task: str = "caption") -> str:
        try:
            self._ensure_florence()
            proc, model = self._hf_proc, self._hf_model
            if task == "caption":
                text = "<CAPTION>"
            else:
                text = f"<VQA> {prompt}"
            inputs = proc(text=text, images=img, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=self.max_tokens)
            out = proc.batch_decode(gen, skip_special_tokens=True)[0]
            return out.strip()
        except Exception as e:
            return f"[Florence-2 error] {e}"
