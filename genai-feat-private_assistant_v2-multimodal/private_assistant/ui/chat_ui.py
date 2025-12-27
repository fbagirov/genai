# private_assistant/ui/chat_ui.py
import os
import mimetypes
import requests
import gradio as gr

# FastAPI endpoint that accepts multipart form-data:
#   message (Form field) + files (UploadFile list)
API = os.getenv("PRIVATE_ASSISTANT_API", "http://127.0.0.1:8000/v2/chat_with_files")

SUPPORTED_EXT = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".doc"}

DEFAULT_FILE_PROMPT = (
    "Use the uploaded file(s) as context. If they are student submissions, grade them per the rubric. "
    "Return exactly the required JSON schema."
)

def _guess_content_type(filename: str) -> str:
    ctype, _ = mimetypes.guess_type(filename)
    return ctype or "application/octet-stream"

def _as_path(item):
    # gr.Files(type="filepath") returns strings, but keep this flexible
    if isinstance(item, str):
        return item
    return getattr(item, "name", None)

def chat_fn(message, history, files):
    """
    message: user text (str)
    history: list of (user, assistant)
    files: list of file paths (from gr.Files(type="filepath"))
    """
    history = history or []
    files = files or []

    msg = (message or "").strip()

    # If user uploads files but leaves message empty, FastAPI will 422 unless we still send message.
    if not msg and files:
        msg = DEFAULT_FILE_PROMPT

    if not msg and not files:
        return "", history, files  # nothing to send

    # Build multipart request
    opened = []
    multipart_files = []

    try:
        for item in files:
            path = _as_path(item)
            if not path:
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext and ext not in SUPPORTED_EXT:
                # Don't hard-fail; just skip unsupported files
                continue

            filename = os.path.basename(path)
            ctype = _guess_content_type(filename)

            fh = open(path, "rb")
            opened.append(fh)
            multipart_files.append(("files", (filename, fh, ctype)))

        data = {"message": msg}  # MUST be present for your FastAPI signature

        r = requests.post(API, data=data, files=multipart_files, timeout=300)
        r.raise_for_status()
        j = r.json()

        # Expecting {"answer": "...", "attachments_used": [...]}
        answer = j.get("answer") or j.get("content") or str(j)
        used = j.get("attachments_used") or j.get("attachments") or []

        if used:
            answer += "\n\n---\nAttachments used: " + ", ".join(map(str, used))

        user_label = message.strip() if (message or "").strip() else "[uploaded files]"
        history = history + [(user_label, answer)]

        # Clear textbox after send; keep files as-is or clear them (your choice).
        # Here: clear textbox, and clear file picker so users don't accidentally resend.
        return "", history, None

    except requests.exceptions.RequestException as e:
        err = f"Request failed: {e}"
        user_label = message.strip() if (message or "").strip() else "[uploaded files]"
        history = history + [(user_label, err)]
        return "", history, files

    finally:
        for fh in opened:
            try:
                fh.close()
            except Exception:
                pass


with gr.Blocks(title="Private Chat (Multi-Modal)") as demo:
    gr.Markdown(
        "# 🔒 Private Chat (Multi-Modal)\n"
        "_Local-first assistant. Files are uploaded to your local API only._\n\n"
        "**Tip:** Upload PDFs/images/text, then ask questions about them."
    )

    chat = gr.Chatbot(height=520)

    msg = gr.Textbox(
        placeholder="Ask something... (or upload files + click Send)",
        lines=4,
    )

    files = gr.Files(
        label="Upload files (txt, md, pdf, docx, doc, png, jpg, jpeg, webp)",
        file_count="multiple",
        type="filepath",
    )

    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear Chat")

    msg.submit(chat_fn, [msg, chat, files], [msg, chat, files])
    send.click(chat_fn, [msg, chat, files], [msg, chat, files])

    def clear_chat():
        return [], None, ""

    clear.click(clear_chat, [], [chat, files, msg])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8001)
