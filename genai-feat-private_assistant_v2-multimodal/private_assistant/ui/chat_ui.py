# private_assistant/ui/chat_ui.py
import os
import mimetypes
import requests
import gradio as gr

# Point this to your FastAPI endpoint that accepts multipart form-data:
# message (Form field) + files (UploadFile list)
API = os.getenv("PRIVATE_ASSISTANT_API", "http://127.0.0.1:8000/v2/chat_with_files")


def _build_multipart(message: str, files):
    """
    Build multipart payload for FastAPI:
      - data: {"message": "..."}
      - files: [("files", (filename, fileobj, content_type)), ...]
    """
    multipart_files = []
    if files:
        for f in files:
            # Gradio can return temp file paths (str)
            path = f if isinstance(f, str) else getattr(f, "name", None)
            if not path:
                continue

            filename = os.path.basename(path)
            ctype, _ = mimetypes.guess_type(filename)
            ctype = ctype or "application/octet-stream"

            # Important: open in binary mode
            multipart_files.append(("files", (filename, open(path, "rb"), ctype)))

    data = {"message": message}
    return data, multipart_files


def chat_fn(message, history, files):
    """
    - message: user text
    - history: list of (user, assistant) tuples
    - files: list of uploaded file paths
    """
    message = (message or "").strip()
    if not message and not files:
        return "", history, None  # nothing to do

    data, multipart_files = _build_multipart(message, files)

    try:
        r = requests.post(API, data=data, files=multipart_files, timeout=300)
        r.raise_for_status()
        j = r.json()

        # Expecting {"answer": "...", "attachments_used": [...]}
        answer = j.get("answer") or j.get("content") or str(j)
        used = j.get("attachments_used") or []

        if used:
            answer += "\n\n---\n**Attachments used:** " + ", ".join(used)

        history = history + [(message if message else "[uploaded files]", answer)]
        return "", history, None

    except requests.exceptions.RequestException as e:
        err = f"Request failed: {e}"
        history = history + [(message if message else "[uploaded files]", err)]
        return "", history, None

    finally:
        # Close any open file handles we created
        for item in multipart_files:
            try:
                item[1][1].close()
            except Exception:
                pass


with gr.Blocks(title="Private Chat (Multi-Modal)") as demo:
    gr.Markdown(
        "# 🔒 Private Chat (Multi-Modal)\n"
        "_Local-first assistant. Files are uploaded to your local API only._\n\n"
        "**Tip:** Upload PDFs/images/text, then ask questions about them."
    )

    chat = gr.Chatbot(height=520)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask something... (e.g., 'Summarize the PDF' or 'What does the image show?')",
            scale=3,
        )

    files = gr.Files(
        label="Upload files (txt, md, pdf, docx, png, jpg, jpeg, webp)",
        file_count="multiple",
        type="filepath",
    )

    with gr.Row():
        send = gr.Button("Send")
        clear = gr.Button("Clear Chat")

    # Send on Enter
    msg.submit(chat_fn, [msg, chat, files], [msg, chat, files])
    # Send on Button click
    send.click(chat_fn, [msg, chat, files], [msg, chat, files])

    # Clear chat
    def clear_chat():
        return [], None, ""

    clear.click(clear_chat, [], [chat, files, msg])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8001)
