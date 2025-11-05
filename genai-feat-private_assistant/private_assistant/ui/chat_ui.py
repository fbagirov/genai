# ui/chat_ui.py
import requests
import gradio as gr

API = "http://127.0.0.1:8000/v1/chat"

def chat_fn(message, history):
    payload = {"messages": [{"role": "user", "content": message}]}
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    answer = r.json()["content"]
    history.append((message, answer))
    return "", history

with gr.Blocks(title="Private Chat - Local LLM") as demo:
    gr.Markdown("# 🔒 Private Chat\n_Local Ollama model — nothing sent to cloud._")
    chat = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask something...")
    msg.submit(chat_fn, [msg, chat], [msg, chat])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8001)
