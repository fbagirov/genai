import requests

def test_chat_roundtrip():
    payload = {
        "messages":[
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"Say hello in one sentence."}
        ]
    }
    r = requests.post("http://localhost:8000/api/chat", json=payload, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "content" in data and len(data["content"]) > 0
