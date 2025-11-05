import time, requests

def main(n=5):
    t0 = time.time()
    for _ in range(n):
        r = requests.post("http://localhost:8000/api/chat", json={
            "messages":[
                {"role":"system","content":"Be concise."},
                {"role":"user","content":"Summarize the benefits of local LLMs in one sentence."}
            ]
        }, timeout=120)
        r.raise_for_status()
    dt = time.time() - t0
    print(f"Completed {n} requests in {dt:.2f}s. Avg: {dt/n:.2f}s/req")

if __name__ == "__main__":
    main()
