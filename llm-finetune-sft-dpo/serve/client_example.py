
import requests, json

resp = requests.post("http://localhost:8000/v1/generate",
                     json={"prompt":"Give two bullet points on this project.","max_new_tokens":128})
print(resp.json())
