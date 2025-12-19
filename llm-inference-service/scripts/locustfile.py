
from locust import HttpUser, task, between
import json, random

PROMPTS = [
    "Write a single-sentence product tagline about eco-friendly bottles.",
    "Explain quantum computing in one short paragraph for a 10th grader.",
    "Give three bullet points about benefits of daily walking.",
    "Summarize the causes of the French Revolution in 3 bullets."
]

class InferenceUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def generate(self):
        prompt = random.choice(PROMPTS)
        payload = {"prompt": prompt, "max_new_tokens": 64, "temperature": 0.2}
        headers = {"Content-Type":"application/json"}
        self.client.post("/v1/generate", data=json.dumps(payload), headers=headers)
