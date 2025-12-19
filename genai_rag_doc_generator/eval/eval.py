import argparse
import random
import yaml
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(int(cfg.get("eval", {}).get("seed", 7)))
    api = f"http://{cfg['server']['host']}:{cfg['server']['port']}"
    n = int(cfg.get("eval", {}).get("test_prompts", 10))

    industries = ["SaaS","Healthcare","FinTech","Manufacturing","Retail","Defense","Education","Real Estate"]
    personas = ["VP of Sales","Head of Ops","CFO","CTO","Program Manager","Procurement Lead","Marketing Director"]

    ok = 0
    for i in range(n):
        payload = {
            "industry": random.choice(industries),
            "persona": random.choice(personas),
            "product": random.choice(["SecureRAG","OpsPulse","CloudCostGuard","DataBridge"]),
            "value_prop": random.choice(["private doc Q&A with citations","cloud cost optimization","workflow automation","data quality monitoring"]),
            "tone": random.choice(["consultative","executive","direct"]),
            "goal": "book a 15-min call",
        }
        try:
            r = requests.post(f"{api}/v1/generate_email", json=payload, timeout=60)
            r.raise_for_status()
            d = r.json()
            assert d.get("subject") and d.get("body")
            ok += 1
        except Exception as e:
            print("FAILED", i, type(e).__name__, e)

    print(f"Smoke eval: {ok}/{n} succeeded")

if __name__ == "__main__":
    main()
