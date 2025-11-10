
import json, subprocess, time, os, sys
from pathlib import Path

TASKS = "eval/tasks.jsonl"

def run(goal: str):
    t0 = time.time()
    p = subprocess.run([sys.executable, "run_agent.py", "--goal", goal], capture_output=True, text=True)
    dt = time.time() - t0
    ok = "FINAL ANSWER" in p.stdout
    return ok, dt, p.stdout

def main():
    results = []
    Path("runs").mkdir(exist_ok=True)
    with open(TASKS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            goal = json.loads(line)["goal"]
            ok, dt, out = run(goal)
            results.append({"goal": goal, "ok": ok, "latency_sec": round(dt,2)})
            print(f"[{ 'OK' if ok else '..' }] {goal}  ({dt:.2f}s)")
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    print(f"\nSuccess: {passed}/{total} ({passed/total*100:.0f}%)")
    with open("eval/report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
