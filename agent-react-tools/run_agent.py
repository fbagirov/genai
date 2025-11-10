
import argparse, os, uuid, yaml
from typing import List
from agent.planner import Planner
from agent.verifier import Verifier
from tools.web_search import web_search
from tools.file_tools import file_read, file_write
from tools.python_exec import python_exec
from memory.scratchpad import Scratchpad
from dotenv import load_dotenv

load_dotenv()

ACTIONS = {
    "web_search": lambda s: ("OBSERVATION: " + str(web_search(s))[:1000], True),
    "file_read":  lambda s: ("OBSERVATION: " + file_read(s)[:1000], False),
    "file_write": lambda s: ("OBSERVATION: " + file_write(s), False),
    "python_exec":lambda s: ("OBSERVATION: " + python_exec(s), False),
    "final":      lambda s: ("FINAL: " + s, False),
    "decide":     lambda s: ("OBSERVATION: ok", False),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True, help="Natural language goal")
    ap.add_argument("--config", default="configs/config.example.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = str(uuid.uuid4())[:8]
    scratch = Scratchpad(run_id)
    planner = Planner(provider=cfg.get("model", {}).get("provider", "local"))
    verifier = Verifier(cfg)

    history: List[str] = []
    used_web = False

    step = planner.first_step(args.goal)
    history.append(step); scratch.log("plan", step)
    if cfg.get("logging", {}).get("verbose", True):
        print(step)

    for i in range(int(cfg.get("max_steps", 12))):
        action = None; payload = ""
        for tag in ("ACTION:", "INPUT:"):
            if tag not in step:
                action, payload = "final", "Unable to proceed; returning best effort."
                break
        if not action:
            try:
                ahead, ainp = step.split("ACTION:",1)[1].split("INPUT:",1)
                action = ahead.strip().splitlines()[0].strip()
                payload = ainp.strip()
            except Exception:
                action, payload = "final", "Internal parse error; best effort."

        func = ACTIONS.get(action)
        if not func:
            func = ACTIONS["final"]; payload = f"Unknown action '{action}'. Ending."
        obs, web_used = func(payload)
        used_web = used_web or web_used
        history.append(obs); scratch.log("obs", obs)
        if cfg.get("logging", {}).get("verbose", True):
            print(obs)

        if obs.startswith("FINAL:"):
            break
        if verifier.should_stop(history):
            break

        step = planner.next(args.goal, history)
        if not step:
            break
        history.append(step); scratch.log("plan", step)
        if cfg.get("logging", {}).get("verbose", True):
            print(step)

    final = next((h[7:] for h in history if h.startswith("FINAL:")), "No final produced.")
    final = verifier.enforce(final, used_web)
    print("\n=== FINAL ANSWER ===\n" + final)

if __name__ == "__main__":
    main()
