from dotenv import load_dotenv
import argparse, os, uuid, yaml
from typing import List
from agent.planner import Planner
from agent.verifier import Verifier
from tools.web_search import web_search
from tools.file_tools import file_read, file_write
from tools.python_exec import python_exec
from memory.scratchpad import Scratchpad
import re, ast

load_dotenv(dotenv_path=".env")

def extract_answer_from_history(history: List[str]) -> str:
    """
    Look back through OBSERVATION lines for a web_search result list,
    pick the best snippet (prefer a 'Direct answer'), and return it
    with a simple citation when available.
    Works even if pretty-printing changed, and tries regex if literal parsing fails.
    """
    for h in reversed(history):
        if not h.startswith("OBSERVATION: "):
            continue
        raw = h[len("OBSERVATION: "):].strip()

        # Fast path: try parsing a Python literal list of dicts
        if raw.startswith("[") and "'title':" in raw:
            try:
                data = ast.literal_eval(raw)  # safe parse of Python literal
                if isinstance(data, list) and data:
                    # Prefer a 'Direct answer' item; else use first result
                    direct = next((d for d in data
                                    if str(d.get("title", "")).lower().startswith("direct answer")), None)
                    item = direct or data[0]
                    snippet = (item.get("snippet") or "").strip()
                    title   = (item.get("title")   or "").strip()
                    url     = (item.get("url")     or "").strip()
                    answer_text = snippet or title or ""
                    if answer_text:
                        cite = f"\n\n[source: {url or 'web_search'}]" if url or 'web_search' else ""
                        return (answer_text + cite).strip()
            except Exception:
                # fall back to regex below
                pass

        # Fallback: regex extraction from a truncated line
        # 1) Try to capture 'Direct answer' snippet
        m = re.search(r"title'\s*:\s*'Direct answer'.*?snippet'\s*:\s*'([^']+)'", raw, flags=re.I|re.S)
        if m:
            snippet = m.group(1).strip()
            if snippet:
                return snippet + "\n\n[source: web_search]"
        # 2) Otherwise capture first snippet
        m2 = re.search(r"snippet'\s*:\s*'([^']+)'", raw, flags=re.I)
        if m2:
            snippet = m2.group(1).strip()
            if snippet:
                return snippet + "\n\n[source: web_search]"

    return ""

# ACTIONS: do NOT truncate observations – store the full tool output in history
ACTIONS = {
    "web_search": lambda s: ("OBSERVATION: " + str(web_search(s)), True),
    "file_read":  lambda s: ("OBSERVATION: " + file_read(s), False),
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
        # Parse action
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

        # Route tool
        func = ACTIONS.get(action)
        if not func:
            func = ACTIONS["final"]; payload = f"Unknown action '{action}'. Ending."
        obs, web_used = func(payload)
        used_web = used_web or web_used
        history.append(obs); scratch.log("obs", obs)
        if cfg.get("logging", {}).get("verbose", True):
            print(obs)

        # Stop conditions
        if obs.startswith("FINAL:"):
            break
        if verifier.should_stop(history):
            break

        # Planner next step
        step = planner.next(args.goal, history)
        if not step:
            break
        history.append(step); scratch.log("plan", step)
        if cfg.get("logging", {}).get("verbose", True):
            print(step)

    # Extract final
    final = next((h[7:] for h in history if h.startswith("FINAL:")), "No final produced.")
    final = verifier.enforce(final, used_web)

    # If the planner left a template, try to build from last web_search observation
    if final.strip().lower().startswith(("provide a concise answer", "provide the answer extracted")):
        fallback = extract_answer_from_history(history)
        if fallback:
            final = fallback

    # Post-hook: if the goal asks to "save ... to <file>", write the FINAL content there
    m = re.search(r"save (?:it|the result|the answer|this)? ?to ([\w\.\-\\/:]+)", args.goal, flags=re.I)
    if m:
        target = m.group(1)
        obs, _ = ACTIONS["file_write"](f"{target} :: {final}")
        print(obs)

    print("\n=== FINAL ANSWER ===\n" + final)

if __name__ == "__main__":
    main()
