
from typing import List
import re

class Planner:
    """A tiny ReAct-style planner.
    Produces blocks like:
    THOUGHT: ...
    ACTION: <tool_name>
    INPUT: <input for tool>
    """
    def __init__(self, provider: str = "local"):
        self.provider = provider

    def first_step(self, goal: str) -> str:
        return f"""THOUGHT: I need to solve the goal step-by-step.
ACTION: decide
INPUT: {goal}"""

    def next(self, goal: str, history: List[str]) -> str:
        transcript = "\n".join(history[-6:])
        if "FINAL:" in transcript:
            return ""

        last_obs = ""
        for line in reversed(history):
            if line.startswith("OBSERVATION:"):
                last_obs = line
                break

        if re.search(r"save|write to|store in|output\.txt", goal, re.I) and "file_write" not in transcript:
            return "THOUGHT: I should persist the result to a file.\nACTION: file_write\nINPUT: output.txt :: <your text here>"

        if re.search(r"capital|price|define|who|what|when|where|why|how", goal, re.I) and "web_search" not in transcript:
            q = re.sub(r"^Find|search for|lookup", "", goal, flags=re.I).strip()
            return f"THOUGHT: I should search the web for this.\nACTION: web_search\nINPUT: {q}"

        if "Traceback" in last_obs or "Error:" in last_obs:
            return "THOUGHT: The code failed; try a simpler snippet.\nACTION: python_exec\nINPUT: print('hello')"

        return "THOUGHT: I can now answer succinctly.\nACTION: final\nINPUT: Provide a concise answer based on observations."
