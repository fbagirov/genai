# agent/planner.py
from typing import List
import re

class Planner:
    """
    Tiny ReAct-style planner that emits:
      THOUGHT: ...
      ACTION: <tool_name>
      INPUT: <payload>
    Tools expected by run_agent.py: web_search, file_write, file_read, python_exec, final, decide
    """

    def __init__(self, provider: str = "local"):
        self.provider = provider

    def first_step(self, goal: str) -> str:
        # Seed the loop
        return f"""THOUGHT: I need to solve the goal step-by-step.
ACTION: decide
INPUT: {goal}"""

    def next(self, goal: str, history: List[str]) -> str:
        """
        Decide the next step based on the goal and latest observations.
        Strategy:
          1) For factual queries (who/what/when/where/how, job title, capital...), do web_search first.
          2) If the goal asks to "save ... to <file>", call file_write AFTER search (placeholder content; post-hook overwrites).
          3) Then produce final.
        """
        transcript = "\n".join(history[-10:])

        # Already finished?
        if "FINAL:" in transcript:
            return ""

        # 1) SEARCH FIRST for factual queries
        factual = re.search(r"(capital|price|define|who|what|when|where|why|how|job title|title of|role of)", goal, re.I)
        not_yet_searched = "web_search" not in transcript
        if factual and not_yet_searched:
            # strip "and save ... to ..." from the query for better search
            q = re.sub(r"(?i)\s*and\s*save.*$", "", goal).strip()
            q = re.sub(r"(?i)^(find|search for|lookup)\s+", "", q).strip()
            return f"THOUGHT: I should search the web for this.\nACTION: web_search\nINPUT: {q}"

        # 2) Then write to file if the goal requests saving (post-hook will overwrite with real final)
        wants_save = re.search(r"(save|write to|store in|output\.txt)", goal, re.I)
        not_yet_wrote = "file_write" not in transcript
        if wants_save and not_yet_wrote:
            return "THOUGHT: Persist the result to a file.\nACTION: file_write\nINPUT: output.txt :: <placeholder>"

        # 3) Finalize (the post-hook in run_agent.py will replace template with best snippet if needed)
        return "THOUGHT: I can now answer succinctly.\nACTION: final\nINPUT: Provide the answer extracted from the last observation."
