
# agent-react-tools

## What is Agentic AI? 

Agentic AI are AI systems that act like an agent, that can decide what to do next use tools and work towards a goal instead of just answering inquries. 

Agentic AI can: 
1. Plan - "what's the next step to reach the goal?"
2. Act - call tools (a calculator, web search, etc.)
3. Observe - Read the result of the previous action. 
4. Decide based on the outcome - pick the next step.
5. Repeat until the job is finished. 

If a regular GenAI application would answer your question about the wether (what's the weather like today?), Agentic AI would be asked to "plan the day based on the weather" and will use multiple tools to: 
- Tool 1 - Check the weather
- Tool 2 - Look into the calendar
- Tool 3 - Create a to-do list 
- Tool 4 - generate the plan for the day


Agentic AI: a **planner → tools → verifier** pipeline that uses a ReAct(Reason + Act)-style loop.
Includes a small suite of tools (web search, file I/O, Python exec sandbox), stop conditions,
and an evaluation harness (10 tasks) to measure **success rate, cost, and latency**.

The app process: 
1. Starts with a goal you give it ("Find the capital of Azerbaijan and save it in the file")
2. Thinks (THOUGHT) - "I need to look up the capital of Azerbaijan". 
3. Choose an ACTION - web_search
4. Provide INPUT - "Azerbaijan capital"
5. Get OBSERVATION - "The capital of Azerbaijan is Baku". 
6. Think again - "Now, I should save it into a file"
7. ACTION - file_write
8. Stop - "goal accomplished"

> Framework-free core loop (pure Python). You can later swap in LangGraph/AutoGen/crewai if desired.

## Features
- **Planner** proposes next step using ReAct (THOUGHT, ACTION, INPUT).
- **Tools**: `web_search` (Tavily API optional), `file_read`, `file_write`, `python_exec` (subprocess, time-limited).
- **Memory**: simple JSONL scratchpad of steps and final answers (per run).
- **Verifier**: checks stop conditions and basic answer sanity (e.g., non-empty, size limits, required citations if used).
- **Config-driven** via `configs/config.yaml`.
- **CLI**: `python run_agent.py --goal "..."`.
- **Eval**: `python eval/run_tasks.py` executes 10 sample tasks and prints success rate.
- **Dockerfile** and **Makefile** for quick use.

## Obtaining the Tavily API key

Tavily is a web search API built for AI agents. Instead of scraping Google search results, Tavily can take a question and get back a JSON response that does not need cleaning, including summaries, titles, URLs and snippets. It is free for developers (free tier gives a 1000 searches per month). 

To obtain the API key > go to https: tavily.com > Create an account > Dashboard > Settings > API Keys. 


TAVILY_API_key is an environment variable that stores your key for Tavily's API. 


## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (Optional) for web_search tool
export TAVILY_API_KEY=<your API key>

# On Windows:
#   setx TAVILY_API_KEY "your_key"   # new shells
#   $env:TAVILY_API_KEY = "your_key" # current shell

# (Alternatively, store in .env file): 
TAVILY_API_KEY=your_real_key_here
SEARCH_PROVIDER=tavily
# If using .env, make sure .env is in .gitignore!


# Run a single goal
python run_agent.py --goal "Find the capital of France and save it to output.txt"

# Evaluate built-in tasks (10 goals)
python eval/run_tasks.py
```

## Architecture
```
[Planner] --(next action)--> [Tool Router] --(call)--> [Tool]
     ^                                                 |
     |----------------------[Observation]--------------|
                                  |
                             [Verifier] -- stop? --> Done
```

## Flow diagram (ReAct loop)
```
THOUGHT -> ACTION -> INPUT -> OBSERVATION -> (repeat) ... -> FINAL ANSWER
Stop when: max_steps, explicit 'final', or verifier deems the goal satisfied.
```

## Config toggles (configs/config.yaml)
- `max_steps`: hard cap on tool calls per run.
- `model.provider`: `openai` (optional) or `local` (rule-based planner for demo).
- `search.provider`: `tavily` (requires API key) or `stub` (returns placeholders).
- `python_exec.timeout_sec`: subprocess timeout.
- `verifier.max_answer_chars`: clamp final answer length.
- `logging.verbose`: print each step.

## Limits
- The `python_exec` tool is a simple subprocess sandbox; do **not** run untrusted code from unknown sources.
- `web_search` requires a Tavily API key to get real results; otherwise returns stub data.
- This is a teaching scaffold; swap in LangGraph/AutoGen when you need concurrency, multi-agent chat, or tracing.
