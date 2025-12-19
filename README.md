This repository contains a curated collection of Generative AI system examples covering modern patterns such as RAG, fine-tuning, agentic AI, multimodal understanding, inference services, safety testing, and analytics copilots.

These projects are for educational reference only. They are implementations designed to demonstrate architecture, engineering tradeoffs, and production-ready patterns, not polished end-user products.

**Apps in this repo**
1. *genai-feat-private_assisant* - a privacy-first, local-only ChatGPT-like AI chat assistant with minimal data exposure, and configurable guardrails. No wifi needed.
2. *genai_rag_alice* - a Retrieval Augumented Generation (RAG) app, that answers questions from "Alice in Wonderland" book
3. *genai_rag_citations* - a RAG app that ingests your PDFs, builds **hybrid search** (BM25 + dense), and answers questions with **inline citations** and a **confidence score**. 
4. *genai_rag_doc_generator* - a RAG app that, given a labeled set of emails resulting in successful outcomes (simulated) generates an email text that will likely to result in a successful outcome. 
5. *llm-finetune-sft-dpo* -  example of fine-tuning of a small instruction model with Supervised Fine-Tuning (SFT) and comparing against preference optimization (DPO). 
6. *multimodal-mini-app* - an app that takes input of images or PDFs, captions images, run OCR, and answers questions about images or PDFs.
7. *llm-inference-service* - an example of a prdocution-ready inference microservice 

**What this repo is:**

- A portfolio of GenAI system patterns
- A learning and experimentation environment
- A reference for recruiters, engineers, and architects
- A demonstration of how to reason about GenAI systems end-to-end
- A realistic example of AI-augmented software development

**Why AI-assisted code is used intentionally instead of coding from scratch:**

- Generative AI systems are complex, multi-component, and rapidly evolving.
- The value of an engineer in this space is not typing speed, but:
- Correct system decomposition
- Safe and testable architectures
- Evaluation design (latency, faithfulness, robustness)
- Security, privacy, and data-handling decisions
- Understanding failure modes
- Making tradeoffs explicit and configurable

**This repo was not built using vibe coding!**

Vibe coding, they usually mean:
- Prompting an LLM with something vague like “Build me a GenAI app”
- Accepting whatever code comes out
- Little or no understanding of:
    - Architecture
    - Failure modes
    - Tradeoffs
    - Evaluation
- Minimal debugging
- Minimal iteration
- Minimal ownership of decisions

In vibe coding:
- The model leads
- The human mostly reacts
- The code “looks fancy” but is fragile
- The author often cannot explain why things are structured the way they are

**This repository was built using AI-assisted systems engineering, not vibe coding!**

**How this repository was built**

This repository was developed using a **human-in-the-loop** workflow:
- Architecture, system design, and evaluation criteria were explicitly defined up front
- Code, scaffolding, and boilerplate were accelerated using AI-assisted generation (ChatGPT Pro, ChatGPT 5.1/5.2)
- Each module was then reviewed, modified, debugged, and validated manually
- Errors, edge cases, and configuration pitfalls were intentionally preserved and documented where useful for learning

**This process replicates how todays engineering teams actually work:**

- AI is used as a force multiplier, not a replacement for engineering judgment.
- AI was used to accelerate implementation.
- Engineering decisions, system design, and integration were human-led.

**Intended audience:**
This repository is designed for:
- Students & practitioners learning modern GenAI patterns
- Senior engineers reviewing architectural thinking
- AI Teams prototyping or validating GenAI approaches
