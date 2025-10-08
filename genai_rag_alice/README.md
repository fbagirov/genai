
# genai | RAG
This is a repository of a sample RAG application


The environment requirements are for CPU-only

To create a virtual environment:

cd rag_app
python -m venv .venv
.\.venv\Scripts\Activate

# CPU wheels
pip install --upgrade pip
pip install -r requirements.txt

(If using CUDA 12.1 builds instead of CPU)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements.txt



run: 
python main.py

Ask a sample question "Why did Alice follow the White Rabbit?"



Questions and answers: 

Q: Why chunk size 1200/overlap 200? 
A: Balanced for recall/precision and typical prompt sizes. Tune to your LLM and document structure. 

Q: What can be done to improve the quality? 
A: Quality improvements: 
- better chunking (by chapters or headings)
- add a reranker (e.g., bge-reranker-base)
- user a stronger generator (e.g., Llama-3-Instruct)

Q: How can this be scaled? 
A: For a larger corpora, store embeddings in FAISS (ANN index) or a vector DB. Add a tiny evaluation set (e.g. 15 Q/A pairs) and track accuracy. 


To use a public link, set `share=True` in `launch()`.
