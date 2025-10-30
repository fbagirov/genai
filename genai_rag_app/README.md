
# genai | RAG (Retrieval + Generation)
This is a repository of a sample RAG application
The application uses open data text (Alice from Wonderland):
- **ETL** - Loads and chunks the data (each chunk is a row (contains approximately a paragraph)). Retrieval precision improves with chunking, because RAG works better on small text units rather than entire documents.
- **Embed chunks** - The chunks are embedded with Sentence-Tranformers - loads a pre-trained encoder model that maps text to a multidimensional vector space. The embeddings can also be done via Word2vec, but the similarity will be less precise, because Word2vec learns word-level vectors from local co-occurence patterns in the data. So, for the "Why did Alice follow the White Rabbit?" query, W2v model will only compare word overlap patterns ("rabbit" is near/similar "hole", "run", etc.) and does not provide deep semantics to discover relationships ("Alice followed the rabbit because she was curious"). 
- **Embed user query** - The submitted user query (for example "Why did Alice follow the White Rabbit?") is encoded into the same vector space.
- **Cosine similarity search** - The cosine similarity between the user query and the text chunks is computed and the top k most relevant chunks are returned. 

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
