import os
from pathlib import Path
from dotenv import load_dotenv

# Lambda's LAMBDA_TASK_ROOT is read-only; all writes must go to /tmp.
_IS_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if _IS_LAMBDA:
    _DATA_DIR = Path("/tmp/rag_data")
    _CHROMA_DIR = Path("/tmp/chroma_db")
else:
    _project_root = Path(__file__).resolve().parents[1]
    load_dotenv(_project_root / ".env")
    _DATA_DIR = _project_root / "data" / "pdfs"
    _CHROMA_DIR = _project_root / "data" / "chroma_db"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_initialized = False
_llm = None
_vector_store = None
_retriever = None
_prompt_template = None


def _initialize():
    global _initialized, _llm, _vector_store, _retriever, _prompt_template

    if _initialized:
        return

    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN is not set")

    # Use the HuggingFace Inference API — do NOT load the model locally.
    # Loading Llama-3-8B weights locally would require ~16 GB and crash Lambda.
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=256,
        temperature=0.1,
        huggingfacehub_api_token=token,
    )

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load persisted store if it exists; otherwise start empty.
    vector_store = Chroma(
        persist_directory=str(_CHROMA_DIR),
        embedding_function=embedding,
    )

    # Seed from any PDFs already present in the data directory.
    seed_pdfs = sorted(_DATA_DIR.glob("*.pdf"))
    if seed_pdfs:
        _ingest_pdfs(seed_pdfs, vector_store)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt_template = (
        "Answer the following question using only the context provided.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    _llm = llm
    _vector_store = vector_store
    _retriever = retriever
    _prompt_template = prompt_template
    _initialized = True


def _ingest_pdfs(pdf_paths, vector_store=None):
    """Chunk PDFs and add them to the vector store. Returns number of chunks added."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import TokenTextSplitter

    if vector_store is None:
        vector_store = _vector_store

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        all_docs.extend(loader.load())

    if not all_docs:
        return 0

    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    vector_store.add_documents(chunks)
    return len(chunks)


def add_pdf_to_store(pdf_path: str) -> int:
    """Ingest a single PDF file into the live Chroma vector store.

    The caller is responsible for writing the file to disk before calling this.
    Returns the number of chunks added.
    """
    _initialize()
    return _ingest_pdfs([Path(pdf_path)])


def answer_from_input(input_obj, top_k=5):
    """Accept either a plain question string or a dict with a ``question`` key."""
    _initialize()

    if isinstance(input_obj, dict):
        question = input_obj.get("question")
    else:
        question = input_obj

    if not question:
        raise ValueError("No question provided")

    docs = _retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs[:top_k])
    prompt = _prompt_template.format(context=context, question=question)
    return _llm.invoke(prompt)


def rag_chain_invoke(input_obj):
    return answer_from_input(input_obj)
