from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=env_path)

_initialized = False
_llm = None
_retriever = None
_message = None


def _initialize():
    global _initialized, _llm, _retriever, _message

    if _initialized:
        return

    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_classic.text_splitter import TokenTextSplitter
    from langchain_community.vectorstores.chroma import Chroma

    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Meta-Llama-3-8B",
        task="text-generation",
        pipeline_kwargs={"max_length": 512, "max_new_tokens": 100},
    )
    embedding = HuggingFaceEmbeddings()

    pdf_folder = project_root / "data" / "chroma"
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_documents.extend(docs)

    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(project_root / "data" / "chroma" / "chroma_db"),
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    message = """
      Answer the following question using the context provided:

      Context: {context}
      Question: {question}
      Answer:
    """

    _llm = llm
    _retriever = retriever
    _message = message
    _initialized = True


def rag_chain_invoke(input_obj):
    return answer_from_input(input_obj)


def answer_from_input(input_obj, top_k=5):
    """Accept either a plain question string or a dict with a `question` key."""
    _initialize()

    if isinstance(input_obj, dict):
        question = input_obj.get("question")
    else:
        question = input_obj

    if question is None:
        raise ValueError("No question provided in input dict")

    try:
        docs = _retriever.get_relevant_documents(question)
    except Exception:
        docs = _retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs[:top_k]])
    prompt_text = _message.format(context=context, question=question)
    resp = _llm.invoke(prompt_text)
    return resp
