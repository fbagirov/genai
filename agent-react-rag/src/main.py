from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools 


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import CharacterTextSplitter, TokenTextSplitter

from langchain_community.vectorstores.chroma import Chroma


# load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / ".env"
project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=env_path)

# import transformers
# import torch


llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Meta-Llama-3-8B",
    task="text-generation",
    pipeline_kwargs={"max_length": 512, "max_new_tokens": 100},
)

embedding = HuggingFaceEmbeddings()

# load PDF documents from the specified folder into Chroma
pdf_folder = project_root / "data" / "chroma"
pdf_files = sorted(pdf_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {pdf_folder}")

all_documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    # print(f"Loaded {len(docs)} pages from {pdf_file.name}")
    all_documents.extend(docs)

print(f"Total loaded documents: {len(all_documents)}")

# Splits by character boundaries and may respect separators, so chunk boundaries can be cleaner.
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)


# encoding = tiktoken.encoding_for_model("meta-llama/Meta-Llama-3-8B")

# splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200, encoding=encoding)

chunks = splitter.split_documents(all_documents)

# semantic_splitter = SemanticChunker(embedding=embedding, chunk_size=1000, chunk_overlap=200, breakpoint_threshold_type = "gradient", breakpoint_threshold_amount=0.8)
# chunks = semantic_splitter.split_documents(all_documents)

# Create a Chroma vector store and embed the chunks
vector_store = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./data/chroma/chroma_db")


print(chunks[0])
# print([len(chunk) for chunk in chunks])
message = """
      Answer the following question using the context provided: 

      Context: {context}
      Question: {question}
      Answer:
    
"""

# Chat prompt template
    

# Example usage: call `rag_chain_invoke(...)` after `rag_chain` is defined.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", message)
])


retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

# Create a chain to link retriever, prompt_template and LLM
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt_template 
    | llm
    | StrOutputParser()
)
    
# Wrapper to accept either a plain string or an input dict with a `question` key
def rag_chain_invoke(input_obj):
    if isinstance(input_obj, dict):
        question = input_obj.get("question")
    else:
        question = input_obj

    if question is None:
        raise ValueError("No question provided in input")

    return rag_chain.invoke(question)


# Invoke the chain
def answer_from_input(input_obj, top_k=5):
    """Accept either a plain question string or a dict with a `question` key.
    When given a dict, run the retriever on the `question`, build the prompt
    using `message`, and call the LLM directly. This avoids passing a dict
    into the embedding pipeline.
    """
    if isinstance(input_obj, dict):
        question = input_obj.get("question")
    else:
        question = input_obj

    if question is None:
        raise ValueError("No question provided in input dict")

    # Get relevant documents from the retriever (support both sync API names).
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception:
        # Fallback to invoke if the retriever is a Runnable
        docs = retriever.invoke(question)

    # Build a short context from the top_k documents
    context = "\n\n".join([d.page_content for d in docs[:top_k]])

    prompt_text = message.format(context=context, question=question)

    # Call the LLM with the rendered prompt
    resp = llm.invoke(prompt_text)
    return resp


# Try the dict-friendly path first (this will also work with a plain string)
response = answer_from_input({"question": "Summarize the documents in the database"})
print(getattr(response, "content", response))



# FAISS as a store
# vector_store = FAISSRetriever(index_path="data/faiss.index")    

# prompt = "What is the capital of Azerbaijan? "

# response = llm.invoke(prompt)
# print(response)

