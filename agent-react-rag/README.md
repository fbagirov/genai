This repository has an agentic RAG AI system, that, given a question on the documentation, evaluates if the documents are sufficient enough to answer the question, and if they are not, re-writes the query and tries again.  

Tech Stack: 
Python 3.12

Usage Instructions: 
1. Go to Huggingface > Profile > Settings > Access Token > Create new token > Wait until it is approved. 
2. Once approved, in powershell, set the environmental variable to your token: 
        $env:HUGGINGFACEHUB_API_TOKEN = "Your Token"
    or use Huggingface CLI login: 
        huggingface-cli login
    If you need to change your token in your environment, use: 
        setx HUGGINGFACEHUB_API_TOKEN "YOUR_TOKEN"

3. Load your pdfs into the FAISS: 
    if the files are encrypted, install PyCryptodome
    Run: 
        python src/load_data.py --pdf-dir data/pdfs --index-path data/faiss.index