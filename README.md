# RAG App: SQLite + SentenceTransformers + Groq + Streamlit

A small Retrieval-Augmented Generation (RAG) app that:
- Ingests PDFs/TXT/Markdown files
- Splits text into chunks (~1000 tokens) using RecursiveCharacterTextSplitter
- Stores embeddings and metadata in a SQLite-backed vector store
- Retrieves most similar chunks for a user query
- Answers via Groq LLM with citations
- Persists docs and chat thread in SQLite

## Features
- SQLite persistence (`rag.db` auto-created)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: Groq via `langchain_groq` (models: llama-3.1-8b-instant / 70b-versatile)
- Streamlit UI with upload, query, retrieved context, and citations

## Setup

1) Python 3.10+

2) Create a virtual environment and install dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Set Groq API Key
```bash
$env:GROQ_API_KEY="your_key_here"   # Windows PowerShell
# or
export GROQ_API_KEY=your_key_here    # macOS/Linux
```

## Run
```bash
streamlit run app.py
```

- Upload PDFs/TXT/MD files in the UI
- Ask questions; answers include citations like `[docname:chunk]`

## Files
- `app.py`: Streamlit UI
- `backend.py`: RAG pipeline (SQLite schema, ingestion, retriever, Groq LLM)
- `requirements.txt`: dependencies
- `rag.db`: auto-created SQLite database

## Notes
- If PDFs fail to parse, ensure `pypdf` is installed (it is in requirements).
- Retrieval uses cosine similarity over stored vectors.

## Resume
- Resume: https://drive.google.com/file/d/18jJ5Vu5udj0xlEw8em66jRxEhmIU-sqw/view?usp=drive_link

## Deploy
- Works locally via `streamlit run app.py`
- For Streamlit Cloud/Render/Railway, set environment variable `GROQ_API_KEY` and deploy the repo.
- LINK: https://mkr2177-mini-rag-app-b1g0hh.streamlit.app/
