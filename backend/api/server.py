"""
RAGBrain — FastAPI Server
Supports Ollama (offline) and Groq (online) providers.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.embeddings.embedder import Embedder
from backend.vectordb.vector_store import VectorStore
from backend.llm.generator import Generator
from backend.ingestion.document_loader import load_documents_from_dir, load_single_pdf
from backend.utils.extractor import extract_projects, is_listing_question

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="RAGBrain API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR     = os.getenv("DATA_DIR",     "data/documents")
LLM_MODEL    = os.getenv("LLM_MODEL",    "llama3")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
TOP_K        = int(os.getenv("TOP_K",    "15"))

embedder:     Optional[Embedder]    = None
vector_store: Optional[VectorStore] = None
generator:    Optional[Generator]   = None


@app.on_event("startup")
async def startup():
    global embedder, vector_store, generator
    print("[Server] Starting RAGBrain...")
    print(f"[Server] Default provider={LLM_PROVIDER} model={LLM_MODEL}")
    embedder     = Embedder()
    vector_store = VectorStore(dimension=embedder.get_dimension())
    generator    = Generator(model=LLM_MODEL, provider=LLM_PROVIDER)
    loaded = vector_store.load()
    if not loaded:
        print("[Server] No existing index. Upload a PDF to get started.")
    print("[Server] Ready → http://localhost:8000")


class AskRequest(BaseModel):
    question: str
    top_k:    Optional[int] = None
    model:    Optional[str] = None
    provider: Optional[str] = None   # "ollama" or "groq"

class AskResponse(BaseModel):
    answer:             str
    sources:            list
    model:              str
    num_context_chunks: int

class IngestResponse(BaseModel):
    message:        str
    chunks_indexed: int

class StatusResponse(BaseModel):
    status:         str
    indexed_chunks: int
    model:          str
    provider:       str
    data_dir:       str


@app.get("/status", response_model=StatusResponse)
async def status():
    return {
        "status":         "ok",
        "indexed_chunks": vector_store.count() if vector_store else 0,
        "model":          generator.model    if generator else LLM_MODEL,
        "provider":       generator.provider if generator else LLM_PROVIDER,
        "data_dir":       DATA_DIR,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    if request.model:    generator.model    = request.model
    if request.provider: generator.provider = request.provider

    total_chunks = vector_store.count()
    if total_chunks <= 30:
        context_chunks = vector_store.metadata
    else:
        top_k           = request.top_k or TOP_K
        query_embedding = embedder.embed(request.question)
        context_chunks  = vector_store.search(query_embedding, top_k=top_k)

    if not context_chunks:
        return AskResponse(answer="No relevant context found.", sources=[], model=generator.model, num_context_chunks=0)

    # For listing questions: extract directly with Python, no LLM needed
    if is_listing_question(request.question):
        print('[Server] Listing question detected — using Python extractor')
        projects = extract_projects(context_chunks)
        if projects:
            answer = "Here are the projects mentioned in the document:\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(projects))
            sources = list({c.get("source", "unknown") for c in context_chunks})
            return AskResponse(answer=answer, sources=sources, model=generator.model, num_context_chunks=len(context_chunks))

    try:
        result = generator.generate(request.question, context_chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return AskResponse(**result)


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    os.makedirs(DATA_DIR, exist_ok=True)
    doc_chunks = load_documents_from_dir(DATA_DIR)
    if not doc_chunks:
        raise HTTPException(status_code=400, detail=f"No PDF files found in '{DATA_DIR}'.")
    texts      = [c["text"] for c in doc_chunks]
    embeddings = embedder.embed_batch(texts)
    vector_store.clear()
    vector_store.add(embeddings, doc_chunks)
    vector_store.save()
    return IngestResponse(message=f"Ingested {len(doc_chunks)} chunks", chunks_indexed=len(doc_chunks))


@app.post("/upload", response_model=IngestResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        doc_chunks = load_single_pdf(tmp_path)
        for chunk in doc_chunks:
            chunk["source"] = file.filename
    finally:
        os.unlink(tmp_path)
    if not doc_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    texts      = [c["text"] for c in doc_chunks]
    embeddings = embedder.embed_batch(texts)
    vector_store.add(embeddings, doc_chunks)
    vector_store.save()
    return IngestResponse(message=f"Indexed '{file.filename}' ({len(doc_chunks)} chunks)", chunks_indexed=len(doc_chunks))


@app.delete("/index")
async def clear_index():
    vector_store.clear()
    for path in ["data/faiss.index", "data/metadata.pkl"]:
        if os.path.exists(path):
            os.remove(path)
    return {"message": "Index cleared."}


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")