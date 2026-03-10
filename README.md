# 🧠 RAGBrain

A local **Retrieval-Augmented Generation (RAG)** AI assistant that lets you chat with your PDF documents. Upload a PDF, ask questions, and get accurate AI-generated answers — powered by your own machine or Groq's cloud.



---

## ✨ Features

- 📄 Upload PDFs and index them instantly
- 💬 Ask natural language questions about your documents
- 🖥️ **Offline mode** — runs fully locally using Ollama + llama3
- ⚡ **Online mode** — uses Groq API for faster, smarter answers (llama3.3-70B)
- 🔒 Nothing leaves your machine in offline mode
- 🎨 Clean dark-theme UI with real-time status

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM (Offline) | Ollama — llama3 / llama3.1:8b |
| LLM (Online) | Groq API — llama3.3-70B / llama3.1-8B |
| Frontend | Vanilla HTML/CSS/JS |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed (for offline mode)
- [Groq API key](https://console.groq.com) (for online mode — free)

### Installation

```bash
# Clone the repo
git clone https://github.com/LitFix0/RAGBrain.git
cd RAGBrain

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model (for offline mode)
ollama pull llama3
```

### Configuration

Create a `.env` file in the root folder:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
GROQ_API_KEY=your_groq_key_here
TOP_K=15
DATA_DIR=data/documents
```

### Run

**Windows (recommended):**
```
Double-click start.bat
```

**Manual:**
```bash
# Start Ollama (in a separate terminal)
ollama serve

# Start RAGBrain
python backend/main.py
```

Open **http://localhost:8000** in your browser.

---

## 📖 Usage

1. **Upload a PDF** — drag and drop or click to browse in the sidebar
2. **Choose provider** — Ollama (offline) or Groq (online)
3. **Choose model** — llama3 for offline, llama3.3-70B for best results
4. **Ask anything** — "What are the projects in this resume?" or "Summarize the experience section"

---

## 📁 Project Structure

```
RAGBrain/
├── backend/
│   ├── api/
│   │   └── server.py          # FastAPI routes
│   ├── embeddings/
│   │   └── embedder.py        # SentenceTransformers
│   ├── vectordb/
│   │   └── vector_store.py    # FAISS index
│   ├── llm/
│   │   └── generator.py       # Ollama + Groq inference
│   ├── ingestion/
│   │   └── document_loader.py # PDF chunking
│   ├── utils/
│   │   └── extractor.py       # Smart project extractor
│   └── main.py
├── frontend/
│   └── index.html             # Single-file UI
├── data/
│   └── documents/             # Drop PDFs here for bulk ingest
├── .env                       # Your config (not committed)
├── requirements.txt
├── start.bat                  # One-click Windows launcher
└── README.md
```

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/status` | Server health + stats |
| POST | `/ask` | Ask a question |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ingest` | Ingest all PDFs from data/documents/ |
| DELETE | `/index` | Clear the vector index |

---

## 🗺️ Roadmap

- [ ] Chat history persistence
- [ ] Multi-document comparison
- [ ] Deploy to cloud (Vercel + Railway)
- [ ] Swap FAISS → Pinecone for cloud vector DB
- [ ] Support for .docx and .txt files

---

## 👤 Author

**Shashank Rawat**  
[GitHub](https://github.com/LitFix0)

---

