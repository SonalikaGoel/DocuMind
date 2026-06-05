# 🧠 DocuMind — AI-Powered Document Q&A Chatbot

> Ask questions about any PDF in plain English. DocuMind retrieves the most relevant passages using semantic search and generates accurate, context-aware answers using **Ollama** (free, local LLM) — with full source citations. **No API costs. No quotas. Unlimited usage.**

---

## 🖼️ Demo

Upload one or more PDFs → ask questions → get answers with page-level citations.

![DocuMind Screenshot](./assets/screenshot.png)

---

## ✨ Features

- **PDF Upload & Processing** — Upload multiple PDFs; pages are parsed, chunked, and embedded into a FAISS vector store
- **Semantic Search (RAG)** — Retrieves the top-K most relevant chunks using cosine similarity before answering
- **Conversational Memory** — Follow-up questions retain context from the entire chat session via `ConversationBufferMemory`
- **Source Citations** — Every answer shows the exact source filename and page number
- **Clean Chat UI** — ChatGPT-style interface built in Streamlit with user/bot message bubbles
- **Error Handling** — Graceful handling of missing API keys, empty uploads, image-only PDFs, and failed retrievals

---

## 🏗️ Architecture

```
User (Streamlit UI)
        │
        ▼
  app.py  ──────────────────────────────────────────────────
        │                                                   │
        ▼                                                   ▼
  utils.py                                          rag_pipeline.py
  ─────────────────                                 ──────────────────────────
  • pdfplumber PDF parse                            • HuggingFace Embeddings (free)
  • RecursiveCharacterTextSplitter                  • FAISS vector store
  • Returns Document chunks                         • ConversationalRetrievalChain
  • with source + page metadata                     • Ollama (local LLM) - Mistral
```

### RAG Pipeline Flow

```
PDF Upload → Parse pages → Split into chunks → Embed chunks → FAISS index
                                                                    │
User Question ──────────────────────────────────────────────────────┘
        │
        ▼
Condense question (with chat history) → Semantic search (top-4 chunks)
        │
        ▼
Ollama (local LLM) generates answer from retrieved context
        │
        ▼
Answer + Source (filename, page number) → Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (Mistral, local, free) |
| Embeddings | HuggingFace BGE (free, local) |
| RAG Framework | LangChain `ConversationalRetrievalChain` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Memory | LangChain `ConversationBufferMemory` |
| PDF Parsing | pdfplumber |
| Frontend | Streamlit |
| Config | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Ollama installed from [ollama.ai](https://ollama.ai)
- 8GB+ RAM, 10GB+ disk space

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DocuMind.git
cd DocuMind
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

Ollama must be running before you start DocuMind. In a new terminal:

```bash
ollama serve
```

First time: download the model:
```bash
ollama pull mistral
```

### 5. Run the app

In your venv terminal:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**That's it!** No API keys. No quotas. Completely free.

👉 See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for detailed Ollama setup and troubleshooting.

---

## 📁 Project Structure

```
DocuMind/
├── app.py                 # Streamlit UI — chat interface, sidebar, state management
├── rag_pipeline.py        # Core RAG logic — embeddings, FAISS, LangChain chain
├── utils.py               # PDF parsing and text chunking utilities
├── requirements.txt       # Pinned Python dependencies
├── OLLAMA_SETUP.md        # Ollama installation & troubleshooting guide
├── .gitignore             # Excludes __pycache__, venv, etc.
└── README.md              # This file
```

Each file has a **single clear responsibility** — no cross-cutting concerns.

---

## 💡 How It Works

### 1. Document Ingestion
PDFs are parsed page-by-page using `pdfplumber`. Each page becomes a LangChain `Document` object carrying `source` (filename) and `page` (page number) metadata. Pages are then split into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter` to preserve sentence boundaries.

### 2. Embedding & Indexing
Each chunk is converted to a 384-dimensional vector using HuggingFace's free `BGE-small` embedding model. All vectors are stored in a FAISS flat index, enabling sub-millisecond cosine similarity search across thousands of chunks.

### 3. Conversational Retrieval
When a user asks a question, `ConversationalRetrievalChain` first condenses the question with the chat history into a standalone question (so follow-ups like "what about in chapter 3?" work correctly). This standalone question is then used to retrieve the top-4 most semantically similar chunks.

### 4. Answer Generation
The retrieved chunks are passed as context to Ollama (running locally on your machine) along with the question. The Mistral model generates a grounded answer. Source metadata from the retrieved chunks is extracted and displayed as citation badges below each answer.

---

## 🔮 Future Enhancements

- **LangGraph** — Migrate the RAG pipeline to a stateful LangGraph agent for more complex multi-step reasoning
- **Persistent Vector Store** — Save/load FAISS index to disk so documents don't need re-processing on every restart
- **Larger models** — Use larger Ollama models (Dolphin Mixtral 8x7B) for higher quality answers
- **Streaming responses** — Stream Ollama's output token-by-token for faster perceived response time
- **Hybrid search** — Combine dense (FAISS) + sparse (BM25) retrieval for better recall

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built as a portfolio project demonstrating production-ready RAG implementation with LangChain, FAISS, and Google Gemini.