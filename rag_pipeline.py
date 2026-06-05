"""
rag_pipeline.py — DocuMind
---------------------------
Responsible for:
  - Embedding document chunks → FAISS vector store
  - Building a ConversationalRetrievalChain (RAG + memory)
  - Querying the chain and returning answers + source metadata

All LLM / embedding / retrieval logic lives here.
No UI code, no PDF parsing — single clear responsibility.
"""

import os
from typing import List, Dict, Any, Tuple

from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


# ── Constants ─────────────────────────────────────────────────────────────────

# Ollama Settings (completely free, runs locally)
OLLAMA_BASE_URL = "http://localhost:11434"     # Ollama server URL
LLM_MODEL = "mistral"                          # Fast, lightweight Ollama model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # HuggingFace embedding model (free)
TOP_K_RETRIEVAL = 4                             # Number of chunks to retrieve per query
LLM_TEMPERATURE = 0.3                           # Lower = more factual, less creative


# ── Ollama Setup Check ────────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """
    Check if Ollama server is running at OLLAMA_BASE_URL.
    Ollama should be started before using the RAG pipeline.

    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


# ── Vector Store Builder ──────────────────────────────────────────────────────

def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed document chunks and store them in a FAISS vector index.

    Each chunk's text is converted to a high-dimensional embedding vector
    using HuggingFace's free embedding model. FAISS indexes these vectors for
    fast nearest-neighbour (semantic) search at query time.

    Args:
        chunks: List of text chunks (LangChain Document objects with metadata)

    Returns:
        FAISS: An in-memory FAISS vector store ready for retrieval

    Raises:
        ValueError: If chunks are empty
        Exception: If embedding API call fails
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU; change to "cuda" if you have GPU
    )

    if not chunks:
        raise ValueError("No text chunks to embed. The PDF may be empty or image-only.")

    # FAISS.from_documents embeds all chunks in batch and builds the index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


# ── Conversational RAG Chain ──────────────────────────────────────────────────

def build_rag_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with memory from a FAISS vector store.

    Architecture:
      User question
        → ConversationBufferMemory (append chat history for context)
        → Question condenser (rewrites follow-up Qs as standalone questions)
        → FAISS retriever (top-K semantic search)
        → Ollama LLM (generates answer from retrieved context)
        → Answer + source documents returned

    Args:
        vector_store: Populated FAISS vector store

    Returns:
        ConversationalRetrievalChain: Ready-to-query RAG chain
    """
    # ── LLM ──
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    # ── Retriever ──
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL},
    )

    # ── Memory ──
    # return_messages=True keeps messages as HumanMessage/AIMessage objects
    # (required by ConversationalRetrievalChain)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",          # Chain returns both 'answer' and 'source_documents'
    )

    # ── Chain ──
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Expose which chunks were used ← key for RAG proof
        verbose=False,
    )

    return chain


# ── Query Interface ───────────────────────────────────────────────────────────

def query_rag_chain(
    chain: ConversationalRetrievalChain,
    question: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Send a question to the RAG chain and return the answer + source info.

    Args:
        chain:    Built ConversationalRetrievalChain
        question: User's natural-language question (str)

    Returns:
        Tuple of:
          - answer (str): LLM-generated answer
          - sources (List[dict]): Each dict has 'source' (filename) and 'page' (int)

    Raises:
        Exception: Propagates API or retrieval errors to be caught by the UI
    """
    response = chain.invoke({"question": question})

    answer: str = response.get("answer", "I could not generate an answer.")

    # Extract unique source references from retrieved chunks
    source_documents: List[Document] = response.get("source_documents", [])
    sources = _deduplicate_sources(source_documents)

    return answer, sources


# ── Private Helpers ───────────────────────────────────────────────────────────

def _deduplicate_sources(source_docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert source Document objects to clean dicts, removing duplicates.

    Two chunks are considered the same source if they share the same
    filename AND page number.

    Args:
        source_docs: List of Document objects with metadata

    Returns:
        List of unique {'source': filename, 'page': page_num} dicts
    """
    seen = set()
    sources = []

    for doc in source_docs:
        filename = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "?")
        key = (filename, page)

        if key not in seen:
            seen.add(key)
            sources.append({"source": filename, "page": page})

    return sources