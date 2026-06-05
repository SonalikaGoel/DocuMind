"""
utils.py — DocuMind
--------------------
Responsible for:
  - Loading and parsing PDF files (text + metadata)
  - Splitting text into overlapping chunks for embedding
  - Returning LangChain Document objects with source/page metadata

No LLM or vector-store logic lives here — pure document pre-processing.
"""

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import streamlit as st


# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between consecutive chunks (preserves context)


# ── PDF Parsing ───────────────────────────────────────────────────────────────

def load_pdf(uploaded_file) -> List[Document]:
    """
    Parse a single Streamlit UploadedFile (PDF) using pdfplumber.

    Returns a list of LangChain Document objects — one per page — each
    carrying 'source' (filename) and 'page' (1-based page number) metadata.

    Args:
        uploaded_file: Streamlit UploadedFile object (PDF)

    Returns:
        List[Document]: One Document per page with text + metadata
    """
    documents: List[Document] = []

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()

                # Skip pages with no extractable text (e.g. pure-image pages)
                if not text or not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num,
                        },
                    )
                )

    except Exception as e:
        st.error(f"❌ Failed to parse '{uploaded_file.name}': {e}")
        return []

    return documents


def load_multiple_pdfs(uploaded_files) -> List[Document]:
    """
    Parse multiple uploaded PDF files.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List[Document]: Combined documents from all PDFs
    """
    all_documents: List[Document] = []

    for uploaded_file in uploaded_files:
        docs = load_pdf(uploaded_file)
        all_documents.extend(docs)

    return all_documents


# ── Text Splitting ────────────────────────────────────────────────────────────

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split page-level Documents into smaller overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on paragraph
    boundaries first, then sentences, then words — preserving semantic units.

    Metadata (source filename + page number) is inherited by every chunk,
    so we always know where an answer came from.

    Args:
        documents: List of page-level Document objects

    Returns:
        List[Document]: Smaller chunk-level Document objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split hierarchy: paragraph → newline → sentence → word → char
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ── Validation Helpers ────────────────────────────────────────────────────────

def validate_pdf_uploads(uploaded_files) -> bool:
    """
    Check that the user actually uploaded at least one file.

    Args:
        uploaded_files: Value from st.file_uploader (None or list)

    Returns:
        bool: True if valid, False otherwise (also shows st.warning)
    """
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF before processing.")
        return False
    return True


def validate_question(question: str) -> bool:
    """
    Check that the user typed a non-empty question.

    Args:
        question: Raw string from the chat input box

    Returns:
        bool: True if valid, False otherwise
    """
    if not question or not question.strip():
        return False
    return True