"""
app.py — DocuMind
------------------
The Streamlit frontend. Single responsibility: render the UI and
wire user interactions to utils.py and rag_pipeline.py.

No PDF parsing, no embedding logic lives here.
Run with: streamlit run app.py
"""

import streamlit as st
from utils import (
    load_multiple_pdfs,
    split_documents,
    validate_pdf_uploads,
    validate_question,
)
from rag_pipeline import build_vector_store, build_rag_chain, query_rag_chain


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocuMind — AI Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base & Background ── */
* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: #0a0b0f;
}
[data-testid="stMain"] {
    background: #0a0b0f;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111218 !important;
    border-right: 1px solid #1e2030;
}
[data-testid="stSidebar"] * {
    color: #c9d1e0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 1.5rem 1rem;
}

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(135deg, #7c6ff7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2px;
    letter-spacing: -0.3px;
}
.sidebar-subtitle {
    font-size: 12px;
    color: #555e75 !important;
    margin-bottom: 24px;
    letter-spacing: 0.3px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #16181f !important;
    border: 1.5px dashed #2a2d3e !important;
    border-radius: 12px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c6ff7 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #8891aa !important;
}

/* ── Process button ── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6c5ce7, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108, 92, 231, 0.45) !important;
}

/* ── Status box ── */
.status-box {
    background: #16181f;
    border: 1px solid #1e2030;
    border-left: 3px solid #6c5ce7;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 10px 0;
    font-size: 13px;
    color: #8891aa !important;
    line-height: 1.6;
}

/* ── Divider ── */
hr {
    border-color: #1e2030 !important;
    margin: 16px 0 !important;
}

/* ── Main area heading ── */
h1, h2, h3 {
    color: #e2e8f5 !important;
}
h2 {
    font-size: 26px !important;
    font-weight: 600 !important;
    letter-spacing: -0.4px !important;
}

/* ── Info banner ── */
[data-testid="stAlert"] {
    background: #13151e !important;
    border: 1px solid #1e2030 !important;
    border-left: 3px solid #6c5ce7 !important;
    border-radius: 10px !important;
    color: #8891aa !important;
}
[data-testid="stAlert"] p {
    color: #8891aa !important;
}

/* ── Chat messages ── */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
    animation: slideInRight 0.2s ease;
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(10px); }
    to   { opacity: 1; transform: translateX(0); }
}
.user-bubble {
    background: linear-gradient(135deg, #6c5ce7, #8b5cf6);
    color: #ffffff;
    padding: 13px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 14.5px;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(108, 92, 231, 0.25);
    font-weight: 400;
}

.bot-message {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    animation: slideInLeft 0.2s ease;
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}
.bot-avatar {
    width: 34px;
    height: 34px;
    min-width: 34px;
    background: linear-gradient(135deg, #6c5ce7, #a78bfa);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    margin-top: 2px;
}
.bot-bubble {
    background: #16181f;
    color: #dde3f0;
    padding: 13px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 70%;
    font-size: 14.5px;
    line-height: 1.7;
    border: 1px solid #1e2030;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ── Source badges ── */
.source-container {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-left: 44px;
}
.source-badge {
    background: #13151e;
    border: 1px solid #2a2d3e;
    color: #7c6ff7;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.2px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #111218 !important;
    border-top: 1px solid #1e2030 !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] textarea {
    background: #16181f !important;
    color: #e2e8f5 !important;
    border: 1.5px solid #2a2d3e !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    caret-color: #7c6ff7 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6c5ce7 !important;
    box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.15) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #3d4259 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #7c6ff7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0b0f; }
::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #6c5ce7; }

/* ── Success / Error / Warning messages ── */
[data-testid="stSuccess"] {
    background: #0d1f17 !important;
    border: 1px solid #1a3a28 !important;
    border-left: 3px solid #22c55e !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
}
[data-testid="stError"] {
    background: #1f0d0d !important;
    border: 1px solid #3a1a1a !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 10px !important;
    color: #f87171 !important;
}
[data-testid="stWarning"] {
    background: #1f180d !important;
    border: 1px solid #3a2e1a !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 10px !important;
    color: #fbbf24 !important;
}

/* ── Section labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #8891aa !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────

def init_session_state():
    """Initialise all session state keys on first load."""
    defaults = {
        "chat_history": [],          # List of {"role": "user"|"assistant", "content": str, "sources": list}
        "vector_store": None,        # FAISS vector store (built after PDF processing)
        "rag_chain": None,           # ConversationalRetrievalChain
        "pdfs_processed": False,     # Flag: have PDFs been embedded?
        "processed_filenames": [],   # Names of currently loaded PDFs
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 DocuMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">AI-Powered Document Q&A</div>', unsafe_allow_html=True)

    st.divider()

    # ── PDF Upload ──
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        label="Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload the PDF documents you want to ask questions about.",
    )

    # ── Process Button ──
    process_clicked = st.button(
        "⚡ Process Documents",
        use_container_width=True,
        type="primary",
        disabled=(not uploaded_files),
    )

    if process_clicked:
        if not validate_pdf_uploads(uploaded_files):
            pass  # validate_pdf_uploads shows the warning itself
        else:
            with st.spinner("📖 Reading PDFs..."):
                raw_docs = load_multiple_pdfs(uploaded_files)

            if not raw_docs:
                st.error("❌ No text could be extracted. PDFs may be image-only or corrupt.")
            else:
                with st.spinner("✂️ Splitting into chunks..."):
                    chunks = split_documents(raw_docs)

                with st.spinner("🔢 Embedding & building vector store..."):
                    try:
                        vector_store = build_vector_store(chunks)
                        rag_chain = build_rag_chain(vector_store)

                        # Persist to session state
                        st.session_state.vector_store = vector_store
                        st.session_state.rag_chain = rag_chain
                        st.session_state.pdfs_processed = True
                        st.session_state.processed_filenames = [f.name for f in uploaded_files]
                        st.session_state.chat_history = []  # Reset chat for new docs

                        st.success(f"✅ {len(chunks)} chunks embedded from {len(uploaded_files)} file(s)!")

                    except ValueError as e:
                        st.error(f"❌ Configuration error: {e}")
                    except Exception as e:
                        st.error(f"❌ Failed to build vector store: {e}")

    st.divider()

    # ── Status Box ──
    if st.session_state.pdfs_processed:
        st.markdown('<div class="status-box">', unsafe_allow_html=True)
        st.markdown("**📚 Loaded documents:**")
        for name in st.session_state.processed_filenames:
            st.markdown(f"• {name}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-box">⏳ No documents loaded yet.<br>Upload PDFs and click Process.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Clear Chat Button ──
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        # Reset chain memory too (rebuild chain to wipe LangChain's internal memory)
        if st.session_state.vector_store:
            st.session_state.rag_chain = build_rag_chain(st.session_state.vector_store)
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#475569;text-align:center;">'
        'Built with LangChain · FAISS · Gemini · Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.markdown("## 💬 Ask your documents anything")

if not st.session_state.pdfs_processed:
    st.info("👈 Upload your PDF documents in the sidebar and click **Process Documents** to get started.")

# ── Render chat history ──
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">'
                f'<div class="user-bubble">{message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            # Bot message with avatar
            sources_html = ""
            if message.get("sources"):
                badges = "".join(
                    f'<span class="source-badge">📄 {s["source"]} — p.{s["page"]}</span>'
                    for s in message["sources"]
                )
                sources_html = f'<div class="source-container">{badges}</div>'

            st.markdown(
                f'<div class="bot-message">'
                f'<div class="bot-avatar">🧠</div>'
                f'<div>'
                f'<div class="bot-bubble">{message["content"]}</div>'
                f'{sources_html}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Chat Input ────────────────────────────────────────────────────────────────

user_question = st.chat_input(
    placeholder="Ask a question about your documents...",
    disabled=not st.session_state.pdfs_processed,
)

if user_question:
    if not validate_question(user_question):
        st.warning("⚠️ Please type a question before sending.")
    elif not st.session_state.rag_chain:
        st.error("❌ Documents not processed yet. Please upload and process PDFs first.")
    else:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
            "sources": [],
        })

        # Query the RAG chain with a spinner
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                answer, sources = query_rag_chain(
                    chain=st.session_state.rag_chain,
                    question=user_question,
                )
            except Exception as e:
                answer = f"Sorry, I encountered an error: {str(e)}"
                sources = []

        # Add bot response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        # Rerun to render updated chat
        st.rerun()