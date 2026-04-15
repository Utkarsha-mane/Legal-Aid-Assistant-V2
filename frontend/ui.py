import streamlit as st
import requests
from pathlib import Path
import time

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_pdf(file):
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Upload failed: {str(e)}"}

def query_assistant(question):
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Query failed: {str(e)}"}

def delete_all_chunks():
    try:
        response = requests.delete(f"{BACKEND_URL}/delete-all", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Delete failed: {str(e)}"}

def get_system_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None


# Page configuration
st.set_page_config(
    page_title="Legal Aid Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1F4788;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .status-online  { background-color:#D4EDDA; border:2px solid #28A745; color:#155724; }
    .status-offline { background-color:#F8D7DA; border:2px solid #DC3545; color:#721C24; }
    .warning-box    { background-color:#FFF3CD; border:2px solid #FFC107; color:#856404;
                      padding:1rem; border-radius:0.5rem; margin-bottom:1rem; }
    .answer-box {
        background-color:#F0F7FF; padding:1.5rem; border-radius:0.5rem;
        border-left:4px solid #1F4788; margin:1rem 0; color:#000;
    }
    .source-box {
        background-color:#FFF9E6; padding:1rem; border-radius:0.5rem;
        border-left:3px solid #FFC107; margin:0.5rem 0; font-size:0.9rem; color:#000;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚖️ Legal Aid Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload legal documents and ask questions</p>',
    unsafe_allow_html=True,
)

backend_online = check_backend_status()

if backend_online:
    st.markdown(
        '<div class="status-box status-online">✅ <b>Backend Status:</b> Online</div>',
        unsafe_allow_html=True,
    )

    status = get_system_status()
    if status:
        gemini_key_set = status.get("gemini_api_key_set", False)
        gemini_status = status.get("gemini_status", "unknown")

        if not gemini_key_set:
            st.error("""
🚨 **Gemini API Key Missing**

Create a `.env` file in your **backend/** directory with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your **free** API key at: https://aistudio.google.com/app/apikey
            """)
            st.stop()

        if gemini_status not in ("connected", "unknown"):
            st.warning(f"")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vs_status = "✓ Initialized" if status.get("vector_store_initialized") else "⚠ Empty"
            st.metric("Vector Store", vs_status)
        with col2:
            st.metric("Chunks Indexed", status.get("documents_indexed", 0))
        with col3:
            st.metric("LLM Backend", "Gemini")
        with col4:
            st.write("")
            if st.button("🗑️ Delete All Chunks", use_container_width=True, key="delete_metrics"):
                st.session_state.show_delete_confirm = True
else:
    st.markdown(
        '<div class="status-box status-offline">❌ <b>Backend Status:</b> Offline</div>',
        unsafe_allow_html=True,
    )
    st.error("""
🚨 **Backend server is not running**

Start it with:
```bash
cd backend
python app.py
```
    """)
    st.stop()

# Delete confirmation
if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = False

if st.session_state.show_delete_confirm:
    st.markdown(
        '<div class="warning-box">⚠️ You are about to delete all chunks. This cannot be undone!</div>',
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
            with st.spinner("Deleting..."):
                result = delete_all_chunks()
                if result.get("success"):
                    st.success(result.get("message"))
                    st.session_state.chat_history = []
                    time.sleep(1)
                    st.session_state.show_delete_confirm = False
                    st.rerun()
                else:
                    st.error(result.get("message"))
            st.session_state.show_delete_confirm = False
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_delete_confirm = False
            st.rerun()

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "💬 Ask Questions", "🗑️ Manage Database"])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Upload Legal Documents")
    st.write("Upload PDF files containing case histories, judgments, or other legal documents.")

    status = get_system_status()
    if status and status.get("documents_indexed", 0) > 0:
        st.info(f"ℹ️ Database contains **{status.get('documents_indexed')} chunks**. New uploads are additive.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], key="pdf_uploader"
    )

    if uploaded_file is not None:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"📎 **Selected:** {uploaded_file.name}")
        with c2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document... (embedding may take 10–30s on first run)"):
                progress = st.progress(0)
                progress.progress(20)
                result = upload_pdf(uploaded_file)
                progress.progress(100)
                time.sleep(0.3)
                progress.empty()

                if result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Chunks Processed", result.get("chunks_processed", "N/A"))
                    with c2:
                        st.metric("Filename", result.get("filename", "N/A"))
                    st.info("Ask questions in the 'Ask Questions' tab!")
                else:
                    st.error(result.get("message"))

# ── Tab 2: Query ───────────────────────────────────────────────────────────────
with tab2:
    st.header("Ask Legal Questions")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_area(
        "Enter your legal question:",
        height=120,
        placeholder=(
            "Examples:\n"
            "• What was the final verdict?\n"
            "• What legal precedents were cited?\n"
            "• Summarise the petitioner's arguments"
        ),
        key="question_input",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        submit = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with c2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if submit:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Searching documents and generating answer via Gemini..."):
                response = query_assistant(question)

            if "error" in response:
                st.error(f"❌ {response['error']}")
            else:
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": response.get("answer"),
                        "sources": response.get("sources", []),
                    }
                )
                st.rerun()

    if st.session_state.chat_history:
        st.markdown("---")
        st.header("📋 Query History")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            label = f"❓ Q{len(st.session_state.chat_history) - idx}: {chat['question'][:80]}..."
            with st.expander(label, expanded=(idx == 0)):
                st.markdown("**Question:**")
                st.info(chat["question"])
                st.markdown("**Answer:**")
                st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
                if chat["sources"]:
                    st.markdown("**📚 Sources Referenced:**")
                    for i, src in enumerate(chat["sources"], 1):
                        st.markdown(f'<div class="source-box"><b>Source {i}:</b> {src}</div>', unsafe_allow_html=True)

# ── Tab 3: Manage Database ─────────────────────────────────────────────────────
with tab3:
    st.header("🗑️ Manage Vector Database")

    status = get_system_status()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📊 Database Status")
        if status:
            if status.get("documents_indexed", 0) > 0:
                st.success(f"✅ Active: {status.get('documents_indexed')} chunks")
                st.info(f"📁 {status.get('vector_store_path')}")
            else:
                st.warning("⚠️ Empty — no chunks stored")
    with c2:
        st.markdown("### ⚠️ Important Notes")
        st.markdown(
            "- Deletion is **permanent**\n"
            "- All uploaded case data will be removed\n"
            "- Query history will be cleared"
        )

    st.markdown("---")
    st.markdown("### 🗑️ Delete All Chunks")

    if status and status.get("documents_indexed", 0) > 0:
        st.warning(f"⚠️ About to delete **{status.get('documents_indexed')} chunks**.")
        confirm = st.checkbox("I understand this action cannot be undone", key="confirm_delete")
        if confirm:
            if st.button("🗑️ DELETE ALL CHUNKS", type="primary", use_container_width=True):
                with st.spinner("Deleting..."):
                    result = delete_all_chunks()
                    if result.get("success"):
                        st.success(result.get("message"))
                        st.session_state.chat_history = []
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result.get("message"))
        else:
            st.info("👆 Check the box above to enable deletion")
    else:
        st.info("ℹ️ Database is already empty.")

    st.markdown("---")
    st.markdown("### 🔧 Tools")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("📊 View Detailed Stats", use_container_width=True):
            if status:
                st.json(status)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Legal Aid Assistant** uses RAG to give accurate answers grounded in your documents.

**Features:**
- ✨ Google Gemini 1.5 Flash for fast generation
- 🔒 Local embeddings — no data leaves your machine
- 📄 PDF Upload & Semantic Search
- 🚫 No hallucinations — answers from docs only
    """)

    st.markdown("---")
    st.header("📖 Quick Start")
    st.markdown(
        "1. Add `GEMINI_API_KEY` to `backend/.env`\n"
        "2. Upload a legal PDF\n"
        "3. Ask questions\n\n"
        "Get a free key → [aistudio.google.com](https://aistudio.google.com/app/apikey)"
    )

    st.markdown("---")
    st.header("🛠️ System Check")
    if st.button("🔄 Refresh", use_container_width=True, key="sidebar_refresh"):
        st.rerun()

    status = get_system_status()
    if status:
        st.success("✅ Backend Connected")
        st.info(f"📊 Chunks: {status.get('documents_indexed', 0)}")
        gemini_status = status.get("gemini_status", "unknown")
        if gemini_status == "connected":
            st.success("✅ Gemini API Connected")
        elif status.get("gemini_api_key_set"):
            st.warning(f"⚠️ Gemini: {gemini_status}")
        else:
            st.error("❌ GEMINI_API_KEY missing in backend/.env")
    else:
        st.error("❌ Backend Offline")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.9rem;padding:1rem;'>"
    "🔐 <b>Privacy-First:</b> Embeddings run locally. Only your questions reach Gemini's API."
    "</div>",
    unsafe_allow_html=True,
)
