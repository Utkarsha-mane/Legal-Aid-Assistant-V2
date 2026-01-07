import streamlit as st
import requests
from pathlib import Path
import time

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

def check_backend_status():
    """Check if backend is running and accessible."""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_pdf(file):
    """
    Upload a PDF file to the backend for processing.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        dict: Response from backend API
    """
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Upload failed: {str(e)}"}

def query_assistant(question):
    """
    Send a query to the legal assistant.
    
    Args:
        question: User's legal question
        
    Returns:
        dict: Response containing answer and sources
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Query failed: {str(e)}"}

def get_system_status():
    """Get current system status from backend."""
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
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .status-online {
        background-color: #D4EDDA;
        border: 2px solid #28A745;
        color: #155724;
    }
    .status-offline {
        background-color: #F8D7DA;
        border: 2px solid #DC3545;
        color: #721C24;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #DEE2E6;
    }
    .answer-box {
        background-color: #F0F7FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1F4788;
        margin: 1rem 0;
        color: #000 !important;
    }
    .answer-box * {
        color: #000 !important;
    }
    .source-box {
        background-color: #FFF9E6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #FFC107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #000 !important;
    }
    .source-box * {
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ Legal Aid Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload legal documents and ask questions powered by local AI (Ollama + FAISS)</p>', unsafe_allow_html=True)

# Check backend status
backend_online = check_backend_status()

if backend_online:
    st.markdown(
        '<div class="status-box status-online">✅ <b>Backend Status:</b> Online and Ready | All processing happens locally on your machine</div>',
        unsafe_allow_html=True
    )
    
    # Get system status
    status = get_system_status()
    if status:
        col1, col2, col3 = st.columns(3)
        with col1:
            vs_status = "✓ Initialized" if status.get("vector_store_initialized") else "⚠ Not Initialized"
            st.metric("Vector Store", vs_status)
        with col2:
            docs = status.get("documents_indexed", 0)
            st.metric("Chunks Indexed", f"{docs}" if isinstance(docs, int) else docs)
        with col3:
            st.metric("Storage Path", "data/vector_store")
else:
    st.markdown(
        '<div class="status-box status-offline">❌ <b>Backend Status:</b> Offline - Please start the backend server first</div>',
        unsafe_allow_html=True
    )
    st.error("**Backend server is not running.** Please start it with: `python backend/app.py`")
    st.info("Make sure Ollama is also running with models: `llama3:8b` and `nomic-embed-text`")
    st.stop()

st.markdown("---")

# Create tabs for better organization
tab1, tab2 = st.tabs(["📄 Upload Documents", "💬 Ask Questions"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload Legal Documents")
    st.write("Upload PDF files containing case histories, legal precedents, judgments, or other legal documents.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload legal documents in PDF format for RAG-based analysis",
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📎 **Selected:** {uploaded_file.name}")
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing document... This may take 1-2 minutes depending on size."):
                # Show progress steps
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("⏳ Uploading PDF...")
                progress_bar.progress(20)
                
                # Upload and process the PDF
                result = upload_pdf(uploaded_file)
                
                progress_bar.progress(100)
                progress_text.text("✅ Processing complete!")
                
                time.sleep(0.5)
                progress_text.empty()
                progress_bar.empty()
                
                if result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Chunks Processed", result.get('chunks_processed', 'N/A'))
                    with col2:
                        st.metric("📁 Filename", result.get('filename', 'N/A'))
                    
                    st.balloons()
                    st.info("You can now ask questions about this document in the 'Ask Questions' tab!")
                else:
                    st.error(f"❌ {result.get('message')}")
                    st.warning("**Troubleshooting tips:**\n- Ensure PDF is not password-protected\n- Check if Ollama is running\n- Verify models are installed: `ollama list`")

# Tab 2: Query Interface
with tab2:
    st.header("Ask Legal Questions")
    st.write("Ask questions about the uploaded legal documents. The AI will provide answers grounded in the document content.")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    question = st.text_area(
        "Enter your legal question:",
        height=120,
        placeholder="Examples:\n• What was the final verdict in this case?\n• What legal precedents were cited?\n• Summarize the petitioner's arguments\n• What were the key issues in this case?",
        help="Ask specific questions about the content of uploaded documents",
        key="question_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if submit_button:
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("🤔 Searching documents and generating answer... (this may take 30-60 seconds)"):
                # Show processing steps
                status_text = st.empty()
                
                status_text.text("1/3: Embedding your question...")
                time.sleep(0.5)
                
                status_text.text("2/3: Searching vector database...")
                time.sleep(0.5)
                
                status_text.text("3/3: Generating grounded answer with LLaMA3...")
                
                # Query the assistant
                response = query_assistant(question)
                
                status_text.empty()
                
                if "error" in response:
                    st.error(f"❌ {response['error']}")
                    st.warning("**Troubleshooting:**\n- Check if documents are uploaded\n- Verify Ollama is running\n- Ensure `llama3:8b` model is available")
                else:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response.get("answer"),
                        "sources": response.get("sources", [])
                    })
                    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("📋 Query History")
    st.write(f"Total questions asked: **{len(st.session_state.chat_history)}**")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"❓ Q{len(st.session_state.chat_history) - idx}: {chat['question'][:80]}...", expanded=(idx == 0)):
            st.markdown(f"**Question:**")
            st.info(chat['question'])
            
            st.markdown(f"**Answer:**")
            st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            if chat['sources']:
                st.markdown("**📚 Sources Referenced:**")
                for i, source in enumerate(chat['sources'], 1):
                    st.markdown(f'<div class="source-box"><b>Source {i}:</b> {source}</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Legal Aid Assistant** uses Retrieval-Augmented Generation (RAG) to provide accurate answers grounded in your legal documents.
    
    **Technology Stack:**
    - 🤖 **LLM:** Ollama (llama3:8b)
    - 🔤 **Embeddings:** nomic-embed-text
    - 🗂️ **Vector DB:** FAISS
    - 🚀 **Backend:** FastAPI
    - 🎨 **Frontend:** Streamlit
    
    **Features:**
    - 🔒 100% Local Processing
    - 📄 PDF Document Upload
    - 🔍 Semantic Search
    - 💡 Grounded AI Answers
    - 🚫 No Data Leaves Your Machine
    """)
    
    st.markdown("---")
    
    st.header("📖 Quick Start")
    st.markdown("""
    1. **Upload** a legal PDF document
    2. **Wait** for processing (1-2 min)
    3. **Ask** questions in natural language
    4. **Review** AI-generated answers
    """)
    
    st.markdown("---")
    
    st.header("🛠️ System Check")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    status = get_system_status()
    if status:
        st.success("✅ Backend Connected")
        st.info(f"📊 Chunks: {status.get('documents_indexed', 0)}")
    else:
        st.error("❌ Backend Offline")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9rem; padding: 1rem;'>
    <p>🔐 <b>Privacy-First:</b> All processing happens locally. No data is sent to external servers.</p>
    <p>⚖️ Powered by Ollama (llama3:8b + nomic-embed-text) • Built with FastAPI & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

