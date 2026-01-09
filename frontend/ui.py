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

def delete_all_chunks():
    """
    Delete all chunks from the vector store.
    
    Returns:
        dict: Response from backend API
    """
    try:
        response = requests.delete(f"{BACKEND_URL}/delete-all", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"Delete failed: {str(e)}"}

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
    .warning-box {
        background-color: #FFF3CD;
        border: 2px solid #FFC107;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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
        color: #000
    }
    .source-box {
        background-color: #FFF9E6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #FFC107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #000
    }
    .delete-button {
        background-color: #DC3545 !important;
        color: white !important;
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
        ollama_status = status.get('ollama_status', 'unknown')
        
        # Show Ollama status warning if not connected
        if ollama_status != "connected":
            st.error(f"""
🚨 **Ollama Connection Error:** {ollama_status}

**Please ensure Ollama is running:**
```bash
ollama serve
```

**Then verify required models are installed:**
```bash
ollama list
```

You should see:
- `nomic-embed-text` (for embeddings)
- `llama3:8b` (for question answering)

If missing, pull them:
```bash
ollama pull nomic-embed-text
ollama pull llama3:8b
```
            """)
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vs_status = "✓ Initialized" if status.get("vector_store_initialized") else "⚠ Not Initialized"
            st.metric("Vector Store", vs_status)
        with col2:
            docs = status.get("documents_indexed", 0)
            st.metric("Chunks Indexed", f"{docs}" if isinstance(docs, int) else docs)
        with col3:
            st.metric("Storage Path", "data/vector_store")
        with col4:
            # Add delete button in metrics row
            st.write("")  # Spacer
            if st.button("🗑️ Delete All Chunks", type="secondary", use_container_width=True, key="delete_metrics"):
                # Show confirmation dialog
                st.session_state.show_delete_confirm = True
else:
    st.markdown(
        '<div class="status-box status-offline">❌ <b>Backend Status:</b> Offline - Please start the backend server first</div>',
        unsafe_allow_html=True
    )
    st.error("""
🚨 **Backend server is not running**

**Start the backend:**
```bash
cd backend
python app.py
```

Or in a separate terminal with auto-reload:
```bash
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

**Also ensure Ollama is running:**
```bash
ollama serve
```

**Then download required models:**
```bash
ollama pull nomic-embed-text
ollama pull llama3:8b
```
    """)
    st.stop()

# Delete confirmation dialog
if 'show_delete_confirm' not in st.session_state:
    st.session_state.show_delete_confirm = False

if st.session_state.show_delete_confirm:
    st.markdown('<div class="warning-box">⚠️ <b>Warning:</b> You are about to delete all chunks from the vector database. This action cannot be undone!</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
            with st.spinner("Deleting all chunks..."):
                result = delete_all_chunks()
                if result.get("success"):
                    st.success(f"✅ {result.get('message')}")
                    st.info(f"🗑️ Deleted {result.get('chunks_deleted', 0)} chunks")
                    # Clear chat history
                    st.session_state.chat_history = []
                    time.sleep(1)
                    st.session_state.show_delete_confirm = False
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
            st.session_state.show_delete_confirm = False
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_delete_confirm = False
            st.rerun()

st.markdown("---")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📄 Upload Documents", "💬 Ask Questions", "🗑️ Manage Database"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload Legal Documents")
    st.write("Upload PDF files containing case histories, legal precedents, judgments, or other legal documents.")
    
    # Show info about existing chunks
    status = get_system_status()
    if status and status.get("documents_indexed", 0) > 0:
        st.info(f"ℹ️ **Current database contains {status.get('documents_indexed')} chunks.** New uploads will be added to existing data. Use 'Manage Database' tab to clear old data.")
    
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

# Tab 3: Manage Database
with tab3:
    st.header("🗑️ Manage Vector Database")
    st.write("Clear all stored document chunks to start fresh with new cases.")
    
    # Get current status
    status = get_system_status()
    
    # Display current database info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Current Database Status")
        if status:
            if status.get("documents_indexed", 0) > 0:
                st.success(f"✅ **Active:** {status.get('documents_indexed')} chunks stored")
                st.info(f"📁 **Storage:** {status.get('vector_store_path')}")
            else:
                st.warning("⚠️ **Empty:** No chunks currently stored")
    
    with col2:
        st.markdown("### ⚠️ Important Notes")
        st.markdown("""
        - Deleting chunks is **permanent** and cannot be undone
        - This will remove all uploaded case data
        - Query history will be cleared
        - You'll need to upload new documents to ask questions
        """)
    
    st.markdown("---")
    
    # Delete section
    st.markdown("### 🗑️ Delete All Chunks")
    
    if status and status.get("documents_indexed", 0) > 0:
        st.warning(f"⚠️ You are about to delete **{status.get('documents_indexed')} chunks** from the database.")
        
        # Confirmation checkbox
        confirm = st.checkbox("I understand this action cannot be undone", key="confirm_delete")
        
        if confirm:
            if st.button("🗑️ DELETE ALL CHUNKS", type="primary", use_container_width=True):
                with st.spinner("Deleting all chunks from vector database..."):
                    result = delete_all_chunks()
                    
                    if result.get("success"):
                        st.success(f"✅ {result.get('message')}")
                        st.info(f"🗑️ Successfully deleted {result.get('chunks_deleted', 0)} chunks")
                        
                        # Clear chat history
                        st.session_state.chat_history = []
                        
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('message')}")
        else:
            st.info("👆 Please check the confirmation box above to enable deletion")
    else:
        st.info("ℹ️ Database is already empty. Upload documents in the 'Upload Documents' tab.")
    
    st.markdown("---")
    
    # Additional tools
    st.markdown("### 🔧 Additional Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 View Detailed Stats", use_container_width=True):
            if status:
                st.json(status)

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
    - 📊 Case Summaries
    - 📅 Case Timelines
    - 🚫 No Data Leaves Your Machine
    """)
    
    st.markdown("---")
    
    st.header("📖 Quick Start")
    st.markdown("""
    1. **Upload** a legal PDF document
    2. **Wait** for processing (1-2 min)
    3. **Ask** questions in natural language
    4. **Generate** case summaries or timelines
    5. **Review** AI-generated insights
    """)
    
    st.markdown("---")
    
    st.header("🛠️ System Check")
    if st.button("🔄 Refresh Status", use_container_width=True, key="sidebar_refresh"):
        st.rerun()
    
    status = get_system_status()
    if status:
        st.success("✅ Backend Connected")
        st.info(f"📊 Chunks Indexed: {status.get('documents_indexed', 0)}")
        
        # Check Ollama status
        ollama_status = status.get('ollama_status', 'unknown')
        embedding_available = status.get('embedding_model_available', False)
        llm_available = status.get('llm_model_available', False)
        
        st.markdown("**Ollama Models:**")
        if ollama_status == "connected":
            if embedding_available:
                st.success("✅ Embedding Model (nomic-embed-text)")
            else:
                st.warning("⚠️ Embedding Model Missing")
                st.code("ollama pull nomic-embed-text", language="bash")
            
            if llm_available:
                st.success("✅ LLM Model (llama3:8b)")
            else:
                st.warning("⚠️ LLM Model Missing")
                st.code("ollama pull llama3:8b", language="bash")
        else:
            st.error(f"❌ Ollama Connection Error")
            st.error(f"Status: {ollama_status}")
            st.markdown("""
**Fix:** Start Ollama with:
```bash
ollama serve
```
            """)
    else:
        st.error("❌ Backend Offline")
        st.markdown("""
**Troubleshooting:**
1. Ensure backend is running:
   ```bash
   cd backend
   python app.py
   ```
2. Check if port 8000 is available
3. Verify all dependencies are installed
        """)


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
