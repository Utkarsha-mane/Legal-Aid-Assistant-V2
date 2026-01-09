# Legal Aid Assistant - Quick Start & Troubleshooting

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed and running

### 1. Install Ollama & Models

```bash
# Install Ollama from https://ollama.ai

# Start Ollama (run in separate terminal)
ollama serve

# In another terminal, download required models
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### 2. Install Dependencies

```bash
cd "c:\legal aid assistsnt model"

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies  
pip install -r frontend/requirements.txt
```

### 3. Start Backend (Terminal 1)

```bash
cd "c:\legal aid assistsnt model"
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✓ Existing vector store loaded successfully
```

### 4. Start Frontend (Terminal 2)

```bash
cd "c:\legal aid assistsnt model"
streamlit run frontend/ui.py
```

Browser will automatically open to `http://localhost:8501`

### 5. Run Diagnostics

To check everything is working:

```bash
cd "c:\legal aid assistsnt model"
python scripts/diagnose.py
```

## 🐛 Troubleshooting

### ❌ Upload fails with 500 error

**Check Ollama is running:**
```bash
ollama list
```

You should see:
```
nomic-embed-text
llama3:8b
```

If missing:
```bash
ollama pull nomic-embed-text
ollama pull llama3:8b
```

**Check Ollama is accessible:**
```bash
curl http://localhost:11434/api/tags
```

### ❌ "Cannot connect to Ollama"

**Make sure Ollama is running:**
```bash
ollama serve
```

### ❌ "Backend Offline"

**Ensure backend is running:**
```bash
cd backend
python app.py
```

**Or with auto-reload:**
```bash
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### ❌ "Port 8000 already in use"

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with the one from above)
taskkill /PID <PID> /F

# Or use different port
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8001
# Then update BACKEND_URL in frontend/ui.py
```

### ❌ Import errors or module not found

**Ensure virtual environment is activated:**
```bash
# On Windows
venv\Scripts\activate

# Then reinstall dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 📊 Check System Status

Use the **System Check** panel in the sidebar to monitor:
- ✅/❌ Backend connection
- ✅/❌ Ollama connectivity
- ✅/❌ Model availability
- 📊 Chunks indexed

Click **🔄 Refresh Status** to update diagnostics.

## 📁 Project Structure

```
legal aid assistsnt model/
├── backend/
│   ├── app.py              # FastAPI main application
│   ├── chunking.py         # Document chunking
│   ├── embeddings.py       # Ollama embeddings interface
│   ├── generation.py       # LLM answer generation
│   ├── retrieval.py        # Semantic search
│   ├── vector_store.py     # FAISS vector database
│   └── requirements.txt
├── frontend/
│   ├── ui.py              # Streamlit interface
│   └── requirements.txt
├── data/
│   ├── uploads/           # Uploaded PDFs
│   └── vector_store/      # FAISS index + metadata
├── scripts/
│   ├── diagnose.py        # Diagnostic tool
│   ├── check_backend.py   # Backend checker
│   └── test_upload.py     # Upload tester
└── venv/                  # Python virtual environment
```

## 🔧 Backend Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload and process PDF |
| POST | `/query` | Ask question about documents |
| POST | `/generate-summary` | Generate case summary |
| POST | `/generate-timeline` | Generate case timeline |
| DELETE | `/delete-all` | Clear vector store |
| GET | `/status` | System status & diagnostics |

## 📝 API Examples

### Upload PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

### Query
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the verdict?"}' \
  http://localhost:8000/query
```

### Generate Summary
```bash
curl -X POST http://localhost:8000/generate-summary
```

### Get Status
```bash
curl http://localhost:8000/status
```

## 🎯 Tips

1. **Large Documents:** May take 1-2 minutes to process. This is normal.
2. **Accuracy:** Upload high-quality PDFs. Scanned documents work best.
3. **Models:** First run downloads models (~5-10 minutes depending on internet).
4. **Performance:** Local processing means zero data leaves your machine!
5. **Multiple Tabs:** Upload multiple documents. They'll be indexed together.

## 🆘 Still Having Issues?

1. Run diagnostics: `python scripts/diagnose.py`
2. Check all three processes are running:
   - Ollama: `ollama serve`
   - Backend: `python -m uvicorn backend.app:app --reload`
   - Frontend: `streamlit run frontend/ui.py`
3. Verify port availability (8000, 8501, 11434)
4. Check logs for error messages
5. Try a simple test document first

## 📞 Support

For issues, try:
1. Restart all services
2. Re-run diagnostics
3. Check `backend/app.py` console output for error messages
4. Verify Ollama is responsive: `curl http://localhost:11434/api/tags`
