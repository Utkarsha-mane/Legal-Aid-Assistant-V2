# Legal Aid Assistant - Quick Reference

## 🚀 Commands

### Start Everything (Windows)
```bash
start.bat
```

### Start Manually (Any OS)

Terminal 1 - Ollama:
```bash
ollama serve
```

Terminal 2 - Backend:
```bash
cd backend
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Terminal 3 - Frontend:
```bash
streamlit run frontend/ui.py
```

### Check System Health
```bash
python scripts/diagnose.py
```

## 🔗 URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| Backend | http://localhost:8000 |
| Ollama API | http://localhost:11434 |

## 📦 Install Models

```bash
ollama pull nomic-embed-text
ollama pull llama3:8b
ollama list  # Verify installed
```

## ✅ Before Uploading

Verify in Sidebar "🛠️ System Check":
- ✅ Backend Connected
- ✅ Embedding Model (nomic-embed-text)
- ✅ LLM Model (llama3:8b)

## 📝 Common Commands

### Upload test
```bash
curl -X POST -F "file=@test.pdf" http://localhost:8000/upload
```

### Query test
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What happened?"}' \
  http://localhost:8000/query
```

### Status check
```bash
curl http://localhost:8000/status
```

### Check Ollama models
```bash
curl http://localhost:11434/api/tags
```

## 🆘 Quick Fixes

| Issue | Fix |
|-------|-----|
| "Cannot connect to Ollama" | `ollama serve` |
| "Model not found" | `ollama pull nomic-embed-text` or `ollama pull llama3:8b` |
| "Backend Offline" | `python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000` |
| "Port already in use" | Kill process or use different port |
| Import errors | `pip install -r backend/requirements.txt` |

## 📊 Upload Workflow

1. Upload PDF → 1-2 min processing
2. Chunks created → Stored in FAISS index
3. Ask question → 5-10 sec response
4. Generate summary → 10-15 sec
5. Generate timeline → 10-15 sec

## 🎯 Key Features

✅ Semantic search over documents
✅ AI-powered Q&A grounded in documents
✅ Case summaries
✅ Timeline generation
✅ 100% local processing
✅ No data leaves your machine
✅ Works with any legal PDF

## 📂 Key Files

| File | Purpose |
|------|---------|
| `backend/app.py` | FastAPI server with endpoints |
| `frontend/ui.py` | Streamlit interface |
| `backend/embeddings.py` | Ollama integration |
| `backend/generation.py` | LLM answer generation |
| `backend/retrieval.py` | Semantic search |
| `scripts/diagnose.py` | System diagnostic tool |
| `TROUBLESHOOTING.md` | Full troubleshooting guide |

## 💡 Tips

- Upload multiple related documents - they all get indexed together
- Ask specific legal questions for best results
- First model load takes ~10 seconds
- Use "Case Summary" for quick document overview
- Use "Case Timeline" to understand sequence of events
- "Manage Database" to clear old documents and start fresh

## 🔒 Privacy

- All processing happens locally
- No data sent to external servers
- Models run on your machine
- No internet required after startup
- PDFs stored in `data/uploads/` locally

## 📞 Troubleshooting Flowchart

```
Upload fails?
├─ Check sidebar System Check
│  ├─ ❌ Backend → Start backend
│  └─ ❌ Ollama → Run: ollama serve
└─ Try diagnostic: python scripts/diagnose.py
```

## ⚡ Performance Notes

- Modern system: ✅ 1-2 min per document
- Slower system: ⚠️  2-5 min per document
- Query response: 5-20 sec (depends on document size)
- First query after startup: +10 sec (model loading)

## 🔧 System Requirements

- RAM: 8GB+ recommended (16GB+ for faster processing)
- Disk: 10GB+ free (for models and data)
- CPU: Any modern CPU (more cores = faster processing)
- Python 3.10+
- Ollama installed

## 📋 Startup Checklist

- [ ] Ollama installed from https://ollama.ai
- [ ] Python 3.10+ installed
- [ ] Project dependencies installed: `pip install -r backend/requirements.txt && pip install -r frontend/requirements.txt`
- [ ] Run `python scripts/diagnose.py` - all checks ✅
- [ ] Start with `start.bat` or manual commands
- [ ] Check sidebar System Check - all ✅
- [ ] Upload first test PDF
- [ ] Try asking a question
- [ ] Generate summary and timeline

---

**Last Updated:** January 2026
**Version:** 1.0.0
**Status:** Production Ready ✅
