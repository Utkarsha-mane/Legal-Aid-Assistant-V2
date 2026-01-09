# Upload Error Fix - Summary

## Problem
Upload was failing with **500 Server Error**, preventing users from processing PDFs.

## Root Causes Identified

1. **Missing global declarations** in `/upload` endpoint
   - Variables `embeddings_client` and `chunker` weren't declared as global
   - Caused "local variable referenced before assignment" errors

2. **Ollama connection issues not properly handled**
   - Connection errors to Ollama weren't caught with helpful error messages
   - Missing models weren't reported clearly

3. **Limited diagnostics**
   - No way to check Ollama status from the UI
   - Users couldn't easily verify backend health
   - No system status reporting for models

## Solutions Implemented

### 1. Backend Fixes (`backend/app.py`)

✅ **Fixed global variable declarations**
```python
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, embeddings_client, chunker  # Added missing declarations
```

✅ **Enhanced error handling for Ollama**
```python
try:
    embeddings = embeddings_client.embed_batch(texts)
except Exception as e:
    error_msg = str(e)
    if "Connection" in error_msg or "refused" in error_msg:
        raise Exception(
            "Cannot connect to Ollama. Please ensure Ollama is running on http://localhost:11434. "
            "Start it with: ollama serve"
        )
    elif "nomic-embed-text" in error_msg:
        raise Exception(
            "Embedding model 'nomic-embed-text' not found. "
            "Pull it with: ollama pull nomic-embed-text"
        )
```

✅ **Enhanced `/status` endpoint**
- Now checks Ollama connectivity
- Reports available models (embedding and LLM)
- Returns diagnostic information for frontend

### 2. Frontend Improvements (`frontend/ui.py`)

✅ **Added Ollama status check at startup**
- Shows clear error if Ollama is not running
- Provides exact commands to fix the issue
- Prevents upload attempts when Ollama is unavailable

✅ **Enhanced System Check sidebar**
- Displays Ollama connection status
- Shows which models are available
- Provides quick-fix commands to pull missing models

✅ **Improved error messages**
- Backend offline: Shows all startup commands
- Ollama error: Explains how to start Ollama and pull models
- Actionable troubleshooting steps

### 3. Diagnostic Tools

✅ **Created `scripts/diagnose.py`**
- Comprehensive system health check
- Checks Python dependencies
- Verifies Ollama availability
- Validates data directories
- Reports clear pass/fail status

✅ **Created `TROUBLESHOOTING.md`**
- Complete troubleshooting guide
- Quick start instructions
- Common issues and solutions
- API endpoint documentation

✅ **Created `start.bat`**
- One-click startup script for Windows
- Starts both backend and frontend
- Checks Ollama is running first

## How to Prevent Future Upload Errors

### Startup Checklist

1. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```

2. **Verify models are installed**
   ```bash
   ollama list
   # Should show: nomic-embed-text and llama3:8b
   ```

3. **Start backend**
   ```bash
   python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start frontend**
   ```bash
   streamlit run frontend/ui.py
   ```

5. **Check System Status in UI**
   - Look at sidebar "🛠️ System Check"
   - Should show all ✅ green checks

### Quick Diagnostics

Run before uploading:
```bash
python scripts/diagnose.py
```

Should show:
```
✅ Dependencies: OK
✅ Data Directories: OK
✅ Ollama: OK
✅ Backend: OK
```

## Testing Upload After Fix

1. Ensure Ollama is running with both models
2. Start backend and frontend
3. Open http://localhost:8501
4. Go to "📄 Upload Documents" tab
5. Upload a PDF
6. Should see: "✅ PDF processed successfully"
7. Check "Documents indexed" in sidebar

## Files Modified

| File | Changes |
|------|---------|
| `backend/app.py` | Added global declarations, Ollama error handling, enhanced /status endpoint, added requests import |
| `frontend/ui.py` | Added Ollama startup check, enhanced diagnostics display, improved error messages |
| `scripts/diagnose.py` | **New** - Comprehensive diagnostic tool |
| `TROUBLESHOOTING.md` | **New** - Complete troubleshooting guide |
| `start.bat` | **New** - Windows quick-start script |

## Error Messages Users May See (and how they're now fixed)

| Old Error | New Experience |
|-----------|-----------------|
| "500 Server Error" (generic) | Specific message: "Cannot connect to Ollama" or "Model not found" with fix instructions |
| No status info | Sidebar shows ✅/❌ status with quick fixes |
| No way to diagnose | `python scripts/diagnose.py` provides full system check |
| Hard to start services | `start.bat` starts everything with one click |

## Next Steps for User

1. ✅ Run `python scripts/diagnose.py` to verify setup
2. ✅ Use `start.bat` to start services or follow TROUBLESHOOTING.md
3. ✅ Check sidebar System Check - should show all ✅
4. ✅ Try uploading a test PDF
5. ✅ Test query and summary/timeline features

## Performance Notes

- PDF processing: 1-2 minutes for typical documents (this is normal, includes embedding generation)
- First query: May take 10-20 seconds (Ollama loading model into memory)
- Subsequent queries: 5-10 seconds
- All processing is local - no data sent to external services
