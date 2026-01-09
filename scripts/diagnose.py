#!/usr/bin/env python
"""
Diagnostic script to check if all components are properly configured.
Run this to troubleshoot issues with the Legal Aid Assistant.
"""

import subprocess
import requests
import sys
from pathlib import Path

def check_ollama():
    """Check if Ollama is running and models are available."""
    print("\n🔍 Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running")
            
            model_names = [m['name'] for m in models]
            print(f"   Available models: {model_names}")
            
            embedding_ok = any('nomic-embed-text' in m for m in model_names)
            llm_ok = any('llama3:8b' in m or 'llama3' in m for m in model_names)
            
            if not embedding_ok:
                print("❌ Missing: nomic-embed-text")
                print("   Fix: ollama pull nomic-embed-text")
            else:
                print("✅ Embedding model (nomic-embed-text) found")
            
            if not llm_ok:
                print("❌ Missing: llama3:8b")
                print("   Fix: ollama pull llama3:8b")
            else:
                print("✅ LLM model (llama3:8b) found")
            
            return embedding_ok and llm_ok
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Start Ollama: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_backend():
    """Check if backend is running."""
    print("\n🔍 Checking Backend...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running on http://localhost:8000")
            
            # Check status endpoint for more details
            try:
                status = requests.get("http://localhost:8000/status", timeout=5).json()
                print(f"   Vector store: {'✅ Initialized' if status.get('vector_store_initialized') else '⚠️ Not initialized'}")
                print(f"   Documents indexed: {status.get('documents_indexed', 0)}")
            except:
                pass
            
            return True
        else:
            print(f"❌ Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend on port 8000")
        print("   Start backend: python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error checking backend: {e}")
        return False

def check_dependencies():
    """Check if Python dependencies are installed."""
    print("\n🔍 Checking Python dependencies...")
    required = {
        'fastapi': 'FastAPI framework',
        'uvicorn': 'ASGI server',
        'streamlit': 'Frontend framework',
        'faiss': 'Vector database',
        'PyPDF2': 'PDF processing',
        'requests': 'HTTP client',
        'numpy': 'Numerical processing',
        'pydantic': 'Data validation',
    }
    
    all_ok = True
    for package, description in required.items():
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_data_directories():
    """Check if data directories exist and are writable."""
    print("\n🔍 Checking data directories...")
    dirs = {
        'data/uploads': 'PDF upload directory',
        'data/vector_store': 'Vector store directory',
    }
    
    all_ok = True
    for dir_path, description in dirs.items():
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                print(f"✅ {dir_path}: {description}")
            else:
                print(f"❌ {dir_path}: Exists but is not a directory")
                all_ok = False
        else:
            print(f"⚠️  {dir_path}: {description} - Will be created on first use")
    
    return all_ok

def main():
    """Run all diagnostics."""
    print("=" * 60)
    print("🏥 Legal Aid Assistant - Diagnostic Check")
    print("=" * 60)
    
    results = {
        'Dependencies': check_dependencies(),
        'Data Directories': check_data_directories(),
        'Ollama': check_ollama(),
        'Backend': check_backend(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    for component, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {component}: {'OK' if status else 'FAILED'}")
    
    all_ok = all(results.values())
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed! You can now:")
        print("   1. Start Streamlit frontend: streamlit run frontend/ui.py")
        print("   2. Open browser to: http://localhost:8501")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("   - Start Ollama: ollama serve")
        print("   - Pull models: ollama pull nomic-embed-text && ollama pull llama3:8b")
        print("   - Start backend: python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000")
        print("   - Install deps: pip install -r backend/requirements.txt && pip install -r frontend/requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
