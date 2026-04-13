from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import shutil
from pathlib import Path
import PyPDF2
from typing import Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load .env early so all modules pick up GEMINI_API_KEY
load_dotenv()

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import existing modules
from chunking import LegalChunker
from embeddings import OllamaEmbeddings          # class name kept for compatibility
from vector_store import FAISSVectorStore
from retrieval import RetrievalPipeline
from generation import LegalAnswerGenerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load existing vector store and initialize clients on startup."""
    global vector_store, embeddings_client, chunker

    try:
        # Validate Gemini API key early
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("[WARN] GEMINI_API_KEY not set. Generation will fail until .env is configured.")
        else:
            print("[OK] GEMINI_API_KEY detected")

        # Initialize helpers
        print("[INFO] Initializing chunker...")
        chunker = LegalChunker()
        print("[OK] Chunker initialized")

        print("[INFO] Initializing embeddings (all-MiniLM-L6-v2)...")
        embeddings_client = OllamaEmbeddings()   # now uses sentence-transformers locally
        print("[OK] Embeddings initialized")

        # Initialize or load FAISS vector store
        print("[INFO] Loading vector store...")
        vs = FAISSVectorStore(str(VECTOR_STORE_PATH))
        loaded = vs.load()
        if loaded:
            # Check if dimensions match current embedding model
            current_dim = embeddings_client.get_embedding_dimension()
            if vs.dimension != current_dim:
                print(f"[WARN] Vector store dimension ({vs.dimension}) doesn't match current model ({current_dim}). Recreating...")
                if VECTOR_STORE_PATH.exists():
                    shutil.rmtree(VECTOR_STORE_PATH)
                VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
                vs = FAISSVectorStore(str(VECTOR_STORE_PATH))
                vs.initialize_index(current_dim)
                vs.save()
                vector_store = vs
                print("[OK] New vector store created with correct dimensions")
            else:
                vector_store = vs
                print("[OK] Existing vector store loaded successfully")
        else:
            vector_store = vs
            print("[INFO] No existing vector store found. Ready to create on first upload.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        vector_store = None
        embeddings_client = None
        chunker = None

    yield


app = FastAPI(title="Legal Aid Assistant API", lifespan=lifespan)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("data/uploads")
VECTOR_STORE_PATH = Path("data/vector_store")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

# Global clients (loaded on startup)
vector_store: FAISSVectorStore | None = None
embeddings_client: OllamaEmbeddings | None = None
chunker: LegalChunker | None = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    chunks_processed: Optional[int] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str
    chunks_deleted: Optional[int] = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as page_e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {str(page_e)}")
                    continue
            return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Legal Aid Assistant API",
        "version": "3.0.0",
        "llm_backend": "Gemini 1.5 Flash",
        "embedding_backend": "sentence-transformers (all-MiniLM-L6-v2)",
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file for legal case retrieval."""
    global vector_store, embeddings_client, chunker

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[OK] PDF saved: {file.filename}")

        # Extract text
        try:
            text = extract_text_from_pdf(file_path)
            if not text.strip():
                raise Exception("No text could be extracted from PDF")
            print(f"[OK] Text extracted: {len(text)} characters")
        except Exception as e:
            print(f"[ERROR] Text extraction failed: {str(e)}")
            raise

        # Chunk
        try:
            if chunker is None:
                chunker = LegalChunker()
            chunk_objs = chunker.chunk_document(text, case_name=file.filename)
            if not chunk_objs:
                from chunking import LegalChunk
                chunk_objs = [LegalChunk(
                    chunk_id="fallback_chunk",
                    chunk_type="document",
                    content=text[:chunker.target_chunk_size],
                    metadata={"case_name": file.filename, "fallback": True}
                )]
                print(f"[WARN] Using fallback chunking: created 1 chunk")
            print(f"[OK] Text chunked: {len(chunk_objs)} chunks created")
        except Exception as e:
            print(f"[ERROR] Chunking failed: {str(e)}")
            from chunking import LegalChunk
            chunk_objs = [LegalChunk(
                chunk_id="fallback_chunk",
                chunk_type="document",
                content=text[:2000] if len(text) > 2000 else text,
                metadata={"case_name": file.filename, "fallback": True}
            )]
            print(f"[WARN] Using fallback chunking due to error: created 1 chunk")

        # Embed
        try:
            if embeddings_client is None:
                embeddings_client = OllamaEmbeddings()
            texts = [c.content for c in chunk_objs]
            embeddings = embeddings_client.embed_batch(texts)
            print(f"[OK] Embeddings generated: {len(embeddings)} vectors")
        except Exception as e:
            print(f"[ERROR] Embedding failed: {str(e)}")
            raise

        # Store
        try:
            if vector_store is None or vector_store.index is None:
                vs = FAISSVectorStore(str(VECTOR_STORE_PATH))
                dim = embeddings_client.get_embedding_dimension()
                vs.initialize_index(dim)
                metadata = [
                    {**c.metadata, "content": c.content, "chunk_id": c.chunk_id}
                    for c in chunk_objs
                ]
                vs.add_chunks(embeddings, metadata)
                vs.save()
                vector_store = vs
                print("[OK] New vector store created")
            else:
                metadata = [
                    {**c.metadata, "content": c.content, "chunk_id": c.chunk_id}
                    for c in chunk_objs
                ]
                vector_store.add_chunks(embeddings, metadata)
                vector_store.save()
                print("[OK] Vector store updated")
        except Exception as e:
            print(f"[ERROR] Vector store operation failed: {str(e)}")
            raise

        return UploadResponse(
            success=True,
            message="PDF processed successfully",
            filename=file.filename,
            chunks_processed=len(chunk_objs),
        )

    except Exception as e:
        print(f"[ERROR] Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_legal_assistant(request: QueryRequest):
    """Query the legal assistant with a question."""
    global vector_store, embeddings_client

    if vector_store is None or vector_store.index is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload a PDF first.",
        )

    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        print(f"[QUERY] Processing: {question}")

        # Initialize embeddings if needed
        try:
            if embeddings_client is None:
                embeddings_client = OllamaEmbeddings()
                print("[OK] Embeddings client initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize embeddings: {str(e)}")
            raise

        # Create retrieval pipeline
        try:
            pipeline = RetrievalPipeline(vector_store, embeddings_client)
            print("[OK] Retrieval pipeline created")
        except Exception as e:
            print(f"[ERROR] Failed to create retrieval pipeline: {str(e)}")
            raise

        # Retrieve context
        try:
            retrieval_results = pipeline.retrieve_context(question, top_k=5)
            print(f"[OK] Context retrieval completed: {len(retrieval_results.get('results', []))} results")
        except Exception as e:
            print(f"[ERROR] Context retrieval failed: {str(e)}")
            raise

        if not retrieval_results.get("success") or not retrieval_results.get("results"):
            print("[INFO] No relevant context found")
            return QueryResponse(
                answer="I couldn't find relevant information in the uploaded documents.",
                sources=[],
            )

        # Format context
        try:
            context_str = pipeline.format_context_for_generation(retrieval_results)
            print(f"[OK] Context formatted: {len(context_str)} chars")
        except Exception as e:
            print(f"[ERROR] Context formatting failed: {str(e)}")
            raise

        # Generate answer via Gemini
        try:
            generator = LegalAnswerGenerator()
            gen = generator.generate_answer(question, context_str)
            if not gen.get("success"):
                error = gen.get("error", "Generation failed")
                print(f"[WARN] Answer generation failed: {error}")
                answer = (
                    f"I found relevant information in the documents, but encountered an issue "
                    f"generating the full answer. Here's the relevant context:\n\n{context_str[:1000]}..."
                )
                sources = [
                    f"{r.get('case_name', 'Unknown')} - {r.get('section_type', 'general')}"
                    for r in retrieval_results.get("results", [])[:3]
                ]
                return QueryResponse(answer=answer, sources=sources)
            print("[OK] Answer generated successfully")
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {str(e)}")
            answer = f"I found relevant information but couldn't generate a proper answer. Error: {str(e)}"
            return QueryResponse(answer=answer, sources=[])

        # Format sources
        try:
            sources = [
                f"{r.get('case_name', 'Unknown')} - {r.get('section_type', 'general')}"
                for r in retrieval_results.get("results", [])[:3]
            ]
        except Exception as e:
            print(f"[ERROR] Source formatting failed: {str(e)}")
            sources = []

        return QueryResponse(answer=gen.get("answer", ""), sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.delete("/delete-all", response_model=DeleteResponse)
async def delete_all_chunks():
    """Delete all chunks from the vector store and reset."""
    global vector_store

    try:
        chunks_deleted = 0
        if vector_store is not None and vector_store.index is not None:
            chunks_deleted = vector_store.index.ntotal

        if vector_store is not None:
            vector_store.clear()
            print("[OK] Vector store cleared")

        if VECTOR_STORE_PATH.exists():
            for file_path in VECTOR_STORE_PATH.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    print(f"[OK] Deleted: {file_path.name}")

        vector_store = FAISSVectorStore(str(VECTOR_STORE_PATH))
        print("[OK] Vector store reinitialized")

        return DeleteResponse(
            success=True,
            message="All chunks deleted successfully. Vector store has been reset.",
            chunks_deleted=chunks_deleted,
        )

    except Exception as e:
        print(f"[ERROR] Error deleting chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {str(e)}")


@app.get("/status")
async def get_status():
    """Get system status including document count and API key presence."""
    global vector_store

    doc_count = 0
    if vector_store is not None and vector_store.index is not None:
        try:
            doc_count = vector_store.index.ntotal
        except Exception:
            doc_count = "unknown"

    gemini_key_set = bool(os.getenv("GEMINI_API_KEY"))

    # Quick Gemini connectivity check
    gemini_status = "unknown"
    if gemini_key_set:
        try:
            gen = LegalAnswerGenerator()
            gemini_status = "connected" if gen.check_model_availability() else "error"
        except Exception as e:
            gemini_status = f"error: {str(e)}"
    else:
        gemini_status = "api_key_missing"

    return {
        "vector_store_initialized": vector_store is not None and vector_store.index is not None,
        "documents_indexed": doc_count,
        "upload_directory": str(UPLOAD_DIR),
        "vector_store_path": str(VECTOR_STORE_PATH),
        "llm_backend": "Gemini 1.5 Flash",
        "embedding_backend": "sentence-transformers (all-MiniLM-L6-v2)",
        "gemini_api_key_set": gemini_key_set,
        "gemini_status": gemini_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)