from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import PyPDF2
from typing import Optional

# Import existing modules
from .chunking import LegalChunker
from .embeddings import OllamaEmbeddings
from .vector_store import FAISSVectorStore
from .retrieval import RetrievalPipeline
from .generation import LegalAnswerGenerator

app = FastAPI(title="Legal Aid Assistant API")

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

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load existing vector store on startup if it exists."""
    global vector_store
    global embeddings_client, chunker

    # Initialize helpers
    chunker = LegalChunker()
    embeddings_client = OllamaEmbeddings()

    # Initialize or load FAISS vector store
    try:
        vs = FAISSVectorStore(str(VECTOR_STORE_PATH))
        loaded = vs.load()
        if loaded:
            vector_store = vs
            print("✓ Existing vector store loaded successfully")
        else:
            vector_store = vs
            print("ℹ No existing vector store found. Ready to create a new one on first upload.")
    except Exception as e:
        print(f"✗ Error initializing vector store: {e}")
        vector_store = None

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Legal Aid Assistant API",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file for legal case retrieval.
    
    Steps:
    1. Save uploaded PDF
    2. Extract text from PDF
    3. Chunk text using legal-aware chunking
    4. Generate embeddings for chunks
    5. Add to vector store
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        UploadResponse with success status and details
    """
    global vector_store, chunker, embeddings_client
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ PDF saved: {file.filename}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise Exception("No text could be extracted from PDF")
        
        print(f"✓ Text extracted: {len(text)} characters")
        
        # Chunk the text using legal-aware chunking
        if chunker is None:
            chunker = LegalChunker()

        chunk_objs = chunker.chunk_document(text, case_name=file.filename)
        if not chunk_objs:
            raise Exception("No chunks generated from text")

        print(f"✓ Text chunked: {len(chunk_objs)} chunks created")

        # Prepare texts for embedding
        texts = [c.content for c in chunk_objs]

        # Generate embeddings for chunks
        if embeddings_client is None:
            embeddings_client = OllamaEmbeddings()

        embeddings = embeddings_client.embed_batch(texts)
        print(f"✓ Embeddings generated: {len(embeddings)} vectors")

        # Create or update vector store
        if vector_store is None or vector_store.index is None:
            vs = FAISSVectorStore(str(VECTOR_STORE_PATH))
            dim = embeddings_client.get_embedding_dimension()
            vs.initialize_index(dim)

            # Build metadata list (include content + chunk metadata)
            metadata = [ {**c.metadata, 'content': c.content, 'chunk_id': c.chunk_id} for c in chunk_objs ]
            vs.add_chunks(embeddings, metadata)
            vs.save()
            vector_store = vs
            print(f"✓ New vector store created")
        else:
            # Add to existing vector store
            metadata = [ {**c.metadata, 'content': c.content, 'chunk_id': c.chunk_id} for c in chunk_objs ]
            vector_store.add_chunks(embeddings, metadata)
            vector_store.save()
            print(f"✓ Vector store updated")
        
        return UploadResponse(
            success=True,
            message="PDF processed successfully",
            filename=file.filename,
            chunks_processed=len(chunk_objs)
        )
        
    except Exception as e:
        print(f"✗ Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_legal_assistant(request: QueryRequest):
    """
    Query the legal assistant with a question.
    
    Steps:
    1. Embed the user's question
    2. Retrieve relevant chunks from vector store
    3. Generate grounded answer using LLM
    
    Args:
        request: QueryRequest containing the user's question
        
    Returns:
        QueryResponse with generated answer and source chunks
    """
    global vector_store, embeddings_client
    
    if vector_store is None or vector_store.index is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Please upload a PDF first."
        )
    
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        print(f"📝 Query received: {question}")
        
        # Build retrieval pipeline and retrieve context
        if embeddings_client is None:
            embeddings_client = OllamaEmbeddings()

        pipeline = RetrievalPipeline(vector_store, embeddings_client)
        retrieval_results = pipeline.retrieve_context(question, top_k=5)

        if not retrieval_results.get('success') or not retrieval_results.get('results'):
            return QueryResponse(
                answer="I couldn't find relevant information in the uploaded documents to answer your question.",
                sources=[]
            )

        print(f"✓ Retrieved {len(retrieval_results.get('results', []))} relevant chunks")

        # Format context and generate answer
        context_str = pipeline.format_context_for_generation(retrieval_results)

        generator = LegalAnswerGenerator()
        gen = generator.generate_answer(question, context_str)
        if not gen.get('success'):
            raise Exception(gen.get('error', 'Generation failed'))

        answer_text = gen.get('answer', '')

        # Extract source information
        sources = []
        for r in retrieval_results.get('results', [])[:3]:
            source_info = f"{r.get('case_name', 'Unknown')} - {r.get('section_type', 'general')}"
            sources.append(source_info)

        return QueryResponse(
            answer=answer_text,
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status including document count."""
    global vector_store
    
    doc_count = 0
    if vector_store is not None and vector_store.index is not None:
        try:
            doc_count = vector_store.index.ntotal
        except:
            doc_count = "unknown"
    
    return {
        "vector_store_initialized": vector_store is not None and vector_store.index is not None,
        "documents_indexed": doc_count,
        "upload_directory": str(UPLOAD_DIR),
        "vector_store_path": str(VECTOR_STORE_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)