from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import shutil
from pathlib import Path
import PyPDF2
from typing import Optional
import requests

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import existing modules
from chunking import LegalChunker
from embeddings import OllamaEmbeddings
from vector_store import FAISSVectorStore
from retrieval import RetrievalPipeline
from generation import LegalAnswerGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Aid Assistant API - OpenAI GPT-4",
    description="Legal document analysis powered by GPT-4",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
class GlobalState:
    def __init__(self):
        self.answer_generator = None
        self.current_document = None
        self.all_chunks = []
        self.vector_store = None
        self.embeddings = None
        
        # Initialize OpenAI generator
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
            else:
                # You can change the model here: gpt-4, gpt-4-turbo-preview, etc.
                self.answer_generator = OpenAILegalAnswerGenerator(
                    api_key=api_key,
                    model="gpt-4-turbo-preview"  # or "gpt-4-0125-preview"
                )
                logger.info("OpenAI generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI generator: {e}")

state = GlobalState()

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 1000

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    grounded: bool = True
    tokens_used: Optional[Dict[str, int]] = None

class StatusResponse(BaseModel):
    openai_initialized: bool
    documents_indexed: int
    current_document: Optional[str] = None
    model: Optional[str] = None

class SummaryResponse(BaseModel):
    summary: str
    success: bool
    document_name: Optional[str] = None
    tokens_used: Optional[Dict[str, int]] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Legal Aid Assistant API (OpenAI GPT-4) is running",
        "version": "2.0.0"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        openai_init = state.answer_generator is not None
        
        if openai_init:
            # Test connection
            try:
                state.answer_generator.check_api_connection()
            except:
                openai_init = False
        
        return {
            "openai_initialized": openai_init,
            "documents_indexed": len(state.all_chunks),
            "current_document": state.current_document,
            "model": state.answer_generator.model if state.answer_generator else None
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "openai_initialized": False,
            "documents_indexed": 0,
            "current_document": None,
            "model": None
        }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create uploads directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Extract text from PDF
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(str(file_path))
            full_text = ""
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            if not full_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF. File may be scanned/image-based."
                )
            
            # Chunk the text (improved chunking strategy)
            chunk_size = 2000  # Larger chunks for GPT-4
            overlap = 200
            chunks = []
            
            # Split by paragraphs first for better semantic chunking
            paragraphs = full_text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            logger.info(f"Created {len(chunks)} chunks from {len(reader.pages)} pages")
            
            # Store chunks globally
            state.all_chunks = chunks
            state.current_document = file.filename
            
            # TODO: Add to vector store for better retrieval
            # For now, using simple storage
            
            return {
                "success": True,
                "message": f"Document '{file.filename}' processed successfully",
                "chunks_processed": len(chunks),
                "pages_processed": len(reader.pages),
                "filename": file.filename
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the uploaded documents using GPT-4
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Processing query: {request.question}")
        
        # Check if OpenAI is initialized
        if not state.answer_generator:
            raise HTTPException(
                status_code=500,
                detail="OpenAI generator not initialized. Check API key."
            )
        
        # Check if documents are uploaded
        if not state.all_chunks:
            return QueryResponse(
                answer="No documents have been uploaded yet. Please upload a document first.",
                sources=[],
                grounded=False
            )
        
        # Simple retrieval: Use all chunks or first N chunks
        # TODO: Implement semantic search with embeddings for better retrieval
        max_chunks = 5
        relevant_chunks = state.all_chunks[:max_chunks]
        
        # Combine chunks with separators
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Generate answer using GPT-4
        result = state.answer_generator.generate_answer(
            query=request.question,
            context=context,
            max_tokens=request.max_tokens
        )
        
        if result.get('success'):
            return QueryResponse(
                answer=result.get('answer', 'No answer generated'),
                sources=relevant_chunks,
                grounded=result.get('grounded', True),
                tokens_used=result.get('tokens_used')
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Generation error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/summary", response_model=SummaryResponse)
async def generate_summary():
    """
    Generate a comprehensive summary of the uploaded document
    """
    try:
        logger.info("Generating document summary")
        
        # Check if OpenAI is initialized
        if not state.answer_generator:
            raise HTTPException(
                status_code=500,
                detail="OpenAI generator not initialized. Check API key."
            )
        
        # Check if documents are uploaded
        if not state.all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload a document first."
            )
        
        # Generate summary
        result = state.answer_generator.generate_summary(
            chunks=state.all_chunks,
            max_chunks=10  # Use more chunks for comprehensive summary
        )
        
        if result.get('success'):
            return SummaryResponse(
                success=True,
                summary=result.get('summary', 'No summary generated'),
                document_name=state.current_document,
                tokens_used=result.get('tokens_used')
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Summary generation error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")

@app.post("/test-connection")
async def test_openai_connection():
    """
    Test OpenAI API connection
    """
    try:
        if not state.answer_generator:
            return {
                "success": False,
                "message": "OpenAI generator not initialized"
            }
        
        # Run test
        test_result = state.answer_generator.test_generation()
        
        return {
            "success": test_result.get('test_passed', False),
            "message": "Connection test completed",
            "result": test_result
        }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "success": False,
            "message": f"Test failed: {str(e)}"
        }

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting Legal Aid Assistant API with OpenAI GPT-4")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "⚠️  OPENAI_API_KEY not found in environment variables. "
            "Please set it in .env file or environment."
        )
    else:
        logger.info("✓ OpenAI API key found")
    
    # Create uploads directory
    Path("uploads").mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Legal Aid Assistant API")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)