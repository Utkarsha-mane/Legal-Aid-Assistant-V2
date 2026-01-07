import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path


class FAISSVectorStore:
    """
    FAISS-based vector database for semantic search over legal document chunks.
    Stores embeddings and metadata separately for efficient retrieval.
    """
    
    def __init__(self, storage_path: str = "./storage"):
        """
        Initialize FAISS vector store.
        
        Args:
            storage_path: Directory for persisting index and metadata
        """
        self.storage_path = Path(storage_path)
        self.index_path = self.storage_path / "faiss_index"
        self.metadata_path = self.storage_path / "metadata.json"
        
        # Create storage directories
        self.storage_path.mkdir(exist_ok=True)
        self.index_path.mkdir(exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.dimension = None
        
    def initialize_index(self, dimension: int):
        """
        Create a new FAISS index with specified dimension.
        Uses IndexFlatL2 for exact L2 distance search.
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        # Using Flat index for exact search (good for < 1M vectors)
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        print(f"Initialized new FAISS index with dimension {dimension}")
    
    def add_chunks(self, embeddings: List[List[float]], chunk_metadata: List[Dict[str, Any]]):
        """
        Add document chunks to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            chunk_metadata: List of metadata dictionaries (one per embedding)
        """
        if self.index is None:
            raise Exception("Index not initialized. Call initialize_index() first.")
        
        if len(embeddings) != len(chunk_metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(chunk_metadata)
        
        print(f"Added {len(embeddings)} chunks to vector store (total: {self.index.ntotal})")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search: find most similar chunks to query.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with chunk content and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        # Compile results with metadata
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'score': float(dist),  # L2 distance (lower is better)
                    'similarity': 1 / (1 + float(dist)),  # Convert to similarity score
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save(self):
        """
        Persist FAISS index and metadata to disk.
        """
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata as JSON
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dimension': self.dimension,
                'total_chunks': self.index.ntotal,
                'metadata': self.metadata
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Vector store saved to {self.storage_path}")
    
    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if load successful, False otherwise
        """
        index_file = self.index_path / "index.faiss"
        
        if not index_file.exists() or not self.metadata_path.exists():
            print("No saved index found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.dimension = data['dimension']
                self.metadata = data['metadata']
            
            print(f"Loaded vector store: {self.index.ntotal} chunks, dimension {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """
        Clear all data from the vector store.
        """
        self.index = None
        self.metadata = []
        self.dimension = None
        print("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        if self.index is None:
            return {'status': 'empty'}
        
        return {
            'status': 'active',
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata)
        }