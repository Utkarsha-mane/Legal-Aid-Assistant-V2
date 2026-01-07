import requests
import json
from typing import List, Dict, Any
import numpy as np


class OllamaEmbeddings:
    """
    Interface to Ollama's embedding API using nomic-embed-text model.
    Handles batching and error recovery.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """
        Initialize Ollama embeddings client.
        
        Args:
            base_url: Ollama API endpoint
            model: Embedding model name
        """
        self.base_url = base_url
        self.model = model
        self.embed_url = f"{base_url}/api/embeddings"
        
    def check_model_availability(self) -> bool:
        """
        Verify that the embedding model is available in Ollama.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(m['name'].startswith(self.model) for m in models)
        except Exception as e:
            print(f"Error checking model availability: {e}")
        return False
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            Exception if embedding fails
        """
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(
                self.embed_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if not embedding:
                    # Treat empty embeddings as errors so callers can handle fallbacks
                    raise Exception("Empty embedding returned from embedding service")
                return embedding
            else:
                raise Exception(f"Embedding failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with progress tracking.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to print progress updates
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                
                if show_progress and (i + 1) % 5 == 0:
                    print(f"Embedded {i + 1}/{total} chunks")
                    
            except Exception as e:
                print(f"Warning: Failed to embed chunk {i}: {e}")
                # Use zero vector as fallback
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    embeddings.append([0.0] * 768)  # Default nomic-embed-text dimension
        
        if show_progress:
            print(f"Embedding complete: {len(embeddings)}/{total} successful")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings from this model.
        
        Returns:
            Embedding vector dimension
        """
        try:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except:
            # Default for nomic-embed-text
            return 768