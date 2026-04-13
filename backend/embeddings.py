from typing import List
from sentence_transformers import SentenceTransformer


class OllamaEmbeddings:  # name kept for compatibility with existing imports
    """
    Local embeddings using sentence-transformers all-MiniLM-L6-v2.
    No Ollama or external service required — runs fully offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        """
        Initialize sentence-transformers embedding model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        print(f"[INFO] Loading sentence-transformer model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print(f"[OK] Embedding model loaded: {model_name}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently (batched).

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show a tqdm progress bar

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=32,
            )
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Error generating batch embeddings: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            384 for all-MiniLM-L6-v2
        """
        return 384