"""
RAGBrain — Embedder
Converts text chunks into vector embeddings using SentenceTransformers.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps SentenceTransformer to produce embeddings for text chunks.

    Pipeline:
        text chunk → embedding model → float32 numpy vector
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            1D numpy array of float32, shape (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns:
            2D numpy array of float32, shape (len(texts), dimension)
        """
        print(f"[Embedder] Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        return self.dimension