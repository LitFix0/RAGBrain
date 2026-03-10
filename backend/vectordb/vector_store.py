"""
RAGBrain — Vector Store
Manages FAISS index: store embeddings, similarity search, persistence.
"""

import os
import json
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss


INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.pkl"


class VectorStore:
    """
    FAISS-backed vector store with metadata.

    Responsibilities:
    - Store embeddings alongside chunk metadata
    - Perform cosine/L2 similarity search
    - Persist and reload index from disk
    """

    def __init__(self, dimension: int, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata: List[Dict] = []  # parallel to FAISS vectors

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatL2(dimension)
        print(f"[VectorStore] Initialized FAISS index (dim={dimension})")

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings and their associated metadata to the index.

        Args:
            embeddings: 2D float32 array, shape (n, dimension)
            metadata: List of dicts with keys: source, chunk_index, text
        """
        assert embeddings.shape[0] == len(metadata), "Embeddings and metadata count must match"
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        print(f"[VectorStore] Added {len(metadata)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Find the top-k most similar chunks to the query embedding.

        Args:
            query_embedding: 1D float32 array of shape (dimension,)
            top_k: Number of results to return

        Returns:
            List of metadata dicts with an added 'score' key (lower = more similar for L2)
        """
        if self.index.ntotal == 0:
            print("[VectorStore] Index is empty. Please ingest documents first.")
            return []

        # FAISS expects 2D input
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(dist)
            results.append(result)

        return results

    def save(self):
        """Persist FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[VectorStore] Saved index ({self.index.ntotal} vectors) to {self.index_path}")

    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("[VectorStore] No saved index found.")
            return False

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors from {self.index_path}")
        return True

    def clear(self):
        """Reset the index and metadata."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        print("[VectorStore] Index cleared.")

    def count(self) -> int:
        return self.index.ntotal