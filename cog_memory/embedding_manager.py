"""Embedding generation for semantic similarity.

Provides embeddings for commitments using OpenAI or sentence-transformers.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Sequence


load_dotenv()


class EmbeddingManager:
    """Generate embeddings for text using OpenAI or sentence-transformers.

    Supports batch embedding for efficiency.
    """

    def __init__(
        self,
        model: str | None = None,
        use_sentence_transformer: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize the embedding manager.

        Args:
            model: Model name (uses OPENAI_EMBEDDING_MODEL env var if None)
            use_sentence_transformer: Use sentence-transformers instead of OpenAI
            device: Device for sentence-transformer models
        """
        self.use_sentence_transformer = use_sentence_transformer

        if use_sentence_transformer:
            from sentence_transformers import SentenceTransformer

            self.model_name = model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            from openai import OpenAI

            self.model_name = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self.client = OpenAI()
            # Set embedding dimension based on model
            if "3-small" in self.model_name:
                self.embedding_dim = 1536
            elif "3-large" in self.model_name:
                self.embedding_dim = 3072
            elif "ada-002" in self.model_name:
                self.embedding_dim = 1536
            else:
                self.embedding_dim = 1536  # Default

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self.use_sentence_transformer:
            return self.model.encode(text, convert_to_numpy=False).tolist()

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self.use_sentence_transformer:
            embeddings = self.model.encode(
                texts, convert_to_numpy=False, show_progress_bar=False
            )
            return [emb.tolist() for emb in embeddings]

        # OpenAI supports batch requests
        response = self.client.embeddings.create(
            model=self.model_name,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def __repr__(self) -> str:
        return f"EmbeddingManager(model={self.model_name}, dim={self.embedding_dim})"
