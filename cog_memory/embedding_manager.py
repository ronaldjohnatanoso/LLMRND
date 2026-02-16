"""Embedding generation for semantic similarity.

Provides embeddings for commitments using:
- Nomic (FREE, high quality, recommended)
- sentence-transformers (FREE, local)
- OpenAI (paid)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Sequence


load_dotenv()


class EmbeddingManager:
    """Generate embeddings for text.

    Options:
    - Nomic: FREE API, 768 dim, high quality (recommended)
    - sentence-transformers: Local, 384 dim, fastest
    - OpenAI: Paid API, 1536 dim
    """

    def __init__(
        self,
        model: str | None = None,
        use_sentence_transformer: bool = False,
        use_nomic: bool = True,  # Changed default to Nomic
        device: str = "cpu",
    ) -> None:
        """Initialize the embedding manager.

        Args:
            model: Model name
            use_sentence_transformer: Use sentence-transformers (local, 384 dim)
            use_nomic: Use Nomic embeddings (FREE API, 768 dim, recommended)
            device: Device for local models
        """
        self.use_nomic = use_nomic
        self.use_sentence_transformer = use_sentence_transformer

        if use_nomic:
            api_key = os.getenv("NOMIC_API_KEY")
            if not api_key:
                raise ValueError("NOMIC_API_KEY environment variable not set")
            self.api_key = api_key
            self.model_name = model or os.getenv("NOMIC_MODEL", "nomic-embed-text-v1.5")
            self.embedding_dim = 768
        elif use_sentence_transformer:
            from sentence_transformers import SentenceTransformer

            self.model_name = model or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            from openai import OpenAI

            self.model_name = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self.client = OpenAI()
            if "3-small" in self.model_name or "ada-002" in self.model_name:
                self.embedding_dim = 1536
            elif "3-large" in self.model_name:
                self.embedding_dim = 3072
            else:
                self.embedding_dim = 1536

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self.use_nomic:
            return self._nomic_embed([text])[0]
        elif self.use_sentence_transformer:
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
        if self.use_nomic:
            return self._nomic_embed(list(texts))
        elif self.use_sentence_transformer:
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

    def _nomic_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Nomic HTTP API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = requests.post(
            "https://api-atlas.nomic.ai/v1/embedding/text",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "texts": texts,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]

    def __repr__(self) -> str:
        return f"EmbeddingManager(model={self.model_name}, dim={self.embedding_dim})"
