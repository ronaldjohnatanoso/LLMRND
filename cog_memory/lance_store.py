"""LanceDB integration for persistent vector storage.

Provides persistent storage for node embeddings with efficient similarity search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence


class NodeRecord(BaseModel):
    """Schema for LanceDB table records."""

    id: str
    text: str
    role: str
    confidence: float
    activation: float
    neighbors: str  # JSON string of neighbor dict
    metadata: str  # JSON string of metadata dict
    vector: Sequence[float] | None = None


class LanceStore:
    """LanceDB wrapper for persistent vector storage.

    Stores node embeddings with metadata for similarity search and retrieval.
    """

    def __init__(
        self,
        db_path: str | Path = "./data/lancedb",
        table_name: str = "nodes",
        embedding_dim: int = 1536,  # Default for OpenAI text-embedding-3-small
    ) -> None:
        """Initialize the LanceDB store.

        Args:
            db_path: Path to LanceDB database directory
            table_name: Name of the table to use/create
            embedding_dim: Dimension of embedding vectors
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.embedding_dim = embedding_dim

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        """Get existing table or create new one."""
        try:
            return self.db.open_table(self.table_name)
        except Exception:
            # Table doesn't exist, create it
            schema = [
                {"name": "id", "data_type": "string"},
                {"name": "text", "data_type": "string"},
                {"name": "role", "data_type": "string"},
                {"name": "confidence", "data_type": "float"},
                {"name": "activation", "data_type": "float"},
                {"name": "neighbors", "data_type": "string"},
                {"name": "metadata", "data_type": "string"},
                {
                    "name": "vector",
                    "data_type": "float",
                    "vec_dim": self.embedding_dim,
                },
            ]
            return self.db.create_table(self.table_name, schema=schema)

    def add_node(
        self,
        node_id: str,
        text: str,
        role: str,
        confidence: float,
        embedding: list[float],
        neighbors: dict | None = None,
        metadata: dict | None = None,
        activation: float = 0.0,
    ) -> None:
        """Add a node to the database.

        Args:
            node_id: Unique identifier for the node
            text: Text content of the node
            role: Meta-role of the node
            confidence: Confidence score
            embedding: Vector embedding
            neighbors: Neighbor dictionary (default: empty)
            metadata: Optional metadata dictionary (default: empty)
            activation: Activation level (default: 0.0)
        """
        record = {
            "id": node_id,
            "text": text,
            "role": role,
            "confidence": confidence,
            "activation": activation,
            "neighbors": json.dumps(neighbors or {}),
            "metadata": json.dumps(metadata or {}),
            "vector": embedding,
        }
        self.table.add([record])

    def query_similar(
        self,
        embedding: list[float],
        k: int = 5,
        filter_role: str | None = None,
        min_confidence: float | None = None,
    ) -> list[dict]:
        """Query for similar nodes by embedding.

        Args:
            embedding: Query vector
            k: Number of results to return
            filter_role: Optional role filter
            min_confidence: Optional minimum confidence filter

        Returns:
            List of matching node records as dictionaries
        """
        # Build query
        query = self.table.search(embedding).limit(k)

        # Apply filters if specified
        if filter_role or min_confidence is not None:
            conditions = []
            if filter_role:
                conditions.append(f"role = '{filter_role}'")
            if min_confidence is not None:
                conditions.append(f"confidence >= {min_confidence}")

            if conditions:
                where_clause = " AND ".join(conditions)
                query = query.where(where_clause)

        results = query.to_pandas()

        # Convert to list of dicts
        records = []
        for _, row in results.iterrows():
            records.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "role": row["role"],
                    "confidence": row["confidence"],
                    "activation": row["activation"],
                    "neighbors": json.loads(row["neighbors"]),
                    "metadata": json.loads(row["metadata"]),
                    "similarity": row.get("_score", 0.0),
                }
            )

        return records

    def get_node(self, node_id: str) -> dict | None:
        """Get a node by ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            Node record as dictionary, or None if not found
        """
        results = self.table.search(f"id = '{node_id}'").limit(1).to_pandas()

        if len(results) == 0:
            return None

        row = results.iloc[0]
        return {
            "id": row["id"],
            "text": row["text"],
            "role": row["role"],
            "confidence": row["confidence"],
            "activation": row["activation"],
            "neighbors": json.loads(row["neighbors"]),
            "metadata": json.loads(row["metadata"]),
        }

    def update_node(
        self,
        node_id: str,
        updates: dict,
    ) -> None:
        """Update a node in the database.

        Args:
            node_id: ID of the node to update
            updates: Dictionary of fields to update
        """
        # Convert dict fields to JSON strings
        if "neighbors" in updates and isinstance(updates["neighbors"], dict):
            updates["neighbors"] = json.dumps(updates["neighbors"])
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        # LanceDB uses SQL for updates
        set_clauses = [f"{k} = '{v}'" for k, v in updates.items()]
        set_clause = ", ".join(set_clauses)

        self.table.update(f"WHERE id = '{node_id}' SET {set_clause}")

    def delete_node(self, node_id: str) -> None:
        """Delete a node from the database.

        Args:
            node_id: ID of the node to delete
        """
        self.table.delete(f"id = '{node_id}'")

    def get_all_nodes(self) -> list[dict]:
        """Get all nodes from the database.

        Returns:
            List of all node records as dictionaries
        """
        results = self.table.search().limit(None).to_pandas()

        records = []
        for _, row in results.iterrows():
            records.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "role": row["role"],
                    "confidence": row["confidence"],
                    "activation": row["activation"],
                    "neighbors": json.loads(row["neighbors"]),
                    "metadata": json.loads(row["metadata"]),
                }
            )

        return records

    def count_nodes(self) -> int:
        """Count total nodes in the database.

        Returns:
            Number of nodes
        """
        return len(self.table.search().limit(None).to_pandas())

    def __repr__(self) -> str:
        return f"LanceStore(path={self.db_path}, table={self.table_name}, nodes={self.count_nodes()})"
