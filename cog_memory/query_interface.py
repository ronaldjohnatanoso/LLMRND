"""Query interface for the cognitive memory system.

Provides high-level API for ingesting text and querying the cognitive graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cog_memory.cognitive_graph import CognitiveGraph
from cog_memory.decay import DecayModule
from cog_memory.deduplication import DeduplicationEngine
from cog_memory.embedding_manager import EmbeddingManager
from cog_memory.llm_extractor import LLMExtractor, Provider
from cog_memory.lance_store import LanceStore
from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Sequence


class CognitiveMemory:
    """Main interface for the cognitive memory system.

    Orchestrates extraction, embedding, storage, and retrieval of commitments.
    """

    def __init__(
        self,
        db_path: str | Path = "./data/lancedb",
        model: str | None = None,
        embedding_model: str | None = None,
        use_sentence_transformer: bool = False,  # Use Nomic by default
        use_nomic: bool = True,  # Nomic via HTTP API (FREE, high quality, no CLI needed)
        use_dummy_extractor: bool = False,
        provider: Provider = "groq",
        llm_api_key: str | None = None,
    ) -> None:
        """Initialize the cognitive memory system.

        Args:
            db_path: Path to LanceDB database
            model: LLM model name for extraction
            embedding_model: Model name for embeddings
            use_sentence_transformer: Use sentence-transformers (local, 384 dim)
            use_nomic: Use Nomic embeddings via HTTP API (FREE, 768 dim, default)
            use_dummy_extractor: Use dummy extractor for testing
            provider: LLM provider ('groq', 'openai', 'huggingface')
            llm_api_key: API key for LLM provider (uses env var if None)
        """
        # Initialize components
        self.store = LanceStore(db_path=db_path)
        self.graph = CognitiveGraph()
        self.extractor = LLMExtractor(
            model=model,
            use_dummy=use_dummy_extractor,
            provider=provider,
            api_key=llm_api_key,
        )
        self.embedding_manager = EmbeddingManager(
            model=embedding_model,
            use_sentence_transformer=use_sentence_transformer,
            use_nomic=use_nomic,
        )
        self.deduplication = DeduplicationEngine()
        self.decay = DecayModule()

        # Load existing nodes into graph
        self._load_graph_from_store()

    def _load_graph_from_store(self) -> None:
        """Load existing nodes from LanceDB into the graph."""
        records = self.store.get_all_nodes()

        for record in records:
            node = Node.from_dict(record)
            self.graph.add_node(node)

    def ingest_paragraph(
        self,
        text: str,
    ) -> list[Node]:
        """Ingest a paragraph and extract commitments.

        Args:
            text: Input text paragraph

        Returns:
            List of newly created nodes
        """
        # Extract commitments
        nodes = self.extractor.extract_commitments(text)

        if not nodes:
            return []

        # Generate embeddings
        embeddings = self.embedding_manager.generate_embeddings_batch(
            [node.text for node in nodes]
        )

        # Process each node
        new_nodes = []
        for node, embedding in zip(nodes, embeddings):
            # Check for deduplication
            should_skip, merge_with_id = self.deduplication.process_node(
                node=node,
                embedding=embedding,
                store=self.store,
            )

            if should_skip and merge_with_id:
                # Merge with existing node
                existing_record = self.store.get_node(merge_with_id)
                if existing_record:
                    merged = self.deduplication.merge_nodes(existing_record, node)
                    self.store.update_node(merge_with_id, merged)
                    # Update graph
                    graph_node = self.graph.get_node(merge_with_id)
                    if graph_node:
                        graph_node.confidence = merged["confidence"]
                continue

            # Add new node
            node.embedding = embedding
            self.store.add_node(
                node_id=node.id,
                text=node.text,
                role=node.role.value,
                confidence=node.confidence,
                embedding=embedding,
                neighbors=node.neighbors,
                metadata=node.metadata,
                activation=node.activation,
            )
            self.graph.add_node(node)
            new_nodes.append(node)

        return new_nodes

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        activate_goals: Sequence[str] | None = None,
    ) -> list[Node]:
        """Query the cognitive memory system.

        Args:
            query_text: Query text
            top_k: Number of results to return
            activate_goals: Optional list of goal IDs to activate

        Returns:
            List of relevant nodes
        """
        # Reset activations
        self.graph.reset_all_activations()

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query_text)

        # Query similar nodes
        similar_records = self.store.query_similar(
            embedding=query_embedding,
            k=top_k * 2,  # Get more candidates
        )

        # Activate similar nodes
        for record in similar_records[:top_k]:
            node_id = record["id"]
            similarity = record.get("similarity", 0.5)
            self.graph.activate_node(node_id, activation=similarity)

        # Activate goals if specified
        if activate_goals:
            self.graph.activate_goals(activate_goals)

        # Return top activated nodes
        results = self.graph.get_top_activated(top_k)

        return results

    def get_nodes_by_role(self, role: Role) -> list[Node]:
        """Get all nodes with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of nodes with the specified role
        """
        return self.graph.get_nodes_by_role(role)

    def get_goals(self) -> list[Node]:
        """Get all goal nodes.

        Returns:
            List of goal nodes
        """
        return self.get_nodes_by_role(Role.GOAL)

    def get_constraints(self) -> list[Node]:
        """Get all constraint nodes.

        Returns:
            List of constraint nodes
        """
        return self.get_nodes_by_role(Role.CONSTRAINT)

    def apply_decay(self) -> dict:
        """Apply decay to all nodes.

        Returns:
            Dictionary with pruning statistics
        """
        nodes = list(self.graph.nodes.values())

        # Decay activations
        self.decay.decay_graph(nodes)

        # Prune weak edges
        pruned = self.decay.prune_weak_edges(nodes)

        # Get inactive nodes
        inactive = self.decay.get_inactive_nodes(nodes)

        return {
            "nodes_decayed": len(nodes),
            "edges_pruned": sum(len(v) for v in pruned.values()),
            "inactive_nodes": len(inactive),
        }

    def get_stats(self) -> dict:
        """Get statistics about the cognitive memory.

        Returns:
            Dictionary with system statistics
        """
        return {
            "total_nodes": len(self.graph),
            "by_role": {
                role.value: len(self.graph.get_nodes_by_role(role))
                for role in Role
            },
            "avg_activation": sum(
                n.activation for n in self.graph.nodes.values()
            ) / max(len(self.graph), 1),
            "total_edges": sum(
                len(n.neighbors) for n in self.graph.nodes.values()
            ),
        }

    def __repr__(self) -> str:
        return f"CognitiveMemory(nodes={len(self.graph)})"
