"""Query interface for the cognitive memory system.

Provides high-level API for ingesting text and querying the cognitive graph.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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

        # Create edges between new nodes based on semantic similarity
        self._create_semantic_edges_batch(nodes, embeddings)

        # Process each node (edges already created, so neighbors are ready)
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

            # Add new node (neighbors already populated by edge creation)
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
                similarity_to_query=node.similarity_to_query,
            )
            self.graph.add_node(node)
            new_nodes.append(node)

        return new_nodes

    def query_debug(
        self,
        query_text: str,
        top_k: int = 10,
        propagation_depth: int = 3,
        min_similarity_threshold: float = 0.55,
        candidate_multiplier: int = 2,
    ) -> dict:
        """Query with detailed debug information for testing.

        Args:
            query_text: Query text
            top_k: Number of results to return
            propagation_depth: How many hops to propagate (default: 3 for testing)
            min_similarity_threshold: Minimum similarity for Layer 1 matches
            candidate_multiplier: Fetch multiplier for vector search (default: 2)

        Returns:
            Dictionary with:
                - direct_matches: List of Layer 1 nodes
                - propagation_tree: Full tree structure with layers
                - all_activated: All activated nodes with their activations
                - layers: Nodes grouped by layer
        """
        from collections import deque

        # Reset activations
        self.graph.reset_all_activations()

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query_text)

        # Query similar nodes
        similar_records = self.store.query_similar(
            embedding=query_embedding,
            k=top_k * candidate_multiplier,
        )

        # Activate Layer 1 (direct matches)
        activated_ids = set()
        layer_1_nodes = []

        for record in similar_records:
            node_id = record["id"]
            similarity = record.get("similarity", 0.5)

            if similarity < min_similarity_threshold and len(activated_ids) >= top_k:
                continue

            node = self.graph.get_node(node_id)
            if node:
                node.similarity_to_query = similarity
                node.activation = similarity
                layer_1_nodes.append({
                    "id": node.id,
                    "text": node.text,
                    "role": node.role.value,
                    "similarity_to_query": similarity,
                    "activation": similarity,
                })
            activated_ids.add(node_id)

            if len(activated_ids) >= top_k:
                break

        # Clear similarity_to_query for non-activated nodes
        all_node_ids = {r["id"] for r in similar_records}
        for node_id in all_node_ids - activated_ids:
            node = self.graph.get_node(node_id)
            if node:
                node.similarity_to_query = 0.0

        # Build propagation tree via BFS
        tree = {}
        layers = {1: []}

        for node_id in activated_ids:
            node = self.graph.get_node(node_id)
            if node:
                tree[node_id] = {
                    "layer": 1,
                    "parent": None,
                    "parent_similarity": node.similarity_to_query,
                    "activation": node.activation,
                    "children": []
                }
                layers[1].append(node_id)

        # BFS to build tree
        visited = set(activated_ids)
        queue = deque([(node_id, 1) for node_id in activated_ids])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= propagation_depth:
                continue

            current_node = self.graph.get_node(current_id)
            if not current_node or current_node.activation < self.graph.activation_threshold:
                continue

            for neighbor_id, weight in current_node.neighbors.items():
                if neighbor_id in visited:
                    continue

                neighbor = self.graph.get_node(neighbor_id)
                if not neighbor:
                    continue

                # Calculate role boost
                role_boost = self.graph.default_boost

                # Calculate activation
                base_similarity = current_node.similarity_to_query if current_node.similarity_to_query > 0 else current_node.activation
                activation_delta = base_similarity * weight * role_boost

                if activation_delta < self.graph.min_delta:
                    continue

                # Update activation
                neighbor.update_activation(activation_delta)
                visited.add(neighbor_id)

                # Add to tree
                child_layer = current_depth + 1
                if child_layer not in layers:
                    layers[child_layer] = []
                layers[child_layer].append(neighbor_id)

                tree[neighbor_id] = {
                    "layer": child_layer,
                    "parent": current_id,
                    "parent_similarity": base_similarity,
                    "edge_weight": weight,
                    "role_boost": role_boost,
                    "activation_delta": activation_delta,
                    "activation": neighbor.activation,
                    "similarity_to_query": neighbor.similarity_to_query,
                    "children": []
                }

                tree[current_id]["children"].append(neighbor_id)
                queue.append((neighbor_id, child_layer))

        # Get all activated nodes with details
        all_activated = []
        for node_id in visited:
            node = self.graph.get_node(node_id)
            if node:
                all_activated.append({
                    "id": node.id,
                    "text": node.text,
                    "role": node.role.value,
                    "activation": node.activation,
                    "similarity_to_query": node.similarity_to_query,
                    "neighbors": list(node.neighbors.keys()),
                })

        return {
            "query": query_text,
            "direct_matches_count": len(layer_1_nodes),
            "direct_matches": layer_1_nodes,
            "total_activated": len(visited),
            "layers": {k: len(v) for k, v in layers.items()},
            "tree": tree,
            "all_activated": all_activated,
            "settings": {
                "min_delta": self.graph.min_delta,
                "activation_threshold": self.graph.activation_threshold,
                "propagation_depth": propagation_depth,
            }
        }

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        activate_goals: Sequence[str] | None = None,
        propagation_depth: int = 2,
        min_similarity_threshold: float = 0.55,
        candidate_multiplier: int = 2,
    ) -> list[Node]:
        """Query the cognitive memory system.

        Args:
            query_text: Query text
            top_k: Number of results to return
            activate_goals: Optional list of goal IDs to activate
            propagation_depth: How many hops to propagate activation (default: 2)
            min_similarity_threshold: Minimum similarity for Layer 1 direct matches (default: 0.55)
            candidate_multiplier: Fetch multiplier for vector search (default: 2)

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
            k=top_k * candidate_multiplier,
        )

        # Activate similar nodes (Layer 1: direct matches)
        # Only activate nodes above threshold to avoid weak matches filling Layer 1
        activated_ids = set()
        for record in similar_records:
            node_id = record["id"]
            similarity = record.get("similarity", 0.5)

            # STRICT: Only activate if above threshold
            if similarity < min_similarity_threshold:
                continue

            node = self.graph.get_node(node_id)
            if node:
                node.similarity_to_query = similarity  # Store original similarity to query
                node.activation = similarity  # Set temporary activation
            activated_ids.add(node_id)

            # Stop if we have enough quality matches
            if len(activated_ids) >= top_k:
                break

        # Clear similarity_to_query for non-activated nodes to prevent "also direct" confusion
        all_node_ids = {r["id"] for r in similar_records}
        for node_id in all_node_ids - activated_ids:
            node = self.graph.get_node(node_id)
            if node:
                node.similarity_to_query = 0.0  # Not a Layer 1 direct match

        # Propagate activation through the graph (multi-hop reasoning)
        for node_id in activated_ids:
            self.graph.propagate_activation(node_id, depth=propagation_depth)

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

    def _create_semantic_edges_batch(
        self,
        new_nodes: list[Node],
        embeddings: list[list[float]],
        similarity_threshold: float = 0.55,  # Lowered from 0.75 to create more connections
        max_edges: int = 3,
    ) -> None:
        """Create edges between new nodes BEFORE storing them.

        Args:
            new_nodes: List of newly created nodes
            embeddings: Embeddings for the new nodes
            similarity_threshold: Minimum similarity to create edge (0-1)
            max_edges: Maximum edges to create per node
        """
        # Create edges between new nodes based on similarity
        for (node1, emb1), (node2, emb2) in combinations(zip(new_nodes, embeddings), 2):
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )

            if similarity >= similarity_threshold:
                # Add edge with weight = similarity
                weight = float(similarity)
                if node2.id not in node1.neighbors:
                    node1.neighbors[node2.id] = weight
                if node1.id not in node2.neighbors:
                    node2.neighbors[node1.id] = weight

        # Connect new nodes to similar existing nodes
        for node, embedding in zip(new_nodes, embeddings):
            # Find similar existing nodes (excluding newly created ones)
            new_ids = {n.id for n in new_nodes}
            results = self.store.query_similar(
                embedding=embedding,
                k=max_edges + len(new_nodes),  # Get extra to filter out new nodes
            )

            edges_added = 0
            for result in results:
                if result["id"] == node.id or result["id"] in new_ids:
                    continue

                similarity = result.get("similarity", 0.0)
                if similarity >= similarity_threshold and edges_added < max_edges:
                    # Add edge to new node
                    weight = float(similarity)
                    node.neighbors[result["id"]] = weight

                    # Update existing node in BOTH database AND in-memory graph
                    existing_record = self.store.get_node(result["id"])
                    if existing_record:
                        neighbors = existing_record.get("neighbors", {})
                        neighbors[node.id] = weight
                        self.store.update_node(result["id"], {"neighbors": neighbors})

                        # Also update the in-memory graph (critical for immediate consistency)
                        existing_graph_node = self.graph.get_node(result["id"])
                        if existing_graph_node:
                            existing_graph_node.neighbors[node.id] = weight

                    edges_added += 1

    def __repr__(self) -> str:
        return f"CognitiveMemory(nodes={len(self.graph)})"
