"""CogMemory: LLM Context Compression & Cognitive Graph System.

A cognitive memory system for LLM applications that extracts commitments
from text, assigns meta-roles, stores embeddings persistently, and maintains
a sparse cognitive graph with activation propagation.
"""

from cog_memory.cognitive_graph import CognitiveGraph
from cog_memory.decay import DecayModule
from cog_memory.deduplication import DeduplicationEngine
from cog_memory.embedding_manager import EmbeddingManager
from cog_memory.llm_extractor import LLMExtractor, Provider
from cog_memory.lance_store import LanceStore
from cog_memory.node import Node, Role
from cog_memory.query_interface import CognitiveMemory
from cog_memory.conflict_detector import ConflictDetector

__version__ = "0.1.0"
__all__ = [
    "CognitiveMemory",
    "CognitiveGraph",
    "Node",
    "Role",
    "LanceStore",
    "EmbeddingManager",
    "LLMExtractor",
    "Provider",
    "DeduplicationEngine",
    "ConflictDetector",
    "DecayModule",
]
