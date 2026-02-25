"""
FastAPI backend for CogMemory system.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from dotenv import load_dotenv

# Get the backend directory and parent directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BACKEND_DIR)

# Load environment variables from multiple locations (in order of priority)
# 1. backend/.env (local override)
# 2. parent .env (shared config)
load_dotenv(os.path.join(BACKEND_DIR, '.env'))
load_dotenv(os.path.join(PARENT_DIR, '.env'))

# Add parent directory to path to import cog_memory
sys.path.insert(0, PARENT_DIR)

from cog_memory.query_interface import CognitiveMemory
from cog_memory.node import Role

# Global memory instance
memory = None

# Check if using local embeddings
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global memory
    try:
        # Initialize with appropriate embedding manager
        if USE_LOCAL_EMBEDDINGS:
            print("🔄 Using local sentence-transformers embeddings")
            memory = CognitiveMemory(use_sentence_transformer=True, use_nomic=False)
        else:
            api_key = os.getenv("NOMIC_API_KEY")
            if not api_key or api_key == "your_nomic_api_key_here":
                print("⚠️  NOMIC_API_KEY not set, falling back to local embeddings")
                print("   Set USE_LOCAL_EMBEDDINGS=true in .env to suppress this warning")
                memory = CognitiveMemory(use_sentence_transformer=True, use_nomic=False)
            else:
                print("🌐 Using Nomic API embeddings")
                memory = CognitiveMemory(use_nomic=True, use_sentence_transformer=False)

        print(f"✅ CogMemory initialized (embeddings: {'local' if USE_LOCAL_EMBEDDINGS or not os.getenv('NOMIC_API_KEY') else 'Nomic API'})")
    except Exception as e:
        print(f"❌ Error initializing CogMemory: {e}")
        import traceback
        traceback.print_exc()

    yield

    # Cleanup on shutdown
    print("🛑 Shutting down...")

app = FastAPI(title="CogMemory API", version="1.0.0", lifespan=lifespan)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8005", "http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AddNodesRequest(BaseModel):
    texts: List[str]
    role: Optional[str] = "FACT"

class PropagationConfigRequest(BaseModel):
    hop_decay_enabled: bool = True
    hop_decay_factor: float = 0.5
    hard_cap_multiplier: float = 2.0
    # Add more config options as needed

class PlasticityConfigRequest(BaseModel):
    learning_rate: float = 0.02
    consolidation_enabled: bool = True
    min_weight: float = 0.05
    max_weight: float = 1.0

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 10
    propagation_depth: int = 3
    activation_threshold: float = 0.55
    decay_per_hop: float = 0.7
    propagation_threshold: float = 0.6
    max_steps: int = 100
    propagation_config: Optional[PropagationConfigRequest] = None
    plasticity_config: Optional[PlasticityConfigRequest] = None
    enable_plasticity: bool = True

class NodeResponse(BaseModel):
    id: str
    text: str
    role: str
    activation: float
    confidence: float
    neighbors: List[str]
    similarity: float = 0.0

class SimulationResponse(BaseModel):
    query: str
    timeline: List[dict]
    total_steps: int
    final_states: List[dict]
    settings: dict
    learning_stats: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "CogMemory API", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "memory_initialized": memory is not None}

@app.post("/nodes", response_model=List[NodeResponse])
async def add_nodes(request: AddNodesRequest):
    """Add new nodes to memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        role = Role[request.role.upper()] if request.role else Role.FACT
        new_nodes = memory.add_commitments(
            texts=request.texts,
            default_role=role
        )

        return [
            NodeResponse(
                id=node.id,
                text=node.text,
                role=node.role.value,
                activation=node.activation,
                confidence=node.confidence,
                neighbors=list(node.neighbors.keys())
            )
            for node in new_nodes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nodes", response_model=List[NodeResponse])
async def get_all_nodes():
    """Get all nodes in memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        node_records = memory.store.get_all_nodes()
        return [
            NodeResponse(
                id=record["id"],
                text=record.get("text", ""),
                role=record.get("role", "FACT"),
                activation=record.get("activation", 0.0),
                confidence=record.get("confidence", 0.0),
                neighbors=list(record.get("neighbors", {}).keys())
            )
            for record in node_records
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=List[NodeResponse])
async def query(request: QueryRequest):
    """Query memory with propagation."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        results = memory.query(
            query_text=request.query_text,
            top_k=request.top_k,
            propagation_depth=request.propagation_depth,
            min_similarity_threshold=request.min_similarity_threshold,
            candidate_multiplier=request.candidate_multiplier
        )

        return [
            NodeResponse(
                id=node.id,
                text=node.text,
                role=node.role.value,
                activation=node.activation,
                confidence=node.confidence,
                neighbors=list(node.neighbors.keys()),
                similarity=getattr(node, 'similarity_to_query', 0.0)
            )
            for node in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/simulation", response_model=SimulationResponse)
async def query_simulation(request: QueryRequest):
    """Query with step-by-step simulation for animation."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        print(f"=== QUERY SIMULATION CALLED ===")
        print(f"query_text: {request.query_text}")
        print(f"max_steps from request: {request.max_steps}")
        print(f"top_k: {request.top_k}")
        print(f"propagation_depth: {request.propagation_depth}")
        print(f"enable_plasticity: {request.enable_plasticity}")

        # Build propagation config if provided
        propagation_config = None
        if request.propagation_config:
            from cog_memory.propagation_config import PropagationConfig
            propagation_config = PropagationConfig(
                hop_decay_enabled=request.propagation_config.hop_decay_enabled,
                hop_decay_factor=request.propagation_config.hop_decay_factor,
                hard_cap_multiplier=request.propagation_config.hard_cap_multiplier,
            )

        # Build plasticity config if provided
        plasticity_config = None
        if request.plasticity_config:
            from cog_memory.plasticity_config import PlasticityConfig
            plasticity_config = PlasticityConfig(
                learning_rate=request.plasticity_config.learning_rate,
                consolidation_enabled=request.plasticity_config.consolidation_enabled,
                min_weight=request.plasticity_config.min_weight,
                max_weight=request.plasticity_config.max_weight,
            )

        result = memory.query_simulation(
            query_text=request.query_text,
            top_k=request.top_k,
            propagation_depth=request.propagation_depth,
            activation_threshold=request.activation_threshold,
            decay_per_hop=request.decay_per_hop,
            propagation_threshold=request.propagation_threshold,
            max_steps=request.max_steps,
            propagation_config=propagation_config,
            enable_plasticity=request.enable_plasticity,
        )

        print(f"=== RESULT ===")
        print(f"total_steps: {result['total_steps']}")
        print(f"timeline length: {len(result['timeline'])}")
        print(f"settings.max_steps: {result['settings']['max_steps']}")
        if result.get('learning_stats'):
            print(f"connections_strengthened: {result['learning_stats'].get('connections_strengthened', 0)}")
        print(f"========================")

        return result
    except Exception as e:
        import traceback
        print(f"Error in query_simulation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")

class IngestRequest(BaseModel):
    text: str

@app.post("/ingest", response_model=List[NodeResponse])
async def ingest_text(request: IngestRequest):
    """Ingest a paragraph and extract commitments using LLM."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        nodes = memory.ingest_paragraph(request.text)
        return [
            NodeResponse(
                id=node.id,
                text=node.text,
                role=node.role.value,
                activation=node.activation,
                confidence=node.confidence,
                neighbors=list(node.neighbors.keys())
            )
            for node in nodes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/nodes")
async def clear_memory():
    """Clear all nodes from memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Clear the in-memory graph
        memory.graph.nodes.clear()

        # Drop and recreate the LanceDB table
        memory.store.db.drop_table(memory.store.table_name)
        memory.store.table = memory.store._get_or_create_table()

        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WeightsRequest(BaseModel):
    filepath: Optional[str] = "./data/learned_weights.json"

@app.post("/weights/save")
async def save_weights(request: WeightsRequest):
    """Save learned connection weights to disk."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        memory.save_learned_weights(request.filepath)
        return {"message": "Weights saved successfully", "filepath": request.filepath}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/weights/load")
async def load_weights(request: WeightsRequest):
    """Load learned connection weights from disk."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        loaded = memory.load_learned_weights(request.filepath)
        if loaded:
            return {"message": "Weights loaded successfully", "filepath": request.filepath}
        else:
            return {"message": "No weights file found, using defaults", "filepath": request.filepath}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weights/stats")
async def get_weights_stats():
    """Get statistics about learned weights."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        return {
            "query_count": memory.neuroplasticity.query_count,
            "last_consolidation_query": memory.neuroplasticity.last_consolidation_query,
            "total_edges": sum(len(n.neighbors) for n in memory.graph.nodes.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
