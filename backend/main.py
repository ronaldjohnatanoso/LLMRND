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

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 10
    propagation_depth: int = 3
    min_similarity_threshold: float = 0.55
    candidate_multiplier: int = 2
    decay_per_hop: float = 0.7

class NodeResponse(BaseModel):
    id: str
    text: str
    role: str
    activation: float
    confidence: float
    neighbors: List[str]

class SimulationResponse(BaseModel):
    query: str
    timeline: List[dict]
    total_steps: int
    final_states: List[dict]
    settings: dict

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
                neighbors=record.get("neighbors", [])
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
                neighbors=list(node.neighbors.keys())
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
        result = memory.query_simulation(
            query_text=request.query_text,
            top_k=request.top_k,
            propagation_depth=request.propagation_depth,
            min_similarity_threshold=request.min_similarity_threshold,
            candidate_multiplier=request.candidate_multiplier,
            decay_per_hop=request.decay_per_hop
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        memory.graph = CognitiveGraph()
        return {"message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
