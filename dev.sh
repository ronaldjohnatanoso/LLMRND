#!/bin/bash

# CogMemory Development Runner
# Runs both backend and frontend services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  CogMemory Development Environment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"

    # Kill backend
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${YELLOW}Killing backend (PID: $BACKEND_PID)${NC}"
        kill $BACKEND_PID 2>/dev/null || true
    fi

    # Kill frontend
    if [ ! -z "$FRONTEND_PID" ]; then
        echo -e "${YELLOW}Killing frontend (PID: $FRONTEND_PID)${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    # Wait for processes to terminate
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true

    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Check if .env exists in root
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${RED}Error: .env file not found in root directory${NC}"
    echo -e "${YELLOW}Create one with your NOMIC_API_KEY or set USE_LOCAL_EMBEDDINGS=true${NC}"
    exit 1
fi

# ============================================
# BACKEND SETUP
# ============================================
echo -e "${BLUE}[1/2]${NC} Starting Backend..."

# Check if backend venv exists
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv backend/venv
fi

# Activate venv and install dependencies
echo -e "${YELLOW}Installing/upgrading backend dependencies...${NC}"
source backend/venv/bin/activate
pip install -q -r backend/requirements.txt

# Start backend in background
echo -e "${GREEN}✓ Starting FastAPI server on http://localhost:8000${NC}"
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend is running (PID: $BACKEND_PID)${NC}"
echo ""

# ============================================
# FRONTEND SETUP
# ============================================
echo -e "${BLUE}[2/2]${NC} Starting Frontend..."

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Start frontend in background
echo -e "${GREEN}✓ Starting Next.js dev server on http://localhost:8005${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}✓ Frontend is running (PID: $FRONTEND_PID)${NC}"
echo ""

# ============================================
# READY
# ============================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All services are running!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Backend:${NC}  http://localhost:8000"
echo -e "${GREEN}Frontend:${NC} http://localhost:8005"
echo -e "${GREEN}API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
wait
