# CogMemory - Frontend Migration

## Architecture

```
LLMRND/
├── backend/          # FastAPI + Python
│   ├── main.py      # API endpoints
│   └── requirements.txt
├── frontend/         # Next.js + React
│   ├── src/
│   │   ├── app/     # Next.js App Router
│   │   ├── components/
│   │   │   └── GraphVisualization.tsx  # Cytoscape.js graph
│   │   ├── lib/
│   │   │   └── api.ts  # API client
│   │   └── types/
│   │       └── index.ts  # TypeScript types
└── cog_memory/       # Core logic (shared)
```

## Setup

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your NOMIC_API_KEY
python main.py
```

Backend runs on `http://localhost:8000`

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

## Environment Variables

**Backend (.env):**
```
NOMIC_API_KEY=your_key_here
LANCEDB_PATH=./data/lancedb
```

**Frontend (.env.local):**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Features

- ✅ Real-time graph visualization with Cytoscape.js
- ✅ Step-by-step propagation animation
- ✅ Query interface with simulation
- ✅ Dark mode UI
- ✅ Responsive design
- ✅ TypeScript for type safety

## Migration Notes

- Streamlit browser.py moved to `browser.py.legacy`
- New frontend uses React + Next.js
- Backend API provides REST endpoints
- No more `st.rerun()` limitations
- Smooth animations with Cytoscape.js

## TODO

- [ ] Add node details panel
- [ ] Add graph pan/zoom controls
- [ ] Add auto-play animation
- [ ] Add node filtering
- [ ] Add add/edit/delete nodes UI
- [ ] Add export graph functionality
