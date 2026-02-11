# DDN Multimodal Semantic Search - Architecture Guide

This document explains the complete structure of the DDN Multimodal Semantic Search application, a modular React + FastAPI architecture for AI-powered content search.

---

## Quick Start (For Your Colleague)

### 1. Run the Installation Script

```bash
./install.sh
```

This will:
- Check Python and Node.js versions
- Create Python virtual environment
- Install all backend dependencies
- Install all frontend dependencies

### 2. Run Both Servers

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 3. Open the App

- **Frontend:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

---

## What This Application Does

### Multimodal Content Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        UPLOAD                                │
├─────────────────────────────────────────────────────────────┤
│  Image → CLIP embedding + BLIP caption → DDN INFINIA        │
│  Video → Frame extraction + scene analysis → DDN INFINIA    │
│  Document → Text extraction + BART summary → DDN INFINIA    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        SEARCH                                │
├─────────────────────────────────────────────────────────────┤
│  Query → Match against metadata → Rank by relevance         │
│  Returns: Presigned URLs + metadata for matching content    │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ddn-multimodal-semantic-search/
├── backend/                      # FastAPI Python Backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py        # All REST API endpoints
│   │   ├── core/
│   │   │   └── config.py        # Configuration settings
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   └── services/
│   │       ├── storage.py       # S3Handler for DDN INFINIA
│   │       └── ai_models.py     # CLIP, BLIP, document analysis
│   ├── main.py                  # FastAPI app entry point
│   └── requirements.txt
│
├── frontend/                     # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx       # Top navigation (About/Demo)
│   │   │   └── DemoSidebar.tsx  # Sidebar for demo sections
│   │   ├── pages/
│   │   │   ├── About.tsx        # Landing page
│   │   │   ├── Configuration.tsx # Storage config
│   │   │   ├── Upload.tsx       # Content upload
│   │   │   ├── Search.tsx       # Semantic search
│   │   │   └── Browse.tsx       # Content browser
│   │   ├── contexts/
│   │   │   └── ThemeContext.tsx # Dark/light theme
│   │   ├── services/api.ts      # API client with TypeScript types
│   │   └── styles/              # Tailwind CSS
│   ├── public/                  # Production assets
│   └── package.json
│
├── README.md
├── ARCHITECTURE.md              # This file
└── install.sh                   # One-click setup script
```

---

## Backend Services

### Storage Service (`storage.py`)

Handles DDN INFINIA S3-compatible storage operations:

```python
S3Handler(provider: str)  # 'ddn_infinia' or 'aws'
  - test_connection()     # Verify credentials
  - upload_bytes()        # Upload with metadata
  - list_objects()        # List all objects with metadata
  - generate_presigned_url()  # Secure access URLs
  - delete_object()       # Remove objects
```

### AI Models Service (`ai_models.py`)

Provides AI analysis for each content type:

```python
ImageAnalyzer
  - CLIP model for embeddings
  - BLIP model for captions
  - Returns: caption, detected_objects, dimensions, embedding

VideoAnalyzer
  - OpenCV frame extraction
  - CLIP analysis on keyframes
  - Returns: summary, duration, fps, detected_objects

DocumentAnalyzer
  - PyPDF2 / python-docx text extraction
  - BART summarization
  - Returns: summary, word_count, key_terms
```

---

## Frontend Pages

| Page | Purpose | Key Features |
|------|---------|--------------|
| About | Landing page | Hero, features, CTA |
| Configuration | Storage setup | DDN/AWS credentials |
| Upload | Content ingestion | Image/Video/Document tabs |
| Search | Semantic search | Natural language queries |
| Browse | Content library | Grid view, delete, preview |

---

## Navigation Structure

```
┌─────────────────────────────────────────────────────────────┐
│  [Logo]  Multimodal Search       [About] [Demo]    [Theme]  │ ← Header
└─────────────────────────────────────────────────────────────┘

When "About" is selected:
┌─────────────────────────────────────────────────────────────┐
│                    Landing Page                              │
│  - Hero with DDN + NVIDIA branding                          │
│  - Feature cards (Image, Video, Document)                   │
│  - Technology stack                                          │
│  - "Start Demo" CTA button                                  │
└─────────────────────────────────────────────────────────────┘

When "Demo" is selected:
┌──────────────┬──────────────────────────────────────────────┐
│  Sidebar     │                                              │
│  ──────────  │         Demo Content Area                    │
│  Config      │                                              │
│  Upload      │         (Selected page renders here)         │
│  Search      │                                              │
│  Browse      │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

---

## API Endpoints

All endpoints are prefixed with `/api` (except health check):

### Configuration
```
POST /api/config/ddn          # Configure DDN INFINIA credentials
POST /api/config/aws          # Configure AWS S3 credentials
GET  /api/config/test/{provider}  # Test connection (ddn_infinia or aws)
```

### Upload
```
POST /api/upload/image        # Upload and analyze image
POST /api/upload/video        # Upload and analyze video
POST /api/upload/document     # Upload and analyze document
```

### Search & Browse
```
POST   /api/search            # Semantic search across content
POST   /api/browse            # List all stored content
DELETE /api/browse/{key}      # Delete specific object
```

### Health & Metrics
```
GET /health   # Health check with status of all services
GET /metrics  # Content counts and storage usage
```

---

## Key API Examples

### Configure DDN INFINIA
```bash
curl -X POST http://localhost:8000/api/config/ddn \
  -H "Content-Type: application/json" \
  -d '{
    "access_key": "your-key",
    "secret_key": "your-secret",
    "bucket_name": "your-bucket",
    "endpoint_url": "https://your-ddn-endpoint",
    "region": "us-east-1"
  }'
```

### Test Connection
```bash
curl http://localhost:8000/api/config/test/ddn_infinia
```

### Upload Image
```bash
curl -X POST http://localhost:8000/api/upload/image \
  -F "file=@photo.jpg"
```

### Semantic Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sunset over mountains",
    "modality": "all",
    "top_k": 10
  }'
```

---

## Dependencies

### Backend (Python 3.9+)
- FastAPI + Uvicorn (web server)
- PyTorch + Transformers (AI models)
- boto3 (S3/DDN storage)
- Pillow, OpenCV (image/video processing)
- PyPDF2, python-docx (document processing)

### Frontend (Node 18+)
- React 18
- TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- TanStack Query (data fetching)
- Framer Motion (animations)
- Lucide React (icons)

---

## AI Models

| Model | Size | Purpose |
|-------|------|---------|
| CLIP ViT-B/32 | ~600MB | Vision-language embeddings |
| BLIP Base | ~800MB | Image captioning |
| BART CNN | ~400MB | Document summarization |

**Note:** Models are downloaded on first run (~2GB total). Subsequent runs use cached models.

---

## Troubleshooting

### Backend won't start
```bash
# Check Python version (need 3.9+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Storage connection fails
- Verify endpoint URL includes `https://`
- Check bucket exists and credentials have access
- The app disables SSL verification for self-signed certs

### Frontend API calls fail
- Ensure backend is running on port 8000
- Vite proxy is configured in `vite.config.ts`

### GPU not detected
- Install CUDA toolkit matching your GPU
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## Questions?

The codebase is modular and each file has a single responsibility. Start with:
- `backend/app/api/routes.py` - to see all endpoints
- `backend/app/services/` - to understand backend logic
- `frontend/src/pages/` - to see UI components
- `frontend/src/services/api.ts` - to see API client

Happy coding!
