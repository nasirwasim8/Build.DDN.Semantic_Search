# DDN INFINIA Multimodal Semantic Search Demo

An AI-powered semantic search application for images, videos, and documents using **DDN INFINIA** storage and **NVIDIA CLIP/BLIP** models.

## Overview

This demo showcases DDN INFINIA high-performance storage combined with NVIDIA's vision-language AI models for enterprise multimodal search applications. Upload content, and search across all media types using natural language queries.

### Key Features

| Feature | Description |
|---------|-------------|
| Image Search | CLIP embeddings + BLIP captions for semantic image understanding |
| Video Search | Frame extraction and scene analysis |
| Document Search | PDF/DOCX text extraction with AI summarization |
| Natural Language | Search using descriptive queries, not just keywords |

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- NVIDIA GPU (optional, but recommended for faster inference)

### Installation

```bash
# Clone and install
./install.sh

# Or manually:
cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cd ../frontend && npm install
```

### Running

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
# App: http://localhost:5173
```

## Architecture

```
ddn-multimodal-semantic-search/
├── backend/                    # FastAPI Python backend
│   ├── app/
│   │   ├── api/routes.py       # REST API endpoints
│   │   ├── core/config.py      # Configuration management
│   │   ├── models/schemas.py   # Pydantic models
│   │   └── services/
│   │       ├── storage.py      # DDN INFINIA S3 handler
│   │       └── ai_models.py    # CLIP, BLIP, document analysis
│   ├── main.py
│   └── requirements.txt
│
├── frontend/                   # React + Vite + TypeScript
│   ├── src/
│   │   ├── components/         # Header, Sidebar
│   │   ├── pages/              # About, Config, Upload, Search, Browse
│   │   ├── services/api.ts     # API client
│   │   └── styles/             # Tailwind CSS + DDN colors
│   ├── public/                 # Production assets
│   └── package.json
├── ARCHITECTURE.md             # Detailed architecture guide
└── install.sh                  # One-click installation
```

## Features

### Multimodal Content Support
- **Images** — JPEG, PNG, WebP, GIF, BMP
- **Videos** — MP4, AVI, MOV, WebM
- **Documents** — PDF, DOCX, TXT

### AI Analysis Pipeline
- **CLIP** — Vision-language embeddings for semantic search
- **BLIP** — Automatic image captioning
- **BART** — Document summarization
- **Object Detection** — Automatic tagging of detected objects

### Storage Integration
- **DDN INFINIA** — High-performance S3-compatible object storage
- **Metadata Enrichment** — AI-generated metadata stored with objects
- **Presigned URLs** — Secure access to stored media

## API Reference

### Configuration
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/config/ddn` | Configure DDN INFINIA |
| POST | `/api/config/aws` | Configure AWS S3 |
| GET | `/api/config/test/{provider}` | Test connection |

### Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload/image` | Upload and analyze image |
| POST | `/api/upload/video` | Upload and analyze video |
| POST | `/api/upload/document` | Upload and analyze document |

### Search & Browse
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search` | Semantic search |
| POST | `/api/browse` | Browse all content |
| DELETE | `/api/browse/{key}` | Delete object |

### Health & Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Storage metrics |

## Tech Stack

### Backend
- FastAPI — High-performance Python web framework
- PyTorch — Deep learning framework
- Transformers — Hugging Face model hub
- boto3 — AWS SDK for S3-compatible storage

### Frontend
- React 18 + TypeScript
- Vite — Build tool
- Tailwind CSS — Styling
- TanStack Query — Data fetching
- Framer Motion — Animations

## AI Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| CLIP (ViT-B/32) | Image embeddings | OpenAI |
| BLIP | Image captioning | Salesforce |
| BART | Document summarization | Facebook |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Check Python 3.9+, reinstall deps |
| Models download slow | First run downloads ~2GB of models |
| Storage connection fails | Check endpoint URL has `https://` |
| Frontend API fails | Ensure backend runs on port 8000 |
| GPU not detected | Install CUDA toolkit, check PyTorch CUDA build |

## License

MIT

---

**DDN INFINIA** | **NVIDIA CLIP/BLIP** | **Multimodal Search**
