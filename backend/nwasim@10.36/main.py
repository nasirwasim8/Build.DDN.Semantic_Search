"""
DDN Multimodal Semantic Search - FastAPI Backend
AI-powered semantic search using DDN INFINIA accelerated by NVIDIA GPU technology.
"""
import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from app.api.routes import (
    config_router,
    upload_router,
    search_router,
    browse_router,
    health_router,
    ingestion_router
)
from app.services.ai_models import get_image_analyzer, get_video_analyzer, get_document_analyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize AI models on startup."""
    print("=" * 60)
    print("  DDN Multimodal Semantic Search")
    print("  Powered by DDN INFINIA & NVIDIA GPU Technology")
    print("=" * 60)

    print("\nInitializing AI models...")
    # Pre-load models (this will download on first run)
    get_image_analyzer()
    get_video_analyzer()
    get_document_analyzer()
    print("AI models ready.\n")

    yield

    print("\nShutting down...")


# Create FastAPI app
app = FastAPI(
    title="DDN Multimodal Semantic Search",
    description="AI-powered semantic search using DDN INFINIA and NVIDIA GPU technology",
    version="2.1.1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(browse_router, prefix="/api")
app.include_router(ingestion_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DDN Multimodal Semantic Search API",
        "version": "2.1.1",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Changed from 8000 to avoid conflict with RAG backend
        reload=True
    )
