"""
Pydantic schemas for API request/response models.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# Storage Configuration
class DDNConfigRequest(BaseModel):
    access_key: str
    secret_key: str
    bucket_name: str
    endpoint_url: str
    region: str = "us-east-1"


class AWSConfigRequest(BaseModel):
    access_key: str
    secret_key: str
    bucket_name: str
    region: str = "us-east-1"


class LocalCacheConfigRequest(BaseModel):
    enabled: bool
    videos_path: str = ""
    embeddings_path: str = ""


class LocalCacheConfigResponse(BaseModel):
    success: bool
    message: str
    enabled: bool
    videos_path: str
    embeddings_path: str


class StorageConfigResponse(BaseModel):
    success: bool
    message: str
    ddn_configured: bool = False
    aws_configured: bool = False


class ConnectionTestResponse(BaseModel):
    provider: str
    success: bool
    message: str
    latency_ms: Optional[float] = None


# Upload Responses
class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    object_key: str
    caption: str
    detected_objects: str
    width: int
    height: int
    has_embedding: bool


class VideoUploadResponse(BaseModel):
    success: bool
    message: str
    object_key: str
    summary: str
    duration_seconds: float
    detected_objects: str
    frame_count: int
    presigned_url: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    object_key: str
    summary: str
    word_count: int
    key_terms: str


# Search
class SearchRequest(BaseModel):
    query: str
    modality: str = Field(default="all", description="Filter: all, image, video, document")
    top_k: int = Field(default=20, ge=1, le=100)
    threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="Semantic similarity threshold")


class StorageInfo(BaseModel):
    """Object storage metadata and capabilities"""
    source: str = Field(description="Storage source: 'local_cache', 'ddn_infinia', or 'aws_s3'")
    storage_class: str = Field(default="STANDARD", description="S3 storage class")
    access_control: Dict[str, bool] = Field(description="Read/Write permissions")
    protocol: str = Field(default="S3", description="Storage protocol")
    encryption: Optional[str] = Field(default=None, description="Encryption type if enabled")
    versioning_enabled: bool = Field(default=False, description="Object versioning status")
    etag: Optional[str] = Field(default=None, description="Object ETag for integrity")
    retrieval_time_ms: Optional[float] = Field(default=None, description="Object retrieval latency")


class SearchResult(BaseModel):
    object_key: str
    modality: str
    relevance_score: float
    metadata: Dict[str, Any]
    size_bytes: int
    last_modified: str
    presigned_url: Optional[str] = None
    storage_info: Optional[StorageInfo] = None  # NEW: Object storage capabilities


class SearchResponse(BaseModel):
    success: bool
    query: str
    total_results: int
    results: List[SearchResult]
    search_time_ms: float


# Browse
class BrowseRequest(BaseModel):
    modality: str = Field(default="all", description="Filter: all, image, video, document")
    prefix: str = ""


class ObjectInfo(BaseModel):
    key: str
    modality: str
    size_bytes: int
    last_modified: str
    metadata: Dict[str, Any]
    presigned_url: Optional[str] = None


class BrowseResponse(BaseModel):
    success: bool
    total_objects: int
    objects: List[ObjectInfo]


# Health
class HealthResponse(BaseModel):
    status: str
    ddn_configured: bool
    aws_configured: bool
    ai_models_loaded: bool
    gpu_available: bool
    device: str


# Error
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# Metrics
class MetricsResponse(BaseModel):
    total_images: int
    total_videos: int
    total_documents: int
    total_storage_bytes: int
    gpu_memory_used_mb: Optional[float] = None


# Video Frame Search
class VideoFrameSearchRequest(BaseModel):
    video_key: str
    query: str
    threshold: float = Field(default=0.20, ge=0.1, le=0.7, description="CLIP similarity threshold")


class MatchingFrame(BaseModel):
    frame_index: int
    timestamp: float
    caption: str
    clip_score: float
    frame_base64: str


class VideoFrameSearchResponse(BaseModel):
    success: bool
    video_key: str
    query: str
    matching_frames: List[MatchingFrame]
    total_frames_analyzed: int
