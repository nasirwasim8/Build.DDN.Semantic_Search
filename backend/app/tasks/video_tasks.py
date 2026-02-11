"""
Celery tasks for video processing.

This module contains the main video processing task that orchestrates:
- Video download and metadata extraction
- Chunk-based video processing
- Keyframe extraction and AI analysis
- Metadata generation and storage in DDN Infinia
"""

import logging
import os
import tempfile
import time
import uuid
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from celery import Task
from PIL import Image
import io

from app.celery_app import celery_app
from app.core.config import settings, storage_config
from app.services.storage import S3Handler
from app.services.video_chunker import VideoChunker
from app.services.keyframe_extractor import KeyframeExtractor
from app.services.artifact_manager import artifact_manager
from app.services.ai_models import get_image_analyzer
from app.models.manifest import (
    AssetManifest,
    ChunkAnalysis,
    KeyframeMetadata,
    StructuredTags
)

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks for progress updates."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"✅ Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"❌ Task {task_id} failed: {exc}")


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="app.tasks.video_tasks.process_video_task",
    queue="video_processing",
    max_retries=2,
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=4000  # Kill after 4000 seconds
)
def process_video_task(
    self,
    asset_id: str,
    s3_key: str,
    custom_summary: str = "",
    custom_tags: str = "",
    gpu_id: int = 0
) -> Dict[str, Any]:
    """
    Main task for processing a video asset.
    
    Args:
        asset_id: Unique asset ID
        s3_key: S3 key of raw video
        custom_summary: User-provided summary
        custom_tags: User-provided tags (comma-separated)
        gpu_id: GPU device ID to use (0 for CPU if no GPUs)
    
    Returns:
        Dict with processing results and statistics
    """
    start_time = time.time()
    temp_video_path = None
    temp_dir = None
    
    try:
        # Update manifest status to processing
        manifest = artifact_manager.load_manifest(asset_id)
        if not manifest:
            raise ValueError(f"Manifest not found for asset {asset_id}")
        
        manifest.processing_status = "processing"
        manifest.processing_timestamp = datetime.utcnow()
        artifact_manager.save_manifest(manifest)
        
        logger.info(f"🎬 Processing video {asset_id} on GPU {gpu_id}")
        
        # Step 1: Download video from Infinia to temp file
        storage = S3Handler('ddn_infinia', storage_config.local_cache_config)
        video_bytes, msg = storage.download_bytes(s3_key)
        
        if not video_bytes:
            raise Exception(f"Failed to  download video: {msg}")
        
        # Save to temp file
        suffix = Path(s3_key).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            temp_video_path = tmp.name
        
        logger.info(f"📥 Downloaded video to {temp_video_path}")
        
        # Step 2: Extract video metadata
        chunker = VideoChunker(chunk_duration=settings.VIDEO_CHUNK_DURATION)
        video_info = chunker.get_video_info(temp_video_path)
        
        if not video_info:
            raise Exception("Failed to extract video metadata")
        
        # Update manifest with video properties
        manifest.width = video_info['width']
        manifest.height = video_info['height']
        manifest.fps = video_info['fps']
        manifest.duration_seconds = video_info['duration']
        
        # Step 3: Calculate chunks
        chunks = chunker.calculate_chunks(video_info['duration'])
        manifest.total_chunks = len(chunks)
        
        logger.info(f"📊 Video: {video_info['width']}x{video_info['height']}, {video_info['duration']:.2f}s, {len(chunks)} chunks")
        
        # Create temp dir for keyframes
        temp_dir = tempfile.mkdtemp()
        
        # Step 4: Process each chunk
        extractor = KeyframeExtractor(fps=settings.KEYFRAME_FPS)
        image_analyzer = get_image_analyzer()
        
        all_captions = []
        all_detected_objects = []
        
        for chunk in chunks:
            chunk_result = _process_video_chunk_inline(
                video_path=temp_video_path,
                asset_id=asset_id,
                chunk=chunk,
                temp_dir=temp_dir,
                extractor=extractor,
                image_analyzer=image_analyzer,
                storage=storage
            )
            
            if chunk_result:
                manifest.chunks.append(ChunkAnalysis(**chunk_result))
                
                # Collect metadata for summary
                for kf in chunk_result.get('keyframes', []):
                    if kf.get('caption'):
                        all_captions.append(kf['caption'])
                    if kf.get('tags') and kf['tags'].get('objects'):
                        all_detected_objects.extend(kf['tags']['objects'])
        
        # Step 5: Generate video summary and detected objects
        if not custom_summary and all_captions:
            # Use first few captions as summary
            manifest.video_summary = ". ".join(all_captions[:3]) + "."
        else:
            manifest.video_summary = custom_summary
        
        if not custom_tags:
            # Aggregate detected objects
            unique_objects = list(set(all_detected_objects))[:10]  # Top 10
            manifest.detected_objects = ", ".join(unique_objects)
        else:
            manifest.detected_objects = custom_tags
        
        manifest.custom_summary = custom_summary
        manifest.custom_tags = custom_tags
        
        # Step 6: Update statistics
        manifest.update_statistics()
        manifest.processing_status = "completed"
        manifest.processing_timestamp = datetime.utcnow()
        
        # Save final manifest
        artifact_manager.save_manifest(manifest)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Completed processing {asset_id} in {processing_time:.2f}s")
        logger.info(f"   Total keyframes: {manifest.total_keyframes}, chunks: {manifest.total_chunks}")
        
        return {
            'success': True,
            'asset_id': asset_id,
            'total_chunks': manifest.total_chunks,
            'total_keyframes': manifest.total_keyframes,
            'processing_time_seconds': processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing video {asset_id}: {e}", exc_info=True)
        
        # Update manifest with error
        try:
            manifest = artifact_manager.load_manifest(asset_id)
            if manifest:
                manifest.processing_status = "failed"
                manifest.processing_error = str(e)
                manifest.processing_timestamp = datetime.utcnow()
                artifact_manager.save_manifest(manifest)
        except:
            pass
        
        raise
        
    finally:
        # Cleanup temp files
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass


def _process_video_chunk_inline(
    video_path: str,
    asset_id: str,
    chunk,
    temp_dir: str,
    extractor: KeyframeExtractor,
    image_analyzer,
    storage: S3Handler
) -> Dict[str, Any]:
    """
    Process a single video chunk inline (not as separate task).
    
    Args:
        video_path: Path to video file
        asset_id: Asset ID
        chunk: VideoChunk object
        temp_dir: Temp directory for keyframes
        extractor: KeyframeExtractor instance
        image_analyzer: ImageAnalyzer instance
        storage: S3Handler instance
    
    Returns:
        Dict with chunk analysis results
    """
    try:
        chunk_start_time = time.time()
        
        logger.info(f"  📹 Processing chunk {chunk.chunk_id}: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s")
        
        # Extract keyframes
        chunk_temp_dir = os.path.join(temp_dir, f"chunk_{chunk.chunk_id}")
        os.makedirs(chunk_temp_dir, exist_ok=True)
        
        keyframes = extractor.extract_keyframes_from_chunk(
            video_path,
            chunk.start_time,
            chunk.end_time,
            chunk_temp_dir
        )
        
        if not keyframes:
            logger.warning(f"  ⚠️ No keyframes extracted for chunk {chunk.chunk_id}")
            return None
        
        # Process keyframes in batches
        keyframe_metadata_list = []
        all_objects = []
        all_captions = []
        
        batch_size = settings.BATCH_SIZE
        for i in range(0, len(keyframes), batch_size):
            batch = keyframes[i:i + batch_size]
            
            # Load images
            images = []
            for kf in batch:
                img = Image.open(kf.file_path).convert('RGB')
                images.append(img)
            
            # Batch AI analysis
            embeddings = image_analyzer.generate_embeddings_batch(images)
            captions = image_analyzer.generate_captions_batch(images)
            
            # Process each keyframe in batch
            for idx, kf in enumerate(batch):
                frame_id = f"{asset_id}_chunk{chunk.chunk_id}_frame{kf.frame_id}"
                
                # Upload keyframe to Infinia
                keyframe_s3_key = storage.get_keyframe_key(asset_id, chunk.chunk_id, kf.frame_id)
                
                with open(kf.file_path, 'rb') as f:
                    keyframe_bytes = f.read()
                
                metadata = {
                    'asset_id': asset_id,
                    'chunk_id': str(chunk.chunk_id),
                    'frame_id': str(kf.frame_id),
                    'timestamp': str(kf.timestamp),
                    'caption': captions[idx] if idx < len(captions) else ""
                }
                
                storage.upload_bytes(keyframe_bytes, keyframe_s3_key, metadata, 'image/jpeg')
                
                # Create keyframe metadata
                # Simple tag extraction from caption
                caption_text = captions[idx] if idx < len(captions) else ""
                detected_objects = _extract_objects_from_caption(caption_text)
                
                structured_tags = StructuredTags(
                    objects=detected_objects,
                    actions=[],
                    scenes=[],
                    safety=[],
                    people_count=0
                )
                
                keyframe_meta = KeyframeMetadata(
                    frame_id=frame_id,
                    timestamp=kf.timestamp,
                    s3_key=keyframe_s3_key,
                    embedding_id=f"{frame_id}_emb",
                    caption=caption_text,
                    tags=structured_tags,
                    confidence_score=1.0
                )
                
                keyframe_metadata_list.append(keyframe_meta.model_dump())
                
                all_captions.append(caption_text)
                all_objects.extend(detected_objects)
        
        # Save embeddings to JSON (if needed for search)
        # TODO: Integrate with FAISS or vector DB
        
        # Create chunk analysis
        chunk_analysis = {
            'chunk_id': chunk.chunk_id,
            'start_time': chunk.start_time,
            'end_time': chunk.end_time,
            'duration': chunk.duration,
            'keyframes': keyframe_metadata_list,
            'total_keyframes': len(keyframe_metadata_list),
            'dominant_tags': StructuredTags(
                objects=list(set(all_objects))[:5],  # Top 5 unique objects
                actions=[],
                scenes=[],
                safety=[]
            ).model_dump(),
            'summary_caption': ". ".join(all_captions[:2]) + "." if all_captions else "",
            's3_key': storage.get_chunk_analysis_key(asset_id, chunk.chunk_id),
            'processing_time_ms': (time.time() - chunk_start_time) * 1000
        }
        
        # Upload chunk analysis JSON
        storage.upload_json(chunk_analysis, chunk_analysis['s3_key'])
        
        logger.info(f"  ✅ Chunk {chunk.chunk_id} processed: {len(keyframe_metadata_list)} keyframes")
        
        return chunk_analysis
        
    except Exception as e:
        logger.error(f"  ❌ Error processing chunk {chunk.chunk_id}: {e}", exc_info=True)
        return None


def _extract_objects_from_caption(caption: str) -> List[str]:
    """Extract likely object names from caption text."""
    # Simple noun extraction (this is basic - could use NLP)
    # Common objects to look for
    common_objects = [
        'person', 'people', 'man', 'woman', 'child', 'dog', 'cat',
        'car', 'bike', 'motorcycle', 'tree', 'building', 'street',
        'sky', 'water', 'mountain', 'beach', 'grass', 'flower',
        'table', 'chair', 'sofa', 'bed', 'computer', 'phone'
    ]
    
    caption_lower = caption.lower()
    detected = [obj for obj in common_objects if obj in caption_lower]
    
    return detected[:5]  # Max 5 objects
