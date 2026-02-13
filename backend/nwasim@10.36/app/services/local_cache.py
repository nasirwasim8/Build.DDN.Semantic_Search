"""
Local disk cache handler for faster demo performance.
Falls back to S3 if files not found locally.
"""
import os
import json
from typing import Tuple, Optional, List, Dict
from pathlib import Path

class LocalCacheHandler:
    """Handle local disk cache for videos and embeddings."""
    
    def __init__(self, videos_path: str = None, embeddings_path: str = None):
        """Initialize local cache handler with custom or default paths."""
        if videos_path and embeddings_path:
            # Use provided custom paths
            self.videos_dir = Path(videos_path)
            self.embeddings_dir = Path(embeddings_path)
            self.cache_dir = self.videos_dir.parent
        else:
            # Default to cache directory in project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.cache_dir = project_root / "cache"
            self.videos_dir = self.cache_dir / "videos"
            self.embeddings_dir = self.cache_dir / "embeddings"
        
        # Check if cache directories exist
        self.cache_available = (
            self.videos_dir.exists() and
            self.embeddings_dir.exists()
        )
        
        if self.cache_available:
            print(f"✅ Local cache available at: {self.cache_dir}")
        else:
            print(f"⚠️  Local cache not found at: {self.cache_dir}")
    
    def get_video_path(self, object_key: str) -> Optional[Path]:
        """Get local path to video file if it exists."""
        if not self.cache_available:
            return None
        
        # Extract filename from object_key (e.g., "videos/20251215_191541_Jensen_Alex.mp4")
        filename = Path(object_key).name
        
        # Check if file exists in cache
        video_path = self.videos_dir / filename
        if video_path.exists():
            print(f"📁 Cache HIT: {filename}")
            return video_path
        
        print(f"❌ Cache MISS: {filename}")
        return None
    
    def get_video_bytes(self, object_key: str) -> Tuple[Optional[bytes], str]:
        """Read video bytes from local cache."""
        video_path = self.get_video_path(object_key)
        
        if video_path:
            try:
                with open(video_path, 'rb') as f:
                    return f.read(), f"Loaded from local cache"
            except Exception as e:
                return None, f"Error reading from cache: {e}"
        
        return None, "Not found in local cache"
    
    def get_metadata(self, object_key: str) -> Tuple[Optional[Dict], str]:
        """Get metadata from JSON sidecar file."""
        if not self.cache_available:
            return None, "Cache not available"
        
        # Extract filename and look for .json file
        filename = Path(object_key).name
        json_path = self.embeddings_dir / f"{filename}.json"
        
        # For demo: create simple metadata from filename
        # The JSON files contain embeddings, not metadata
        # We'll infer metadata from the filename
        video_name = filename.replace('.mp4', '').split('_', 2)[-1]  # e.g., "Jensen_Alex"
        
        metadata = {
            'metadata': {
                'modality': 'video',
                'summary': f'Video: {video_name}',
                'tags': video_name.replace('_', ', '),
                'detected_objects': video_name.replace('_', ', '),
                'source': 'local_cache'
            }
        }
        
        # Check if embeddings file exists to confirm video has been processed
        if json_path.exists():
            return metadata, "Generated metadata from cache"
        
        return metadata, "Generated default metadata"

    
    def list_videos(self) -> List[Dict]:
        """List all videos in local cache with metadata."""
        if not self.cache_available:
            return []
        
        videos = []
        
        # Get all MP4 files in videos directory
        for video_file in self.videos_dir.glob("*.mp4"):
            # Generate metadata from filename
            video_name = video_file.name.replace('.mp4', '').split('_', 2)[-1]
            
            metadata = {
                'modality': 'video',
                'summary': f'Video: {video_name}',
                'tags': video_name.replace('_', ', '),
                'detected_objects': video_name.replace('_', ', '),
                'source': 'local_cache'
            }
            
            # Create object info with proper timestamp format
            file_stats = video_file.stat()
            from datetime import datetime
            
            videos.append({
                'key': f"videos/{video_file.name}",
                'size': file_stats.st_size,
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'metadata': metadata
            })
        
        return videos
    
    def is_available(self) -> bool:
        """Check if local cache is available."""
        return self.cache_available
