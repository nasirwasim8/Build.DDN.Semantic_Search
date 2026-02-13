"""
Video chunking service for segmenting videos into processable chunks.

This service handles video metadata extraction and chunk calculation
for parallel processing of large video files.
"""

import logging
import cv2
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoChunk:
    """Represents a video chunk for processing."""
    chunk_id: int
    start_time: float
    end_time: float
    duration: float


class VideoChunker:
    """Handles video chunking operations."""
    
    def __init__(self, chunk_duration: float = 10.0):
        """
        Initialize video chunker.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 10s)
        """
        self.chunk_duration = chunk_duration
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract video metadata using OpenCV.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with video properties or None on error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
            
            logger.info(f"📹 Video info: {width}x{height} @ {fps:.2f}fps, duration: {duration:.2f}s")
            return info
            
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return None
    
    def calculate_chunks(self, duration: float) -> List[VideoChunk]:
        """
        Calculate chunk boundaries for a video.
        
        Args:
            duration: Total video duration in seconds
            
        Returns:
            List of VideoChunk objects
        """
        chunks = []
        chunk_id = 0
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + self.chunk_duration, duration)
            chunk_duration = end_time - current_time
            
            chunks.append(VideoChunk(
                chunk_id=chunk_id,
                start_time=current_time,
                end_time=end_time,
                duration=chunk_duration
            ))
            
            current_time = end_time
            chunk_id += 1
        
        logger.info(f"📊 Calculated {len(chunks)} chunks @ {self.chunk_duration}s each")
        return chunks
