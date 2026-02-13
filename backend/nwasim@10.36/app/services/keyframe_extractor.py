"""
Keyframe extraction service for video processing.

Extracts keyframes from video chunks at specified intervals or using scene detection.
"""

import logging
import cv2
import os
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedKeyframe:
    """Represents an extracted keyframe."""
    frame_id: int
    timestamp: float
    file_path: str
    scene_score: Optional[float] = None


class KeyframeExtractor:
    """Extracts keyframes from video chunks."""
    
    def __init__(
        self,
        fps: float = 1.0,
        enable_scene_detection: bool = False,
        min_scene_score: float = 0.3
    ):
        """
        Initialize keyframe extractor.
        
        Args:
            fps: Keyframes per second to extract (default: 1.0 = 1 frame per second)
            enable_scene_detection: Use scene detection instead of fixed FPS
            min_scene_score: Minimum scene change score for detection
        """
        self.fps = fps
        self.enable_scene_detection = enable_scene_detection
        self.min_scene_score = min_scene_score
    
    def extract_keyframes_from_chunk(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_dir: str
    ) -> List[ExtractedKeyframe]:
        """
        Extract keyframes from a specific time range in video.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save keyframes
            
        Returns:
            List of ExtractedKeyframe objects
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame positions
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            frame_interval = int(video_fps / self.fps) if self.fps > 0 else int(video_fps)
            
            keyframes = []
            frame_id = 0
            
            # Seek to start position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            
            while current_frame <= end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if this is a keyframe position
                if (current_frame - start_frame) % frame_interval == 0:
                    # Save keyframe
                    timestamp = current_frame / video_fps
                    filename = f"frame_{frame_id:04d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save frame
                    cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    keyframes.append(ExtractedKeyframe(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        file_path=filepath
                    ))
                    
                    frame_id += 1
                
                current_frame += 1
            
            cap.release()
            
            logger.info(f"✅ Extracted {len(keyframes)} keyframes from {start_time:.2f}s to {end_time:.2f}s")
            return keyframes
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
            return []
