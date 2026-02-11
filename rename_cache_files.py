#!/usr/bin/env python3
"""
Rename all cache files to random UUIDs to prevent filename-based search impression during demos.
This script:
1. Generates random UUIDs for all video files
2. Renames videos and their corresponding embeddings
3. Creates a mapping file for reference
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime

# Directories
VIDEOS_DIR = Path("cache/videos")
EMBEDDINGS_DIR = Path("cache/embeddings")
MAPPING_FILE = Path("cache/filename_mapping.json")

def generate_random_id():
    """Generate a short random identifier (8 characters)"""
    return str(uuid.uuid4())[:8]

def main():
    print("🔄 Starting file renaming process...")
    print("=" * 60)
    
    # Storage for mappings
    mapping = {
        "renamed_at": datetime.now().isoformat(),
        "files": []
    }
    
    # Get all video files
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"📹 Found {len(video_files)} video files\n")
    
    renamed_count = 0
    
    for video_path in video_files:
        old_name = video_path.name
        
        # Generate new random name
        new_name = f"{generate_random_id()}.mp4"
        new_video_path = VIDEOS_DIR / new_name
        
        # Find corresponding embedding file
        # Embedding files are named as: {video_filename}.json
        embedding_pattern = f"{old_name}.json"
        old_embedding_path = EMBEDDINGS_DIR / embedding_pattern
        new_embedding_path = EMBEDDINGS_DIR / f"{new_name}.json"
        
        # Check if files exist before renaming
        video_exists = video_path.exists()
        embedding_exists = old_embedding_path.exists()
        
        if not video_exists:
            print(f"⚠️  Video not found: {old_name}")
            continue
        
        # Rename video file
        try:
            video_path.rename(new_video_path)
            print(f"✅ Video: {old_name} -> {new_name}")
            
            # Rename embedding file if it exists
            if embedding_exists:
                old_embedding_path.rename(new_embedding_path)
                print(f"   📊 Embedding: {embedding_pattern} -> {new_name}.json")
            else:
                print(f"   ⚠️  No embedding found for {old_name}")
            
            # Store mapping
            mapping["files"].append({
                "old_video": old_name,
                "new_video": new_name,
                "old_embedding": embedding_pattern if embedding_exists else None,
                "new_embedding": f"{new_name}.json" if embedding_exists else None,
                "had_embedding": embedding_exists
            })
            
            renamed_count += 1
            print()
            
        except Exception as e:
            print(f"❌ Error renaming {old_name}: {e}\n")
    
    # Save mapping file
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("=" * 60)
    print(f"✅ Renamed {renamed_count} video files")
    print(f"📝 Mapping saved to: {MAPPING_FILE}")
    print("\n🎯 Cache files now have random names - search will show true semantic capability!")

if __name__ == "__main__":
    main()
