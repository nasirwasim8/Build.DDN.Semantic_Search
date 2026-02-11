"""
Artifact manager for manifest CRUD operations.

This service handles loading/saving asset manifests from/to DDN Infinia storage.
Manifests track all metadata, processing status, and artifacts for each uploaded asset.
"""

import logging
import json
from typing import Optional
from app.models.manifest import AssetManifest
from app.services.storage import S3Handler
from app.core.config import storage_config

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages asset manifests in Infinia storage."""
    
    def __init__(self):
        """Initialize artifact manager."""
        self.storage = S3Handler('ddn_infinia', storage_config.local_cache_config)
    
    def get_manifest_key(self, asset_id: str, version: int = 1) -> str:
        """Generate S3 key for manifest."""
        return f"media/derived/manifests/{asset_id}/manifest_v{version}.json"
    
    def save_manifest(self, manifest: AssetManifest) -> bool:
        """
        Save manifest to Infinia storage.
        
        Args:
            manifest: AssetManifest to save
            
        Returns:
            bool: True if successful
        """
        try:
            manifest_key = self.get_manifest_key(manifest.asset_id, manifest.version)
            manifest_data = manifest.model_dump(mode='json')
            
            # Upload as JSON
            success, msg = self.storage.upload_json(manifest_data, manifest_key)
            
            if success:
                logger.info(f"✅ Saved manifest: {manifest_key}")
                return True
            else:
                logger.error(f"❌ Failed to save manifest: {msg}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving manifest for {manifest.asset_id}:  {e}")
            return False
    
    def load_manifest(self, asset_id: str, version: int = 1) -> Optional[AssetManifest]:
        """
        Load manifest from Infinia storage.
        
        Args:
            asset_id: Asset ID to load
            version: Manifest version (default: 1)
            
        Returns:
            AssetManifest or None if not found
        """
        try:
            manifest_key = self.get_manifest_key(asset_id, version)
            manifest_data, msg = self.storage.download_json(manifest_key)
            
            if manifest_data:
                manifest = AssetManifest(**manifest_data)
                logger.info(f"✅ Loaded manifest: {manifest_key}")
                return manifest
            else:
                logger.warning(f"⚠️ Manifest not found: {manifest_key}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading manifest for {asset_id}: {e}")
            return None
    
    def manifest_exists(self, asset_id: str, version: int = 1) -> bool:
        """Check if manifest exists."""
        manifest_key = self.get_manifest_key(asset_id, version)
        # Check if object exists in S3
        try:
            metadata, _ = self.storage.get_object_metadata(manifest_key)
            return metadata is not None
        except:
            return False


# Global instance
artifact_manager = ArtifactManager()
