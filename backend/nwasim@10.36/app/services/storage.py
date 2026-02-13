from typing import Dict, Optional, List, Tuple
from botocore.config import Config
from botocore.exceptions import ClientError
import boto3
from app.services.local_cache import LocalCacheHandler


class S3Handler:
    """Unified S3 handler supporting both AWS S3 and DDN INFINIA storage."""

    def __init__(self, config_type: str = 'ddn_infinia', local_cache_config: Dict = None):
        """
        Initialize S3 handler.
        config_type: 'aws' or 'ddn_infinia'
        local_cache_config: Optional dict with {'enabled': bool, 'videos_path': str, 'embeddings_path': str}
        """
        self.config_type = config_type
        self.config = None
        self.client = None
        
        # Initialize local cache if enabled in config
        if local_cache_config and local_cache_config.get('enabled'):
            videos_path = local_cache_config.get('videos_path')
            embeddings_path = local_cache_config.get('embeddings_path')
            self.local_cache = LocalCacheHandler(videos_path, embeddings_path)
        else:
            self.local_cache = None

    def create_client(self) -> bool:
        """Create S3 client based on configuration type."""
        try:
            # Import here to avoid circular imports
            from app.core.config import storage_config
            
            # Validate configuration
            is_valid, message = storage_config.validate_config(self.config_type)
            if not is_valid:
                print(f"Invalid {self.config_type} configuration: {message}")
                return False

            if self.config_type == 'aws':
                self.config = storage_config.aws_config
            elif self.config_type == 'ddn_infinia':
                self.config = storage_config.ddn_infinia_config
            else:
                print(f"Unsupported config type: {self.config_type}")
                return False

            # Create the boto3 client directly
            if self.config_type == 'ddn_infinia':
                # Keep HTTPS for boto3 client - it can handle self-signed certs with verify=False
                # Only convert to HTTP in presigned URLs for browser
                boto_config = Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'},
                    retries={'max_attempts': 3},
                    connect_timeout=60,
                    read_timeout=60
                )

                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.config['access_key'],
                    aws_secret_access_key=self.config['secret_key'],
                    endpoint_url=self.config['endpoint_url'],  # Use original HTTPS endpoint
                    region_name=self.config['region'],
                    config=boto_config,
                    verify=False  # Disable SSL verification for self-signed certs
                )
                
                # Suppress SSL warnings
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            elif self.config_type == 'aws':
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=self.config['access_key'],
                    aws_secret_access_key=self.config['secret_key'],
                    region_name=self.config['region']
                )
            
            return True
        except Exception as e:
            print(f"Error creating {self.config_type} S3 client: {e}")
            return False

    def _ensure_client(self) -> bool:
        """Ensure client is initialized."""
        if not self.client:
            return self.create_client()
        return True

    def upload_bytes(self, data_bytes: bytes, object_key: str,
                     metadata: Dict[str, str] = None,
                     content_type: str = None) -> Tuple[bool, str]:
        """Upload bytes data to S3 bucket with metadata."""
        if not self._ensure_client():
            return False, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            if content_type:
                extra_args['ContentType'] = content_type

            from io import BytesIO
            self.client.upload_fileobj(
                BytesIO(data_bytes),
                bucket_name,
                object_key,
                ExtraArgs=extra_args if extra_args else None
            )
            return True, f"Successfully uploaded to {self.config['provider']}"
        except Exception as e:
            return False, f"Upload error: {e}"

    def download_bytes(self, object_key: str) -> Tuple[Optional[bytes], str]:
        """Download object as bytes from S3 bucket."""
        # Try local cache first for faster performance
        if self.local_cache and self.local_cache.is_available():
            cached_bytes, cache_msg = self.local_cache.get_video_bytes(object_key)
            if cached_bytes:
                return cached_bytes, cache_msg
        
        # Fallback to S3 if not in cache
        if not self._ensure_client():
            return None, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            from io import BytesIO
            buffer = BytesIO()
            self.client.download_fileobj(bucket_name, object_key, buffer)
            buffer.seek(0)
            return buffer.read(), f"Downloaded from {self.config['provider']}"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return None, f"Download failed ({error_code}): {e}"
        except Exception as e:
            return None, f"Download error: {e}"

    def list_objects(self, prefix: str = '', max_keys: int = 1000, include_metadata: bool = False) -> Tuple[List[Dict], str]:
        """List objects in S3 bucket with optional prefix filter."""
        # DEMO MODE: Use local cache ONLY, no S3 calls
        if self.local_cache and self.local_cache.is_available():
            print("🚀 DEMO MODE: Using local cache only")
            videos = self.local_cache.list_videos()
            
            # Fetch metadata from local cache
            for video in videos:
                metadata_dict, _ = self.local_cache.get_metadata(video['key'])
                if metadata_dict:
                    video['metadata'] = metadata_dict.get('metadata', {})
                else:
                    video['metadata'] = {}
                        
            return videos, f"Listed {len(videos)} objects from local cache"
        
        # Production mode: Use S3
        if not self._ensure_client():
            return [], "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            kwargs = {'Bucket': bucket_name, 'MaxKeys': max_keys}
            if prefix:
                kwargs['Prefix'] = prefix

            response = self.client.list_objects_v2(**kwargs)

            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    obj_dict = {
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    }
                    
                    # Optionally fetch metadata for each object
                    if include_metadata:
                        try:
                            head_response = self.client.head_object(Bucket=bucket_name, Key=obj['Key'])
                            obj_dict['metadata'] = head_response.get('Metadata', {})
                        except Exception:
                            # If metadata fetch fails, just set empty dict
                            obj_dict['metadata'] = {}
                    
                    objects.append(obj_dict)

            return objects, f"Listed {len(objects)} objects from {self.config['provider']}"
        except Exception as e:
            return [], f"List error: {e}"


    def get_object_metadata(self, object_key: str) -> Tuple[Optional[Dict], str]:
        """Get object metadata from S3."""
        # Use local cache in demo mode
        if self.local_cache and self.local_cache.is_available():
            metadata_dict, msg = self.local_cache.get_metadata(object_key)
            if metadata_dict:
                # Convert to S3-like format
                video_path = self.local_cache.get_video_path(object_key)
                if video_path:
                    import os
                    stats = os.stat(video_path)
                    return {
                        'size': stats.st_size,
                        'last_modified': None,
                        'content_type': 'video/mp4',
                        'metadata': metadata_dict.get('metadata', {})
                    }, "Retrieved from local cache"
        
        # Fallback to S3
        if not self._ensure_client():
            return None, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            response = self.client.head_object(Bucket=bucket_name, Key=object_key)
            
            metadata = {
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified').isoformat() if response.get('LastModified') else None,
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
            return metadata, "Successfully retrieved metadata"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return None, f"Metadata retrieval failed ({error_code}): {e}"
        except Exception as e:
            return None, f"Metadata error: {e}"

    def delete_object(self, object_key: str) -> Tuple[bool, str]:
        """Delete object from S3 bucket."""
        if not self._ensure_client():
            return False, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            self.client.delete_object(Bucket=bucket_name, Key=object_key)
            return True, f"Successfully deleted from {self.config['provider']}"
        except Exception as e:
            return False, f"Delete error: {e}"

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to storage provider."""
        if not self._ensure_client():
            return False, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            self.client.head_bucket(Bucket=bucket_name)
            return True, f"Successfully connected to {self.config['provider']}"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return False, f"Connection failed ({error_code}): {e}"
        except Exception as e:
            return False, f"Connection error: {e}"

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned URL for object access."""
        # In demo mode, we don't need presigned URLs (served from local cache)
        if self.local_cache and self.local_cache.is_available():
            print(f"🚀 DEMO MODE: Skipping presigned URL for {object_key}")
            return None
        
        if not self._ensure_client():
            return None

        try:
            bucket_name = self.config['bucket_name']
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            
            # Keep HTTPS for INFINIA - port 8111 only accepts HTTPS
            # Browser will show cert warning for self-signed cert, user can click "proceed anyway"
            # HTTP conversion was causing ERR_CONNECTION_RESET
            # if self.config_type == 'ddn_infinia':
            #     url = url.replace('https://', 'http://')
            
            return url
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None

    def copy_object(self, source_key: str, destination_key: str) -> Tuple[bool, str]:
        """Copy object from source to destination within the same bucket."""
        if not self._ensure_client():
            return False, "Failed to create S3 client"

        try:
            bucket_name = self.config['bucket_name']
            copy_source = {'Bucket': bucket_name, 'Key': source_key}
            
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=bucket_name,
                Key=destination_key
            )
            
            return True, f"Successfully copied {source_key} to {destination_key}"
        except Exception as e:
            return False, f"Copy error: {e}"

    def move_object(self, source_key: str, destination_key: str) -> Tuple[bool, str]:
        """Move object from source to destination (copy + delete)."""
        # First copy the object
        success, message = self.copy_object(source_key, destination_key)
        if not success:
            return False, message
        
        # Then delete the source
        delete_success, delete_message = self.delete_object(source_key)
        if not delete_success:
            return False, f"Copied but failed to delete source: {delete_message}"
        
        return True, f"Successfully moved {source_key} to {destination_key}"
    
    # ========== New Helper Methods for Video Processing ==========
    
    def upload_file(self, file_obj, object_key: str, content_type: str = None, metadata: Dict[str, str] = None, tags: Dict = None) -> Tuple[bool, str]:
        """
        Upload file object with metadata and tags.
        
        Args:
            file_obj: File object to upload
            object_key: S3 key
            content_type: Content type
            metadata: Standard S3 metadata (key-value pairs)
            tags: Custom tags for Infinia (saved as additional metadata)
        
        Returns:
            Success boolean and message
        """
        if not self._ensure_client():
            return False, "Failed to create S3 client"
        
        try:
            bucket_name = self.config['bucket_name']
            extra_args = {}
            
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Combine metadata and tags into S3 metadata
            combined_metadata = {}
            if metadata:
                combined_metadata.update(metadata)
            if tags:
                # Serialize tags as JSON string in metadata
                import json
                for key, value in tags.items():
                    if isinstance(value, (list, dict)):
                        combined_metadata[f'tag_{key}'] = json.dumps(value)
                    else:
                        combined_metadata[f'tag_{key}'] = str(value)
            
            if combined_metadata:
                extra_args['Metadata'] = combined_metadata
            
            self.client.upload_fileobj(
                file_obj,
                bucket_name,
                object_key,
                ExtraArgs=extra_args if extra_args else None
            )
            return True, f"Successfully uploaded to {object_key}"
        except Exception as e:
            return False, f"Upload error: {e}"
    
    def upload_json(self, data: dict, s3_key: str) -> Tuple[bool, str]:
        """Upload JSON data to S3."""
        import json
        try:
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            return self.upload_bytes(json_bytes, s3_key, content_type='application/json')
        except Exception as e:
            return False, f"JSON upload error: {e}"
    
    def download_json(self, s3_key: str) -> Tuple[Optional[dict], str]:
        """Download and parse JSON from S3."""
        import json
        try:
            data_bytes, msg = self.download_bytes(s3_key)
            if data_bytes:
                data = json.loads(data_bytes.decode('utf-8'))
                return data, msg
            return None, msg
        except Exception as e:
            return None, f"JSON download error: {e}"
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3 to local path."""
        if not self._ensure_client():
            return False
        
        try:
            bucket_name = self.config['bucket_name']
            self.client.download_file(bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Download error: {e}")
            return False
    
    # Artifact key generation helpers
    def get_keyframe_key(self, asset_id: str, chunk_id: int, frame_id: int) -> str:
        """Generate S3 key for keyframe."""
        return f"media/derived/keyframes/{asset_id}/chunk_{chunk_id}/frame_{frame_id}.jpg"
    
    def get_chunk_analysis_key(self, asset_id: str, chunk_id: int) -> str:
        """Generate S3 key for chunk analysis JSON."""
        return f"media/derived/chunks/{asset_id}/chunk_{chunk_id}_analysis.json"
    
    def get_embeddings_key(self, asset_id: str, version: int = 1) -> str:
        """Generate S3 key for embeddings JSON."""
        return f"media/derived/embeddings/{asset_id}_v{version}.json"
    
    def get_thumbnail_key(self, asset_id: str) -> str:
        """Generate S3 key for thumbnail."""
        return f"media/derived/thumbnails/{asset_id}/thumbnail.jpg"
