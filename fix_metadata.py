#!/usr/bin/env python3
"""Script to add include_metadata parameter to list_objects method"""
import sys

# Read the current storage.py file
with open('backend/app/services/storage.py', 'r') as f:
    lines = f.readlines()

# Find and replace the list_objects method (lines 128-153)
new_method = '''    def list_objects(self, prefix: str = '', max_keys: int = 1000, include_metadata: bool = False) -> Tuple[List[Dict], str]:
        """List objects in S3 bucket with optional prefix filter."""
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

'''

# Replace lines 127-153 (0-indexed: 127-153)
new_lines = lines[:127] + [new_method] + lines[153:]

# Write back
with open('backend/app/services/storage.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Updated storage.py with include_metadata parameter")
