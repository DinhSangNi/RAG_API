"""
Cloudinary Upload Service
Handles file uploads to Cloudinary cloud storage
"""

import cloudinary
import cloudinary.uploader
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import settings


class CloudinaryService:
    """
    Service for uploading files to Cloudinary
    """

    def __init__(self):
        """Initialize Cloudinary with API credentials"""
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            timeout=60  # Set timeout to 60 seconds
        )

    def upload_file(
        self,
        file_path: str,
        public_id: Optional[str] = None,
        resource_type: str = "raw",
        folder: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Upload a file to Cloudinary using stream upload with retry logic

        Args:
            file_path: Local path to file to upload
            public_id: Optional custom public ID for the file
            resource_type: Type of resource (raw, image, video, auto)
            folder: Folder path in Cloudinary
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Upload result dictionary with:
            {
                'url': CDN URL of the file,
                'secure_url': HTTPS URL,
                'public_id': Public ID,
                'resource_type': Type of resource,
                'size': File size in bytes,
                'created_at': Upload timestamp
            }

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If upload fails after all retries
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = Path(file_path).stem
        upload_folder = folder or settings.CLOUDINARY_UPLOAD_FOLDER

        # Build public ID with folder
        if public_id is None:
            public_id = f"{upload_folder}/{file_name}"
        else:
            public_id = f"{upload_folder}/{public_id}"

        print(f"📤 Uploading to Cloudinary (stream): {file_name}")
        print(f"   Path: {file_path}")
        print(f"   Public ID: {public_id}")
        print(f"   Max retries: {max_retries}")

        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Stream upload: open file and pass file object
                with open(file_path, 'rb') as file_stream:
                    result = cloudinary.uploader.upload(
                        file_stream,  # Pass file object for streaming
                        public_id=public_id,
                        resource_type=resource_type,
                        overwrite=True,
                        unique_filename=False,
                        timeout=60  # 60 second timeout per upload
                    )

                print(f"✅ Stream upload successful!")
                print(f"   URL: {result['url']}")
                print(f"   Size: {result['bytes']} bytes")

                return {
                    'url': result['url'],
                    'secure_url': result.get('secure_url', result['url']),
                    'public_id': result['public_id'],
                    'resource_type': result['resource_type'],
                    'size': result['bytes'],
                    'created_at': result['created_at'],
                    'cloudinary_id': result['version']
                }

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff: 5s, 10s, 15s
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}")
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Failed to upload file to Cloudinary after {max_retries + 1} attempts: {str(e)}"
                    print(f"❌ {error_msg}")
                    raise ValueError(error_msg)
            
            except Exception as e:
                error_msg = f"Failed to upload file to Cloudinary: {str(e)}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
        
        # Should not reach here, but just in case
        error_msg = f"Failed to upload file to Cloudinary: {str(last_exception)}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
