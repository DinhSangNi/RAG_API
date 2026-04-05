"""
Cloudinary Upload Service
Handles file uploads to Cloudinary cloud storage
"""

import cloudinary
import cloudinary.uploader
import os
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
            api_secret=settings.CLOUDINARY_API_SECRET
        )

    def upload_file(
        self,
        file_path: str,
        public_id: Optional[str] = None,
        resource_type: str = "raw",
        folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Cloudinary

        Args:
            file_path: Local path to file to upload
            public_id: Optional custom public ID for the file
            resource_type: Type of resource (raw, image, video, auto)
            folder: Folder path in Cloudinary

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
            ValueError: If upload fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            file_name = Path(file_path).stem
            upload_folder = folder or settings.CLOUDINARY_UPLOAD_FOLDER

            # Build public ID with folder
            if public_id is None:
                public_id = f"{upload_folder}/{file_name}"
            else:
                public_id = f"{upload_folder}/{public_id}"

            print(f"📤 Uploading to Cloudinary: {file_name}")
            print(f"   Path: {file_path}")
            print(f"   Public ID: {public_id}")

            result = cloudinary.uploader.upload(
                file_path,
                public_id=public_id,
                resource_type=resource_type,
                overwrite=True,
                unique_filename=False
            )

            print(f"✅ Upload successful!")
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

        except Exception as e:
            error_msg = f"Failed to upload file to Cloudinary: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)

    def delete_file(self, public_id: str, resource_type: str = "raw") -> bool:
        """
        Delete a file from Cloudinary

        Args:
            public_id: Public ID of the file to delete
            resource_type: Type of resource

        Returns:
            True if deletion successful

        Raises:
            ValueError: If deletion fails
        """
        try:
            print(f"🗑️  Deleting from Cloudinary: {public_id}")
            result = cloudinary.uploader.destroy(
                public_id,
                resource_type=resource_type
            )

            if result.get('result') == 'ok':
                print(f"✅ Deletion successful!")
                return True
            else:
                raise ValueError(f"Deletion failed with result: {result}")

        except Exception as e:
            error_msg = f"Failed to delete file from Cloudinary: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)

    def get_file_info(self, public_id: str, resource_type: str = "raw") -> Dict[str, Any]:
        """
        Get information about a file in Cloudinary

        Args:
            public_id: Public ID of the file
            resource_type: Type of resource

        Returns:
            File information dictionary

        Raises:
            ValueError: If retrieval fails
        """
        try:
            result = cloudinary.api.resource(
                public_id,
                resource_type=resource_type
            )
            return result
        except Exception as e:
            error_msg = f"Failed to get file info from Cloudinary: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)


# Singleton instance
_cloudinary_service: Optional[CloudinaryService] = None


def get_cloudinary_service() -> CloudinaryService:
    """Get or create Cloudinary service instance"""
    global _cloudinary_service
    if _cloudinary_service is None:
        _cloudinary_service = CloudinaryService()
    return _cloudinary_service
