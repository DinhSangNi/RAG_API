"""
Queue Job Handler for Document Processing
Handles file upload to Cloudinary and indexing workflow
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database.models import Document
from app.services.cloudinary_service import get_cloudinary_service
from app.workers.process_worker import process_document


def index_document_job(
    temp_file_path: str,
    source_type: str = "local",
    file_name: Optional[str] = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Queue job: Upload file to Cloudinary and process document through indexing workflow

    Workflow:
    1. Verify temp file exists
    2. Upload to Cloudinary
    3. Create Document record in DB
    4. Run indexing: chunking, embedding, segmentation
    5. Save to database
    6. Cleanup temp file

    Args:
        temp_file_path: Path to file in temp storage
        source_type: Type of source (local, cloud, wikipedia)
        file_name: Optional custom file name (uses basename if not provided)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata dictionary

    Returns:
        Job result with:
        {
            'status': 'completed' | 'failed',
            'document_id': UUID of created document,
            'file_name': Name of file,
            'cloudinary_url': URL to Cloudinary file,
            'message': Status message,
            'error': Error message if failed,
            'timing': Processing timing stats
        }
    """
    job_id = str(uuid.uuid4())
    result = {
        'job_id': job_id,
        'status': 'initiated',
        'temp_file_path': temp_file_path,
    }

    # Setup database session
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=2,
        max_overflow=28
    )
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        print(f"\n{'='*70}")
        print(f"🚀 QUEUE JOB STARTED: {job_id}")
        print(f"{'='*70}")
        print(f"📁 Temp file: {temp_file_path}")
        print(f"📝 File name: {file_name or Path(temp_file_path).name}")
        
        # ====================================================================
        # STEP 1: Verify temp file exists
        # ====================================================================
        if not os.path.exists(temp_file_path):
            raise FileNotFoundError(f"Temp file not found: {temp_file_path}")

        print(f"✅ Temp file verified")

        # ====================================================================
        # STEP 2: Upload to Cloudinary
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"☁️  STEP 1: UPLOAD TO CLOUDINARY")
        print(f"{'='*70}")

        start_time = time.time()
        cloudinary_service = get_cloudinary_service()
        
        upload_result = cloudinary_service.upload_file(
            file_path=temp_file_path,
            resource_type="raw"
        )
        
        upload_duration = time.time() - start_time
        print(f"⏱️  Cloudinary upload took: {upload_duration:.2f}s")

        # ====================================================================
        # STEP 3: Create Document record in database
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"📋 STEP 2: CREATE DOCUMENT RECORD")
        print(f"{'='*70}")

        document_id = str(uuid.uuid4())
        file_name = file_name or Path(temp_file_path).name
        
        # Get file size
        file_size = os.path.getsize(temp_file_path)
        
        # Create document record
        doc_metadata = metadata or {}
        doc_metadata.update({
            'cloudinary_url': upload_result['url'],
            'cloudinary_secure_url': upload_result['secure_url'],
            'cloudinary_public_id': upload_result['public_id'],
            'original_file_name': file_name,
            'file_size': file_size,
            'uploaded_at': upload_result['created_at']
        })

        document = Document(
            id=document_id,
            file_path=upload_result['secure_url'],  # Store Cloudinary URL
            file_name=file_name,
            source_type=source_type,
            status="queued",
            meta_data=doc_metadata
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)

        print(f"✅ Document record created: {document_id}")
        print(f"   File name: {file_name}")
        print(f"   File size: {file_size} bytes")
        print(f"   Cloudinary URL: {upload_result['url']}")

        result['document_id'] = document_id
        result['file_name'] = file_name
        result['cloudinary_url'] = upload_result['url']
        result['file_size'] = file_size

        # ====================================================================
        # STEP 4: Run indexing workflow
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"🔄 STEP 3: DOCUMENT INDEXING WORKFLOW")
        print(f"{'='*70}")

        # Call the existing process_document function
        # Note: We pass the temp_file_path since it's still available
        indexing_result = process_document(
            document_id=document_id,
            file_path=temp_file_path,  # Use original temp file for content processing
            source_type=source_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_id=job_id,
            is_summary=False
        )

        print(f"✅ Indexing workflow completed")
        result['indexing_result'] = indexing_result
        result['timing'] = indexing_result.get('timing', {})

        # Update document status to completed
        document.status = "completed"
        db.commit()

        # ====================================================================
        # STEP 5: Cleanup temp file
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"🧹 STEP 4: CLEANUP")
        print(f"{'='*70}")

        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"✅ Removed temp file: {temp_file_path}")
        except Exception as e:
            print(f"⚠️ Failed to remove temp file: {e}")

        # ====================================================================
        # Final result
        # ====================================================================
        result['status'] = 'completed'
        result['message'] = f'Successfully processed document: {file_name}'

        total_duration = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ JOB COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"📊 Job ID: {job_id}")
        print(f"📊 Document ID: {document_id}")
        print(f"📊 File: {file_name}")
        print(f"⏱️  TOTAL TIME: {total_duration:.2f}s")
        print(f"{'='*70}\n")

        return result

    except Exception as e:
        error_msg = f"Job failed: {str(e)}"
        print(f"\n❌ {error_msg}")
        result['status'] = 'failed'
        result['error'] = error_msg

        # Update document status to failed if created
        try:
            if 'document_id' in result:
                doc = db.query(Document).filter(
                    Document.id == result['document_id']
                ).first()
                if doc:
                    doc.status = "failed"
                    doc.meta_data = doc.meta_data or {}
                    doc.meta_data['error'] = error_msg
                    db.commit()
        except Exception as db_error:
            print(f"⚠️ Failed to update document status: {db_error}")

        # Cleanup temp file even on error
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"🧹 Cleaned up temp file on error")
        except Exception as cleanup_error:
            print(f"⚠️ Failed to cleanup temp file: {cleanup_error}")

        return result

    finally:
        # Ensure database session is closed
        db.close()
