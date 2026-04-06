"""
Background Worker for RAG API
Processes document upload and indexing tasks from Redis queue
"""
import os
import sys
import uuid
import hashlib
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.database.connection import engine, Base, SessionLocal
from app.database.models import Document, ChildChunk
from app.queue.service import RedisQueueService
from app.queue.models import UploadTask, EditTask, TaskResult
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.cloudinary_service import CloudinaryService
from app.services.segmentation_service import get_segmentation_service


class DocumentWorker:
    """Worker for processing document tasks"""
    
    def __init__(self):
        self.queue_service = RedisQueueService(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        self.chunking_service = ChunkingService(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_service = EmbeddingService()
        self.cloudinary_service = CloudinaryService()
        self.segmentation_service = get_segmentation_service()
        self.db = SessionLocal()
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """Process document upload task"""
        print(f"\n{'='*70}")
        print(f"📄 UPLOAD TASK: {task.task_id}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Read file
            print(f"Step 1: Reading file: {task.file_name}")
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found: {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_size = os.path.getsize(task.file_path)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            print(f"✅ File read: {file_size} bytes")
            
            # Step 2: Upload to Cloudinary
            print(f"Step 2: Uploading to Cloudinary...")
            cloudinary_result = self.cloudinary_service.upload_file(
                file_path=task.file_path,
                folder=settings.CLOUDINARY_UPLOAD_FOLDER
            )
            cloudinary_url = cloudinary_result['secure_url']
            print(f"✅ Uploaded to Cloudinary")
            
            # Step 3: Create Document record
            print(f"Step 3: Creating document record...")
            document_id = str(uuid.uuid4())
            document = Document(
                id=document_id,
                file_name=task.file_name,
                file_path=cloudinary_url,
                source_type=task.source_type,
                status="indexing",
                meta_data=task.metadata or {},
                file_size=file_size,
                content_hash=content_hash
            )
            self.db.add(document)
            self.db.commit()
            print(f"✅ Document created: {document_id}")
            
            # Step 4: Chunk content
            print(f"Step 4: Chunking document...")
            chunks_result = self.chunking_service.chunk_markdown(content, task.file_name)
            chunks = chunks_result['child_chunks']
            print(f"✅ Created {len(chunks)} chunks")
            
            # Step 5: Embed and save chunks
            print(f"Step 5: Embedding chunks...")
            chunks_created = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                # Embed
                embedding = self.embedding_service.embed_text(chunk_text)
                
                # Segment
                segments = self.segmentation_service.segment(chunk_text)
                
                # Save
                child_chunk = ChildChunk(
                    document_id=document_id,
                    content=chunk_text,
                    metadata=chunk_metadata,
                    vector=embedding,
                    bm25_text=segments,
                    h1=chunk_metadata.get('h1'),
                    h2=chunk_metadata.get('h2'),
                    h3=chunk_metadata.get('h3'),
                    chunk_index=chunk.get('chunk_index', i)
                )
                self.db.add(child_chunk)
                chunks_created += 1
            
            self.db.commit()
            print(f"✅ Saved {chunks_created} chunks")
            
            # Step 6: Update document status
            print(f"Step 6: Updating document status...")
            document.status = "completed"
            self.db.commit()
            print(f"✅ Document status updated")
            
            # Clean up temp file
            if os.path.exists(task.file_path):
                os.remove(task.file_path)
                print(f"✅ Temp file cleaned up")
            
            print(f"\n{'='*70}")
            print(f"✅ UPLOAD TASK COMPLETED")
            print(f"{'='*70}\n")
            
            return TaskResult(
                task_id=task.task_id,
                status="completed",
                document_id=document_id,
                message="Document uploaded and indexed successfully",
                chunks_created=chunks_created
            )
        
        except Exception as e:
            error_msg = f"Upload task failed: {str(e)}"
            print(f"❌ {error_msg}")
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def process_edit_task(self, task: EditTask) -> TaskResult:
        """Process document edit task"""
        print(f"\n{'='*70}")
        print(f"✏️ EDIT TASK: {task.task_id}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Verify document exists
            print(f"Step 1: Verifying document...")
            document = self.db.query(Document).filter(Document.id == task.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {task.document_id}")
            print(f"✅ Document found: {document.file_name}")
            
            # Step 2: Read new file
            print(f"Step 2: Reading new file...")
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()
            print(f"✅ File read ({len(new_content)} chars)")
            
            # Step 3: Delete old chunks
            print(f"Step 3: Deleting old chunks...")
            old_chunk_count = self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).count()
            self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).delete()
            self.db.commit()
            print(f"✅ Deleted {old_chunk_count} old chunks")
            
            # Step 4: Update document metadata
            print(f"Step 4: Updating document metadata...")
            content_hash = hashlib.sha256(new_content.encode()).hexdigest()
            document.content_hash = content_hash
            document.status = "indexing"
            document.meta_data = document.meta_data or {}
            document.meta_data['last_edited'] = str(datetime.now())
            self.db.commit()
            print(f"✅ Metadata updated")
            
            # Step 5: Chunk content
            print(f"Step 5: Re-chunking document...")
            chunks_result = self.chunking_service.chunk_markdown(new_content, task.file_name)
            chunks = chunks_result['child_chunks']
            print(f"✅ Created {len(chunks)} new chunks")
            
            # Step 6: Embed and save
            print(f"Step 6: Embedding and saving chunks...")
            chunks_created = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                embedding = self.embedding_service.embed_text(chunk_text)
                segments = self.segmentation_service.segment(chunk_text)
                
                child_chunk = ChildChunk(
                    document_id=task.document_id,
                    content=chunk_text,
                    metadata=chunk_metadata,
                    vector=embedding,
                    bm25_text=segments,
                    h1=chunk_metadata.get('h1'),
                    h2=chunk_metadata.get('h2'),
                    h3=chunk_metadata.get('h3'),
                    chunk_index=chunk.get('chunk_index', i)
                )
                self.db.add(child_chunk)
                chunks_created += 1
            
            self.db.commit()
            print(f"✅ Saved {chunks_created} chunks")
            
            # Step 7: Update document status
            print(f"Step 7: Updating document status...")
            document.status = "completed"
            self.db.commit()
            print(f"✅ Document status updated")
            
            # Clean up
            if os.path.exists(task.file_path):
                os.remove(task.file_path)
                print(f"✅ Temp file cleaned up")
            
            print(f"\n{'='*70}")
            print(f"✅ EDIT TASK COMPLETED")
            print(f"{'='*70}\n")
            
            return TaskResult(
                task_id=task.task_id,
                status="completed",
                document_id=task.document_id,
                message="Document updated and re-indexed successfully",
                chunks_created=chunks_created,
                chunks_deleted=old_chunk_count
            )
        
        except Exception as e:
            error_msg = f"Edit task failed: {str(e)}"
            print(f"❌ {error_msg}")
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def run(self):
        """Main worker loop"""
        print(f"\n{'='*70}")
        print(f"🚀 DOCUMENT WORKER STARTED")
        print(f"{'='*70}\n")
        
        # Health check
        if not self.queue_service.health_check():
            print("❌ Redis connection failed!")
            return
        
        print("✅ Redis connection successful")
        print("⏳ Waiting for tasks...\n")
        
        try:
            while True:
                # Check upload queue
                upload_data = self.queue_service.pop_upload_task()
                if upload_data:
                    task = UploadTask.from_dict(upload_data)
                    result = self.process_upload_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    continue
                
                # Check edit queue
                edit_data = self.queue_service.pop_edit_task()
                if edit_data:
                    task = EditTask.from_dict(edit_data)
                    result = self.process_edit_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    continue
        
        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print(f"👋 WORKER STOPPED")
            print(f"{'='*70}\n")
        finally:
            self.db.close()


if __name__ == "__main__":
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Run worker
    worker = DocumentWorker()
    worker.run()
