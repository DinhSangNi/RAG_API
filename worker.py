"""
Background Worker for RAG API
Processes document upload and indexing tasks from Redis queue
"""
import os
import sys
import uuid
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DocumentWorker')

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
        logger.info("Initializing DocumentWorker services...")
        
        logger.debug(f"Connecting to Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        self.queue_service = RedisQueueService(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        logger.debug("✅ RedisQueueService initialized")
        
        logger.debug(f"Initializing ChunkingService (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
        self.chunking_service = ChunkingService(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        logger.debug("✅ ChunkingService initialized")
        
        logger.debug("Initializing EmbeddingService...")
        self.embedding_service = EmbeddingService()
        logger.debug("✅ EmbeddingService initialized")
        
        logger.debug("Initializing CloudinaryService...")
        self.cloudinary_service = CloudinaryService()
        logger.debug("✅ CloudinaryService initialized")
        
        logger.debug("Initializing VietnameseSegmentationService...")
        self.segmentation_service = get_segmentation_service()
        logger.debug("✅ VietnameseSegmentationService initialized")
        
        logger.debug("Connecting to database...")
        self.db = SessionLocal()
        logger.debug("✅ Database connection established")
        
        logger.info("✅ DocumentWorker fully initialized and ready")
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """Process document upload task"""
        logger.info(f"Starting UPLOAD task - ID: {task.task_id}")
        task_start = time.time()
        
        try:
            # Step 1: Read file
            logger.info(f"[{task.task_id}] Step 1: Reading file: {task.file_name}")
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found: {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_size = os.path.getsize(task.file_path)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            logger.info(f"[{task.task_id}] ✅ File read: {file_size} bytes")
            
            # Step 2: Upload to Cloudinary
            logger.info(f"[{task.task_id}] Step 2: Uploading to Cloudinary...")
            cloudinary_result = self.cloudinary_service.upload_file(
                file_path=task.file_path,
                folder=settings.CLOUDINARY_UPLOAD_FOLDER
            )
            cloudinary_url = cloudinary_result['secure_url']
            logger.info(f"[{task.task_id}] ✅ Uploaded to Cloudinary")
            
            # Step 3: Create Document record
            logger.info(f"[{task.task_id}] Step 3: Creating document record...")
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
            logger.info(f"[{task.task_id}] ✅ Document created: {document_id}")
            
            # Step 4: Chunk content
            logger.info(f"[{task.task_id}] Step 4: Chunking document...")
            chunks_result = self.chunking_service.chunk_markdown(content, task.file_name)
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}] ✅ Created {len(chunks)} chunks")
            
            # Step 5: Embed and save chunks
            logger.info(f"[{task.task_id}] Step 5: Embedding chunks...")
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
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"[{task.task_id}] Progress: {i + 1}/{len(chunks)} chunks embedded")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Saved {chunks_created} chunks")
            
            # Step 6: Update document status
            logger.info(f"[{task.task_id}] Step 6: Updating document status...")
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document status updated to COMPLETED")
            
            # Clean up temp file
            if os.path.exists(task.file_path):
                os.remove(task.file_path)
                logger.info(f"[{task.task_id}] ✅ Temp file cleaned up")
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ UPLOAD TASK COMPLETED in {elapsed:.2f}s")
            
            return TaskResult(
                task_id=task.task_id,
                status="completed",
                document_id=document_id,
                message="Document uploaded and indexed successfully",
                chunks_created=chunks_created
            )
        
        except Exception as e:
            error_msg = f"Upload task failed: {str(e)}"
            logger.error(f"[{task.task_id}] ❌ {error_msg}", exc_info=True)
            elapsed = time.time() - task_start
            logger.error(f"[{task.task_id}] Task failed after {elapsed:.2f}s")
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def process_edit_task(self, task: EditTask) -> TaskResult:
        """Process document edit task"""
        logger.info(f"Starting EDIT task - ID: {task.task_id}, Document: {task.document_id}")
        task_start = time.time()
        
        try:
            # Step 1: Verify document exists
            logger.info(f"[{task.task_id}] Step 1: Verifying document...")
            document = self.db.query(Document).filter(Document.id == task.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {task.document_id}")
            logger.info(f"[{task.task_id}] ✅ Document found: {document.file_name}")
            
            # Step 2: Read new file
            logger.info(f"[{task.task_id}] Step 2: Reading new file...")
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()
            logger.info(f"[{task.task_id}] ✅ File read ({len(new_content)} chars)")
            
            # Step 3: Delete old chunks
            logger.info(f"[{task.task_id}] Step 3: Deleting old chunks...")
            old_chunk_count = self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).count()
            self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).delete()
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Deleted {old_chunk_count} old chunks")
            
            # Step 4: Update document metadata
            logger.info(f"[{task.task_id}] Step 4: Updating document metadata...")
            content_hash = hashlib.sha256(new_content.encode()).hexdigest()
            document.content_hash = content_hash
            document.status = "indexing"
            document.meta_data = document.meta_data or {}
            document.meta_data['last_edited'] = str(datetime.now())
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Metadata updated")
            
            # Step 5: Chunk content
            logger.info(f"[{task.task_id}] Step 5: Re-chunking document...")
            chunks_result = self.chunking_service.chunk_markdown(new_content, task.file_name)
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}] ✅ Created {len(chunks)} new chunks")
            
            # Step 6: Embed and save
            logger.info(f"[{task.task_id}] Step 6: Embedding and saving chunks...")
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
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"[{task.task_id}] Progress: {i + 1}/{len(chunks)} chunks embedded")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Saved {chunks_created} chunks")
            
            # Step 7: Update document status
            logger.info(f"[{task.task_id}] Step 7: Updating document status...")
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document status updated to COMPLETED")
            
            # Clean up
            if os.path.exists(task.file_path):
                os.remove(task.file_path)
                logger.info(f"[{task.task_id}] ✅ Temp file cleaned up")
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ EDIT TASK COMPLETED in {elapsed:.2f}s (deleted {old_chunk_count}, created {chunks_created})")
            
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
            logger.error(f"[{task.task_id}] ❌ {error_msg}", exc_info=True)
            elapsed = time.time() - task_start
            logger.error(f"[{task.task_id}] Task failed after {elapsed:.2f}s")
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def run(self):
        """Main worker loop"""
        logger.info("="*70)
        logger.info("🚀 DOCUMENT WORKER STARTING")
        logger.info("="*70)
        
        # Health check
        logger.info("Checking Redis connection...")
        if not self.queue_service.health_check():
            logger.error("❌ Redis connection failed!")
            return
        
        logger.info("✅ Redis connection successful")
        logger.info(f"Redis host: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        logger.info("⏳ Worker is ready - waiting for tasks...")
        
        task_count = 0
        idle_time = 0
        
        try:
            while True:
                # Check upload queue
                upload_data = self.queue_service.pop_upload_task()
                if upload_data:
                    task = UploadTask.from_dict(upload_data)
                    logger.info(f"📩 Received UPLOAD task from queue: {task.task_id}")
                    result = self.process_upload_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    logger.info(f"📤 Result stored for task {task.task_id} - Status: {result.status}")
                    idle_time = 0
                    continue
                
                # Check edit queue
                edit_data = self.queue_service.pop_edit_task()
                if edit_data:
                    task = EditTask.from_dict(edit_data)
                    logger.info(f"📩 Received EDIT task from queue: {task.task_id}")
                    result = self.process_edit_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    logger.info(f"📤 Result stored for task {task.task_id} - Status: {result.status}")
                    idle_time = 0
                    continue
                
                # No tasks available - log idle status periodically
                idle_time += 1
                if idle_time % 30 == 0:  # Log every 30 seconds of idle time
                    logger.debug(f"⏳ No tasks in queue (idle for {idle_time}s, processed {task_count} tasks total)")
                
                # Small sleep to avoid CPU spinning
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("="*70)
            logger.info(f"👋 WORKER STOPPING - Processed {task_count} tasks")
            logger.info("="*70)
        except Exception as e:
            logger.error("❌ Worker encountered fatal error", exc_info=True)
        finally:
            logger.info("Cleaning up database connection...")
            self.db.close()
            logger.info("✅ Worker shutdown complete")


if __name__ == "__main__":
    logger.info("Initializing database tables...")
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")
    
    logger.info("Starting DocumentWorker...")
    # Run worker
    worker = DocumentWorker()
    worker.run()
