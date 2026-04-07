"""
Background Worker for RAG API
Processes document upload and indexing tasks from Redis queue
Runs inside the same container as API
"""
import os
import sys
import uuid
import hashlib
import logging
import time
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


def retry_with_backoff(func, max_retries=5, initial_delay=2):
    """Retry a function with exponential backoff"""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                logger.warning(f"⏳ Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            else:
                logger.error(f"❌ All {max_retries + 1} attempts failed")
    
    raise last_exception


try:
    from app.config import settings
    from app.database.connection import engine, Base, SessionLocal
    from app.database.models import Document, ChildChunk
    from app.queue.service import RedisQueueService
    from app.queue.models import UploadTask, EditTask, TaskResult
    from app.services.chunking_service import ChunkingService
    from app.services.embedding_service import EmbeddingService
    from app.services.cloudinary_service import CloudinaryService
    from app.services.segmentation_service import get_segmentation_service
except Exception as e:
    logger.error(f"❌ Failed to import required modules: {str(e)}", exc_info=True)
    sys.exit(1)


class DocumentWorker:
    """Worker for processing document tasks"""
    
    def __init__(self):
        logger.info("Initializing DocumentWorker...")
        
        try:
            self.queue_service = RedisQueueService(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB
            )
            logger.info("✅ RedisQueueService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RedisQueueService: {str(e)}")
            raise
        
        try:
            self.chunking_service = ChunkingService(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            logger.info("✅ ChunkingService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChunkingService: {str(e)}")
            raise
        
        try:
            self.embedding_service = EmbeddingService()
            logger.info("✅ EmbeddingService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EmbeddingService: {str(e)}")
            raise
        
        try:
            self.cloudinary_service = CloudinaryService()
            logger.info("✅ CloudinaryService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CloudinaryService: {str(e)}")
            raise
        
        try:
            self.segmentation_service = get_segmentation_service()
            logger.info("✅ VietnameseSegmentationService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize VietnameseSegmentationService: {str(e)}")
            raise
        
        try:
            self.db = SessionLocal()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise
        
        logger.info("✅ DocumentWorker fully initialized")
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """Process document upload task"""
        logger.info(f"Starting UPLOAD task - ID: {task.task_id}")
        task_start = time.time()
        
        try:
            logger.info(f"[{task.task_id}] Reading file from: {task.file_path}")
            
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_size = os.path.getsize(task.file_path)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            logger.info(f"[{task.task_id}] ✅ File read: {file_size} bytes")
            
            # Upload to Cloudinary
            logger.info(f"[{task.task_id}] Uploading to Cloudinary...")
            cloudinary_result = self.cloudinary_service.upload_file(
                file_path=task.file_path,
                folder=settings.CLOUDINARY_UPLOAD_FOLDER
            )
            cloudinary_url = cloudinary_result['secure_url']
            logger.info(f"[{task.task_id}] ✅ Uploaded to Cloudinary")
            
            # Create Document record
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
            
            # Process content
            logger.info(f"[{task.task_id}] Processing: chunking, embedding, segmentation...")
            chunks_result = self.chunking_service.chunk_markdown(content, task.file_name)
            chunks = chunks_result['child_chunks']
            
            chunks_created = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                embedding = self.embedding_service.embed_text(chunk_text)
                segments = self.segmentation_service.segment(chunk_text)
                
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
                    logger.debug(f"[{task.task_id}] Progress: {i + 1}/{len(chunks)}")
            
            self.db.commit()
            
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document status: COMPLETED")
            
            # Clean up
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] Cleaned up temp file")
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ UPLOAD COMPLETED in {elapsed:.2f}s")
            
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
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
            except:
                pass
            return TaskResult(task_id=task.task_id, status="failed", error=error_msg)
    
    def process_edit_task(self, task: EditTask) -> TaskResult:
        """Process document edit task"""
        logger.info(f"Starting EDIT task - ID: {task.task_id}")
        task_start = time.time()
        
        try:
            document = self.db.query(Document).filter(Document.id == task.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {task.document_id}")
            
            logger.info(f"[{task.task_id}] Document found: {document.file_name}")
            
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()
            
            # Delete old chunks
            old_chunk_count = self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).count()
            self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).delete()
            self.db.commit()
            logger.info(f"[{task.task_id}] Deleted {old_chunk_count} old chunks")
            
            # Process new content
            content_hash = hashlib.sha256(new_content.encode()).hexdigest()
            document.content_hash = content_hash
            document.status = "indexing"
            self.db.commit()
            
            chunks_result = self.chunking_service.chunk_markdown(new_content, task.file_name)
            chunks = chunks_result['child_chunks']
            
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
            
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document re-indexed")
            
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ EDIT COMPLETED in {elapsed:.2f}s")
            
            return TaskResult(
                task_id=task.task_id,
                status="completed",
                document_id=task.document_id,
                message="Document updated and re-indexed",
                chunks_created=chunks_created,
                chunks_deleted=old_chunk_count
            )
        
        except Exception as e:
            error_msg = f"Edit task failed: {str(e)}"
            logger.error(f"[{task.task_id}] ❌ {error_msg}", exc_info=True)
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
            except:
                pass
            return TaskResult(task_id=task.task_id, status="failed", error=error_msg)
    
    def run(self):
        """Main worker loop"""
        logger.info("="*70)
        logger.info("🚀 WORKER STARTING MAIN LOOP")
        logger.info("="*70)
        
        task_count = 0
        
        try:
            while True:
                # Check upload queue
                upload_data = self.queue_service.pop_upload_task()
                if upload_data:
                    task = UploadTask.from_dict(upload_data)
                    result = self.process_upload_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    continue
                
                # Check edit queue
                edit_data = self.queue_service.pop_edit_task()
                if edit_data:
                    task = EditTask.from_dict(edit_data)
                    result = self.process_edit_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    continue
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info(f"👋 WORKER STOPPING - Processed {task_count} tasks")
        except Exception as e:
            logger.error("❌ Worker error", exc_info=True)
        finally:
            try:
                self.db.close()
                logger.info("✅ Worker shutdown complete")
            except:
                pass


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("DOCUMENT WORKER STARTUP")
    logger.info("="*70)
    
    try:
        # Initialize database
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        sys.exit(1)
    
    try:
        worker = DocumentWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ Worker crashed: {str(e)}", exc_info=True)
        sys.exit(1)
