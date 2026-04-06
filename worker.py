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


def validate_environment():
    """Validate required environment variables before starting worker"""
    logger.info("Validating environment variables...")
    logger.info("📌 NOTE: Environment variables can come from:")
    logger.info("  1. .env file (if present in working directory)")
    logger.info("  2. System environment variables (Azure, Docker, etc)")
    logger.info("  3. Container secrets/mounts")
    logger.info("")
    
    required_vars = {
        'DATABASE_URL': 'PostgreSQL database connection string',
        'REDIS_HOST': 'Redis server hostname/IP',
        'GEMINI_API_KEY': 'Google Gemini API key',
        'CLOUDINARY_CLOUD_NAME': 'Cloudinary cloud name',
        'CLOUDINARY_API_KEY': 'Cloudinary API key',
        'CLOUDINARY_API_SECRET': 'Cloudinary API secret'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var, '').strip()
        if not value:
            missing_vars.append(f"  - {var}: {description}")
            logger.warning(f"❌ Missing {var}")
        else:
            if var in ['GEMINI_API_KEY', 'CLOUDINARY_API_SECRET']:
                logger.info(f"✅ {var}: set (masked)")
            else:
                logger.info(f"✅ {var}: {value[:50]}..." if len(value) > 50 else f"✅ {var}: {value}")
    
    if missing_vars:
        logger.error("")
        logger.error("🚨 CRITICAL: Missing required environment variables!")
        logger.error("Required variables:\n" + "\n".join(missing_vars))
        logger.error("")
        logger.error("HOW TO FIX (for different environments):")
        logger.error("")
        logger.error("LOCAL DEVELOPMENT:")
        logger.error("  - Create .env file in project root with all required variables")
        logger.error("  - Run: python worker.py")
        logger.error("")
        logger.error("AZURE CONTAINER INSTANCES:")
        logger.error("  - Set environment variables when creating container:")
        logger.error("    az container create --environment-variables DATABASE_URL=... REDIS_HOST=...")
        logger.error("  - Or use Azure Container Registry with environment variables in deployment")
        logger.error("")
        logger.error("DOCKER (local):")
        logger.error("  - Pass environment variables: docker run -e DATABASE_URL=... -e REDIS_HOST=...")
        logger.error("  - Or mount .env file: docker run --env-file .env ...")
        logger.error("")
        logger.error("DOCKER COMPOSE:")
        logger.error("  - Set in docker-compose.yml environment section")
        logger.error("  - Or use .env file in same directory as docker-compose.yml")
        logger.error("")
        logger.error("Exiting worker...")
        return False
    
    logger.info("")
    logger.info("✅ All required environment variables are set")
    return True


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
    logger.error("\nPossible causes:")
    logger.error("  1. Missing or invalid .env file")
    logger.error("  2. Missing required environment variables (DATABASE_URL, REDIS_HOST, GEMINI_API_KEY, etc.)")
    logger.error("  3. Invalid database connection string")
    logger.error("\nPlease check your environment configuration and try again.")
    sys.exit(1)


def retry_with_backoff(func, max_retries=5, initial_delay=2):
    """
    Retry a function with exponential backoff.
    Useful for connecting to external services that may not be immediately available.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        
    Returns:
        Result of the function
        
    Raises:
        Exception: If all retries fail
    """
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
                delay = min(delay * 2, 30)  # Exponential backoff, capped at 30s
            else:
                logger.error(f"❌ All {max_retries + 1} attempts failed")
    
    raise last_exception


class DocumentWorker:
    """Worker for processing document tasks"""
    
    def __init__(self):
        logger.info("Initializing DocumentWorker services...")
        
        try:
            def connect_redis():
                logger.debug(f"Connecting to Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                queue_service = RedisQueueService(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB
                )
                return queue_service
            
            self.queue_service = retry_with_backoff(connect_redis, max_retries=5, initial_delay=2)
            logger.debug("✅ RedisQueueService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RedisQueueService after all retries: {str(e)}", exc_info=True)
            logger.error(f"Cannot connect to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            logger.error("\nMake sure:")
            logger.error("  1. Redis server is running and accessible")
            logger.error("  2. REDIS_HOST and REDIS_PORT are set correctly")
            logger.error("  3. Firewall/network allows connection to Redis")
            raise
        
        try:
            logger.debug(f"Initializing ChunkingService (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
            self.chunking_service = ChunkingService(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            logger.debug("✅ ChunkingService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChunkingService: {str(e)}", exc_info=True)
            raise
        
        try:
            logger.debug("Initializing EmbeddingService...")
            self.embedding_service = EmbeddingService()
            logger.debug("✅ EmbeddingService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize EmbeddingService: {str(e)}", exc_info=True)
            logger.error("Make sure GEMINI_API_KEY is set correctly")
            raise
        
        try:
            logger.debug("Initializing CloudinaryService...")
            self.cloudinary_service = CloudinaryService()
            logger.debug("✅ CloudinaryService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CloudinaryService: {str(e)}", exc_info=True)
            logger.error("Make sure Cloudinary credentials are set (CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)")
            raise
        
        try:
            logger.debug("Initializing VietnameseSegmentationService...")
            self.segmentation_service = get_segmentation_service()
            logger.debug("✅ VietnameseSegmentationService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize VietnameseSegmentationService: {str(e)}", exc_info=True)
            raise
        
        try:
            def connect_db():
                logger.debug("Connecting to database...")
                db = SessionLocal()
                return db
            
            self.db = retry_with_backoff(connect_db, max_retries=5, initial_delay=2)
            logger.debug("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database after all retries: {str(e)}", exc_info=True)
            logger.error("Make sure DATABASE_URL is set correctly")
            logger.error("\nMake sure:")
            logger.error("  1. PostgreSQL server is running and accessible")
            logger.error("  2. DATABASE_URL is in correct format: postgresql://user:password@host:5432/dbname")
            logger.error("  3. Firewall/network allows connection to database")
            logger.error("  4. Database credentials are correct")
            raise
        
        logger.info("✅ DocumentWorker fully initialized and ready")
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """
        Process document upload task
        
        Steps:
        1. Read raw file from shared storage (TEMP_UPLOAD_DIR)
        2. Upload content to Cloudinary (streaming - no intermediate files)
        3. Process content: chunk, embed, segment (in-memory)
        4. Save to database
        5. Clean up raw file from shared storage
        """
        logger.info(f"Starting UPLOAD task - ID: {task.task_id}")
        task_start = time.time()
        
        try:
            # Step 1: Read raw file from shared storage
            logger.info(f"[{task.task_id}] Step 1: Reading file from shared storage")
            logger.info(f"[{task.task_id}] Shared storage: {settings.TEMP_UPLOAD_DIR}")
            logger.info(f"[{task.task_id}] File name: {task.file_name}")
            
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_size = os.path.getsize(task.file_path)
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            logger.info(f"[{task.task_id}] ✅ File read from shared storage: {file_size} bytes")
            
            # Step 2: Upload to Cloudinary (streaming content)
            logger.info(f"[{task.task_id}] Step 2: Uploading content to Cloudinary...")
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
            
            # Step 4: Process content (in-memory streaming)
            logger.info(f"[{task.task_id}] Step 4: Processing content (chunking, embedding, segmentation)...")
            
            # 4a: Chunk content
            logger.info(f"[{task.task_id}]   - Chunking...")
            chunks_result = self.chunking_service.chunk_markdown(content, task.file_name)
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}]   - Created {len(chunks)} chunks")
            
            # 4b: Embed and save chunks (streamed, no intermediate files)
            logger.info(f"[{task.task_id}]   - Embedding and indexing chunks...")
            chunks_created = 0
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                # Embed
                embedding = self.embedding_service.embed_text(chunk_text)
                
                # Segment
                segments = self.segmentation_service.segment(chunk_text)
                
                # Save to database (no file I/O)
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
                    logger.debug(f"[{task.task_id}]   - Progress: {i + 1}/{len(chunks)} chunks embedded")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Processed: {len(chunks)} chunks embedded and indexed")
            
            # Step 5: Update document status
            logger.info(f"[{task.task_id}] Step 5: Finalizing...")
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document status updated to COMPLETED")
            
            # Step 6: Clean up raw file from shared storage
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] ✅ Cleaned up raw file from shared storage")
            except Exception as cleanup_error:
                logger.warning(f"[{task.task_id}] ⚠️ Failed to clean up file: {cleanup_error}")
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ UPLOAD TASK COMPLETED in {elapsed:.2f}s")
            logger.info(f"[{task.task_id}] Summary: {chunks_created} chunks indexed, {file_size} bytes processed")
            
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
            
            # Try to clean up even on error
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] Cleaned up file after error")
            except:
                pass
            
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def process_edit_task(self, task: EditTask) -> TaskResult:
        """
        Process document edit task (re-indexing)
        
        Steps:
        1. Verify document exists
        2. Read new raw file from shared storage
        3. Delete old chunks
        4. Process new content (chunk, embed, segment - in-memory)
        5. Save to database
        6. Clean up raw file from shared storage
        """
        logger.info(f"Starting EDIT task - ID: {task.task_id}, Document: {task.document_id}")
        task_start = time.time()
        
        try:
            # Step 1: Verify document exists
            logger.info(f"[{task.task_id}] Step 1: Verifying document...")
            document = self.db.query(Document).filter(Document.id == task.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {task.document_id}")
            logger.info(f"[{task.task_id}] ✅ Document found: {document.file_name}")
            
            # Step 2: Read new file from shared storage
            logger.info(f"[{task.task_id}] Step 2: Reading new file from shared storage...")
            logger.info(f"[{task.task_id}] Shared storage: {settings.TEMP_UPLOAD_DIR}")
            
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
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
            
            # Step 5: Process new content (in-memory streaming)
            logger.info(f"[{task.task_id}] Step 5: Processing new content (chunking, embedding, segmentation)...")
            
            # 5a: Chunk content
            logger.info(f"[{task.task_id}]   - Re-chunking...")
            chunks_result = self.chunking_service.chunk_markdown(new_content, task.file_name)
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}]   - Created {len(chunks)} new chunks")
            
            # 5b: Embed and save (streamed, no intermediate files)
            logger.info(f"[{task.task_id}]   - Embedding and indexing chunks...")
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
                    logger.debug(f"[{task.task_id}]   - Progress: {i + 1}/{len(chunks)} chunks embedded")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Processed: {len(chunks)} chunks embedded and indexed")
            
            # Step 6: Update document status
            logger.info(f"[{task.task_id}] Step 6: Finalizing...")
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document status updated to COMPLETED")
            
            # Step 7: Clean up raw file from shared storage
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] ✅ Cleaned up raw file from shared storage")
            except Exception as cleanup_error:
                logger.warning(f"[{task.task_id}] ⚠️ Failed to clean up file: {cleanup_error}")
            
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
            
            # Try to clean up even on error
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] Cleaned up file after error")
            except:
                pass
            
            return TaskResult(
                task_id=task.task_id,
                status="failed",
                error=error_msg
            )
    
    def run(self):
        """Main worker loop"""
        logger.info("="*70)
        logger.info("🚀 DOCUMENT WORKER STARTING MAIN LOOP")
        logger.info("="*70)
        
        # Health check
        logger.info("Checking Redis connection...")
        try:
            if not self.queue_service.health_check():
                logger.error("❌ Redis health check failed!")
                logger.error("Redis is not responding. Check:")
                logger.error(f"  - Redis server is running at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                logger.error("  - Network connectivity to Redis")
                logger.error("  - Redis credentials/authentication")
                return
        except Exception as e:
            logger.error(f"❌ Redis connection check failed: {str(e)}", exc_info=True)
            logger.error(f"Cannot connect to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            return
        
        logger.info("✅ Redis connection successful")
        logger.info(f"Redis config: {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
        logger.info("⏳ Worker is ready - waiting for tasks...")
        logger.info("="*70)
        
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
            logger.error("❌ Worker encountered error in main loop", exc_info=True)
            raise
        finally:
            logger.info("Cleaning up database connection...")
            try:
                self.db.close()
                logger.info("✅ Worker shutdown complete")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("DOCUMENT WORKER STARTUP")
    logger.info("="*70)
    
    # Step 1: Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed - cannot start worker")
        sys.exit(1)
    
    # Step 2: Initialize database tables
    try:
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database tables: {str(e)}", exc_info=True)
        logger.error("\nPossible causes:")
        logger.error("  1. Invalid DATABASE_URL or connection string")
        logger.error("  2. Database server is not accessible from this location")
        logger.error("  3. Database credentials are incorrect")
        logger.error("  4. Network/firewall blocking database access")
        sys.exit(1)
    
    # Step 3: Start worker
    try:
        logger.info("Starting DocumentWorker...")
        worker = DocumentWorker()
        worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Worker crashed: {str(e)}", exc_info=True)
        logger.error("\nPlease check the logs above for details and contact support if needed.")
        sys.exit(1)
