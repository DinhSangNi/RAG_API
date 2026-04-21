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
import re
import html
import io
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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


def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def clean_wikipedia_html(html_content: str) -> str:
    """
    Clean Wikipedia HTML by removing unwanted elements and classes
    
    Args:
        html_content: Raw HTML content as string
        
    Returns:
        Cleaned HTML content
    """
    if BeautifulSoup is None:
        logger.warning("[Clean] BeautifulSoup not available, skipping HTML cleaning")
        return html_content
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        content = soup.find("div", class_="mw-parser-output")
        
        if not content:
            logger.warning("[Clean] Could not find mw-parser-output div, using full content")
            content = soup
        
        # Remove unwanted tags
        tags_without_class = ['audio', 'style', 'img', 'sup', 'link', 'input']
        for tag in tags_without_class:
            for element in content.find_all(tag):
                element.decompose()
        
        # Remove specific tags with certain classes
        tags_with_class = [
            ('ol', 'references'),
            ('span', 'mw-editsection'),
            ('div', 'hatnote'),
            ('div', 'navbox'),
            ('div', 'navbox-styles'),
            ('div', 'metadata'),
            ('div', 'toc'),
            ('table', 'navbox-inner'),
            ('table', 'navbox'),
            ('table', 'sidebar'),
            ('table', 'infobox'),
            ('table', 'metadata'),
            ('span', 'languageicon'),
            ('span', 'tocnumber'),
            ('span', 'toctext'),
            ('span', 'reference-accessdate'),
            ('span', 'Z3988'),
            ('cite', None),
        ]
        
        for tag, class_name in tags_with_class:
            if class_name:
                for element in content.find_all(tag, class_=class_name):
                    element.decompose()
            else:
                for element in content.find_all(tag):
                    element.decompose()
        
        # Convert cquote tables to paragraphs
        for table in content.find_all('table', class_="cquote"):
            quote_text = table.get_text(separator=" ", strip=True)
            p = soup.new_tag('p')
            p.string = quote_text
            table.replace_with(p)
        
        # Remove empty paragraphs
        for p in content.find_all('p'):
            if not p.get_text(strip=True):
                p.decompose()
        
        # Remove empty spans with IDs
        for span in content.find_all('span'):
            if span.get('id') and not span.get_text(strip=True):
                span.decompose()
        
        # Convert figure captions
        for figure in content.find_all('figure'):
            figcaption = figure.find('figcaption')
            if figcaption:
                new_p = soup.new_tag('p')
                new_p.string = f"[Hình ảnh: {figcaption.get_text(strip=True)}]"
                figure.replace_with(new_p)
            else:
                figure.decompose()
        
        # Remove link, span, bold wrapper tags but keep content
        for a_tag in content.find_all('a'):
            a_tag.unwrap()
        
        for span in content.find_all('span'):
            span.unwrap()
        
        for tag in content.find_all(['b']):
            tag.unwrap()
        
        # Remove specific sections (References, See Also, etc.)
        sections_to_remove = [
            "Tham_khảo", "Tài liệu tham khảo", "Chú giải", "Liên_kết_ngoài", 
            "Danh_mục", "Ghi_chú", "Thư_mục_hậu_cần", "Đọc_thêm", "Chú_thích",
            "Thư_mục", "Nguồn_thứ_cấp", "Nguồn_sơ_cấp", "Nguồn_trích_dẫn"
        ]
        
        for section_id in sections_to_remove:
            header = content.find(['h2', 'h3'], id=section_id)
            if header:
                for sibling in header.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    sibling.decompose()
                header.decompose()
        
        # Remove empty list items
        for li in content.find_all('li'):
            if not li.get_text(strip=True):
                li.decompose()
        
        # Clean attributes
        for tag in content.find_all(True):
            if tag.has_attr('class'):
                del tag['class']
            if tag.has_attr('style'):
                del tag['style']
            if tag.has_attr('id') and tag.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                del tag['id']
            if tag.has_attr('dir'):
                del tag['dir']
            if tag.has_attr('lang'):
                del tag['lang']
        
        return str(content)
    
    except Exception as e:
        logger.warning(f"[Clean] BeautifulSoup cleaning failed: {str(e)}, returning original content")
        return html_content


def normalize_markdown(md_text: str) -> str:
    """
    Normalize markdown content for better structure
    
    Args:
        md_text: Markdown text to normalize
        
    Returns:
        Normalized markdown text
    """
    lines = md_text.split('\n')
    normalized_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Keep headers as-is
        if line.strip().startswith('#'):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Keep special lines (lists, tables, quotes, code)
        if (line.strip().startswith(('* ', '- ', '+ ', '|', '>')) or 
            re.match(r'^\s*\d+\.', line) or
            line.strip() == '' or
            line.strip().startswith(':')):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Merge consecutive non-special lines into paragraphs
        paragraph = line
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Stop at special lines
            if (next_line.strip() == '' or 
                next_line.strip().startswith(('#', '* ', '- ', '+ ', '|', '>', ':')) or
                re.match(r'^\s*\d+\.', next_line)):
                break
            # Merge line
            paragraph += ' ' + next_line.strip()
            i += 1
        
        normalized_lines.append(paragraph)
    
    # Join lines
    result = '\n'.join(normalized_lines)
    
    # Normalize bullet points to *
    result = re.sub(r'^\s*[-+]\s+', '* ', result, flags=re.MULTILINE)
    
    # Remove trailing spaces
    result = re.sub(r' +\n', '\n', result)
    
    # Normalize block quotes
    result = re.sub(r'^:   \*', '>   *', result, flags=re.MULTILINE)
    result = re.sub(r'^:\s+', '> ', result, flags=re.MULTILINE)
    
    # Ensure blank line before headers
    result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', result)
    
    # Remove excessive blank lines (keep max 2)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


class DocumentWorker:
    """Worker for processing document tasks"""
    
    def __init__(self):
        logger.info("Initializing DocumentWorker...")
        
        try:
            self.queue_service = RedisQueueService(
                connection_string=settings.REDIS_CONNECTION_STRING,
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
        
        logger.info("✅ DocumentWorker fully initialized\n")
    
    def convert_html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown format with cleaning and normalization
        
        Pipeline: Clean HTML → Convert to Markdown → Normalize Markdown
        """
        logger.info("[Convert] Starting HTML → Markdown pipeline")
        
        # Step 1: Clean HTML using BeautifulSoup
        logger.info("[Convert] Step 1: Cleaning HTML with BeautifulSoup...")
        cleaned_html = clean_wikipedia_html(html_content)
        
        # Step 2: Convert to Markdown using MarkItDown
        logger.info("[Convert] Step 2: Converting to Markdown with MarkItDown...")
        markdown_content = None
        
        try:
            if MarkItDown is not None:
                md = MarkItDown()
                
                # Write cleaned HTML to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp:
                    tmp.write(cleaned_html)
                    tmp_path = tmp.name
                
                try:
                    result = md.convert(tmp_path)
                    markdown_content = result.text_content
                    logger.info("[Convert] ✅ MarkItDown conversion successful")
                except Exception as e:
                    logger.warning(f"[Convert] MarkItDown conversion failed: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"[Convert] MarkItDown initialization error: {str(e)}")
        
        # Fallback to regex conversion if MarkItDown fails
        if markdown_content is None:
            logger.info("[Convert] Using regex fallback for HTML → Markdown")
            text = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_html, flags=re.DOTALL)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', text, flags=re.DOTALL)
            text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', text, flags=re.DOTALL)
            text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', text, flags=re.DOTALL)
            text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1', text, flags=re.DOTALL)
            text = re.sub(r'<h5[^>]*>(.*?)</h5>', r'##### \1', text, flags=re.DOTALL)
            text = re.sub(r'<h6[^>]*>(.*?)</h6>', r'###### \1', text, flags=re.DOTALL)
            text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
            text = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', text, flags=re.DOTALL)
            text = re.sub(r'<ul[^>]*>|</ul>', '', text)
            text = re.sub(r'<ol[^>]*>|</ol>', '', text)
            text = re.sub(r'<br\s*/?>', '\n', text)
            text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
            text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
            text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
            text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
            text = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            text = html.unescape(text)
            text = re.sub(r'\n\n\n+', '\n\n', text)
            markdown_content = text.strip()
            logger.info("[Convert] ✅ Regex fallback conversion successful")
        
        # Step 3: Normalize Markdown
        logger.info("[Convert] Step 3: Normalizing Markdown structure...")
        normalized_markdown = normalize_markdown(markdown_content)
        logger.info("[Convert] ✅ Markdown normalization complete")
        
        return normalized_markdown
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """Process document upload task with detailed logging"""
        logger.info("\n" + "="*70)
        logger.info(f"📥 UPLOAD TASK STARTED: {task.task_id}")
        logger.info("="*70)
        task_start = time.time()
        document_id = None
        
        try:
            logger.info(f"[{task.task_id}] 📂 Step 1: Reading file")
            logger.info(f"[{task.task_id}]    Path: {task.file_path}")
            
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_size = os.path.getsize(task.file_path)
            original_filename = task.file_name
            
            # Check if file is HTML and convert to Markdown
            if task.file_name.lower().endswith('.html'):
                logger.info(f"[{task.task_id}] 🔄 HTML file detected, converting to Markdown...")
                content = self.convert_html_to_markdown(content)
                
                # Save converted Markdown file to data/temp_uploads
                md_filename = original_filename.rsplit('.', 1)[0] + '.md'
                md_file_path = os.path.join(settings.TEMP_UPLOAD_DIR, md_filename)
                
                os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"[{task.task_id}] ✅ Markdown saved: {md_filename}")
                logger.info(f"[{task.task_id}]    Path: {md_file_path}")
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            logger.info(f"[{task.task_id}] ✅ File loaded: {format_bytes(file_size)}")
            logger.info(f"[{task.task_id}]    Hash: {content_hash[:16]}...")
            logger.info(f"[{task.task_id}]    Content: {len(content)} chars")
            
            # Upload to Cloudinary
            logger.info(f"[{task.task_id}] 🌐 Step 2: Uploading to Cloudinary...")
            cloudinary_result = self.cloudinary_service.upload_file(
                file_path=task.file_path,
                folder=settings.CLOUDINARY_UPLOAD_FOLDER
            )
            cloudinary_url = cloudinary_result['secure_url']
            logger.info(f"[{task.task_id}] ✅ Uploaded: {cloudinary_url[:60]}...")
            
            # Create Document record with 'pending' status
            logger.info(f"[{task.task_id}] 💾 Step 3: Creating document record...")
            document_id = str(uuid.uuid4())
            document = Document(
                id=document_id,
                file_name=task.file_name,
                file_path=cloudinary_url,
                source_type=task.source_type,
                status="pending",
                meta_data=task.metadata or {},
                file_size=file_size,
                content_hash=content_hash
            )
            self.db.add(document)
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Document created: {document_id}")
            logger.info(f"[{task.task_id}]    Initial status: pending")
            
            # Update to 'indexing' status
            document.status = "indexing"
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: pending → indexing")
            
            # Process content - chunking
            logger.info(f"[{task.task_id}] ✂️ Step 4: Chunking document...")
            logger.info(f"[{task.task_id}]    Using chunk_size={task.chunk_size}, overlap={task.chunk_overlap}")
            chunks_result = self.chunking_service.chunk_markdown(
                content, 
                task.file_name,
                chunk_size=task.chunk_size,
                chunk_overlap=task.chunk_overlap
            )
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}]    Created {len(chunks)} chunks")
            
            # Embed and index chunks
            logger.info(f"[{task.task_id}] 🧠 Step 5: Embedding and indexing chunks...")
            chunks_created = 0
            for i, chunk in enumerate(chunks, 1):
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
                    chunk_index=chunk.get('chunk_index', i-1)
                )
                self.db.add(child_chunk)
                chunks_created += 1
                
                if i % 10 == 0:
                    logger.info(f"[{task.task_id}]    Progress: {i}/{len(chunks)} chunks indexed")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ All {chunks_created} chunks indexed")
            
            # Final status update to 'completed'
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: indexing → completed")
            
            # Clean up raw file
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] 🗑️ Temp file cleaned up")
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ UPLOAD COMPLETED")
            logger.info(f"[{task.task_id}]    Duration: {elapsed:.2f}s")
            logger.info(f"[{task.task_id}]    Chunks created: {chunks_created}")
            logger.info(f"[{task.task_id}]    File size: {format_bytes(file_size)}")
            logger.info("="*70 + "\n")
            
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
            
            # Rollback any uncommitted transaction
            try:
                self.db.rollback()
                logger.info(f"[{task.task_id}] ↩️ Database transaction rolled back")
            except Exception as rollback_error:
                logger.error(f"[{task.task_id}] Failed to rollback: {rollback_error}")
            
            # Update document status to 'failed' if document was created
            if document_id:
                try:
                    document = self.db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        document.status = "failed"
                        document.meta_data = document.meta_data or {}
                        document.meta_data['error'] = str(e)
                        self.db.commit()
                        logger.info(f"[{task.task_id}] 📌 Status transition: indexing → failed")
                        logger.info(f"[{task.task_id}]    Error saved: {str(e)[:80]}...")
                except Exception as db_error:
                    logger.error(f"[{task.task_id}] Failed to update status to failed: {db_error}")
                    try:
                        self.db.rollback()
                    except:
                        pass
            
            # Clean up
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.error(f"[{task.task_id}] Elapsed time before failure: {elapsed:.2f}s")
            logger.error("="*70 + "\n")
            
            return TaskResult(task_id=task.task_id, status="failed", error=error_msg)
        
        finally:
            # Reset the session after each task to ensure clean state
            try:
                self.db.expunge_all()
                logger.debug(f"[{task.task_id}] 🔄 Session cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"[{task.task_id}] Warning during session cleanup: {cleanup_error}")
    
    def process_edit_task(self, task: EditTask) -> TaskResult:
        """Process document edit task with detailed logging"""
        logger.info("\n" + "="*70)
        logger.info(f"✏️ EDIT TASK STARTED: {task.task_id}")
        logger.info("="*70)
        task_start = time.time()
        
        try:
            # Find document
            logger.info(f"[{task.task_id}] 📄 Step 1: Finding document...")
            document = self.db.query(Document).filter(Document.id == task.document_id).first()
            if not document:
                raise ValueError(f"Document not found: {task.document_id}")
            
            logger.info(f"[{task.task_id}] ✅ Document found: {document.file_name}")
            logger.info(f"[{task.task_id}]    Current status: {document.status}")
            
            # Read new file
            logger.info(f"[{task.task_id}] 📂 Step 2: Reading new file...")
            if not os.path.exists(task.file_path):
                raise FileNotFoundError(f"File not found at {task.file_path}")
            
            with open(task.file_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()
            logger.info(f"[{task.task_id}]    File loaded: {len(new_content)} chars")
            
            # Delete old chunks
            logger.info(f"[{task.task_id}] 🗑️ Step 3: Removing old chunks...")
            old_chunk_count = self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).count()
            self.db.query(ChildChunk).filter(
                ChildChunk.document_id == task.document_id
            ).delete()
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ Deleted {old_chunk_count} old chunks")
            
            # Update status to 'indexing'
            content_hash = hashlib.sha256(new_content.encode()).hexdigest()
            document.content_hash = content_hash
            old_status = document.status
            document.status = "indexing"
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: {old_status} → indexing")
            
            # Chunk new content
            logger.info(f"[{task.task_id}] ✂️ Step 4: Re-chunking document...")
            logger.info(f"[{task.task_id}]    Using chunk_size={task.chunk_size}, overlap={task.chunk_overlap}")
            chunks_result = self.chunking_service.chunk_markdown(
                new_content, 
                task.file_name,
                chunk_size=task.chunk_size,
                chunk_overlap=task.chunk_overlap
            )
            chunks = chunks_result['child_chunks']
            logger.info(f"[{task.task_id}]    Created {len(chunks)} new chunks")
            
            # Embed and index
            logger.info(f"[{task.task_id}] 🧠 Step 5: Embedding and indexing chunks...")
            chunks_created = 0
            for i, chunk in enumerate(chunks, 1):
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
                    chunk_index=chunk.get('chunk_index', i-1)
                )
                self.db.add(child_chunk)
                chunks_created += 1
                
                if i % 10 == 0:
                    logger.info(f"[{task.task_id}]    Progress: {i}/{len(chunks)} chunks indexed")
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ All {chunks_created} new chunks indexed")
            
            # Final status update to 'completed'
            document.status = "completed"
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: indexing → completed")
            
            # Clean up raw file
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
                    logger.info(f"[{task.task_id}] 🗑️ Temp file cleaned up")
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.info(f"[{task.task_id}] ✅ EDIT COMPLETED")
            logger.info(f"[{task.task_id}]    Duration: {elapsed:.2f}s")
            logger.info(f"[{task.task_id}]    Created: {chunks_created} chunks")
            logger.info(f"[{task.task_id}]    Deleted: {old_chunk_count} chunks")
            logger.info("="*70 + "\n")
            
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
            
            # Rollback any uncommitted transaction
            try:
                self.db.rollback()
                logger.info(f"[{task.task_id}] ↩️ Database transaction rolled back")
            except Exception as rollback_error:
                logger.error(f"[{task.task_id}] Failed to rollback: {rollback_error}")
            
            # Update document status to 'failed'
            try:
                document = self.db.query(Document).filter(Document.id == task.document_id).first()
                if document:
                    document.status = "failed"
                    document.meta_data = document.meta_data or {}
                    document.meta_data['error'] = str(e)
                    self.db.commit()
                    logger.info(f"[{task.task_id}] 📌 Status transition: indexing → failed")
                    logger.info(f"[{task.task_id}]    Error saved: {str(e)[:80]}...")
            except Exception as db_error:
                logger.error(f"[{task.task_id}] Failed to update status to failed: {db_error}")
                try:
                    self.db.rollback()
                except:
                    pass
            
            # Clean up
            try:
                if os.path.exists(task.file_path):
                    os.remove(task.file_path)
            except:
                pass
            
            elapsed = time.time() - task_start
            logger.error(f"[{task.task_id}] Elapsed time before failure: {elapsed:.2f}s")
            logger.error("="*70 + "\n")
            
            return TaskResult(task_id=task.task_id, status="failed", error=error_msg)
        
        finally:
            # Reset the session after each task to ensure clean state
            try:
                self.db.expunge_all()
                logger.debug(f"[{task.task_id}] 🔄 Session cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"[{task.task_id}] Warning during session cleanup: {cleanup_error}")
    
    def run(self):
        """Main worker loop with detailed logging"""
        logger.info("\n" + "="*70)
        logger.info("🚀 WORKER MAIN LOOP STARTED")
        logger.info(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
        logger.info("   Waiting for tasks from queue...")
        logger.info("="*70 + "\n")
        
        task_count = 0
        idle_seconds = 0
        
        try:
            while True:
                # Check upload queue
                upload_data = self.queue_service.pop_upload_task()
                if upload_data:
                    logger.info(f"📋 Upload task data from queue: {list(upload_data.keys())}")
                    task = UploadTask.from_dict(upload_data)
                    logger.info(f"✅ Parsed UploadTask: task_id={task.task_id}, file={task.file_name}, chunk_size={task.chunk_size}, overlap={task.chunk_overlap}")
                    logger.info(f"\n📨 Task received from queue: {task.task_id} (upload)")
                    result = self.process_upload_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    logger.info(f"✅ Result stored in Redis for task {task.task_id}\n")
                    idle_seconds = 0
                    continue
                
                # Check edit queue
                edit_data = self.queue_service.pop_edit_task()
                if edit_data:
                    logger.info(f"📋 Edit task data from queue: {list(edit_data.keys())}")
                    task = EditTask.from_dict(edit_data)
                    logger.info(f"✅ Parsed EditTask: task_id={task.task_id}, doc_id={task.document_id}, file={task.file_name}, chunk_size={task.chunk_size}, overlap={task.chunk_overlap}")
                    logger.info(f"\n📨 Task received from queue: {task.task_id} (edit)")
                    result = self.process_edit_task(task)
                    self.queue_service.set_result(task.task_id, result.to_dict())
                    task_count += 1
                    logger.info(f"✅ Result stored in Redis for task {task.task_id}\n")
                    idle_seconds = 0
                    continue
                
                # No tasks available
                idle_seconds += 1
                if idle_seconds % 30 == 0:
                    logger.debug(f"⏳ Idle: {idle_seconds}s (total tasks: {task_count})")
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info(f"\n👋 WORKER STOPPING")
            logger.info(f"   Total tasks processed: {task_count}")
        except Exception as e:
            logger.error("❌ Worker error", exc_info=True)
        finally:
            try:
                # Ensure any pending transaction is rolled back
                self.db.rollback()
            except:
                pass
            
            try:
                self.db.close()
                logger.info("✅ Database connection closed at worker shutdown")
            except:
                pass


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("DOCUMENT WORKER STARTUP")
    logger.info("="*70)
    
    try:
        # Initialize database
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized\n")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        sys.exit(1)
    
    try:
        worker = DocumentWorker()
        worker.run()
    except Exception as e:
        logger.error(f"❌ Worker crashed: {str(e)}", exc_info=True)
        sys.exit(1)
