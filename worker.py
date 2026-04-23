"""
Background Worker for RAG API
Processes document upload and indexing tasks from Redis queue
Runs inside the same container as API
"""
import os
import sys
import uuid
import hashlib
import json
import logging
import time
import re
import html
import io
import signal
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError

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


class RetryableTaskError(Exception):
    """Task failed due to a transient dependency issue and should be retried."""

    def __init__(self, message, *, original_error=None):
        super().__init__(message)
        self.original_error = original_error

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
    from app.queue.service import (
        RedisQueueService,
        TASK_STATUS_PROCESSING,
        TASK_STATUS_COMPLETED,
        TASK_STATUS_FAILED,
    )
    from app.queue.models import UploadTask, EditTask, TaskResult
    from app.services.chunking_service import ChunkingService
    from app.services.embedding_service import EmbeddingService
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
        self._shutdown_requested = False
        self._current_task_id = None
        self._current_task_raw_payload = None
        self.pipeline_webhook_url = os.getenv("PIPELINE_WEBHOOK_URL", "").strip()
        self.pipeline_webhook_token = os.getenv("PIPELINE_WEBHOOK_TOKEN", "").strip()
        self.pipeline_webhook_timeout = int(os.getenv("PIPELINE_WEBHOOK_TIMEOUT_SECONDS", "10"))

        # Handle SIGINT/SIGTERM for graceful shutdown.
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        
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
            self.segmentation_service = get_segmentation_service()
            logger.info("✅ VietnameseSegmentationService initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize VietnameseSegmentationService: {str(e)}")
            raise
        
        try:
            self.db = None
            self._refresh_db_session()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise
        
        logger.info("✅ DocumentWorker fully initialized\n")

    def _handle_shutdown_signal(self, signum, _frame):
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.warning(f"⚠️ Received {signal_name}. Worker will stop after current task finishes.")
        self._shutdown_requested = True

    def _set_processing_progress(
        self,
        task_id: str,
        current: int,
        total: int,
        message: str,
        **extra,
    ):
        percent = int((current / total) * 100) if total > 0 else 0
        self.queue_service.set_task_status(
            task_id=task_id,
            status=TASK_STATUS_PROCESSING,
            message=message,
            progress={
                "current": current,
                "total": total,
                "percent": percent,
            },
            **extra,
        )

    def _close_db_session(self):
        if self.db is None:
            return

        try:
            self.db.close()
        except Exception as close_error:
            logger.warning(f"Warning while closing DB session: {close_error}")
        finally:
            self.db = None

    def _refresh_db_session(self):
        self._close_db_session()
        self.db = SessionLocal()
        return self.db

    @staticmethod
    def _is_transient_db_error(error):
        if isinstance(error, (OperationalError, DisconnectionError)):
            return True

        if isinstance(error, DBAPIError) and getattr(error, "connection_invalidated", False):
            return True

        message = str(error).lower()
        transient_markers = (
            "server closed the connection unexpectedly",
            "connection not open",
            "connection reset by peer",
            "connection refused",
            "terminating connection",
            "could not receive data from server",
            "could not send data to server",
            "ssl connection has been closed unexpectedly",
        )
        return any(marker in message for marker in transient_markers)

    def _requeue_retryable_task(self, task_data, task_id, task_type, error):
        retry_count = int(task_data.get("retry_count", 0)) + 1
        task_data["retry_count"] = retry_count

        if retry_count > settings.WORKER_MAX_RETRIES:
            self.queue_service.push_dead_letter_task(
                task_data,
                reason=(
                    "Retryable task exceeded configured retry limit "
                    f"({settings.WORKER_MAX_RETRIES})"
                ),
                error=str(error),
            )
            self.queue_service.set_task_status(
                task_id=task_id,
                status=TASK_STATUS_FAILED,
                message="Task moved to dead-letter queue after retry limit exceeded",
                error=str(error),
                document_id=task_data.get("document_id"),
                retry_count=retry_count,
                task_type=task_type,
                dead_lettered=True,
            )
            return

        self.queue_service._push_queue_item(self.queue_service.main_queue, task_data)
        self.queue_service.set_task_status(
            task_id=task_id,
            status="PENDING",
            message=f"Transient failure detected, task requeued ({retry_count}/{settings.WORKER_MAX_RETRIES})",
            error=str(error),
            document_id=task_data.get("document_id"),
            retry_count=retry_count,
            task_type=task_type,
        )

    def _notify_pipeline_webhook(self, *, task_id, document_id, file_name, status, message="", error=None):
        if not self.pipeline_webhook_url:
            return

        payload = {
            "documentId": document_id,
            "taskId": task_id,
            "pipeline": "rag",
            "status": status,
            "message": message,
            "error": error,
            "fileName": file_name,
        }
        headers = {"Content-Type": "application/json"}
        if self.pipeline_webhook_token:
            headers["X-Webhook-Token"] = self.pipeline_webhook_token

        try:
            response = requests.post(
                self.pipeline_webhook_url,
                json=payload,
                headers=headers,
                timeout=self.pipeline_webhook_timeout,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Pipeline webhook returned %s for task=%s body=%s",
                    response.status_code,
                    task_id,
                    response.text[:300],
                )
            else:
                logger.info("Pipeline webhook sent successfully for task=%s pipeline=rag", task_id)
        except Exception as exc:
            logger.warning("Pipeline webhook failed for task=%s pipeline=rag: %s", task_id, str(exc))

    @staticmethod
    def _normalize_pipeline_status(value):
        if not value:
            return "not_found"
        normalized = str(value).strip().lower()
        if normalized in {"pending", "queued"}:
            return "pending"
        if normalized in {"processing", "indexing", "updating"}:
            return "processing"
        if normalized in {"completed", "complete", "done"}:
            return "completed"
        if normalized in {"failed", "error"}:
            return "failed"
        return normalized

    @staticmethod
    def _compose_document_status(rag_status, graph_status):
        rag = DocumentWorker._normalize_pipeline_status(rag_status)
        graph = DocumentWorker._normalize_pipeline_status(graph_status)

        if rag == "failed" or graph == "failed":
            return "failed"
        if rag == "completed" and graph == "completed":
            return "completed"
        if rag == "completed":
            return "rag_completed_waiting_graph"
        if rag in {"processing", "pending"}:
            return "indexing"
        return rag

    def _get_graph_status(self, task_id):
        if not task_id:
            return "not_found"

        key = f"graphrag:task:status:{task_id}"
        raw = self.queue_service.redis_client.get(key)
        if not raw:
            return "not_found"

        try:
            payload = json.loads(raw)
            return self._normalize_pipeline_status(payload.get("status"))
        except Exception:
            return self._normalize_pipeline_status(raw)

    def _update_document_pipeline_status(self, document, task_id, rag_status, error=None):
        metadata = dict(document.meta_data or {})
        pipeline_status = dict(metadata.get("pipeline_status") or {})
        graph_status = self._get_graph_status(task_id)

        pipeline_status["rag"] = self._normalize_pipeline_status(rag_status)
        pipeline_status["graph"] = graph_status
        pipeline_status["updated_at"] = datetime.now().astimezone().isoformat()

        if task_id:
            metadata["task_id"] = task_id

        if error:
            metadata["error"] = str(error)
        elif metadata.get("error"):
            metadata.pop("error", None)

        metadata["pipeline_status"] = pipeline_status
        document.meta_data = metadata
        document.status = self._compose_document_status(
            rag_status=pipeline_status.get("rag"),
            graph_status=pipeline_status.get("graph"),
        )

    def _normalize_payload(self, task_data):
        """Normalize queue payloads from both RAG API and Wiki Backend formats."""
        normalized = dict(task_data or {})
        original = dict(task_data or {})

        raw_type = str(normalized.get("type") or normalized.get("pipeline") or "").strip().lower()
        if raw_type in {"rag", "rag-api", "rag_api", "ragworker", "rag-worker"}:
            normalized["type"] = "rag"
        elif raw_type in {"graph", "graph-rag", "graph_rag", "graphrag", "graphworker", "graph-worker"}:
            normalized["type"] = "graph"
        else:
            normalized["type"] = "rag"
        normalized["pipeline"] = normalized["type"]

        # Backward compatibility with Wiki Backend payload shape.
        if not normalized.get("task_id") and normalized.get("job_id"):
            normalized["task_id"] = normalized.get("job_id")

        if not normalized.get("file_path") and normalized.get("file_url"):
            normalized["file_path"] = normalized.get("file_url")

        if not normalized.get("file_name"):
            normalized["file_name"] = (
                normalized.get("fileName")
                or normalized.get("filename")
                or normalized.get("FileName")
            )

        file_type = str(normalized.get("file_type") or "").strip().lower()
        if not normalized.get("file_name"):
            derived_file_name = None
            file_path = str(normalized.get("file_path") or "")
            if file_path.startswith(("http://", "https://")):
                parsed = urlparse(file_path)
                name_from_url = unquote(Path(parsed.path).name)
                if name_from_url:
                    derived_file_name = name_from_url
            elif file_path:
                name_from_path = Path(file_path).name
                if name_from_path:
                    derived_file_name = name_from_path

            if derived_file_name:
                normalized["file_name"] = derived_file_name
            else:
                normalized["file_name"] = f"document.{file_type}" if file_type else "document.unknown"

        if not normalized.get("source_type"):
            normalized["source_type"] = "cloud"

        if not normalized.get("chunk_size"):
            normalized["chunk_size"] = settings.CHUNK_SIZE

        if not normalized.get("chunk_overlap"):
            normalized["chunk_overlap"] = settings.CHUNK_OVERLAP

        normalized["retry_count"] = int(normalized.get("retry_count") or 0)

        if not normalized.get("task_type"):
            # Wiki Backend legacy payload (job_id + file_url) always represents upload tasks.
            if original.get("job_id") or original.get("file_url"):
                normalized["task_type"] = "upload"

        metadata = dict(normalized.get("metadata") or {})
        if normalized.get("task_id"):
            metadata.setdefault("task_id", normalized.get("task_id"))
        if normalized.get("document_id"):
            metadata.setdefault("wiki_document_id", normalized.get("document_id"))
        if normalized.get("user_id"):
            metadata.setdefault("wiki_user_id", normalized.get("user_id"))
        if normalized.get("created_at"):
            metadata.setdefault("created_at", normalized.get("created_at"))
        metadata.setdefault("retry_count", normalized.get("retry_count", 0))
        normalized["metadata"] = metadata

        return normalized

    @staticmethod
    def _build_upload_task_data(task_data):
        """Keep only fields accepted by UploadTask dataclass."""
        return {
            "task_id": task_data.get("task_id"),
            "file_path": task_data.get("file_path"),
            "file_name": task_data.get("file_name"),
            "source_type": task_data.get("source_type", "local"),
            "chunk_size": task_data.get("chunk_size", settings.CHUNK_SIZE),
            "chunk_overlap": task_data.get("chunk_overlap", settings.CHUNK_OVERLAP),
            "metadata": task_data.get("metadata") or {},
            "created_at": task_data.get("created_at") or datetime.now().isoformat(),
        }

    @staticmethod
    def _build_edit_task_data(task_data):
        """Keep only fields accepted by EditTask dataclass."""
        return {
            "task_id": task_data.get("task_id"),
            "document_id": task_data.get("document_id"),
            "file_path": task_data.get("file_path"),
            "file_name": task_data.get("file_name"),
            "chunk_size": task_data.get("chunk_size", settings.CHUNK_SIZE),
            "chunk_overlap": task_data.get("chunk_overlap", settings.CHUNK_OVERLAP),
            "metadata": task_data.get("metadata") or {},
            "created_at": task_data.get("created_at") or datetime.now().isoformat(),
        }

    @staticmethod
    def _truncate_header(value, max_length=512):
        if not value:
            return None
        return str(value)[:max_length]
    
    def convert_html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown format with cleaning and normalization
        
        Pipeline: Clean HTML → Convert to Markdown → Normalize Markdown
        """
        logger.info("[Convert] Starting HTML → Markdown pipeline")
        
        # Step 1: Clean HTML using BeautifulSoup
        logger.info("[Convert] Step 1: Cleaning HTML with BeautifulSoup...")
        cleaned_html = clean_wikipedia_html(html_content)
        
        # Step 2: Convert to Markdown without temp files
        logger.info("[Convert] Step 2: Converting to Markdown without temp files...")
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
        logger.info("[Convert] ✅ Fileless HTML → Markdown conversion successful")

        # Step 3: Normalize Markdown
        logger.info("[Convert] Step 3: Normalizing Markdown structure...")
        normalized_markdown = normalize_markdown(markdown_content)
        logger.info("[Convert] ✅ Markdown normalization complete")
        
        return normalized_markdown

    def _read_source_content(self, file_path: str) -> tuple[str, int]:
        """Stream source content from a local file or URL into memory."""
        if file_path.lower().startswith(("http://", "https://")):
            response = requests.get(file_path, stream=True, timeout=120)
            response.raise_for_status()

            buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    buffer.write(chunk)

            raw_bytes = buffer.getvalue()
            try:
                return raw_bytes.decode("utf-8"), len(raw_bytes)
            except UnicodeDecodeError:
                guessed_encoding = response.apparent_encoding or response.encoding or "utf-8"
                try:
                    return raw_bytes.decode(guessed_encoding), len(raw_bytes)
                except UnicodeDecodeError:
                    return raw_bytes.decode("utf-8", errors="replace"), len(raw_bytes)

        with open(file_path, "rb") as source_file:
            raw_bytes = source_file.read()

        try:
            return raw_bytes.decode("utf-8"), len(raw_bytes)
        except UnicodeDecodeError:
            return raw_bytes.decode("utf-8", errors="replace"), len(raw_bytes)
    
    def process_upload_task(self, task: UploadTask) -> TaskResult:
        """Process document upload task with detailed logging"""
        logger.info("\n" + "="*70)
        logger.info(f"📥 UPLOAD TASK STARTED: {task.task_id}")
        logger.info("="*70)
        task_start = time.time()
        document_id = None
        
        try:
            self._set_processing_progress(
                task_id=task.task_id,
                current=1,
                total=5,
                message="Reading and validating input file",
                file_name=task.file_name,
                task_type="upload",
            )

            logger.info(f"[{task.task_id}] 📂 Step 1: Reading file")
            logger.info(f"[{task.task_id}]    Path: {task.file_path}")

            content, file_size = self._read_source_content(task.file_path)
            original_filename = task.file_name
            
            # Check if file is HTML and convert to Markdown
            if task.file_name.lower().endswith('.html'):
                logger.info(f"[{task.task_id}] 🔄 HTML file detected, converting to Markdown...")
                content = self.convert_html_to_markdown(content)
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            logger.info(f"[{task.task_id}] ✅ File loaded: {format_bytes(file_size)}")
            logger.info(f"[{task.task_id}]    Hash: {content_hash[:16]}...")
            logger.info(f"[{task.task_id}]    Content: {len(content)} chars")
            
            # Prepare document for indexing
            self._set_processing_progress(
                task_id=task.task_id,
                current=2,
                total=5,
                message="Preparing content for indexing",
            )
            
            # Create Document record with 'pending' status
            self._set_processing_progress(
                task_id=task.task_id,
                current=3,
                total=5,
                message="Creating document record",
            )
            logger.info(f"[{task.task_id}] 💾 Step 3: Creating document record...")
            payload_metadata = dict(task.metadata or {})
            payload_document_id = (
                payload_metadata.get("wiki_document_id")
                or payload_metadata.get("document_id")
            )

            document = None
            if payload_document_id:
                document = self.db.query(Document).filter(Document.id == payload_document_id).one_or_none()
                if document is None:
                    raise ValueError(
                        f"Document not found for provided document_id: {payload_document_id}"
                    )

            if document is None:
                document = self.db.query(Document).from_statement(
                    text("SELECT * FROM documents WHERE metadata->>'task_id' = :task_id LIMIT 1")
                ).params(task_id=task.task_id).one_or_none()

            if document is not None:
                logger.info(f"[{task.task_id}] ♻️ Reusing existing document: {document.id}")
                if document.parent_chunks:
                    document.parent_chunks.clear()
                if document.child_chunks:
                    document.child_chunks.clear()
            else:
                document = Document(
                    id=str(uuid.uuid4()),
                    file_name=task.file_name,
                    file_path=task.file_path,
                    source_type=task.source_type,
                    status="pending",
                    meta_data=payload_metadata,
                    file_size=file_size,
                    content_hash=content_hash
                )
                self.db.add(document)

            document.file_name = task.file_name
            document.file_path = task.file_path
            document.source_type = task.source_type
            document.meta_data = task.metadata or {}
            document.file_size = file_size
            document.content_hash = content_hash
            self._update_document_pipeline_status(document, task.task_id, "pending")
            self.db.commit()
            document_id = str(document.id)
            logger.info(f"[{task.task_id}] ✅ Document ready: {document_id}")
            logger.info(f"[{task.task_id}]    Initial status: pending")
            
            # Update to 'indexing' status
            self._update_document_pipeline_status(document, task.task_id, "processing")
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: pending → indexing")
            
            # Process content - chunking
            self._set_processing_progress(
                task_id=task.task_id,
                current=4,
                total=5,
                message="Chunking and embedding content",
                document_id=document_id,
            )
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
                    h1=self._truncate_header(chunk_metadata.get('h1')),
                    h2=self._truncate_header(chunk_metadata.get('h2')),
                    h3=self._truncate_header(chunk_metadata.get('h3')),
                    chunk_index=chunk.get('chunk_index', i-1)
                )
                self.db.add(child_chunk)
                chunks_created += 1
                
                if i % 10 == 0:
                    logger.info(f"[{task.task_id}]    Progress: {i}/{len(chunks)} chunks indexed")
                    self._set_processing_progress(
                        task_id=task.task_id,
                        current=4,
                        total=5,
                        message=f"Embedding chunks ({i}/{len(chunks)})",
                        document_id=document_id,
                        chunk_progress={
                            "current": i,
                            "total": len(chunks),
                        },
                    )
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ All {chunks_created} chunks indexed")
            
            # Final status update is composite: RAG completed + Graph status
            self._update_document_pipeline_status(document, task.task_id, "completed")
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: indexing → {document.status}")

            self._set_processing_progress(
                task_id=task.task_id,
                current=5,
                total=5,
                message="Upload task completed",
                document_id=document_id,
            )
            
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
                        self._update_document_pipeline_status(document, task.task_id, "failed", e)
                        self.db.commit()
                        logger.info(f"[{task.task_id}] 📌 Status transition: indexing → failed")
                        logger.info(f"[{task.task_id}]    Error saved: {str(e)[:80]}...")
                except Exception as db_error:
                    logger.error(f"[{task.task_id}] Failed to update status to failed: {db_error}")
                    try:
                        self.db.rollback()
                    except:
                        pass
            
            elapsed = time.time() - task_start
            logger.error(f"[{task.task_id}] Elapsed time before failure: {elapsed:.2f}s")
            logger.error("="*70 + "\n")

            if self._is_transient_db_error(e):
                try:
                    engine.dispose()
                except Exception as dispose_error:
                    logger.warning(f"[{task.task_id}] Failed to dispose SQLAlchemy engine: {dispose_error}")
                raise RetryableTaskError(error_msg, original_error=e) from e
            
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
            self._set_processing_progress(
                task_id=task.task_id,
                current=1,
                total=4,
                message="Validating target document and input file",
                document_id=task.document_id,
                file_name=task.file_name,
                task_type="edit",
            )

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
            self._set_processing_progress(
                task_id=task.task_id,
                current=2,
                total=4,
                message="Removing old chunks",
                document_id=task.document_id,
            )
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
            self._update_document_pipeline_status(document, task.task_id, "processing")
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: {old_status} → indexing")
            
            # Chunk new content
            self._set_processing_progress(
                task_id=task.task_id,
                current=3,
                total=4,
                message="Re-chunking and embedding new content",
                document_id=task.document_id,
            )
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
                    h1=self._truncate_header(chunk_metadata.get('h1')),
                    h2=self._truncate_header(chunk_metadata.get('h2')),
                    h3=self._truncate_header(chunk_metadata.get('h3')),
                    chunk_index=chunk.get('chunk_index', i-1)
                )
                self.db.add(child_chunk)
                chunks_created += 1
                
                if i % 10 == 0:
                    logger.info(f"[{task.task_id}]    Progress: {i}/{len(chunks)} chunks indexed")
                    self._set_processing_progress(
                        task_id=task.task_id,
                        current=3,
                        total=4,
                        message=f"Re-indexing chunks ({i}/{len(chunks)})",
                        document_id=task.document_id,
                        chunk_progress={
                            "current": i,
                            "total": len(chunks),
                        },
                    )
            
            self.db.commit()
            logger.info(f"[{task.task_id}] ✅ All {chunks_created} new chunks indexed")
            
            # Final status update is composite: RAG completed + Graph status
            self._update_document_pipeline_status(document, task.task_id, "completed")
            self.db.commit()
            logger.info(f"[{task.task_id}] 📌 Status transition: indexing → {document.status}")

            self._set_processing_progress(
                task_id=task.task_id,
                current=4,
                total=4,
                message="Edit task completed",
                document_id=task.document_id,
            )
            
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
                    self._update_document_pipeline_status(document, task.task_id, "failed", e)
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

            if self._is_transient_db_error(e):
                try:
                    engine.dispose()
                except Exception as dispose_error:
                    logger.warning(f"[{task.task_id}] Failed to dispose SQLAlchemy engine: {dispose_error}")
                raise RetryableTaskError(error_msg, original_error=e) from e
            
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
            while not self._shutdown_requested:
                claimed = self.queue_service.claim_task_blocking(timeout=5)
                if not claimed:
                    idle_seconds += 5
                    if idle_seconds % 30 == 0:
                        logger.debug(f"⏳ Idle: {idle_seconds}s (total tasks: {task_count})")
                    continue

                idle_seconds = 0
                task_data = claimed["task"]
                raw_payload = claimed["raw_payload"]
                task_data = self._normalize_payload(task_data)
                should_ack = True
                task_owner = str(task_data.get("type") or "").lower()

                if task_owner not in {"rag", "graph", ""}:
                    released = self.queue_service.release_unhandled_task(raw_payload)
                    should_ack = False
                    logger.info(
                        "↩️ Released non-RAG task back to shared queue: owner=%s removed=%s",
                        task_owner,
                        released,
                    )
                    continue

                task_type = str(task_data.get("task_type") or "").lower()
                retry_count = int(task_data.get("retry_count", 0))

                if task_type not in {"upload", "edit"}:
                    # Backward compatibility: infer type from payload shape.
                    task_type = "edit" if "document_id" in task_data else "upload"

                task_id = str(task_data.get("task_id") or "")
                self._current_task_id = task_id
                self._current_task_raw_payload = raw_payload

                try:
                    if not task_id:
                        raise ValueError("Task payload missing task_id")

                    if retry_count > settings.WORKER_MAX_RETRIES:
                        self.queue_service.push_dead_letter_task(
                            task_data,
                            reason=(
                                "Task retry_count exceeded configured limit "
                                f"({settings.WORKER_MAX_RETRIES})"
                            ),
                        )
                        self.queue_service.set_task_status(
                            task_id=task_id,
                            status=TASK_STATUS_FAILED,
                            message="Task moved to dead-letter queue before execution",
                            retry_count=retry_count,
                            dead_lettered=True,
                        )
                        logger.error(
                            f"❌ Poison-pill task moved to dead-letter queue: {task_id}, retry_count={retry_count}"
                        )
                        continue

                    logger.info(f"\n📨 Task received from queue: {task_id} ({task_type})")
                    self.queue_service.set_task_status(
                        task_id=task_id,
                        status=TASK_STATUS_PROCESSING,
                        message="Worker picked up task",
                        task_type=task_type,
                        file_name=task_data.get("file_name"),
                        document_id=task_data.get("document_id"),
                        progress={"current": 0, "total": 1, "percent": 0},
                        retry_count=retry_count,
                        processing_started_at=datetime.now().astimezone().isoformat(),
                    )

                    self._refresh_db_session()

                    if task_type == "upload":
                        task = UploadTask.from_dict(self._build_upload_task_data(task_data))
                        result = self.process_upload_task(task)
                    else:
                        task = EditTask.from_dict(self._build_edit_task_data(task_data))
                        result = self.process_edit_task(task)

                    self.queue_service.set_result(task_id, result.to_dict())

                    if result.status == "completed":
                        self.queue_service.set_task_status(
                            task_id=task_id,
                            status=TASK_STATUS_COMPLETED,
                            message=result.message or "Task completed",
                            document_id=result.document_id,
                            progress={"current": 1, "total": 1, "percent": 100},
                        )
                        self._notify_pipeline_webhook(
                            task_id=task_id,
                            document_id=result.document_id,
                            file_name=task_data.get("file_name"),
                            status="completed",
                            message=result.message or "Task completed",
                        )
                    else:
                        self.queue_service.set_task_status(
                            task_id=task_id,
                            status=TASK_STATUS_FAILED,
                            message="Task failed",
                            error=result.error,
                            document_id=result.document_id,
                        )
                        self.queue_service.push_failed_task(
                            task_data,
                            reason=result.error or "Task failed",
                            error=result.error,
                        )
                        self._notify_pipeline_webhook(
                            task_id=task_id,
                            document_id=result.document_id,
                            file_name=task_data.get("file_name"),
                            status="failed",
                            message="Task failed",
                            error=result.error,
                        )

                    task_count += 1
                    logger.info(f"✅ Result stored in Redis for task {task_id}\n")

                except RetryableTaskError as retryable_error:
                    logger.warning(f"⚠️ Retryable task failure: {retryable_error}")
                    self._requeue_retryable_task(task_data, task_id, task_type, retryable_error)

                except Exception as task_error:
                    logger.error(f"❌ Task processing crashed: {task_error}", exc_info=True)
                    if task_id:
                        self.queue_service.set_task_status(
                            task_id=task_id,
                            status=TASK_STATUS_FAILED,
                            message="Worker crashed while processing task",
                            error=str(task_error),
                            document_id=task_data.get("document_id"),
                        )
                        self.queue_service.push_failed_task(
                            task_data,
                            reason="Worker crashed while processing task",
                            error=str(task_error),
                        )
                        self._notify_pipeline_webhook(
                            task_id=task_id,
                            document_id=task_data.get("document_id"),
                            file_name=task_data.get("file_name"),
                            status="failed",
                            message="Worker crashed while processing task",
                            error=str(task_error),
                        )
                finally:
                    self._close_db_session()
                    if raw_payload and should_ack:
                        removed = self.queue_service.ack_processing_task(raw_payload)
                        logger.debug(f"Ack processing queue: removed={removed}, task_id={task_id}")
                    self._current_task_id = None
                    self._current_task_raw_payload = None

            logger.info("🛑 Shutdown requested, worker stopped accepting new tasks.")
        
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
