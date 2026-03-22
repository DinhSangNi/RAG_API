"""
Combined worker for processing documents: ingest + chunk in one job
Handles document ingestion, chunking, embedding, and database storage
"""

import os
import re
import time
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import markitdown
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database.models import Document, ChildChunk, SummaryDocument
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service
from app.services.segmentation_service import get_segmentation_service


# ============================================================================
# HTML Cleaning Functions (from src/preprocessing/html_cleaner.py)
# ============================================================================

def clean_wikipedia_html(html_file_path):
    """
    Clean Wikipedia HTML content by extracting main content and removing unwanted elements

    Args:
        html_file_path: Path to HTML file

    Returns:
        Cleaned markdown text
    """
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find Wikipedia content area, if not found use body or entire content
    content = soup.find("div", class_="mw-parser-output")

    if not content:
        print("Không tìm thấy vùng Wikipedia, sẽ clean toàn bộ HTML")
        # Try to find body tag, if not use soup
        content = soup.find("body")
        if not content:
            content = soup

    # Remove unwanted tags and elements
    tags_without_class = ['audio', 'style', 'img', 'sup', 'link', 'input', 'script', 'meta', 'noscript']
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
        ('div', 'toc'),  # remove table of contents (left side)
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

    # Convert quote tables to paragraphs
    for table in content.find_all('table', class_="cquote"):
        quote_text = table.get_text(separator=" ", strip=True)
        p = soup.new_tag('p')
        p.string = quote_text
        table.replace_with(p)

    # Remove empty paragraphs
    empty_paragraphs = [p for p in content.find_all('p') if p is not None and not p.get_text(strip=True)]
    for p in empty_paragraphs:
        p.decompose()

    # Remove empty spans with IDs
    empty_spans = [span for span in content.find_all('span') if span is not None and hasattr(span, 'get') and span.get('id') and not span.get_text(strip=True)]
    for span in empty_spans:
        span.decompose()

    # Handle figure captions
    figures = [f for f in content.find_all('figure') if f is not None]
    for figure in figures:
        figcaption = figure.find('figcaption')
        if figcaption:
            new_p = soup.new_tag('p')
            new_p.string = f"[Hình ảnh: {figcaption.get_text(strip=True)}]"
            figure.replace_with(new_p)
        else:
            figure.decompose()

    # Unwrap anchor tags
    anchor_tags = [a for a in content.find_all('a') if a is not None]
    for a_tag in anchor_tags:
        a_tag.unwrap()

    # Unwrap span tags
    span_tags = [span for span in content.find_all('span') if span is not None]
    for span in span_tags:
        span.unwrap()

    # Unwrap bold tags
    for tag in content.find_all(['b']):
        tag.unwrap()

    # Remove unwanted sections
    sections_to_kill = [
        "Tham_khảo",
        "Tài liệu tham khảo",
        "Chú giải",
        "Liên_kết_ngoài",
        "Danh_mục",
        "Ghi_chú",
        "Thư_mục_hậu_cần",
        "Đọc_thêm",
        "Chú_thích",
        "Thư_mục",
        "Nguồn_thứ_cấp",
        "Nguồn_sơ_cấp",
        "Nguồn_trích_dẫn",
        "Diễn_văn_của_Hồ_Chí_Minh",
        "Tác_phẩm_của_Hồ_Chí_Minh",
        "Viết_về_Hồ_Chí_Minh",
        "Những_người_từng_gặp_Hồ_Chí_Minh_kể_về_ông"
    ]

    for section_id in sections_to_kill:
        header = content.find(['h2', 'h3'], id=section_id)
        if header:
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                sibling.decompose()
            header.decompose()

    # Remove empty <li> elements
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

    input_file_name = Path(html_file_path).stem
    temp_html_file_path = f"data/raw_data/wikipedia/temp_clean_html/{input_file_name}.html"
    os.makedirs(os.path.dirname(temp_html_file_path), exist_ok=True)
    with open(temp_html_file_path, "w", encoding="utf-8") as f:
        f.write(str(content))

    print(f"Đã lưu HTML đã làm sạch vào: {temp_html_file_path}")

    return temp_html_file_path


# ============================================================================
# Markdown Normalization Functions (from src/preprocessing/normalize_markdown.py)
# ============================================================================

def normalize_markdown(md_text):
    lines = md_text.split('\n')
    normalized_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Nếu là header, giữ nguyên
        if line.strip().startswith('#'):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Nếu là bullet point, table, hoặc dòng trống, giữ nguyên
        if (line.strip().startswith(('* ', '- ', '+ ', '|', '>')) or 
            re.match(r'^\s*\d+\.', line) or
            line.strip() == '' or
            line.strip().startswith(':')):
            normalized_lines.append(line)
            i += 1
            continue
        
        # Gộp các dòng liên tiếp không phải là đoạn đặc biệt
        paragraph = line
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Dừng nếu gặp dòng trống, header, bullet, table
            if (next_line.strip() == '' or 
                next_line.strip().startswith(('#', '* ', '- ', '+ ', '|', '>', ':')) or
                re.match(r'^\s*\d+\.', next_line)):
                break
            # Gộp dòng
            paragraph += ' ' + next_line.strip()
            i += 1
        
        normalized_lines.append(paragraph)
    
    # Join và làm sạch khoảng trắng thừa
    result = '\n'.join(normalized_lines)
    
    # Chuẩn hóa bullet points: chuyển tất cả thành *
    result = re.sub(r'^\s*[-+]\s+', '* ', result, flags=re.MULTILINE)
    
    # Loại bỏ khoảng trắng thừa ở cuối dòng
    result = re.sub(r' +\n', '\n', result)
    
    # Chuẩn hóa block quotes: chuyển : thành >
    result = re.sub(r'^:   \*', '>   *', result, flags=re.MULTILINE)
    result = re.sub(r'^:\s+', '> ', result, flags=re.MULTILINE)
    
    # Đảm bảo có dòng trống trước header
    result = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', result)
    # Loại bỏ nhiều dòng trống liên tiếp (giữ tối đa 2)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def convert_html_to_normalized_md(html_file_path, output_md_file_path=None):
    md = markitdown.MarkItDown()   
    rs = md.convert(html_file_path)
    normalized_md = normalize_markdown(rs.text_content)

    if output_md_file_path is None:
        name = Path(html_file_path).stem
        output_md_file_path = "data/processed_data/{}.md".format(name)
    
    os.makedirs(os.path.dirname(output_md_file_path), exist_ok=True)
    with open(output_md_file_path, "w", encoding="utf-8") as f:
        f.write(normalized_md)

    print(f"Đã lưu markdown đã chuẩn hóa vào: {output_md_file_path}")

    return output_md_file_path


def extract_text_from_html_file(html_file_path: str) -> str:
    """Extract plain text from HTML as a fallback when markdown conversion returns empty content."""
    with open(html_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    content = soup.find("div", class_="mw-parser-output") or soup.find("body") or soup
    text = content.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file_fallback(file_path: str) -> str:
    """Read text from file using tolerant UTF-8 decoding as a last-resort fallback."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


# Tạo database session cho worker
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def process_document(
    job_id: str,
    document_id: str,
    file_path: str,
    source_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    batch_id: str | None = None,
    is_summary: bool = False,
    summary_id: str | None = None,
    target_document_ids: list | None = None
):
    """
    Worker function to process complete document: ingest + chunk + embeddings

    Workflow:
    1. Clean HTML (if HTML)
    2. Convert to Markdown
    3. Normalize Markdown
    4. If is_summary=True: Create summary document
       If is_summary=False: Chunk document + embeddings
    5. Save to PostgreSQL

    Args:
        job_id: Unique job identifier
        document_id: Document UUID
        file_path: Path to file to process
        source_type: Source type (local, cloud, wikipedia)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_id: Batch identifier for tracking
        is_summary: If True, file is processed as summary document
        summary_id: UUID of summary document - child chunks will be linked to this summary
        target_document_ids: List of document IDs to link this summary to (for summary documents)
    """
    db = SessionLocal()

    # Initialize timing dictionary
    timing_stats = {
        'total_start': time.time(),
        'phases': {}
    }

    try:
        # Lấy document record (chỉ khi không phải summary)
        document = None
        if not is_summary:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document không tồn tại: {document_id}")
            
            document.status = "processing"  # type: ignore[assignment]
            db.commit()
        
        # ========================================================================
        # PHASE 1: INGEST (0-30%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"📦 PHASE 1: INGEST - Starting for document {document_id}")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Kiểm tra loại file
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension in ['.md', '.markdown']:
            # File markdown - sử dụng trực tiếp
            print(f"📝 Detected Markdown file: {file_path}")
            md_file_path = file_path
            
            # Update document metadata
            if document:
                meta_data = document.meta_data or {}  # type: ignore[assignment]
                if isinstance(meta_data, dict):
                    meta_data['processed_file'] = md_file_path
                    meta_data['original_file'] = file_path
                    meta_data['file_type'] = 'markdown'
                    document.meta_data = meta_data  # type: ignore[assignment]
                db.commit()
            
            print(f"✅ Using markdown file directly (no conversion needed)")
            timing_stats['phases']['ingest_total'] = time.time() - phase_start
            
        elif file_extension in ['.html', '.htm']:
            # File HTML - cần clean và convert
            print(f"🌐 Detected HTML file: {file_path}")
            
            # Bước 1: Clean HTML
            if job:
                job.meta['progress'] = {'step': 'cleaning_html', 'current': 5, 'total': 100}
                job.save_meta()
            
            step_start = time.time()
            print(f"🧹 Cleaning HTML: {file_path}")
            cleaned_file_path = clean_wikipedia_html(file_path)
            step_duration = time.time() - step_start
            print(f"✅ Cleaned HTML: {cleaned_file_path}")
            print(f"⏱️  HTML Cleaning took: {step_duration:.2f}s")
            timing_stats['phases']['html_cleaning'] = step_duration
            
            # Bước 2: Convert to Markdown
            if job:
                job.meta['progress'] = {'step': 'converting_to_markdown', 'current': 15, 'total': 100}
                job.save_meta()
            
            step_start = time.time()
            print(f"📝 Converting to Markdown: {cleaned_file_path}")
            md_file_path = convert_html_to_normalized_md(cleaned_file_path)
            step_duration = time.time() - step_start
            print(f"✅ Converted to Markdown: {md_file_path}")
            print(f"⏱️  Markdown Conversion took: {step_duration:.2f}s")
            timing_stats['phases']['markdown_conversion'] = step_duration
            
            # Bước 3: Update document metadata (skip for summary)
            if job:
                job.meta['progress'] = {'step': 'updating_metadata', 'current': 30, 'total': 100}
                job.save_meta()
            
            if document:
                meta_data = document.meta_data or {}  # type: ignore[assignment]
                if isinstance(meta_data, dict):
                    meta_data['processed_file'] = md_file_path
                    meta_data['original_file'] = file_path
                    meta_data['file_type'] = 'html'
                    document.meta_data = meta_data  # type: ignore[assignment]
                db.commit()
            
            phase_duration = time.time() - phase_start
            timing_stats['phases']['ingest_total'] = phase_duration
            print(f"✅ Ingest phase completed")
            print(f"⏱️  Total INGEST time: {phase_duration:.2f}s")
            
        else:
            # Loại file khác - thử convert trực tiếp bằng MarkItDown
            print(f"📄 Detected file type: {file_extension}")
            
            if job:
                job.meta['progress'] = {'step': 'converting_to_markdown', 'current': 15, 'total': 100}
                job.save_meta()
            
            step_start = time.time()
            print(f"📝 Converting to Markdown: {file_path}")
            md_file_path = convert_html_to_normalized_md(file_path)
            step_duration = time.time() - step_start
            print(f"✅ Converted to Markdown: {md_file_path}")
            print(f"⏱️  Conversion took: {step_duration:.2f}s")
            timing_stats['phases']['markdown_conversion'] = step_duration
            
            if job:
                job.meta['progress'] = {'step': 'updating_metadata', 'current': 30, 'total': 100}
                job.save_meta()
            
            if document:
                meta_data = document.meta_data or {}  # type: ignore[assignment]
                if isinstance(meta_data, dict):
                    meta_data['processed_file'] = md_file_path
                    meta_data['original_file'] = file_path
                    meta_data['file_type'] = file_extension.lstrip('.')
                    document.meta_data = meta_data  # type: ignore[assignment]
                db.commit()
            
            timing_stats['phases']['ingest_total'] = time.time() - phase_start
        
        # Load markdown content (needed for both paths)
        step_start = time.time()
        with open(md_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.strip()

        # Some pages can produce empty markdown via converter.
        # For summary jobs, fallback to HTML extraction and then raw text loading.
        if is_summary and not text and file_extension in ['.html', '.htm']:
            fallback_candidates = [file_path]
            if 'cleaned_file_path' in locals():
                fallback_candidates.append(cleaned_file_path)

            for candidate in fallback_candidates:
                try:
                    fallback_text = extract_text_from_html_file(candidate)
                    if fallback_text:
                        text = fallback_text
                        print(f"⚠️ Markdown conversion produced empty text; used HTML fallback from: {candidate}")
                        break
                except Exception as fallback_error:
                    print(f"⚠️ HTML fallback failed for {candidate}: {fallback_error}")

        if is_summary and not text:
            raw_candidates = [file_path]
            if 'md_file_path' in locals():
                raw_candidates.append(md_file_path)
            if 'cleaned_file_path' in locals():
                raw_candidates.append(cleaned_file_path)

            seen = set()
            for candidate in raw_candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    fallback_text = read_text_file_fallback(candidate)
                    if fallback_text:
                        text = fallback_text
                        print(f"⚠️ Used raw text fallback from: {candidate}")
                        break
                except Exception as fallback_error:
                    print(f"⚠️ Raw text fallback failed for {candidate}: {fallback_error}")

        if not text:
            raise ValueError("Nội dung sau khi ingest rỗng, không thể tạo embedding/chunks")

        step_duration = time.time() - step_start
        print(f"📄 Loaded file: {md_file_path} ({len(text)} chars)")
        print(f"⏱️  File Loading took: {step_duration:.2f}s")
        timing_stats['phases']['file_loading'] = step_duration
        
        # ========================================================================
        # BRANCH: Process as Summary Document or Regular Document
        # ========================================================================
        if is_summary:
            # ====================================================================
            # SUMMARY DOCUMENT PATH: Create summary_document record
            # ====================================================================
            print(f"\n{'='*70}")
            print(f"📋 PROCESSING AS SUMMARY DOCUMENT")
            print(f"{'='*70}")
            
            # Update progress
            if job:
                job.meta['progress'] = {'step': 'creating_summary_embedding', 'current': 50, 'total': 100}
                job.save_meta()
            
            # Create embedding for summary content
            phase_start = time.time()
            embedding_service = get_embedding_service()
            summary_embedding = embedding_service.embed_text(text)
            step_duration = time.time() - phase_start
            print(f"🎯 Created summary embedding")
            print(f"⏱️  Embedding generation took: {step_duration:.2f}s")
            timing_stats['phases']['embedding_generation'] = step_duration
            timing_stats['phases']['embedding_total'] = step_duration
            
            # Update existing summary document (created by route)
            if job:
                job.meta['progress'] = {'step': 'saving_summary_to_db', 'current': 80, 'total': 100}
                job.save_meta()
            
            phase_start = time.time()
            # Tính content hash cho summary
            content_hash_for_summary = hashlib.sha256(text.encode('utf-8')).hexdigest()
            
            # Get existing summary document (created by route)
            if summary_id:
                summary_doc = db.query(SummaryDocument).filter(SummaryDocument.id == summary_id).first()
                if not summary_doc:
                    raise ValueError(f"Summary document {summary_id} not found")
                
                            # Update existing summary
                summary_doc.summary_content = text
                summary_doc.content_hash = content_hash_for_summary
                summary_doc.vector = summary_embedding
                summary_doc.bm25_text = get_segmentation_service().segment(text)
                summary_doc.status = 'completed'
                
                # Update metadata
                current_meta = summary_doc.meta_data or {}
                current_meta['processed_file'] = md_file_path
                current_meta['original_file'] = file_path
                current_meta['processing_time'] = timing_stats
                summary_doc.meta_data = current_meta
                
                db.commit()
                db.refresh(summary_doc)
                print(f"💾 Updated existing summary document: {summary_doc.id}")
            else:
                            # Fallback: create new summary if summary_id not provided (shouldn't happen)
                summary_doc = SummaryDocument(
                    summary_content=text,
                    content_hash=content_hash_for_summary,
                    vector=summary_embedding,
                    bm25_text=get_segmentation_service().segment(text),
                    status='completed',
                    meta_data={
                        'processed_file': md_file_path,
                        'original_file': file_path,
                        'processing_time': timing_stats
                    }
                )
                db.add(summary_doc)
                db.commit()
                db.refresh(summary_doc)
                print(f"💾 Created new summary document: {summary_doc.id}")
            
            step_duration = time.time() - phase_start
            print(f"⏱️  Database save took: {step_duration:.2f}s")
            timing_stats['phases']['database_save'] = step_duration
            timing_stats['phases']['database_total'] = step_duration
            
            # Finalize job progress
            if job:
                job.meta['progress'] = {'step': 'finalizing', 'current': 95, 'total': 100}
                job.save_meta()
            
            # Update child chunks with summary_id if target_document_ids provided
            if target_document_ids and len(target_document_ids) > 0:
                print(f"\n{'='*70}")
                print(f"🔗 LINKING SUMMARY TO TARGET DOCUMENTS")
                print(f"{'='*70}")
                
                for target_doc_id in target_document_ids:
                    # Get target document to add to association table
                    target_doc = db.query(Document).filter(Document.id == target_doc_id).first()
                    if not target_doc:
                        print(f"⚠️ Document {target_doc_id} not found, skipping")
                        continue
                    
                    # Add document to summary's documents (auto insert to association table)
                    if target_doc not in summary_doc.documents:
                        summary_doc.documents.append(target_doc)
                        print(f"✅ Added document {target_doc_id} to summary associations")
                    
                    # Link all child chunks of target document with this summary (many-to-many)
                    child_chunks = db.query(ChildChunk).filter(ChildChunk.document_id == target_doc_id).all()
                    chunks_linked = 0
                    for chunk in child_chunks:
                        if summary_doc not in chunk.summary_documents:
                            chunk.summary_documents.append(summary_doc)
                            chunks_linked += 1
                    
                    print(f"✅ Linked {chunks_linked} child chunks to summary for document {target_doc_id}")
                
                # Commit all association and chunk updates
                db.commit()
                
                print(f"🔗 Total documents linked: {len(target_document_ids)}")
                print(f"{'='*70}\n")
            
            # Calculate total time
            total_duration = time.time() - timing_stats['total_start']
            timing_stats['total_duration'] = total_duration
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"✅ SUMMARY DOCUMENT PROCESSING COMPLETED")
            print(f"{'='*70}")
            print(f"📊 Document ID: {document_id}")
            print(f"📊 Summary ID: {summary_doc.id}")
            print(f"📊 Content length: {len(text)} chars")
            if target_document_ids:
                print(f"🔗 Linked to {len(target_document_ids)} documents")
            print(f"⏱️  TOTAL TIME: {total_duration:.2f}s")
            print(f"{'='*70}\n")
            
            # Update batch tracking if needed
            if batch_id:
                # Batch tracking removed (Redis dependency removed)
                pass
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'message': f'Đã xử lý summary document thành công',
                'document_id': document_id,
                'summary_id': str(summary_doc.id),
                'progress': {'step': 'completed', 'current': 100, 'total': 100},
                'timing': timing_stats
            }
        
        # ========================================================================
        # REGULAR DOCUMENT PATH: CHUNKING (30-50%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"🔪 PHASE 2: CHUNKING - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 4 was file loading - moved above
        
        # Bước 5: Chunk document
        if job:
            job.meta['progress'] = {'step': 'chunking', 'current': 40, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        chunking_service = get_chunking_service(chunk_size, chunk_overlap)
        chunking_result = chunking_service.chunk_markdown(text, md_file_path)
        parent_chunks = chunking_result['parent_chunks']
        child_chunks = chunking_result['child_chunks']
        step_duration = time.time() - step_start
        print(f"🔪 Created {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")
        print(f"⏱️  Chunking took: {step_duration:.2f}s")
        timing_stats['phases']['chunking'] = step_duration
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['chunking_total'] = phase_duration
        print(f"⏱️  Total CHUNKING time: {phase_duration:.2f}s")

        # ========================================================================
        # PHASE 2b: WORD SEGMENTATION FOR BM25 (child chunks)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"🔤 PHASE 2b: WORD SEGMENTATION (VnCoreNLP) - Starting")
        print(f"{'='*70}")
        seg_start = time.time()

        if job:
            job.meta['progress'] = {'step': 'word_segmentation', 'current': 48, 'total': 100}
            job.save_meta()

        segmentation_service = get_segmentation_service()
        child_bm25_texts = []
        for child_data in child_chunks:
            bm25_text = segmentation_service.segment(child_data['content'])
            child_bm25_texts.append(bm25_text)

        seg_duration = time.time() - seg_start
        print(f"🔤 Segmented {len(child_bm25_texts)} child chunks for BM25")
        print(f"⏱️  Word segmentation took: {seg_duration:.2f}s")
        timing_stats['phases']['word_segmentation'] = seg_duration
        
        # ========================================================================
        # PHASE 3: EMBEDDING (50-95%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"🎯 PHASE 3: EMBEDDING - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 6: Tạo embeddings
        if job:
            job.meta['progress'] = {'step': 'creating_embeddings', 'current': 50, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        embedding_service = get_embedding_service()
        
        # Generate embeddings for parent chunks
        parent_texts = [chunk['content'] for chunk in parent_chunks]
        parent_embeddings = embedding_service.embed_documents(parent_texts) if parent_texts else []
        
        # Generate embeddings for child chunks
        child_texts = [chunk['content'] for chunk in child_chunks]
        child_embeddings = embedding_service.embed_documents(child_texts)
        
        total_embeddings = len(parent_embeddings) + len(child_embeddings)
        step_duration = time.time() - step_start
        print(f"🎯 Created {len(parent_embeddings)} parent embeddings, {len(child_embeddings)} child embeddings")
        print(f"⏱️  Embedding generation took: {step_duration:.2f}s")
        print(f"⏱️  Average per chunk: {step_duration/total_embeddings:.3f}s")
        timing_stats['phases']['embedding_generation'] = step_duration
        timing_stats['phases']['embedding_per_chunk'] = step_duration / total_embeddings if total_embeddings > 0 else 0
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['embedding_total'] = phase_duration
        print(f"⏱️  Total EMBEDDING time: {phase_duration:.2f}s")
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['embedding_total'] = phase_duration
        
        # ========================================================================
        # PHASE 4: DATABASE SAVE (95-100%)
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"💾 PHASE 4: DATABASE SAVE - Starting")
        print(f"{'='*70}")
        phase_start = time.time()
        
        # Bước 7: Xóa chunks cũ (nếu có)
        from app.database.models import ParentChunk
        db.query(ParentChunk).filter(ParentChunk.document_id == document_id).delete()
        db.query(ChildChunk).filter(ChildChunk.document_id == document_id).delete()
        db.commit()
        
        # Bước 8: Lưu parent chunks vào database trước
        if job:
            job.meta['progress'] = {'step': 'saving_parent_chunks', 'current': 60, 'total': 100}
            job.save_meta()
        
        step_start = time.time()
        parent_id_mapping = {}  # Map section_id -> parent_chunk.id
        
        for idx, (parent_data, embedding) in enumerate(zip(parent_chunks, parent_embeddings)):
            parent_chunk = ParentChunk(
                document_id=document_id,
                content=parent_data['content'],
                embedding=embedding,
                chunk_index=parent_data['chunk_index'],
                h1=parent_data['metadata'].get('h1'),
                h2=parent_data['metadata'].get('h2'),
                h3=parent_data['metadata'].get('h3'),
                meta_data=parent_data['metadata']
            )
            db.add(parent_chunk)
            db.flush()  # Get ID immediately
            
            # Map section_id to parent_chunk.id
            parent_id_mapping[parent_data['section_id']] = parent_chunk.id
        
        db.commit()
        print(f"💾 Saved {len(parent_chunks)} parent chunks")
        
        # Bước 9: Lưu child chunks vào database
        if job:
            job.meta['progress'] = {'step': 'saving_child_chunks', 'current': 70, 'total': 100}
            job.save_meta()
        
        total_child_chunks = len(child_chunks)
        for idx, (child_data, embedding) in enumerate(zip(child_chunks, child_embeddings)):
            # Lấy parent_id từ mapping nếu chunk này có parent
            parent_section_id = child_data.get('parent_section_id')
            parent_id = parent_id_mapping.get(parent_section_id) if parent_section_id is not None else None
            
            chunk = ChildChunk(
                document_id=document_id,
                parent_id=parent_id,  # Link to parent chunk
                content=child_data['content'],
                vector=embedding,
                bm25_text=child_bm25_texts[idx] if idx < len(child_bm25_texts) else child_data['content'],
                chunk_index=child_data['chunk_index'],
                section_id=child_data['metadata'].get('section_id'),
                sub_chunk_id=child_data['metadata'].get('sub_chunk_id'),
                h1=child_data['metadata'].get('h1'),
                h2=child_data['metadata'].get('h2'),
                h3=child_data['metadata'].get('h3'),
                meta_data=child_data['metadata']
            )
            db.add(chunk)
            
            # Update progress
            if job and idx % 10 == 0:
                progress = 70 + int((idx / total_child_chunks) * 25)
                job.meta['progress'] = {
                    'step': 'saving_child_chunks',
                    'current': progress,
                    'total': 100,
                    'chunks_saved': idx,
                    'total_chunks': total_child_chunks
                }
                job.save_meta()
        
        db.commit()
        step_duration = time.time() - step_start
        print(f"💾 Saved {len(child_chunks)} child chunks to database")
        print(f"⏱️  Database save took: {step_duration:.2f}s")
        
        # Link child chunks với summary nếu summary_id được provide
        if summary_id:
            print(f"\n🔗 Linking {len(child_chunks)} child chunks to summary {summary_id}")
            summary_doc = db.query(SummaryDocument).filter(SummaryDocument.id == summary_id).first()
            if summary_doc:
                # Query all child chunks của document này
                created_chunks = db.query(ChildChunk).filter(ChildChunk.document_id == document_id).all()
                chunks_linked = 0
                for chunk in created_chunks:
                    if summary_doc not in chunk.summary_documents:
                        chunk.summary_documents.append(summary_doc)
                        chunks_linked += 1
                db.commit()
                print(f"✅ Linked {chunks_linked} child chunks to summary")
            else:
                print(f"⚠️ Summary {summary_id} not found, skipping link")
        
        total_chunks = len(parent_chunks) + len(child_chunks)
        print(f"⏱️  Average per chunk: {step_duration/total_chunks:.3f}s")
        timing_stats['phases']['database_save'] = step_duration
        timing_stats['phases']['db_save_per_chunk'] = step_duration / total_chunks if total_chunks > 0 else 0
        
        # Bước 9: Update document metadata với chunk info
        if job:
            job.meta['progress'] = {'step': 'finalizing', 'current': 95, 'total': 100}
            job.save_meta()
        
        # Update document metadata (only for regular documents, not summaries)
        if document:
            current_meta = document.meta_data or {}  # type: ignore[assignment]
            if isinstance(current_meta, dict):
                current_meta['parent_chunk_count'] = len(parent_chunks)
                current_meta['child_chunk_count'] = len(child_chunks)
                current_meta['chunk_count'] = len(child_chunks)  # Backward compatibility
                current_meta['chunk_size'] = chunk_size
                current_meta['chunk_overlap'] = chunk_overlap
                # Add timing stats to metadata
                current_meta['processing_time'] = timing_stats
                document.meta_data = current_meta  # type: ignore[assignment]
            
            document.status = "completed"  # type: ignore[assignment]
            db.commit()
        
        phase_duration = time.time() - phase_start
        timing_stats['phases']['database_total'] = phase_duration
        print(f"⏱️  Total DATABASE time: {phase_duration:.2f}s")
        
        # Calculate total time
        total_duration = time.time() - timing_stats['total_start']
        timing_stats['total_duration'] = total_duration
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETED - Summary")
        print(f"{'='*70}")
        print(f"📊 Document ID: {document_id}")
        print(f"📊 Total chunks created: {len(parent_chunks) + len(child_chunks)}")
        print(f"⏱️  TOTAL TIME: {total_duration:.2f}s")
        print(f"\n📈 Time Breakdown:")
        print(f"   • Ingest:    {timing_stats['phases'].get('ingest_total', 0):.2f}s ({timing_stats['phases'].get('ingest_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Chunking:  {timing_stats['phases'].get('chunking_total', 0):.2f}s ({timing_stats['phases'].get('chunking_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Embedding: {timing_stats['phases'].get('embedding_total', 0):.2f}s ({timing_stats['phases'].get('embedding_total', 0)/total_duration*100:.1f}%)")
        print(f"   • Database:  {timing_stats['phases'].get('database_total', 0):.2f}s ({timing_stats['phases'].get('database_total', 0)/total_duration*100:.1f}%)")
        print(f"{'='*70}\n")
        
        # Batch tracking removed (Redis dependency removed)
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'message': f'Đã xử lý document và tạo {len(parent_chunks) + len(child_chunks)} chunks thành công',
            'document_id': document_id,
            'progress': {
                'step': 'completed',
                'current': 100,
                'total': 100,
                'chunks_saved': len(child_chunks)
            },
            'timing': timing_stats
        }
        
    except Exception as e:
        # Update summary status to failed for summary jobs.
        if is_summary and summary_id:
            summary_doc = db.query(SummaryDocument).filter(SummaryDocument.id == summary_id).first()
            if summary_doc:
                summary_doc.status = "failed"
                current_meta = summary_doc.meta_data or {}
                current_meta['error'] = str(e)
                summary_doc.meta_data = current_meta
                db.commit()

        # Update document status to failed
        if 'document' in locals() and document:
            document.status = "failed"  # type: ignore[assignment]
            meta_data = document.meta_data or {}  # type: ignore[assignment]
            if isinstance(meta_data, dict):
                meta_data['error'] = str(e)
                document.meta_data = meta_data  # type: ignore[assignment]
            db.commit()
        
        print(f"❌ Error processing document {document_id}: {str(e)}")
        raise
        
    finally:
        db.close()
