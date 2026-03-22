"""
Chunking Service for document processing
Handles hierarchical markdown splitting and text chunking
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document as LangChainDocument

class ChunkingService:
    """
    Service for chunking documents with hierarchical structure
    Supports markdown header splitting and recursive character splitting
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize chunking service

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Markdown header splitter configuration
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        self.section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

        # Recursive character splitter for sub-chunking
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_markdown(self, text: str, source_file: str = "") -> Dict[str, Any]:
        """
        Chunk a markdown document with hierarchical structure

        Args:
            text: Markdown text content
            source_file: Source file path for metadata

        Returns:
            Dictionary with:
            {
                'parent_chunks': List[Dict] - Large sections that were split,
                'child_chunks': List[Dict] - All chunks (with parent_id if belongs to parent)
            }
        """
        # Step 1: Split by headers
        section_docs = self.section_splitter.split_text(text)

        parent_chunks = []
        child_chunks = []
        child_chunk_index = 0

        # Step 2: Process each section
        for section_idx, section_doc in enumerate(section_docs):
            # Extract headers from metadata
            headers = {
                'h1': section_doc.metadata.get('h1', ''),
                'h2': section_doc.metadata.get('h2', ''),
                'h3': section_doc.metadata.get('h3', ''),
            }

            # If section is too large, split further
            if len(section_doc.page_content) > self.chunk_size:
                # Save parent chunk (original section)
                parent_chunk = {
                    'content': section_doc.page_content,
                    'chunk_index': section_idx,
                    'metadata': {
                        **headers,
                        'source': source_file
                    },
                    'section_id': section_idx
                }
                parent_chunks.append(parent_chunk)

                # Split section into sub-chunks
                sub_chunks = self.recursive_splitter.split_documents([section_doc])

                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    child_chunks.append({
                        'content': sub_chunk.page_content,
                        'chunk_index': child_chunk_index,
                        'parent_section_id': section_idx,  # Mark which parent it belongs to
                        'metadata': {
                            **headers,
                            'section_id': section_idx,
                            'sub_chunk_id': sub_idx,
                            'source': source_file
                        }
                    })
                    child_chunk_index += 1
            else:
                # Section is small, save directly to child_chunks (no parent)
                child_chunks.append({
                    'content': section_doc.page_content,
                    'chunk_index': child_chunk_index,
                    'parent_section_id': None,  # No parent
                    'metadata': {
                        **headers,
                        'section_id': section_idx,
                        'sub_chunk_id': None,
                        'source': source_file
                    }
                })
                child_chunk_index += 1

        return {
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks
        }


def get_chunking_service(chunk_size: int = 800, chunk_overlap: int = 150) -> ChunkingService:
    """
    Factory function to create ChunkingService

    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        ChunkingService instance
    """
    return ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
