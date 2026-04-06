"""
Vietnamese Word Segmentation Service using Underthesea.

Outputs multi-word tokens joined by underscore
(e.g., "Hà Nội" → "Hà_Nội", "tổng khởi nghĩa" → "tổng_khởi_nghĩa").
These are stored in the `bm25_text` column and used for BM25 indexing via
ParadeDB.  The ParadeDB `unicode_words` tokenizer treats underscore as a
word connector, so "Hà_Nội" stays as a single token "hà_nội".

Underthesea is pure Python, no Java needed, instant startup.
"""
from typing import Optional, Callable


def _load_underthesea():
    """Safely load Underthesea's word_tokenize function."""
    try:
        from underthesea import word_tokenize
        return word_tokenize
    except ImportError:
        print('⚠️ underthesea not installed. Run: pip install underthesea')
        return None
    except Exception as e:
        print(f'⚠️ Failed to load Underthesea: {e}')
        return None


# Load word_tokenize function at module level (instant, no Java)
_word_tokenize: Optional[Callable] = _load_underthesea()
if _word_tokenize:
    print('✅ Underthesea loaded successfully')



class VietnameseSegmentationService:
    """
    Singleton service for Vietnamese word segmentation using Underthesea.
    
    Pure Python implementation - no Java needed, instant initialization.
    """

    _instance: Optional['VietnameseSegmentationService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, text: str) -> str:
        """Segment Vietnamese text; multi-word tokens are joined by underscore.

        Example::

            Input:  "Hà Nội chiếm dinh Khâm sai Bắc bộ"
            Output: "Hà_Nội chiếm dinh Khâm_sai Bắc_bộ"

        Falls back to the original text when word_tokenize is unavailable.
        """
        if not text or not text.strip():
            return ''
        
        if _word_tokenize is None:
            return text.strip()
        
        try:
            # Normalize pre-existing underscore word-joins (e.g. from Wikipedia
            # source or previous segmentation passes) back to spaces so that
            # Underthesea receives clean Vietnamese text and can re-segment
            # correctly.  Without this, "Lê_Thái_Tông" may be mishandled.
            clean = text.replace('_', ' ')
            # word_tokenize returns text with multi-word tokens joined by underscore
            result = _word_tokenize(clean, format='text')
            return result
        except Exception as e:
            print(f'⚠️ Underthesea segmentation error: {e}')
            return text.strip()

    def segment_query(self, query: str) -> str:
        """Segment a search query to match bm25_text tokens in the index."""
        import re
        # Strip punctuation so "ai?" and "ai ?" are treated identically
        clean = re.sub(r'[^\w\s]', ' ', query.strip(), flags=re.UNICODE)
        clean = re.sub(r' +', ' ', clean).strip()
        return self.segment(clean)


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------

_segmentation_service: Optional[VietnameseSegmentationService] = None


def get_segmentation_service() -> VietnameseSegmentationService:
    """Return the singleton VietnameseSegmentationService."""
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = VietnameseSegmentationService()
    return _segmentation_service

