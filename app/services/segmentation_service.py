"""
Vietnamese Word Segmentation Service using Underthesea.

Outputs multi-word tokens joined by underscore
(e.g., "Hà Nội" → "Hà_Nội", "tổng khởi nghĩa" → "tổng_khởi_nghĩa").
These are stored in the `bm25_text` column and used for BM25 indexing via
ParadeDB.  The ParadeDB `unicode_words` tokenizer treats underscore as a
word connector, so "Hà_Nội" stays as a single token "hà_nội".

Underthesea is pure Python, no Java needed, instant startup.
"""
from typing import Optional





class VietnameseSegmentationService:
    """
    Singleton service for Vietnamese word segmentation using Underthesea.
    
    Pure Python implementation - no Java needed, instant initialization.
    """

    _instance: Optional['VietnameseSegmentationService'] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_model()
            self._initialized = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load Underthesea word tokenizer (instant, no Java)."""
        try:
            from underthesea import word_tokenize
            print('✅ Underthesea loaded successfully')
        except ImportError:
            print('⚠️ underthesea not installed. Run: pip install underthesea')
        except Exception as e:
            print(f'⚠️ Failed to initialize Underthesea: {e}')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, text: str) -> str:
        """Segment Vietnamese text; multi-word tokens are joined by underscore.

        Example::

            Input:  "Hà Nội chiếm dinh Khâm sai Bắc bộ"
            Output: "Hà_Nội chiếm dinh Khâm_sai Bắc_bộ"

        Falls back to the original text when the model is unavailable.
        """
        if not text or not text.strip():
            return ''
        if self._model is None:
            return text.strip()
        try:
            # Normalize pre-existing underscore word-joins (e.g. from Wikipedia
            # source or previous segmentation passes) back to spaces so that
            # VnCoreNLP receives clean Vietnamese text and can re-segment
            # correctly.  Without this, "Lê_Thái_Tông" is mishandled and
            # produces garbled output like "Lê__ Thái _ Tông".
            clean = text.replace('_', ' ')
            sentences = self._model.tokenize(clean)
            return ' '.join(' '.join(sent) for sent in sentences)
        except Exception as e:
            print(f'⚠️ VnCoreNLP segmentation error: {e}')
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

