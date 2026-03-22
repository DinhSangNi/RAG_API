"""
Vietnamese Word Segmentation Service using VnCoreNLP (wseg annotator).

VnCoreNLP outputs multi-word tokens joined by underscore
(e.g., "Hà Nội" → "Hà_Nội", "tổng khởi nghĩa" → "tổng_khởi_nghĩa").
These are stored in the `bm25_text` column and used for BM25 indexing via
ParadeDB.  The ParadeDB `unicode_words` tokenizer treats underscore as a
word connector, so "Hà_Nội" stays as a single token "hà_nội".
"""
import os
import urllib.request
from typing import Optional


VNCORENLP_DIR = '/app/vncorenlp'
VNCORENLP_JAR = f'{VNCORENLP_DIR}/VnCoreNLP-1.2.jar'

_DOWNLOAD_FILES = {
    VNCORENLP_JAR:
        'https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.2.jar',
    f'{VNCORENLP_DIR}/models/wordsegmenter/vi-vocab':
        'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab',
    f'{VNCORENLP_DIR}/models/wordsegmenter/wordsegmenter.rdr':
        'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr',
}


def _ensure_vncorenlp_files() -> bool:
    """Download VnCoreNLP JAR + model files if not already present.

    Returns True when all files exist (whether pre-existing or just downloaded).
    """
    os.makedirs(f'{VNCORENLP_DIR}/models/wordsegmenter', exist_ok=True)
    try:
        for dest, url in _DOWNLOAD_FILES.items():
            if not os.path.exists(dest):
                print(f'📥 Downloading {os.path.basename(dest)} ...')
                urllib.request.urlretrieve(url, dest)
                print(f'✅ Downloaded {os.path.basename(dest)}')
        return True
    except Exception as e:
        print(f'⚠️  Failed to download VnCoreNLP files: {e}')
        return False


class VietnameseSegmentationService:
    """
    Singleton service for Vietnamese word segmentation using VnCoreNLP wseg.

    Falls back to returning the original text when VnCoreNLP is unavailable
    (Java not found, download failed, etc.).
    """

    _instance: Optional['VietnameseSegmentationService'] = None
    _model = None
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
        """Load VnCoreNLP JVM model (once per process)."""
        try:
            import vncorenlp  # noqa: F401 — check import before heavy work

            if not _ensure_vncorenlp_files():
                self._model = None
                return

            print('🔧 Initializing VnCoreNLP for Vietnamese word segmentation...')
            import vncorenlp as _vncorenlp
            self._model = _vncorenlp.VnCoreNLP(
                VNCORENLP_JAR,
                annotators='wseg',
                max_heap_size='-Xmx512m',
            )
            print('✅ VnCoreNLP loaded successfully')

        except ImportError:
            print('⚠️ vncorenlp not installed. Run: pip install vncorenlp')
            self._model = None
        except Exception as e:
            print(f'⚠️ Failed to load VnCoreNLP: {e}')
            self._model = None

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
        # Strip punctuation before VnCoreNLP so "ai?" and "ai ?" are treated identically
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

