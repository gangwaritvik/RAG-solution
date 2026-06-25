import os  
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """Return a required env var, or fail fast with a clear message."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Required environment variable '{name}' is missing or empty. "
            f"Set it in your .env file."
        )
    return value


def _require_int(name: str) -> int:
    """Return a required integer env var, or fail fast with a clear message."""
    raw = _require(name)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f"Environment variable '{name}' must be an integer, got '{raw}'."
        )


AZURE_ENDPOINT    = _require("AZURE_ENDPOINT")
AZURE_API_KEY     = _require("AZURE_API_KEY")
AZURE_API_VERSION = _require("AZURE_API_VERSION")
EMBEDDING_MODEL   = _require("EMBEDDING_MODEL")
CHAT_MODEL        = _require("CHAT_MODEL")
CHUNK_SIZE        = _require_int("CHUNK_SIZE")
CHUNK_OVERLAP     = _require_int("CHUNK_OVERLAP")
TOP_K             = int(os.getenv("TOP_K", 5))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 50))

# ── Upload limits ──
# Max size (bytes) accepted for a single uploaded file. Guards against unbounded
# in-memory reads exhausting RAM. Default 50 MB; override via MAX_UPLOAD_MB.
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", 50))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ── Semantic chunker tuning ──
SEMANTIC_BREAKPOINT_THRESHOLD = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", 0.3))
SEMANTIC_MIN_CHUNK_SIZE       = int(os.getenv("SEMANTIC_MIN_CHUNK_SIZE", 100))
SEMANTIC_MAX_CHUNK_SIZE       = int(os.getenv("SEMANTIC_MAX_CHUNK_SIZE", 1000))

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  
UPLOAD_DIR   = os.path.join(BASE_DIR, "storage", "uploads")  
CHROMA_DIR   = os.path.join(BASE_DIR, "storage", "chroma_db")  
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")  
DOC_RELEVANCE_THRESHOLD   = float(os.getenv("DOC_RELEVANCE_THRESHOLD", 0.25))  
CHUNK_RELEVANCE_THRESHOLD = float(os.getenv("CHUNK_RELEVANCE_THRESHOLD", 0.20))



# ── Vision (multimodal ingestion) ──
# Some PDFs render math/symbols with embedded symbol fonts that DON'T map to real
# characters, so the text layer comes out garbled or empty (e.g. a formula like
# "6x^3 + \u221a2 x^2 - 10x - 4\u221a2" extracts as "  "). When VISION_ENABLED is on, a page
# whose extraction shows this signature is re-read with the vision model, which
# transcribes it faithfully (text + LaTeX). Gated by detection, so ordinary text pages
# never trigger a vision call. VISION_MODEL defaults to the chat model (gpt-4.1 is
# vision-capable); VISION_DPI controls the render resolution sent to the model.
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
VISION_MODEL   = os.getenv("VISION_MODEL", CHAT_MODEL or "gpt-4.1")
VISION_DPI     = int(os.getenv("VISION_DPI", 200))
# All per-page/per-figure vision calls for one PDF run in parallel (moderate waves, like
# the generator's map-reduce) so ingesting a math/figure-heavy document isn't a long
# sequential chain of API round-trips. Kept modest to avoid Azure per-minute token bursts.
VISION_MAX_WORKERS = int(os.getenv("VISION_MAX_WORKERS", 8))

# ── Figure / diagram captioning (multimodal retrieval) ──
# When VISION_CAPTION_FIGURES is on, embedded raster images and vector diagrams (graphs,
# charts) are cropped during ingestion and described by the vision model; the caption text
# is embedded as its own chunk so the figure's information is searchable by a normal text
# query. Detection is size-gated, so pages without real figures cost nothing. (Images are
# NOT stored or displayed — only their extracted caption data is kept.)
VISION_CAPTION_FIGURES = os.getenv("VISION_CAPTION_FIGURES", "true").strip().lower() in ("1", "true", "yes", "on")
# Min figure size in PDF points (72pt = 1 inch) — smaller drawings (rules, the √ glyph,
# underlines) are ignored so only genuine figures are captioned.
FIGURE_MIN_WIDTH  = int(os.getenv("FIGURE_MIN_WIDTH", 60))
FIGURE_MIN_HEIGHT = int(os.getenv("FIGURE_MIN_HEIGHT", 50))
# Max figures captioned per page.
FIGURE_MAX_PER_PAGE = int(os.getenv("FIGURE_MAX_PER_PAGE", 6))

  
