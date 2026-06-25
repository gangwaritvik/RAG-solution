"""Chunking strategies — the four ways a loaded document is split into chunks.

Re-exported here so callers import a strategy from one place:
    from backend.ingestion.chunkers import Chunker, SemanticChunker
"""

from backend.ingestion.chunkers.chunker import Chunker
from backend.ingestion.chunkers.semantic_chunker import SemanticChunker
from backend.ingestion.chunkers.sliding_window import SlidingWindowChunker
from backend.ingestion.chunkers.fixed_chunker import FixedChunker

__all__ = [
    "Chunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "FixedChunker",
]
