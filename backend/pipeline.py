#!/usr/bin/env python3
"""
Pipeline initialization module.

This module wires up and exposes the shared pipeline singletons (loader, embedder,
vector_store, retriever, generator, memory_manager, context_resolver, bg_summarizer)
plus the ingestion helpers. The FastAPI app (backend/app.py) imports these.

It no longer runs an HTTP server itself — serving is handled by uvicorn + FastAPI.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import (
    UPLOAD_DIR, CHROMA_DIR, FRONTEND_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE,
    SEMANTIC_BREAKPOINT_THRESHOLD, SEMANTIC_MIN_CHUNK_SIZE, SEMANTIC_MAX_CHUNK_SIZE,
)  
from backend.ingestion.document_loader import DocumentLoader  
from backend.ingestion.chunkers import Chunker, SemanticChunker, SlidingWindowChunker, FixedChunker  
from backend.ingestion.embedder import Embedder  
from backend.ingestion.vector_store import VectorStore  
from backend.retrieval.retriever import Retriever  
from backend.generation.generator import Generator  
from backend.memory import ConversationMemoryManager, ContextResolver  
from backend.processing import MultiGroupProcessor
from backend.summarization import BackgroundSummarizer
from backend.utils.logger import get_logger
  

log = get_logger("pipeline")

os.makedirs(UPLOAD_DIR, exist_ok=True)  
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── Pipeline components ──  
loader       = DocumentLoader(upload_dir=UPLOAD_DIR)  
chunker      = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  
embedder     = Embedder()  
vector_store = VectorStore(persist_dir=CHROMA_DIR)  
retriever = Retriever(embedder, vector_store)
generator    = Generator()
# The conversation memory manager is REQUIRED — app.py calls core.memory_manager.*
# unconditionally on every query and group endpoint. Fail fast with a clear message
# rather than starting in a broken state that crashes with AttributeError later.
try:
    memory_manager = ConversationMemoryManager(chroma_db_path=CHROMA_DIR)
    log.info("[STARTUP] Conversation memory manager initialized")
except Exception as e:
    log.error(f"[STARTUP] Failed to initialize conversation memory manager: {e}", exc_info=True)
    raise RuntimeError(
        "Conversation memory manager failed to initialize — cannot start. "
        f"Underlying error: {e}"
    ) from e
context_resolver = ContextResolver(memory_manager, embedder)
multi_group_processor = MultiGroupProcessor(
    retriever=retriever,
    max_workers=5,
)
bg_summarizer = BackgroundSummarizer(
    memory_manager=memory_manager,
    generator=generator,
    embedder=embedder,
    max_workers=3
)

# ── Track ingestion status independently from vector count ──
ingestion_status = {}  # {filename: {status: 'processing'|'completed'|'failed', chunks: 0}}
ingestion_lock = threading.Lock()

# Serialize ChromaDB writes. Chroma is SQLite-backed and can raise "database is locked"
# under concurrent writers, and the vector_store filename cache is a plain set. Holding
# this lock ONLY around the store step keeps the slow load/chunk/embed work of different
# files fully concurrent while making the actual write single-writer-safe.
_store_lock = threading.Lock()

# Cap how many files ingest CONCURRENTLY. Each file's embed step already fans out to
# ~10 workers internally, so unbounded file concurrency could stampede the Azure
# embedding endpoint into rate-limiting. A small bound overlaps the slow per-file steps
# (one file embeds while another loads/chunks) without flooding the API.
INGEST_MAX_FILE_WORKERS = 3

# ── Clear stale vectors on startup ──
# Always start fresh: every backend (re)start wipes the vector store so the count is 0.
vector_store.clear()
log.info("[STARTUP] Vector store cleared — starting fresh (0 vectors)")


def get_chunker(mode: str):  
    """Returns the appropriate chunker based on selected mode."""  
    if mode == "semantic":  
        log.info("[CHUNKER] Selected: SemanticChunker")  
        return SemanticChunker(  
            breakpoint_threshold=SEMANTIC_BREAKPOINT_THRESHOLD,  
            min_chunk_size=SEMANTIC_MIN_CHUNK_SIZE,  
            max_chunk_size=SEMANTIC_MAX_CHUNK_SIZE,  
        )  
    elif mode == "sliding":  
        log.info("[CHUNKER] Selected: SlidingWindowChunker")  
        return SlidingWindowChunker(  
            chunk_size=CHUNK_SIZE,  
            chunk_overlap=CHUNK_OVERLAP,  
        )  
    elif mode == "fixed":  
        log.info("[CHUNKER] Selected: FixedChunker")  
        return FixedChunker(chunk_size=CHUNK_SIZE)  
    else:  
        log.info("[CHUNKER] Selected: RecursiveCharacterTextSplitter (default)")  
        return chunker


# ══════════════════════════════════════════════════════════════════════
#  INGEST BACKGROUND WORKER (module-level)
# ══════════════════════════════════════════════════════════════════════
def process_ingest_background(files, chunk_mode):
    """Background-thread worker: ingest all uploaded files CONCURRENTLY.

    Each file runs its OWN load → chunk → embed → store pipeline in a worker thread
    (bounded by INGEST_MAX_FILE_WORKERS), so the chunking/embedding of different files
    OVERLAP instead of running strictly one-after-another. ChromaDB writes are
    serialized via ``_store_lock`` and each file's errors are isolated, so one bad file
    neither corrupts the store nor stalls the others.

    NOTE: within a SINGLE file, the default recursive/fixed/sliding chunkers are pure
    Python (GIL-bound), so threading one file's chunk step gives no speedup — the win
    here is across files. The semantic chunker (which makes embedding calls) overlaps
    naturally with other files under this model.
    """
    log.info(f"[INGEST_BG] Starting background processing for {len(files)} file(s)")

    all_results = []
    max_workers = max(1, min(len(files), INGEST_MAX_FILE_WORKERS))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(_ingest_one_file, filename, content, chunk_mode): filename
                for filename, content in files
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    log.error(f"[INGEST_BG] ❌ Worker for {name} crashed — {type(e).__name__}: {e}", exc_info=True)

        log.info(f"[INGEST_BG] ✅ Complete — {len(all_results)} doc(s) | {vector_store.count} total vectors")

    except Exception as e:
        log.error(f"[INGEST_BG] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)


def _ingest_one_file(filename, content, chunk_mode):
    """Run the full load → chunk → embed → store pipeline for ONE file (worker thread).

    Returns the list of per-document result dicts (usually one). On any failure it marks
    the file 'failed' in ingestion_status — so a bad/unsupported file surfaces an error
    instead of hanging as a perpetual 'processing' spinner — and returns [] without
    affecting the other files being ingested in parallel.
    """
    file_results = []
    try:
        # Each thread gets its own chunker instance (or the stateless shared recursive
        # singleton) so there is no shared mutable chunker state across files.
        active_chunker = get_chunker(chunk_mode)

        log.info(f"[INGEST_BG] Processing: {filename} ({len(content)} bytes)")

        # Step 1 — Load
        log.info(f"[INGEST_BG] STEP 1 — Loading {filename}")
        doc_results = loader.load_documents([(filename, content)])

        # Step 2 — Chunk (overlaps with other files' steps)
        log.info(f"[INGEST_BG] STEP 2 — Chunking {filename} (mode: {chunk_mode})")
        chunked = active_chunker.chunk_documents(doc_results)

        for doc in chunked:
            doc_filename = doc.get('filename', filename)

            if not doc.get("chunks"):
                log.warning(f"[INGEST_BG] ⚠️ No chunks for {doc_filename}")
                with ingestion_lock:
                    ingestion_status[doc_filename] = {"status": "completed", "chunks": 0}
                continue

            texts = [c.page_content for c in doc["chunks"]]
            total_chunks = len(texts)

            log.info(f"[INGEST_BG] STEP 2.5 — Created {total_chunks} chunks from {doc_filename}")

            # Step 3 — Embed (embedder fans out internally)
            log.info(f"[INGEST_BG] STEP 3 — Embedding {total_chunks} chunks from {doc_filename} (batch_size: {EMBEDDING_BATCH_SIZE})")
            vecs = embedder.embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
            log.info(f"[INGEST_BG] ✅ Embedded {len(vecs)} vectors")

            # Step 4 — Store (serialized: one writer at a time)
            log.info(f"[INGEST_BG] STEP 4 — Storing {len(vecs)} vectors in ChromaDB")
            meta = [
                {
                    "text":        c.page_content,
                    "filename":    doc["filename"],
                    "doc_id":      doc["doc_id"],
                    "chunk_index": c.metadata.get("chunk_index"),
                    "page":        c.metadata.get("page"),
                    "chunk_type":  c.metadata.get("chunk_type", chunk_mode),
                }
                for c in doc["chunks"]
            ]
            with _store_lock:
                # Replace-on-reingest: if this filename already has vectors, delete them
                # FIRST so re-uploading the same file updates it in place instead of
                # appending a duplicate copy of every chunk. Kept inside the store lock so
                # the delete+add is atomic w.r.t. the single serialized ChromaDB writer.
                if doc["filename"] in vector_store.list_filenames():
                    removed = vector_store.delete_by_filename(doc["filename"])
                    log.info(f"[INGEST_BG] ♻️ Replaced {removed} existing vector(s) for {doc['filename']} (re-ingest)")
                vector_store.add(vecs, meta)

            file_results.append({
                "filename":     doc["filename"],
                "total_pages":  doc["total_pages"],
                "total_chunks": doc["total_chunks"],
                "chunk_mode":   chunk_mode,
                "errors":       doc["errors"],
            })
            log.info(f"[INGEST_BG] ✅ {doc['filename']} — ingested successfully")

            with ingestion_lock:
                ingestion_status[doc['filename']] = {"status": "completed", "chunks": total_chunks}

    except Exception as e:
        # Isolate the failure: mark THIS file failed (so the UI stops showing a spinner)
        # and let the other files continue.
        log.error(f"[INGEST_BG] ❌ Failed to ingest {filename} — {type(e).__name__}: {e}", exc_info=True)
        with ingestion_lock:
            ingestion_status[filename] = {"status": "failed", "chunks": 0, "error": str(e)}

    return file_results
