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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import UPLOAD_DIR, CHROMA_DIR, FRONTEND_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE  
from backend.ingestion.document_loader import DocumentLoader  
from backend.ingestion.chunker import Chunker  
from backend.ingestion.semantic_chunker import SemanticChunker  
from backend.ingestion.sliding_window import SlidingWindowChunker  
from backend.ingestion.fixed_chunker import FixedChunker  
from backend.ingestion.embedder import Embedder  
from backend.ingestion.vector_store import VectorStore  
from backend.retrieval.retriever import Retriever  
from backend.generation.generator import Generator  
from backend.memory import ConversationMemoryManager, ContextResolver  
from backend.processing import MultiGroupProcessor
from backend.summarization import BackgroundSummarizer
from backend.utils.logger import get_logger
  

log = get_logger("main")

os.makedirs(UPLOAD_DIR, exist_ok=True)  
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── Pipeline components ──  
loader       = DocumentLoader(upload_dir=UPLOAD_DIR)  
chunker      = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  
embedder     = Embedder()  
vector_store = VectorStore(persist_dir=CHROMA_DIR)  
retriever = Retriever(embedder, vector_store)
generator    = Generator()
# Memory manager disabled - initialize with None to prevent collection lookup errors
try:
    memory_manager = ConversationMemoryManager(chroma_db_path=CHROMA_DIR)
    log.info("[STARTUP] Conversation memory manager initialized")
except Exception as e:
    log.warning(f"[STARTUP] Memory manager skipped: {e}")
    memory_manager = None
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
ingestion_status = {}  # {filename: {status: 'processing'|'completed', chunks: 0}}
ingestion_lock = threading.Lock()

# ── Clear stale vectors on startup ──  
# vector_store.clear()  # Disabled: Don't clear on startup to preserve state
log.info("[STARTUP] Vector store ready — session started")
log.info("[STARTUP] Conversation memory manager initialized")  


def get_chunker(mode: str):  
    """Returns the appropriate chunker based on selected mode."""  
    if mode == "semantic":  
        log.info("[CHUNKER] Selected: SemanticChunker")  
        return SemanticChunker(  
            breakpoint_threshold=0.3,  
            min_chunk_size=100,  
            max_chunk_size=1000,  
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
    """Background-thread worker for file ingestion.

    Loads → chunks → embeds → stores each uploaded file, updating ingestion_status
    as it goes. Called by the FastAPI /ingest endpoint in a daemon thread.
    """
    log.info(f"[INGEST_BG] Starting background processing for {len(files)} file(s)")

    try:
        active_chunker = get_chunker(chunk_mode)
        results = []

        for filename, content in files:
            log.info(f"[INGEST_BG] Processing: {filename} ({len(content)} bytes)")

            # Step 1 — Load
            log.info(f"[INGEST_BG] STEP 1 — Loading {filename}")
            doc_results = loader.load_documents([(filename, content)])

            # Step 2 — Chunk
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

                log.info(f"[INGEST_BG] STEP 2.5 — Created {total_chunks} chunks from {doc.get('filename')}")
                for i, chunk in enumerate(doc["chunks"], 1):
                    preview = chunk.page_content[:100].replace('\n', ' ')
                    log.debug(f"[INGEST_BG]   Chunk {i}/{total_chunks}: {preview}...")

                # Step 3 — Embed
                log.info(f"[INGEST_BG] STEP 3 — Embedding {total_chunks} chunks from {doc.get('filename')} (batch_size: {EMBEDDING_BATCH_SIZE})")
                vecs = embedder.embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
                log.info(f"[INGEST_BG] ✅ Embedded {len(vecs)} vectors")

                # Step 4 — Store
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
                vector_store.add(vecs, meta)

                results.append({
                    "filename":     doc["filename"],
                    "total_pages":  doc["total_pages"],
                    "total_chunks": doc["total_chunks"],
                    "chunk_mode":   chunk_mode,
                    "errors":       doc["errors"],
                })
                log.info(f"[INGEST_BG] ✅ {doc['filename']} — ingested successfully")

                with ingestion_lock:
                    ingestion_status[doc['filename']] = {"status": "completed", "chunks": total_chunks}

        log.info(f"[INGEST_BG] ✅ Complete — {len(results)} doc(s) | {vector_store.count} total vectors")

    except Exception as e:
        log.error(f"[INGEST_BG] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
