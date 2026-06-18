#!/usr/bin/env python3  
import json  
import os  
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer, BaseHTTPRequestHandler  
from urllib.parse import urlparse

from backend.config import UPLOAD_DIR, CHROMA_DIR, FRONTEND_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP  
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
from backend.utils.parser import parse_multipart_full  
from backend.utils.logger import get_logger
from backend.utils.timeout import QueryTimeout, TimeoutError as QueryTimeoutError
  

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
    memory_manager=memory_manager,
    context_resolver=context_resolver,
    retriever=retriever,
    generator=generator,
    embedder=embedder,
    max_workers=5
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


class RAGHandler(BaseHTTPRequestHandler):

    STATIC = {  
        "/":           ("index.html",  "text/html"),  
        "/index.html": ("index.html",  "text/html"),  
        "/styles.css": ("styles.css",  "text/css"),  
        "/app.js":     ("app.js",      "application/javascript"),  
    }

    def do_GET(self):  
        path = urlparse(self.path).path  
        
        # Status endpoint for polling
        if path == "/status":
            self.handle_status()
        
        # Static files
        elif path in self.STATIC:  
            filename, ctype = self.STATIC[path]  
            self.serve_file(os.path.join(FRONTEND_DIR, filename), ctype)
        
        # Group endpoints
        elif path == "/groups":
            self.handle_get_groups()
        elif path.startswith("/group/"):
            self.handle_group_get(path)
        
        else:  
            self.send_error(404, "Not Found")
    
    def handle_status(self):
        """Handle GET /status - Return current system status with per-file chunks and ingestion status"""
        try:
            total_vectors = vector_store.count
            
            # Get all chunks grouped by filename
            files_data = []
            if total_vectors > 0:
                all_results = vector_store.collection.get(
                    include=["metadatas"]
                )
                
                # Group by filename
                files_dict = {}
                for i, metadata in enumerate(all_results.get("metadatas", [])):
                    filename = metadata.get("filename", "unknown")
                    if filename not in files_dict:
                        files_dict[filename] = []
                    files_dict[filename].append({
                        "chunk_index": metadata.get("chunk_index", 0),
                        "page": metadata.get("page", 0),
                        "text": metadata.get("text", "")
                    })
                
                # Convert to list format
                for filename, chunks in files_dict.items():
                    files_data.append({
                        "filename": filename,
                        "chunk_count": len(chunks),
                        "chunks": sorted(chunks, key=lambda x: x["chunk_index"])
                    })
            
            # Include ingestion status (files with 0 chunks will appear here)
            with ingestion_lock:
                ingestion_data = dict(ingestion_status)
            
            self.send_json({
                "status": "ok",
                "total_vectors": total_vectors,
                "files": files_data,
                "ingestion_status": ingestion_data,
                "server": "running"
            })
        except Exception as e:
            log.error(f"[STATUS] Failed — {e}", exc_info=True)
            self.send_json({"error": str(e), "status": "error"}, 500)
    
    def handle_clear(self):  
        try:  
            vector_store.clear()  
            log.info("[CLEAR] All vectors deleted from ChromaDB")  
            self.send_json({"status": "ok", "message": "All vectors cleared"})  
        except Exception as e:  
            log.error(f"[CLEAR] Failed — {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)  

    def do_OPTIONS(self):  
        self.send_response(200)  
        self._cors()  
        self.end_headers()

    def do_PUT(self):
        """Handle PUT requests for updating resources"""
        path = urlparse(self.path).path
        
        # PUT /group/{group_id} - rename group
        if path.startswith("/group/") and path.count("/") == 2:
            self.handle_put_group(path)
        else:
            self.send_error(404, "Not Found")
    
    def do_DELETE(self):
        """Handle DELETE requests for removing resources"""
        path = urlparse(self.path).path
        
        # DELETE /group/{group_id}/turn/{turn_id} - delete specific turn
        if "/turn/" in path:
            self.handle_delete_turn(path)
        # DELETE /group/{group_id} - delete entire group
        elif path.startswith("/group/") and path.count("/") == 2:
            self.handle_delete_group(path)
        else:
            self.send_error(404, "Not Found")

    def handle_delete(self):  
        log.info("[DELETE] ▶ Delete request received")  
        try:  
            length   = int(self.headers.get("Content-Length", 0))  
            body     = json.loads(self.rfile.read(length))  
            filename = body.get("filename", "").strip()

            if not filename:  
                self.send_json({"error": "No filename provided."}, 400)  
                return

            log.info(f"[DELETE] Deleting vectors for: {filename}")  
            deleted_count = vector_store.delete_by_filename(filename)

            log.info(f"[DELETE] ✅ Deleted {deleted_count} vectors | Total remaining: {vector_store.count}")  
            self.send_json({  
                "status":          "ok",  
                "filename":        filename,  
                "deleted_vectors": deleted_count,  
                "total_vectors":   vector_store.count,  
            })

        except Exception as e:  
            log.error(f"[DELETE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)  


    def do_POST(self):  
        import sys
        import io
        # Write directly to stdout buffer to bypass any buffering
        sys.stdout.buffer.write(b"\n\n========== POST REQUEST RECEIVED ==========\n")
        sys.stdout.buffer.flush()
        try:
            path = urlparse(self.path).path  
            print(f"\n📨 POST {path} received", flush=True)
            sys.stdout.flush()
            if path == "/ingest":  
                print("✅ ROUTING TO HANDLE_INGEST", flush=True)
                sys.stdout.flush()
                self.handle_ingest()  
            elif path == "/query":  
                self.handle_query()  
            elif path == "/delete":          
                self.handle_delete()  
            elif path == "/clear":  
                self.handle_clear()
            
            # Conversation group endpoints
            elif path == "/group/create":
                self.handle_group_create()
            elif path == "/group/list":
                self.handle_group_list()
            elif path == "/group/summarize":
                self.handle_group_summarize()
            elif path == "/group/status":
                self.handle_group_status()

            else:  
                self.send_error(404, "Not Found")
        except Exception as e:
            print(f"❌ CRITICAL ERROR IN do_POST: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            self.send_error(500, f"Server error: {str(e)}")  

    # ──────────────────────────────────  
    #  INGEST  
    # ──────────────────────────────────  
    def handle_ingest(self):  
        print("\n\n🔥🔥🔥 HANDLE_INGEST CALLED 🔥🔥🔥\n", flush=True)
        sys.stdout.flush()
        log.info("=" * 55)  
        log.info("[INGEST] ▶ New ingest request received")

        try:  
            files, fields = parse_multipart_full(self.headers, self.rfile)  
            chunk_mode    = fields.get("chunk_mode", "recursive").lower().strip()

            log.info(f"[INGEST] Files: {len(files)} | Chunk mode: '{chunk_mode}'")

            if not files:  
                log.warning("[INGEST] ⚠️ No files in request")  
                self.send_json({"error": "No files provided."}, 400)  
                return

            # ── Validate files ──  
            valid_files = []
            for filename, content in files:
                if (filename.lower().endswith(".pdf") or  
                    filename.lower().endswith(".docx") or  
                    filename.lower().endswith(".doc")):
                    valid_files.append((filename, content))
                else:
                    log.warning(f"[INGEST] Skipping unsupported file: {filename}")

            if not valid_files:
                log.warning("[INGEST] ⚠️ No valid files in request")
                self.send_json({"error": "No valid PDF/DOCX files."}, 400)
                return

            # ── Mark files as being processed ──
            with ingestion_lock:
                for filename, _ in valid_files:
                    ingestion_status[filename] = {"status": "processing", "chunks": 0}

            # ── Start background thread for processing ──
            print(f"[DEBUG] ABOUT TO START THREAD", flush=True)
            sys.stdout.flush()
            thread = threading.Thread(
                target=self._process_ingest_background,
                args=(valid_files, chunk_mode),
                daemon=True
            )
            print(f"[DEBUG] THREAD OBJECT CREATED: {thread}", flush=True)
            sys.stdout.flush()
            thread.start()
            print(f"[DEBUG] THREAD STARTED: {thread.is_alive()}", flush=True)
            sys.stdout.flush()

            # ── Return 202 Accepted immediately ──
            self.send_json({
                "status": "accepted",
                "message": f"Processing {len(valid_files)} file(s) in background",
                "files": len(valid_files),
            }, 202)

        except Exception as e:  
            log.error(f"[INGEST] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)

    def _process_ingest_background(self, files, chunk_mode):
        """Background thread worker for file ingestion."""
        print("\n\n🚀🚀🚀 BACKGROUND THREAD STARTED 🚀🚀🚀\n", flush=True)
        sys.stdout.flush()
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
                    chunk_count = len(doc.get("chunks", []))
                    
                    if not doc.get("chunks"):  
                        log.warning(f"[INGEST_BG] ⚠️ No chunks for {doc_filename}")
                        # Mark as completed with 0 chunks
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
                    log.info(f"[INGEST_BG] STEP 3 — Embedding {total_chunks} chunks from {doc.get('filename')} (batch_size: 500)")
                    vecs = embedder.embed_texts(texts, batch_size=500)
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
                        "chunks": [  
                            {  
                                "chunk_index": c.metadata.get("chunk_index"),  
                                "page":        c.metadata.get("page"),  
                                "text":        c.page_content,  
                            }  
                            for c in doc["chunks"]  
                        ],  
                    })  
                    log.info(f"[INGEST_BG] ✅ {doc['filename']} — ingested successfully")
                    
                    # Mark as completed with actual chunk count
                    with ingestion_lock:
                        ingestion_status[doc['filename']] = {"status": "completed", "chunks": total_chunks}

            log.info(f"[INGEST_BG] ✅ Complete — {len(results)} doc(s) | {vector_store.count} total vectors")

        except Exception as e:  
            log.error(f"[INGEST_BG] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)


    # ──────────────────────────────────  
    #  QUERY  
    # ──────────────────────────────────

    def handle_query(self):  
        log.info("[QUERY] ▶ New query request received")  
        try:  
            length = int(self.headers.get("Content-Length", 0))  
            body   = json.loads(self.rfile.read(length))  
            query  = body.get("query", "").strip()  
            top_k  = int(body.get("top_k", TOP_K))  
            temp   = float(body.get("temperature", 0.2))
            group_id = body.get("group_id")  # Optional: for conversation memory
            # Optional explicit Top-K override from the frontend's Query Settings.
            # When present (settings enabled), it WINS over each intent's default cap.
            # When absent/None (settings disabled), intent defaults are used.
            top_k_override = body.get("top_k_override")
            if top_k_override is not None:
                try:
                    top_k_override = int(top_k_override)
                    if top_k_override <= 0:
                        top_k_override = None
                except (TypeError, ValueError):
                    top_k_override = None
            # MAX toggle: retrieve EVERY chunk (no threshold, no doc-relevance filter).
            retrieve_all = bool(body.get("retrieve_all", False))

            log.info(f"[QUERY] '{query[:80]}' | top_k: {top_k} | temp: {temp} | top_k_override: {top_k_override} | retrieve_all: {retrieve_all}")

            if not query:  
                self.send_json({"error": "Query is empty."}, 400)  
                return

            if vector_store.count == 0:  
                self.send_json({"error": "No documents ingested yet."}, 400)  
                return

            # Wrap main processing in timeout (30 seconds)
            with QueryTimeout(seconds=30, message="Query processing exceeded 30 second timeout"):
                self._process_query(query, top_k, temp, group_id, top_k_override, retrieve_all)

        except QueryTimeoutError as e:
            log.error(f"[QUERY] ⏱️ TIMEOUT — {e}")
            self.send_json({
                "error": f"Query processing timeout (30s exceeded): {str(e)}",
                "status": "timeout",
            }, 504)  # 504 Gateway Timeout
        except Exception as e:  
            log.error(f"[QUERY] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)
    
    def _process_query(self, query: str, top_k: int, temp: float, group_id: str = None, top_k_override: int = None, retrieve_all: bool = False):
        """
        Internal method to process query (wrapped in timeout).
        
        Args:
            query: User's query
            top_k: Number of chunks to retrieve
            temp: Temperature for generation
            group_id: Optional group ID for conversation memory
            top_k_override: Explicit user Top-K that overrides intent defaults (None = use intent defaults)
            retrieve_all: When True (MAX toggle), retrieve every chunk with no relevance filtering
        """
        # STEP 1 — Context Resolution: Classify query and load memory context
        log.info("[QUERY] STEP 1 — Resolving query context")
        # Tell the resolver which documents are loaded so a whole-corpus request
        # ("summarize the document") resolves instead of being marked ambiguous.
        available_documents = vector_store.list_filenames()
        query_context = context_resolver.resolve(
            query, active_group_id=group_id, available_documents=available_documents
        )
        log.info(f"[QUERY] Standalone query: {query_context.standalone_query[:80]}")

        from backend.memory.resolution.context_resolver import DependencyType

        # CONVERSATION FOLLOW-UP that operates on the assistant's PREVIOUS ANSWER, e.g.
        # "summarize the above", "translate that", "put it in a table". The LLM classifier
        # decides this via answer_source="previous_answer" (with full active-group context),
        # so we use the previous answer as context and skip document retrieval. Any other
        # follow-up (answer_source="document") still retrieves from the document.
        is_followup = (
            query_context.dependency_type == DependencyType.DEPENDENT
            and query_context.belongs_to_active_group
        )
        operates_on_prev_answer = query_context.answer_source == "previous_answer"

        # Load the active group ONCE (cheap in-memory lookup) — reused below for the
        # previous-answer lookup and later for the turn-count context optimization.
        active_group = None
        if query_context.active_group_id:
            active_group = memory_manager.get_conversation_group(query_context.active_group_id)

        prev_answer_chunk = None
        if is_followup and operates_on_prev_answer:
            if active_group and active_group.all_turns:
                for prev_turn in reversed(active_group.all_turns):
                    if getattr(prev_turn, "full_answer", ""):
                        prev_answer_chunk = {
                            "text": prev_turn.full_answer,
                            "filename": "(previous answer)",
                            "page": 0,
                            "chunk_index": 0,
                            "score": 1.0,
                            "chunk_type": "conversation",
                        }
                        log.info("[QUERY] Follow-up operates on previous answer — using it as primary context")
                        break
            if prev_answer_chunk is None:
                # Classifier asked to operate on the previous answer, but none is stored
                # (e.g. the group has no prior answer with full_answer yet). Fall back to
                # normal document retrieval instead of producing an empty-context answer.
                log.warning("[QUERY] answer_source=previous_answer but no stored previous answer found — falling back to document retrieval")

        # DYNAMIC top_k: Adjust based on retrieval intent
        # Summary/extraction/comparison/analysis queries need all relevant chunks for comprehensive synthesis
        retrieval_top_k = top_k
        get_all_relevant = False
        if query_context.retrieval_intent.value in ["summary", "extraction", "comparison", "analysis"]:
            # Skip whole-document retrieval ONLY when the user is operating on the
            # previous answer (e.g. "summarize the above") — otherwise the document
            # drowns it out. All other follow-ups still retrieve from the document.
            if prev_answer_chunk is not None:
                log.info(f"[QUERY] Intent '{query_context.retrieval_intent.value}' on previous answer — skipping document-wide retrieval")
            elif top_k_override is not None:
                # User explicitly set Top-K — respect it instead of pulling everything.
                log.info(f"[QUERY] Intent '{query_context.retrieval_intent.value}' — user Top-K override ({top_k_override}) takes precedence over full retrieval")
            else:
                get_all_relevant = True
                log.info(f"[QUERY] Intent '{query_context.retrieval_intent.value}' — fetching ALL relevant chunks (not limited by top_k)")
        
        # STEP 2 — Retrieval: Use standalone query for document retrieval (skip if ambiguous)
        log.info("[QUERY] STEP 2 — Retrieving relevant chunks")
        if query_context.dependency_type == DependencyType.AMBIGUOUS:
            log.info("[QUERY] ⚠️ AMBIGUOUS query — skipping retrieval, will generate clarification")
            hits = []
        elif prev_answer_chunk is not None:
            # Follow-up operates on the PREVIOUS ANSWER (e.g. "summarize the above") —
            # skip document retrieval entirely so the answer isn't diluted by the document.
            log.info("[QUERY] Operating on previous answer — skipping document retrieval entirely")
            hits = []
        else:
            log.info(f"[QUERY] Retrieving with get_all_relevant={get_all_relevant}, top_k={retrieval_top_k}, intent={query_context.retrieval_intent.value}, override={top_k_override}, retrieve_all={retrieve_all}")
            hits = retriever.retrieve(
                query_context.standalone_query, 
                top_k=retrieval_top_k, 
                get_all_relevant=get_all_relevant,
                retrieval_intent=query_context.retrieval_intent.value,
                top_k_override=top_k_override,
                retrieve_all=retrieve_all
            )
            log.info(f"[QUERY] Retrieved {len(hits)} chunks")
            if hits:
                log.info(f"[QUERY] First chunk preview: {hits[0].get('text', '')[:150]}...")
                # Log chunk variety
                sources = set()
                for hit in hits:
                    sources.add(hit.get('filename', 'unknown'))
                log.info(f"[QUERY] Chunks from {len(sources)} source(s): {', '.join(list(sources)[:5])}")

        # Inject the previous answer as the SOLE context for conversation follow-ups
        # (e.g. "summarize the above"), so generation operates only on it.
        if prev_answer_chunk is not None:
            hits = [prev_answer_chunk] + hits
            log.info(f"[QUERY] Using previous answer as context — {len(hits)} total context chunks")

        # STEP 3 — Generation: Generate answer with memory context
        log.info("[QUERY] STEP 3 — Generating answer with memory context")
        
        # Get turn count from active group for dynamic context optimization
        # (reuse the active_group fetched earlier — avoids a second lookup).
        turn_count = 1
        if active_group:
            turn_count = len(active_group.all_turns)
            log.info(f"[QUERY] Using group {query_context.active_group_id} — turn_count={turn_count}")
        
        # When operating solely on the previous answer, that full answer IS the context.
        # The memory_context (recent-turn summaries) just duplicates it — drop it to save tokens.
        gen_memory_context = query_context.memory_context
        if prev_answer_chunk is not None:
            gen_memory_context = None
            log.info("[QUERY] Operating on previous answer — omitting redundant memory context")

        answer, memory_summary = generator.generate(
            query=query,
            context_chunks=hits,
            temperature=temp,
            memory_context=gen_memory_context,
            retrieval_intent=query_context.retrieval_intent.value,
            turn_count=turn_count  # Pass turn count for dynamic optimization
        )  
        log.info("[QUERY] ✅ Answer generated successfully")

        # STEP 4 — Store turn in conversation memory
        log.info("[QUERY] STEP 4 — Storing turn in conversation memory")
        if query_context.active_group_id:
            memory_manager.add_conversation_turn(
                group_id=query_context.active_group_id,
                query=query,
                memory_summary=memory_summary,
                dependency_type=query_context.dependency_type.value,
                retrieval_intent=query_context.retrieval_intent.value,
                full_answer=answer
            )
            log.info(f"[QUERY] ✅ Turn saved to group {query_context.active_group_id}")
        else:
            log.warning("[QUERY] ⚠️ No active group — turn not saved to memory")

        # STEP 5 — Prepare response
        log.info("[QUERY] ✅ Query processing complete")
        self.send_json({
            "answer": answer, 
            "retrieved_chunks": hits,
            "query": query,
            "group_id": query_context.active_group_id,  # Return active group for frontend to use in next query
        })

    # ──────────────────────────────────  
    #  CONVERSATION GROUPS  
    # ──────────────────────────────────  
    def handle_group_get(self, path):
        """Handle GET /group/{group_id} endpoints"""
        log.info("[GROUP-GET] ▶ Group get request received")
        try:
            parts = path.strip("/").split("/")
            
            if len(parts) < 2:
                self.send_json({"error": "Invalid path"}, 400)
                return
            
            group_id = parts[1]
            endpoint = parts[2] if len(parts) > 2 else None
            
            if endpoint == "history":
                # GET /group/{group_id}/history
                history = memory_manager.get_group_history(group_id)
                if not history:
                    self.send_json({"error": f"Group not found: {group_id}"}, 404)
                    return
                log.info(f"[GROUP-GET] ✅ Retrieved history for {group_id}")
                self.send_json(history)
                
            elif endpoint == "summary":
                # GET /group/{group_id}/summary
                summary = memory_manager.get_group_summary(group_id)
                if not summary:
                    self.send_json({"error": f"Group not found: {group_id}"}, 404)
                    return
                log.info(f"[GROUP-GET] ✅ Retrieved summary for {group_id}")
                self.send_json(summary)
                
            else:
                # GET /group/{group_id} - basic context
                context = memory_manager.get_group_context(group_id, include_all_turns=False)
                if not context:
                    self.send_json({"error": f"Group not found: {group_id}"}, 404)
                    return
                log.info(f"[GROUP-GET] ✅ Retrieved context for {group_id}")
                self.send_json(context)
            
        except Exception as e:
            log.error(f"[GROUP-GET] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_get_groups(self):
        """Handle GET /groups - List all groups with metadata"""
        log.info("[GROUPS-LIST] ▶ List groups request received")
        try:
            groups = memory_manager.list_groups_with_metadata()
            log.info(f"[GROUPS-LIST] ✅ Retrieved {len(groups)} groups")
            self.send_json({
                "status": "ok",
                "groups": groups,
                "total": len(groups),
            })
        except Exception as e:
            log.error(f"[GROUPS-LIST] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_group_create(self):
        """Handle POST /group/create"""
        log.info("[GROUP-CREATE] ▶ Group create request received")
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            topic = body.get("topic", "").strip()
            
            if not topic:
                self.send_json({"error": "Topic is required"}, 400)
                return
            
            log.info(f"[GROUP-CREATE] Creating group with topic: {topic}")
            group = memory_manager.create_conversation_group(topic)
            
            log.info(f"[GROUP-CREATE] ✅ Created group {group.group_id}")
            self.send_json({
                "status": "ok",
                "group_id": group.group_id,
                "topic": group.topic,
                "created_at": group.created_at,
            })
            
        except Exception as e:
            log.error(f"[GROUP-CREATE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_group_list(self):
        """Handle POST /group/list"""
        log.info("[GROUP-LIST] ▶ Group list request received")
        try:
            groups = memory_manager.list_conversation_groups()
            
            log.info(f"[GROUP-LIST] Retrieved {len(groups)} groups")
            self.send_json({
                "status": "ok",
                "total_groups": len(groups),
                "groups": [
                    {
                        "group_id": g.group_id,
                        "topic": g.topic,
                        "summary_ready": g.summary_ready,
                        "recent_turns": len(g.recent_turns),
                        "total_turns": len(g.all_turns),
                        "created_at": g.created_at,
                        "updated_at": g.updated_at,
                    }
                    for g in groups
                ]
            })
            
        except Exception as e:
            log.error(f"[GROUP-LIST] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_group_summarize(self):
        """Handle POST /group/summarize - Manual summarization trigger"""
        log.info("[GROUP-SUMMARIZE] ▶ Manual summarize request received")
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            group_id = body.get("group_id", "").strip()
            
            if not group_id:
                self.send_json({"error": "group_id is required"}, 400)
                return
            
            group = memory_manager.get_conversation_group(group_id)
            if not group:
                self.send_json({"error": f"Group not found: {group_id}"}, 404)
                return
            
            # Check if already summarizing
            if bg_summarizer.is_summarizing(group_id):
                self.send_json({
                    "status": "in_progress",
                    "message": f"Group {group_id} is already being summarized",
                    "group_id": group_id,
                }, 202)
                return
            
            # Check if summarization is needed
            if not memory_manager.should_summarize_group(group_id):
                self.send_json({
                    "status": "not_needed",
                    "message": f"Group has only {group.unsummarized_turn_count()} turns (needs 5+)",
                    "unsummarized_turns": group.unsummarized_turn_count(),
                }, 200)
                return
            
            # Trigger background summarization
            log.info(f"[GROUP-SUMMARIZE] Triggering background summarization for {group_id}")
            bg_summarizer.summarize_if_needed(group_id, threshold=5)
            
            log.info(f"[GROUP-SUMMARIZE] ✅ Summarization triggered for {group_id}")
            self.send_json({
                "status": "triggered",
                "message": f"Background summarization started for group {group_id}",
                "group_id": group_id,
                "unsummarized_turns": group.unsummarized_turn_count(),
            }, 202)  # 202 Accepted
            
        except Exception as e:
            log.error(f"[GROUP-SUMMARIZE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_group_status(self):
        """Handle POST /group/status - Get summarization status"""
        log.info("[GROUP-STATUS] ▶ Status request received")
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
            group_id = body.get("group_id", "").strip()
            
            if not group_id:
                # If no group_id, return status of all active summarizations
                active = bg_summarizer.get_active_summarizations()
                self.send_json({
                    "status": "ok",
                    "active_summarizations": active,
                    "total_active": len(active),
                })
                return
            
            group = memory_manager.get_conversation_group(group_id)
            if not group:
                self.send_json({"error": f"Group not found: {group_id}"}, 404)
                return
            
            is_summarizing = bg_summarizer.is_summarizing(group_id)
            unsummarized_count = group.unsummarized_turn_count()
            needs_summarization = memory_manager.should_summarize_group(group_id)
            
            log.info(f"[GROUP-STATUS] ✅ Retrieved status for {group_id}")
            self.send_json({
                "status": "ok",
                "group_id": group_id,
                "is_summarizing": is_summarizing,
                "summary_ready": group.summary_ready,
                "unsummarized_turns": unsummarized_count,
                "needs_summarization": needs_summarization,
                "total_turns": len(group.all_turns),
            })
            
        except Exception as e:
            log.error(f"[GROUP-STATUS] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_put_group(self, path):
        """Handle PUT /group/{group_id} - Rename group topic"""
        log.info("[GROUP-PUT] ▶ Update group request received")
        try:
            group_id = path.strip("/").split("/")[1]
            
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
            new_topic = body.get("topic", "").strip()
            
            if not new_topic:
                self.send_json({"error": "topic is required"}, 400)
                return
            
            success = memory_manager.update_group_topic(group_id, new_topic)
            
            if success:
                log.info(f"[GROUP-PUT] ✅ Updated group {group_id}")
                self.send_json({
                    "status": "ok",
                    "group_id": group_id,
                    "new_topic": new_topic,
                })
            else:
                self.send_json({"error": f"Group not found: {group_id}"}, 404)
                
        except Exception as e:
            log.error(f"[GROUP-PUT] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_delete_group(self, path):
        """Handle DELETE /group/{group_id} - Delete entire group"""
        log.info("[GROUP-DELETE] ▶ Delete group request received")
        try:
            group_id = path.strip("/").split("/")[1]
            
            if not group_id:
                self.send_json({"error": "group_id is required"}, 400)
                return
            
            success = memory_manager.delete_group(group_id)
            
            if success:
                log.info(f"[GROUP-DELETE] ✅ Deleted group {group_id}")
                self.send_json({
                    "status": "ok",
                    "group_id": group_id,
                    "message": f"Group {group_id} deleted successfully",
                })
            else:
                self.send_json({"error": f"Group not found: {group_id}"}, 404)
                
        except Exception as e:
            log.error(f"[GROUP-DELETE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)
    
    def handle_delete_turn(self, path):
        """Handle DELETE /group/{group_id}/turn/{turn_id} - Delete specific turn"""
        log.info("[TURN-DELETE] ▶ Delete turn request received")
        try:
            parts = path.strip("/").split("/")
            
            if len(parts) < 4:
                self.send_json({"error": "Invalid path"}, 400)
                return
            
            group_id = parts[1]
            turn_id = parts[3]
            
            if not group_id or not turn_id:
                self.send_json({"error": "group_id and turn_id are required"}, 400)
                return
            
            success = memory_manager.delete_group_turn(group_id, turn_id)
            
            if success:
                log.info(f"[TURN-DELETE] ✅ Deleted turn {turn_id} from group {group_id}")
                self.send_json({
                    "status": "ok",
                    "group_id": group_id,
                    "turn_id": turn_id,
                    "message": f"Turn {turn_id} deleted successfully",
                })
            else:
                self.send_json({"error": f"Group or turn not found"}, 404)
                
        except Exception as e:
            log.error(f"[TURN-DELETE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
            self.send_json({"error": str(e)}, 500)

    # ──────────────────────────────────  
    #  Helpers  
    # ──────────────────────────────────  
    def serve_file(self, filepath, content_type):  
        try:  
            with open(filepath, "rb") as f:  
                data = f.read()  
            self.send_response(200)  
            self.send_header("Content-Type",   content_type)  
            self.send_header("Content-Length", len(data))  
            self._cors()  
            self.end_headers()  
            self.wfile.write(data)  
        except FileNotFoundError:  
            self.send_error(404, f"Not found: {filepath}")

    def send_json(self, data, status=200):  
        body = json.dumps(data).encode("utf-8")  
        self.send_response(status)  
        self.send_header("Content-Type",   "application/json")  
        self.send_header("Content-Length", len(body))  
        self._cors()  
        self.end_headers()  
        self.wfile.write(body)

    def _cors(self):  
        self.send_header("Access-Control-Allow-Origin",  "*")  
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")  
        self.send_header("Access-Control-Allow-Headers", "*")

    def log_message(self, format, *args):  
        pass


# ══════════════════════════════════════  
#  START SERVER  
# ══════════════════════════════════════  
if __name__ == "__main__":  
    HOST, PORT = "localhost", 8000  
    server = HTTPServer((HOST, PORT), RAGHandler)
    server.timeout = None  # Infinite timeout for large file uploads
    log.info("=" * 46)  
    log.info("  📄 RAG Pipeline — Enterprise")  
    log.info(f"  🌐  http://{HOST}:{PORT}")  
    log.info("  🛑  Stop: Ctrl + C")  
    log.info("=" * 46)  
    try:  
        server.serve_forever()  
    except KeyboardInterrupt:  
        log.info("[SERVER] Shutting down...")
        log.info("[SERVER] Waiting for background summarization tasks...")
        bg_summarizer.shutdown()
        log.info("[SERVER] Stopped.")  
        server.server_close()  
