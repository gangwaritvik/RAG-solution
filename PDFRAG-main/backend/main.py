#!/usr/bin/env python3  
import json  
import os  
import sys

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
from backend.utils.parser import parse_multipart_full  
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
memory_manager = ConversationMemoryManager(chroma_db_path=CHROMA_DIR)
context_resolver = ContextResolver(memory_manager, embedder)

# ── Clear stale vectors on startup ──  
vector_store.clear()  
log.info("[STARTUP] Vector store cleared — fresh session started")
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
        
        # Static files
        if path in self.STATIC:  
            filename, ctype = self.STATIC[path]  
            self.serve_file(os.path.join(FRONTEND_DIR, filename), ctype)
        
        # Group endpoints
        elif path.startswith("/group/"):
            self.handle_group_get(path)
        
        else:  
            self.send_error(404, "Not Found")
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
        path = urlparse(self.path).path  
        if path == "/ingest":  
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

        else:  
            self.send_error(404, "Not Found")  

    # ──────────────────────────────────  
    #  INGEST  
    # ──────────────────────────────────  
    def handle_ingest(self):  
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

            # ── Select chunker dynamically ──  
            active_chunker = get_chunker(chunk_mode)

            results = []

            for filename, content in files:  
                log.info(f"[INGEST] Processing: {filename} ({len(content)} bytes)")

                if not (filename.lower().endswith(".pdf") or  
                        filename.lower().endswith(".docx") or  
                        filename.lower().endswith(".doc")):  
                    log.warning(f"[INGEST] Skipping unsupported file: {filename}")  
                    continue  

                # Step 1 — Load  
                log.info(f"[INGEST] STEP 1 — Loading {filename}")  
                doc_results = loader.load_documents([(filename, content)])

                # Step 2 — Chunk  
                log.info(f"[INGEST] STEP 2 — Chunking {filename} (mode: {chunk_mode})")  
                chunked = active_chunker.chunk_documents(doc_results)

                for doc in chunked:  
                    if not doc.get("chunks"):  
                        log.warning(f"[INGEST] ⚠️ No chunks for {doc.get('filename')}")  
                        continue

                    texts = [c.page_content for c in doc["chunks"]]

                    # Step 3 — Embed  
                    log.info(f"[INGEST] STEP 3 — Embedding {len(texts)} chunks")  
                    vecs = embedder.embed_texts(texts)

                    # Step 4 — Store  
                    log.info(f"[INGEST] STEP 4 — Storing {len(vecs)} vectors")  
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
                    log.info(f"[INGEST] ✅ {doc['filename']} — ingested successfully")

            log.info(f"[INGEST] ✅ Complete — {len(results)} doc(s) | {vector_store.count} total vectors")  
            self.send_json({  
                "status":        "ok",  
                "chunk_mode":    chunk_mode,  
                "total_vectors": vector_store.count,  
                "documents":     results,  
            })

        except Exception as e:  
            log.error(f"[INGEST] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)

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

            log.info(f"[QUERY] '{query[:80]}' | top_k: {top_k} | temp: {temp}")

            if not query:  
                self.send_json({"error": "Query is empty."}, 400)  
                return

            if vector_store.count == 0:  
                self.send_json({"error": "No documents ingested yet."}, 400)  
                return

            # STEP 1 — Context Resolution: Classify query and load memory context
            log.info("[QUERY] STEP 1 — Resolving query context")
            query_context = context_resolver.resolve(query, active_group_id=group_id)
            log.info(f"[QUERY] Dependency: {query_context.dependency_type.value} | Intent: {query_context.retrieval_intent.value}")
            log.info(f"[QUERY] Standalone query: {query_context.standalone_query[:80]}")

            # STEP 2 — Retrieval: Use standalone query for document retrieval
            log.info("[QUERY] STEP 2 — Retrieving relevant chunks")
            # Use standalone query for better retrieval
            hits = retriever.retrieve(query_context.standalone_query, top_k=top_k)  
            log.info(f"[QUERY] Retrieved {len(hits)} chunks")

            # STEP 3 — Generation: Generate answer with memory context
            log.info("[QUERY] STEP 3 — Generating answer with memory context")
            answer, memory_summary = generator.generate(
                query=query,
                context_chunks=hits,
                temperature=temp,
                memory_context=query_context.memory_context
            )  
            log.info("[QUERY] ✅ Answer generated with memory summary")

            # STEP 4 — Memory Storage: Store in conversation group if provided
            if group_id:
                log.info(f"[QUERY] STEP 4 — Storing in conversation group: {group_id}")
                turn = memory_manager.add_conversation_turn(
                    group_id=group_id,
                    query=query,
                    memory_summary=memory_summary
                )
                if turn:
                    log.info(f"[QUERY] ✅ Turn stored in group {group_id}")
                    # Check if group should be summarized
                    if memory_manager.should_summarize_group(group_id):
                        log.info(f"[QUERY] ⚠️ Group {group_id} has 5+ turns — ready for summarization")
                else:
                    log.warning(f"[QUERY] ⚠️ Failed to store turn in group {group_id}")

            self.send_json({
                "answer": answer, 
                "memory_summary": memory_summary,
                "sources": hits,
                "group_id": group_id,
                "query_context": {
                    "dependency_type": query_context.dependency_type.value,
                    "retrieval_intent": query_context.retrieval_intent.value,
                    "standalone_query": query_context.standalone_query,
                    "relevant_groups": len(query_context.relevant_groups),
                }
            })

        except Exception as e:  
            log.error(f"[QUERY] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
            self.send_json({"error": str(e)}, 500)

    # ──────────────────────────────────  
    #  CONVERSATION GROUPS  
    # ──────────────────────────────────  
    def handle_group_get(self, path):
        """Handle GET /group/{group_id}"""
        log.info("[GROUP-GET] ▶ Group get request received")
        try:
            group_id = path.split("/")[-1].strip()
            
            if not group_id:
                self.send_json({"error": "No group_id provided"}, 400)
                return
            
            log.info(f"[GROUP-GET] Retrieving group: {group_id}")
            context = memory_manager.get_group_context(group_id, include_all_turns=False)
            
            if not context:
                self.send_json({"error": f"Group not found: {group_id}"}, 404)
                return
            
            log.info(f"[GROUP-GET] ✅ Retrieved group {group_id}")
            self.send_json(context)
            
        except Exception as e:
            log.error(f"[GROUP-GET] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
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
        """Handle POST /group/summarize"""
        log.info("[GROUP-SUMMARIZE] ▶ Group summarize request received")
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
            
            if not memory_manager.should_summarize_group(group_id):
                self.send_json({
                    "status": "not_needed",
                    "message": f"Group has only {group.unsummarized_turn_count()} turns (needs 5+)",
                    "unsummarized_turns": group.unsummarized_turn_count(),
                })
                return
            
            log.info(f"[GROUP-SUMMARIZE] Summarizing group {group_id}")
            
            # Generate summary from recent turns
            recent_queries = "\n".join([
                f"Q: {t.query}\nA: {t.memory_summary}"
                for t in group.recent_turns
            ])
            
            summary_prompt = f"""
            Topic: {group.topic}
            
            Recent conversation:
            {recent_queries}
            
            Create a concise summary of the key points discussed (2-3 sentences):
            """
            
            response = generator.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1,
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Generate embedding for the summary
            log.info(f"[GROUP-SUMMARIZE] Embedding summary for group {group_id}")
            embeddings = embedder.embed_texts([summary])
            summary_embedding = embeddings[0] if embeddings else None
            
            # Update group
            memory_manager.update_group_summary(
                group_id=group_id,
                summary=summary,
                summary_embedding=summary_embedding
            )
            
            log.info(f"[GROUP-SUMMARIZE] ✅ Summarized group {group_id}")
            self.send_json({
                "status": "ok",
                "group_id": group_id,
                "summary": summary,
                "summarized_turns": len(group.recent_turns),
            })
            
        except Exception as e:
            log.error(f"[GROUP-SUMMARIZE] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")  
        self.send_header("Access-Control-Allow-Headers", "*")

    def log_message(self, format, *args):  
        pass


# ══════════════════════════════════════  
#  START SERVER  
# ══════════════════════════════════════  
if __name__ == "__main__":  
    HOST, PORT = "localhost", 8000  
    server = HTTPServer((HOST, PORT), RAGHandler)  
    log.info("=" * 46)  
    log.info("  📄 RAG Pipeline — Enterprise")  
    log.info(f"  🌐  http://{HOST}:{PORT}")  
    log.info("  🛑  Stop: Ctrl + C")  
    log.info("=" * 46)  
    try:  
        server.serve_forever()  
    except KeyboardInterrupt:  
        log.info("[SERVER] Stopped.")  
        server.server_close()  
