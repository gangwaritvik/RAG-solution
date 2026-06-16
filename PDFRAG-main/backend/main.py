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
from backend.memory import ConversationMemoryManager  
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
        if path in self.STATIC:  
            filename, ctype = self.STATIC[path]  
            self.serve_file(os.path.join(FRONTEND_DIR, filename), ctype)  
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

            log.info(f"[QUERY] '{query[:80]}' | top_k: {top_k} | temp: {temp}")

            if not query:  
                self.send_json({"error": "Query is empty."}, 400)  
                return

            if vector_store.count == 0:  
                self.send_json({"error": "No documents ingested yet."}, 400)  
                return

            log.info("[QUERY] STEP 1 — Retrieving relevant chunks")  
            hits = retriever.retrieve(query, top_k=top_k)  
            log.info(f"[QUERY] Retrieved {len(hits)} chunks")

            log.info("[QUERY] STEP 2 — Generating answer with GPT-4.1")  
            answer = generator.generate(query, hits, temperature=temp)  
            log.info("[QUERY] ✅ Answer generated")

            self.send_json({"answer": answer, "sources": hits})

        except Exception as e:  
            log.error(f"[QUERY] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)  
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
