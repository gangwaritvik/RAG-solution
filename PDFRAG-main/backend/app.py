#!/usr/bin/env python3
"""
FastAPI application for the PDF-RAG backend.

This is a drop-in replacement for the legacy http.server backend (main.py). It
reuses the EXACT same pipeline singletons (loader, embedder, vector_store,
retriever, generator, memory_manager, ...) by importing them from main.py, so the
behaviour of every endpoint matches the original. The only additions are:

  * A streaming endpoint  POST /query/stream  that streams the answer token-by-token
    via Server-Sent Events (SSE).
  * Native async serving via uvicorn (ThreadingHTTPServer is no longer needed).

Run with:  uvicorn backend.app:app --host localhost --port 8080
"""
import os
import sys
import json
import asyncio
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from backend.config import UPLOAD_DIR, CHROMA_DIR, FRONTEND_DIR, TOP_K
from backend.utils.logger import get_logger

# Reuse the SAME initialized pipeline singletons and helpers as the legacy server.
# Importing main.py runs its module-level initialization (the heavy pipeline wiring)
# but NOT the http.server start, which is guarded by `if __name__ == "__main__"`.
import backend.main as core

log = get_logger("app")

app = FastAPI(title="PDF-RAG API", version="2.0-fastapi")

# CORS — mirror the legacy "_cors()" (allow everything).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Overall safety-net timeout (seconds) for a single query's blocking pipeline work.
# Individual LLM calls already have their own SDK-level timeouts; this guards against
# a total hang. Generous because parallel map-reduce over a whole document can take a
# while. Returns HTTP 504 if exceeded.
QUERY_TIMEOUT_SECONDS = 120


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def _err(message: str, status: int = 500):
    return JSONResponse({"error": message}, status_code=status)


async def _json_body(request: Request) -> dict:
    try:
        raw = await request.body()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────
#  STATIC FRONTEND
# ──────────────────────────────────────────────────────────────────────
_STATIC = {
    "/":           ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js":     ("app.js",     "application/javascript"),
}


@app.get("/")
@app.get("/index.html")
@app.get("/styles.css")
@app.get("/app.js")
async def serve_static(request: Request):
    path = request.url.path
    entry = _STATIC.get(path)
    if not entry:
        return _err("Not found", 404)
    filename, ctype = entry
    full = os.path.join(FRONTEND_DIR, filename)
    if not os.path.exists(full):
        return _err(f"Not found: {full}", 404)
    return FileResponse(full, media_type=ctype)


# ──────────────────────────────────────────────────────────────────────
#  STATUS
# ──────────────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    try:
        total_vectors = core.vector_store.count
        files_data = []
        if total_vectors > 0:
            all_results = core.vector_store.collection.get(include=["metadatas"])
            files_dict = {}
            for metadata in all_results.get("metadatas", []):
                filename = metadata.get("filename", "unknown")
                files_dict.setdefault(filename, []).append({
                    "chunk_index": metadata.get("chunk_index", 0),
                    "page": metadata.get("page", 0),
                    "text": metadata.get("text", ""),
                })
            for filename, chunks in files_dict.items():
                files_data.append({
                    "filename": filename,
                    "chunk_count": len(chunks),
                    "chunks": sorted(chunks, key=lambda x: x["chunk_index"]),
                })

        with core.ingestion_lock:
            ingestion_data = dict(core.ingestion_status)

        return {
            "status": "ok",
            "total_vectors": total_vectors,
            "files": files_data,
            "ingestion_status": ingestion_data,
            "server": "running",
        }
    except Exception as e:
        log.error(f"[STATUS] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


# ──────────────────────────────────────────────────────────────────────
#  INGEST  (multipart, background processing, 202)
# ──────────────────────────────────────────────────────────────────────
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(default=[]), chunk_mode: str = Form("recursive")):
    try:
        if not files:
            return _err("No files provided.", 400)

        valid = []
        for uf in files:
            name = (uf.filename or "").lower()
            if name.endswith(".pdf") or name.endswith(".docx") or name.endswith(".doc"):
                content = await uf.read()
                valid.append((uf.filename, content))

        if not valid:
            return JSONResponse({"error": "No valid PDF/DOCX files.", "status": "error"}, status_code=400)

        # Mark each file as processing immediately.
        with core.ingestion_lock:
            for filename, _ in valid:
                core.ingestion_status[filename] = {"status": "processing", "chunks": 0}

        # Background processing reuses the legacy worker so behaviour is identical.
        thread = threading.Thread(
            target=core.process_ingest_background,
            args=(valid, chunk_mode),
            daemon=True,
        )
        thread.start()

        return JSONResponse(
            {
                "status": "accepted",
                "message": f"Processing {len(valid)} file(s) in background",
                "files": len(valid),
            },
            status_code=202,
        )
    except Exception as e:
        log.error(f"[INGEST] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


# ──────────────────────────────────────────────────────────────────────
#  QUERY  (shared resolve+retrieve, used by streaming and non-streaming)
# ──────────────────────────────────────────────────────────────────────
def _resolve_and_retrieve(query: str, top_k: int, group_id, top_k_override, retrieve_all) -> dict:
    """Resolve query context (STEP 1) and retrieve chunks (STEP 2).

    This is the SINGLE source of truth shared by the non-streaming `/query` and the
    streaming `/query/stream` endpoints, so their retrieval behaviour can never drift.

    Returns a dict: {query_context, hits, turn_count, memory_context}.
    """
    from backend.memory.resolution.context_resolver import DependencyType

    log.info("[QUERY] STEP 1 — Resolving query context")
    available_documents = core.vector_store.list_filenames()
    query_context = core.context_resolver.resolve(
        query, active_group_id=group_id, available_documents=available_documents
    )
    log.info(f"[QUERY] Standalone query: {query_context.standalone_query[:80]}")

    is_followup = (
        query_context.dependency_type == DependencyType.DEPENDENT
        and query_context.belongs_to_active_group
    )
    operates_on_prev_answer = query_context.answer_source == "previous_answer"

    active_group = None
    if query_context.active_group_id:
        active_group = core.memory_manager.get_conversation_group(query_context.active_group_id)

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
            log.warning("[QUERY] answer_source=previous_answer but no stored previous answer found — falling back to document retrieval")

    intent = query_context.retrieval_intent.value
    broad = intent in [
        "targeted_summary", "global_summary",
        "targeted_extraction", "global_extraction",
        "positional", "comparison", "analysis",
    ]

    get_all_relevant = False
    effective_retrieve_all = retrieve_all

    if broad:
        if prev_answer_chunk is not None:
            log.info(f"[QUERY] Intent '{intent}' on previous answer — skipping document-wide retrieval")
        elif top_k_override is not None:
            log.info(f"[QUERY] Intent '{intent}' — user Top-K override ({top_k_override}) takes precedence over full retrieval")
        elif effective_retrieve_all:
            log.info(f"[QUERY] Intent '{intent}' — manual MAX retrieve_all active")
        else:
            get_all_relevant = True
            log.info(f"[QUERY] Intent '{intent}' — fetching ALL relevant chunks (not limited by top_k)")

    log.info("[QUERY] STEP 2 — Retrieving relevant chunks")
    segments = None  # populated only for compound (multi-operation) queries
    if query_context.dependency_type == DependencyType.AMBIGUOUS:
        log.info("[QUERY] ⚠️ AMBIGUOUS query — skipping retrieval, will generate clarification")
        hits = []
    elif prev_answer_chunk is not None:
        log.info("[QUERY] Operating on previous answer — skipping document retrieval entirely")
        hits = []
    elif (
        getattr(query_context, "segments", None)
        and not effective_retrieve_all
        and top_k_override is None
    ):
        # COMPOUND query (e.g. "compare A and B, and extract C"): each segment is a
        # self-contained operation with its OWN intent and optional file pinning. Retrieve
        # each segment independently so it gets its intent-appropriate depth and the right
        # document(s); generation answers each segment in its own section.
        segments = query_context.segments
        log.info(f"[QUERY] Compound query — per-segment retrieval across {len(segments)} segment(s)")
        flat = []
        for seg in segments:
            seg_hits = core.retriever.retrieve(
                seg["query"],
                top_k=top_k,
                get_all_relevant=True,
                retrieval_intent=seg.get("intent"),
                restrict_filenames=seg.get("files"),
            )
            seg["hits"] = seg_hits
            flat.extend(seg_hits)
            log.info(f"[QUERY]   segment '{seg.get('title')}' (intent={seg.get('intent')}) → {len(seg_hits)} chunks")
        # De-duplicate the union for the sources chips shown in the UI.
        seen = set()
        hits = []
        for h in flat:
            k = (h.get("filename"), h.get("chunk_index"))
            if k not in seen:
                seen.add(k)
                hits.append(h)
        log.info(f"[QUERY] Retrieved {len(hits)} unique chunks (compound — {len(segments)} segments)")
    elif (
        query_context.is_multi_group
        and len(query_context.sub_queries) >= 2
        and not effective_retrieve_all
        and top_k_override is None
    ):
        # MULTI-SUBJECT query (e.g. "compare X and Y"): retrieve each subject
        # independently in parallel and merge, for balanced coverage of all subjects.
        log.info(f"[QUERY] Multi-group query — balanced retrieval across {len(query_context.sub_queries)} subjects")
        hits = core.multi_group_processor.retrieve_balanced(
            sub_queries=query_context.sub_queries,
            retrieval_intent=intent,
        )
        if not hits:
            # < 2 valid subjects after cleaning — fall back to normal retrieval.
            hits = core.retriever.retrieve(
                query_context.standalone_query,
                top_k=top_k,
                get_all_relevant=get_all_relevant,
                retrieval_intent=intent,
                top_k_override=top_k_override,
                retrieve_all=effective_retrieve_all,
            )
        log.info(f"[QUERY] Retrieved {len(hits)} chunks (multi-group balanced)")
    else:
        hits = core.retriever.retrieve(
            query_context.standalone_query,
            top_k=top_k,
            get_all_relevant=get_all_relevant,
            retrieval_intent=intent,
            top_k_override=top_k_override,
            retrieve_all=effective_retrieve_all,
            restrict_filenames=query_context.restrict_filenames,
        )
        log.info(f"[QUERY] Retrieved {len(hits)} chunks")

    if prev_answer_chunk is not None:
        hits = [prev_answer_chunk] + hits

    turn_count = len(active_group.all_turns) if active_group else 1
    gen_memory_context = None if prev_answer_chunk is not None else query_context.memory_context

    return {
        "query_context": query_context,
        "hits": hits,
        "segments": segments,
        "turn_count": turn_count,
        "memory_context": gen_memory_context,
    }


def _resolve_query(query: str, top_k: int, temp: float, group_id, top_k_override, retrieve_all):
    """Run the full retrieval+generation pipeline (non-streaming) and return the
    legacy /query response dict."""
    ctx = _resolve_and_retrieve(query, top_k, group_id, top_k_override, retrieve_all)
    query_context = ctx["query_context"]

    log.info("[QUERY] STEP 3 — Generating answer with memory context")
    segments = ctx.get("segments")
    if segments:
        answer, memory_summary = core.generator.generate_segments(
            segments=segments,
            temperature=temp,
            memory_context=ctx["memory_context"],
            turn_count=ctx["turn_count"],
        )
    else:
        # Generate against the RESOLVED standalone query (typos corrected, references
        # like "the above" rewritten), NOT the raw user text — otherwise the model sees
        # gibberish (e.g. a typo'd "what is T aobve") and says the context has nothing,
        # even though retrieval used the resolved query and fetched the right chunks.
        gen_query = getattr(query_context, "standalone_query", None) or query
        answer, memory_summary = core.generator.generate(
            query=gen_query,
            context_chunks=ctx["hits"],
            temperature=temp,
            memory_context=ctx["memory_context"],
            retrieval_intent=query_context.retrieval_intent.value,
            turn_count=ctx["turn_count"],
        )
    log.info("[QUERY] ✅ Answer generated successfully")

    _store_turn(query_context, query, memory_summary, answer)

    return {
        "answer": answer,
        "retrieved_chunks": ctx["hits"],
        "query": query,
        "group_id": query_context.active_group_id,
    }


def _store_turn(query_context, query, memory_summary, answer):
    """Persist the completed turn to conversation memory (STEP 4)."""
    log.info("[QUERY] STEP 4 — Storing turn in conversation memory")
    if query_context.active_group_id:
        core.memory_manager.add_conversation_turn(
            group_id=query_context.active_group_id,
            query=query,
            memory_summary=memory_summary,
            dependency_type=query_context.dependency_type.value,
            retrieval_intent=query_context.retrieval_intent.value,
            full_answer=answer,
            restrict_filenames=query_context.restrict_filenames,
        )
        log.info(f"[QUERY] ✅ Turn saved to group {query_context.active_group_id}")
        # Auto roll-up: once a group reaches the threshold (5) UNSUMMARIZED turns, kick
        # off summarization in the BACKGROUND (non-blocking) so it never delays the user's
        # response. While that summary is being generated (or before it exists), the group's
        # summary_ready flag stays False, so the resolver keeps using the group's RAW turn
        # Q&A as context — a query arriving mid-summarization is fully race-safe.
        try:
            core.bg_summarizer.summarize_if_needed(query_context.active_group_id, threshold=5)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[QUERY] ⚠️ Could not trigger background summarization: {e}")
    else:
        log.warning("[QUERY] ⚠️ No active group — turn not saved to memory")


@app.post("/query")
async def query(request: Request):
    body = await _json_body(request)
    query_text = (body.get("query") or "").strip()
    if not query_text:
        return _err("Query is empty.", 400)
    if core.vector_store.count == 0:
        return _err("No documents ingested yet.", 400)

    top_k = int(body.get("top_k", TOP_K))
    temp = float(body.get("temperature", 0.2))
    group_id = body.get("group_id")
    top_k_override = _parse_override(body.get("top_k_override"))
    retrieve_all = bool(body.get("retrieve_all", False))

    log.info(f"[QUERY] ▶ '{query_text[:80]}' | top_k={top_k} | temp={temp} | override={top_k_override} | retrieve_all={retrieve_all}")
    try:
        # Run the blocking pipeline in a thread so the event loop stays free, capped by
        # an overall timeout so a hung LLM/retrieval call can't block the request forever.
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _resolve_query, query_text, top_k, temp, group_id, top_k_override, retrieve_all
            ),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError:
        log.error(f"[QUERY] ⏱️ TIMEOUT — exceeded {QUERY_TIMEOUT_SECONDS}s")
        return JSONResponse(
            {"error": f"Query processing timeout ({QUERY_TIMEOUT_SECONDS}s exceeded)", "status": "timeout"},
            status_code=504,
        )
    except Exception as e:
        log.error(f"[QUERY] ❌ FATAL — {type(e).__name__}: {e}", exc_info=True)
        return _err(str(e), 500)


def _parse_override(value):
    if value is None:
        return None
    try:
        v = int(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────
#  QUERY (STREAMING) — Server-Sent Events
# ──────────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    """Format a dict as a single SSE 'data:' event."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/query/stream")
async def query_stream(request: Request):
    body = await _json_body(request)
    query_text = (body.get("query") or "").strip()
    if not query_text:
        return _err("Query is empty.", 400)
    if core.vector_store.count == 0:
        return _err("No documents ingested yet.", 400)

    top_k = int(body.get("top_k", TOP_K))
    temp = float(body.get("temperature", 0.2))
    group_id = body.get("group_id")
    top_k_override = _parse_override(body.get("top_k_override"))
    retrieve_all = bool(body.get("retrieve_all", False))

    log.info(f"[QUERY/STREAM] ▶ '{query_text[:80]}' | override={top_k_override} | retrieve_all={retrieve_all}")

    async def event_gen():
        try:
            # 1) Resolve context + retrieve (blocking → thread, capped by timeout).
            #    Uses the SAME shared helper as /query so behaviour can't drift.
            ctx = await asyncio.wait_for(
                asyncio.to_thread(
                    _resolve_and_retrieve, query_text, top_k, group_id, top_k_override, retrieve_all
                ),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
            query_context = ctx["query_context"]
            hits = ctx["hits"]

            # Send metadata first (sources + group id) so the UI can render chips.
            yield _sse({
                "type": "meta",
                "group_id": query_context.active_group_id,
                "retrieved_chunks": hits,
                "intent": query_context.retrieval_intent.value,
            })

            # 2) Stream the answer. generate_stream is a blocking generator, so we
            #    drain it in a worker thread and hand items to the event loop via a queue.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            SENTINEL = object()
            final = {"answer": "", "memory_summary": ""}

            def _produce():
                try:
                    segments = ctx.get("segments")
                    if segments:
                        gen_iter = core.generator.generate_segments_stream(
                            segments=segments,
                            temperature=temp,
                            memory_context=ctx["memory_context"],
                            turn_count=ctx["turn_count"],
                        )
                    else:
                        # Use the RESOLVED standalone query (typos fixed, references
                        # rewritten) for generation too — matching what retrieval used.
                        # Passing the raw user text makes the model answer the literal
                        # typo'd string and wrongly report the context as empty.
                        gen_query = getattr(query_context, "standalone_query", None) or query_text
                        gen_iter = core.generator.generate_stream(
                            query=gen_query,
                            context_chunks=hits,
                            temperature=temp,
                            memory_context=ctx["memory_context"],
                            retrieval_intent=query_context.retrieval_intent.value,
                            turn_count=ctx["turn_count"],
                        )
                    for ev in gen_iter:
                        if ev.get("type") == "done":
                            final["answer"] = ev.get("answer", "")
                            final["memory_summary"] = ev.get("memory_summary", "")
                        loop.call_soon_threadsafe(queue.put_nowait, ev)
                except Exception as e:  # noqa: BLE001
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": str(e)})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

            threading.Thread(target=_produce, daemon=True).start()

            while True:
                ev = await queue.get()
                if ev is SENTINEL:
                    break
                yield _sse(ev)

            # 3) Persist the turn (STEP 4) after streaming completes.
            if final["answer"]:
                await asyncio.to_thread(
                    _store_turn, query_context, query_text, final["memory_summary"], final["answer"]
                )
        except asyncio.TimeoutError:
            log.error(f"[QUERY/STREAM] ⏱️ TIMEOUT — exceeded {QUERY_TIMEOUT_SECONDS}s")
            yield _sse({"type": "error", "error": f"Query processing timeout ({QUERY_TIMEOUT_SECONDS}s exceeded)"})
        except Exception as e:  # noqa: BLE001
            log.error(f"[QUERY/STREAM] ❌ {type(e).__name__}: {e}", exc_info=True)
            yield _sse({"type": "error", "error": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ──────────────────────────────────────────────────────────────────────
#  DELETE / CLEAR
# ──────────────────────────────────────────────────────────────────────
@app.post("/delete")
async def delete(request: Request):
    body = await _json_body(request)
    filename = (body.get("filename") or "").strip()
    if not filename:
        return _err("No filename provided.", 400)
    try:
        deleted = core.vector_store.delete_by_filename(filename)
        with core.ingestion_lock:
            core.ingestion_status.pop(filename, None)
        return {
            "status": "ok",
            "filename": filename,
            "deleted_vectors": deleted,
            "total_vectors": core.vector_store.count,
        }
    except Exception as e:
        log.error(f"[DELETE] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.post("/clear")
async def clear():
    try:
        core.vector_store.clear()
        with core.ingestion_lock:
            core.ingestion_status.clear()
        return {"status": "ok", "message": "All vectors cleared"}
    except Exception as e:
        log.error(f"[CLEAR] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


# ──────────────────────────────────────────────────────────────────────
#  CONVERSATION GROUPS
# ──────────────────────────────────────────────────────────────────────
def _group_summary_dict(g):
    return {
        "group_id": g.group_id,
        "topic": g.topic,
        "summary_ready": g.summary_ready,
        "recent_turns": len(g.recent_turns),
        "total_turns": len(g.all_turns),
        "created_at": g.created_at,
        "updated_at": g.updated_at,
    }


@app.get("/groups")
async def get_groups():
    try:
        groups = core.memory_manager.list_conversation_groups()
        return {
            "status": "ok",
            "groups": [_group_summary_dict(g) for g in groups],
            "total": len(groups),
        }
    except Exception as e:
        log.error(f"[GROUPS] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.post("/group/create")
async def group_create(request: Request):
    body = await _json_body(request)
    topic = (body.get("topic") or "").strip()
    if not topic:
        return _err("No topic provided.", 400)
    try:
        group = core.memory_manager.create_conversation_group(topic)
        return {
            "status": "ok",
            "group_id": group.group_id,
            "topic": group.topic,
            "created_at": group.created_at,
        }
    except Exception as e:
        log.error(f"[GROUP/CREATE] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.post("/group/list")
async def group_list():
    try:
        groups = core.memory_manager.list_conversation_groups()
        return {
            "status": "ok",
            "total_groups": len(groups),
            "groups": [_group_summary_dict(g) for g in groups],
        }
    except Exception as e:
        log.error(f"[GROUP/LIST] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.post("/group/summarize")
async def group_summarize(request: Request):
    body = await _json_body(request)
    group_id = (body.get("group_id") or "").strip()
    if not group_id:
        return _err("No group_id provided.", 400)
    try:
        group = core.memory_manager.get_conversation_group(group_id)
        if not group:
            return _err(f"Group not found: {group_id}", 404)

        if core.bg_summarizer.is_summarizing(group_id):
            return JSONResponse(
                {"status": "in_progress", "message": "Summarization already running", "group_id": group_id},
                status_code=202,
            )

        if not core.memory_manager.should_summarize_group(group_id):
            return {
                "status": "not_needed",
                "message": "Group does not need summarization yet",
                "group_id": group_id,
                "unsummarized_turns": len(group.recent_turns),
            }

        core.bg_summarizer.summarize_if_needed(group_id, threshold=5)
        return JSONResponse(
            {
                "status": "triggered",
                "message": "Summarization started in background",
                "group_id": group_id,
                "unsummarized_turns": len(group.recent_turns),
            },
            status_code=202,
        )
    except Exception as e:
        log.error(f"[GROUP/SUMMARIZE] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.post("/group/status")
async def group_status(request: Request):
    body = await _json_body(request)
    group_id = (body.get("group_id") or "").strip() if body.get("group_id") else None
    try:
        if not group_id:
            active = core.bg_summarizer.get_active_summarizations()
            return {"status": "ok", "active_summarizations": active, "total_active": len(active)}

        group = core.memory_manager.get_conversation_group(group_id)
        if not group:
            return _err(f"Group not found: {group_id}", 404)
        return {
            "status": "ok",
            "group_id": group_id,
            "is_summarizing": core.bg_summarizer.is_summarizing(group_id),
            "summary_ready": group.summary_ready,
            "unsummarized_turns": len(group.recent_turns),
            "needs_summarization": core.memory_manager.should_summarize_group(group_id),
            "total_turns": len(group.all_turns),
        }
    except Exception as e:
        log.error(f"[GROUP/STATUS] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.get("/group/{group_id}")
async def group_get(group_id: str):
    try:
        ctx = core.memory_manager.get_group_context(group_id, include_all_turns=False)
        if not ctx:
            return _err(f"Group not found: {group_id}", 404)
        return ctx
    except Exception as e:
        log.error(f"[GROUP/GET] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.get("/group/{group_id}/history")
async def group_history(group_id: str):
    try:
        history = core.memory_manager.get_group_history(group_id)
        if history is None:
            return _err(f"Group not found: {group_id}", 404)
        return history
    except Exception as e:
        log.error(f"[GROUP/HISTORY] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.get("/group/{group_id}/summary")
async def group_summary(group_id: str):
    try:
        summary = core.memory_manager.get_group_summary(group_id)
        if summary is None:
            return _err(f"Group not found: {group_id}", 404)
        return summary
    except Exception as e:
        log.error(f"[GROUP/SUMMARY] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.put("/group/{group_id}")
async def group_rename(group_id: str, request: Request):
    body = await _json_body(request)
    new_topic = (body.get("topic") or "").strip()
    if not new_topic:
        return _err("No topic provided.", 400)
    try:
        ok = core.memory_manager.update_group_topic(group_id, new_topic)
        if not ok:
            return _err(f"Group not found: {group_id}", 404)
        return {"status": "ok", "group_id": group_id, "new_topic": new_topic}
    except Exception as e:
        log.error(f"[GROUP/RENAME] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.delete("/group/{group_id}/turn/{turn_id}")
async def delete_turn(group_id: str, turn_id: str):
    try:
        ok = core.memory_manager.delete_group_turn(group_id, turn_id)
        if not ok:
            return _err("Group or turn not found", 404)
        return {
            "status": "ok",
            "group_id": group_id,
            "turn_id": turn_id,
            "message": f"Turn {turn_id} deleted successfully",
        }
    except Exception as e:
        log.error(f"[GROUP/DELETE_TURN] Failed — {e}", exc_info=True)
        return _err(str(e), 500)


@app.delete("/group/{group_id}")
async def delete_group(group_id: str):
    try:
        ok = core.memory_manager.delete_group(group_id)
        if not ok:
            return _err(f"Group not found: {group_id}", 404)
        return {
            "status": "ok",
            "group_id": group_id,
            "message": f"Group {group_id} deleted successfully",
        }
    except Exception as e:
        log.error(f"[GROUP/DELETE] Failed — {e}", exc_info=True)
        return _err(str(e), 500)
