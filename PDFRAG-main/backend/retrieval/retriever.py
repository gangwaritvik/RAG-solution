from backend.ingestion.embedder import Embedder  
from backend.ingestion.vector_store import VectorStore  
from backend.config import TOP_K  
from backend.utils.logger import get_logger  
import numpy as np

log = get_logger("retriever")

# ── Document-level threshold ──  
DOC_RELEVANCE_THRESHOLD = 0.25   # min avg score for a doc to be considered relevant  
CHUNK_RELEVANCE_THRESHOLD = 0.20  # default min score for individual chunks (factual)

# ── Per-intent retrieval configuration ──
# Each intent explicitly declares its retrieval strategy:
#   mode      = how chunks are selected:
#                 "top_k"         → best N chunks by similarity (precise answers)
#                 "comprehensive" → ALL chunks above threshold (synthesis)
#                 "exhaustive"    → ENTIRE document, no threshold (enumeration)
#   top_k     = max chunks to return (used by "top_k" mode; None = unbounded)
#   threshold = min similarity score a chunk must have to be included
INTENT_RETRIEVAL_CONFIG = {
    # Precise, single-fact answers — keep tight & relevant
    "factual":    {"mode": "top_k",         "top_k": 10,   "threshold": 0.20},
    # Targeted synthesis — summarize a specific topic/section
    "targeted_summary": {"mode": "comprehensive", "top_k": 30,   "threshold": 0.10},
    # Global synthesis — summarize whole document/corpus
    "global_summary":   {"mode": "exhaustive",    "top_k": None, "threshold": 0.0},
    # Targeted extraction — extract a specific table/list/subset
    "targeted_extraction": {"mode": "comprehensive", "top_k": 30,   "threshold": 0.05},
    # Global extraction — list/enumerate ALL matching items across the document
    "global_extraction":   {"mode": "exhaustive",    "top_k": None, "threshold": 0.0},
    # Positional/ordinal — "the last two", "the 5th item". Needs the WHOLE document in
    # natural order so position can be resolved; exhaustive mode returns every chunk and
    # the retriever emits them in document order (page, then chunk_index).
    "positional":          {"mode": "exhaustive",    "top_k": None, "threshold": 0.0},
    # Reasoning over the topic — same breadth as summary
    "analysis":   {"mode": "comprehensive", "top_k": 30,   "threshold": 0.10},
    # Comparing items — broad coverage of both sides, but capped
    "comparison": {"mode": "comprehensive", "top_k": 30,   "threshold": 0.12},
    # Ambiguous — minimal retrieval (clarification usually generated instead)
    "ambiguous":  {"mode": "top_k",         "top_k": 3,    "threshold": 0.20},
}

# Fallback for unknown / missing intent
DEFAULT_INTENT_CONFIG = {"mode": "top_k", "top_k": 5, "threshold": CHUNK_RELEVANCE_THRESHOLD}


def _filename_filter(restrict_filenames):
    """Build a ChromaDB where-filter pinning search to one or more filenames.

    Accepts None, a single filename (str), or a list of filenames (N-file scoping).
    Returns None when empty, a single-equality filter for one file, or an ``$in``
    filter for N files — so a sub-query that targets any subset of the loaded
    documents searches exactly those files and no others.
    """
    if not restrict_filenames:
        return None
    if isinstance(restrict_filenames, str):
        files = [restrict_filenames]
    else:
        files = [str(f).strip() for f in restrict_filenames if str(f).strip()]
    files = list(dict.fromkeys(files))  # de-dup, preserve order
    if not files:
        return None
    if len(files) == 1:
        return {"filename": files[0]}
    return {"filename": {"$in": files}}


class Retriever:  
    def __init__(self, embedder: Embedder, vector_store: VectorStore):  
        self.embedder     = embedder  
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K, get_all_relevant: bool = False, retrieval_intent: str = None, top_k_override: int = None, retrieve_all: bool = False, restrict_filenames=None):
        log.info(f"[RETRIEVER] Query: '{query[:60]}' | top_k: {top_k} | get_all_relevant: {get_all_relevant} | intent: {retrieval_intent} | top_k_override: {top_k_override} | retrieve_all: {retrieve_all} | restrict_filenames: {restrict_filenames}")

        query_vector = self.embedder.embed_query(query)
        total        = self.vector_store.count

        if total == 0:
            log.warning("[RETRIEVER] Vector store is empty")
            return []

        # Per-subject document pinning: when a sub-query targets specific file(s), the
        # vector search is restricted to those file(s) so the subject never pulls chunks
        # from unrelated documents. Supports ONE file or N files (ChromaDB $in).
        where_filter = _filename_filter(restrict_filenames)

        # ── MAX / retrieve-all: return EVERY chunk in the store, ranked by similarity,
        # with NO threshold and NO document-relevance filtering. This is the explicit
        # "use all chunks" mode triggered by the frontend MAX toggle. ──
        if retrieve_all:
            all_hits = self.vector_store.search(query_vector, top_k=total, where=where_filter)
            log.info(f"[RETRIEVER] RETRIEVE-ALL (MAX) → returning all {len(all_hits)} chunks (no threshold, no doc filter)")
            return all_hits

        # ── Resolve per-intent retrieval config ──
        cfg = INTENT_RETRIEVAL_CONFIG.get(retrieval_intent, DEFAULT_INTENT_CONFIG)
        mode            = cfg["mode"]
        # When the intent declares no cap (top_k is None — e.g. the exhaustive global
        # intents), it is UNBOUNDED: every qualifying chunk is kept. Only fall back to
        # the caller's default ``top_k`` for capped modes; never let it silently shrink
        # an unbounded intent (that would make the log read like a small cap was applied
        # when the whole document is actually returned).
        is_unbounded    = cfg["top_k"] is None
        intent_top_k    = top_k if is_unbounded else cfg["top_k"]
        chunk_threshold = cfg["threshold"]

        # ── User override: when the frontend explicitly sets Top-K, it WINS over the
        # intent's default cap for every mode. An exhaustive intent ("list all") is
        # downgraded to a capped comprehensive pass so the user's K is respected. ──
        if top_k_override is not None and top_k_override > 0:
            intent_top_k = top_k_override
            is_unbounded = False
            if mode == "exhaustive":
                mode = "comprehensive"
            log.info(f"[RETRIEVER] Top-K OVERRIDE active → using user top_k={intent_top_k} (mode now '{mode}')")

        top_k_display = "ALL" if (is_unbounded and mode == "exhaustive") else intent_top_k
        log.info(
            f"[RETRIEVER] Intent '{retrieval_intent}' → mode={mode} | "
            f"top_k={top_k_display} | chunk_threshold={chunk_threshold}"
        )
        # ── Step 1: Fetch ALL vectors ──  
        all_hits = self.vector_store.search(query_vector, top_k=total, where=where_filter)  
        log.info(f"[RETRIEVER] Total vectors fetched: {len(all_hits)}")

        # ── Step 2: Group all hits by document ──  
        per_doc = {}  
        for hit in all_hits:  
            fname = hit.get("filename", "unknown")  
            if fname not in per_doc:  
                per_doc[fname] = []  
            per_doc[fname].append(hit)

        # ── Step 3: Score each document by avg similarity of its TOP 5 chunks ──  
        doc_scores = {}  
        for fname, hits in per_doc.items():  
            top_hits       = hits[:5]   # top 5 chunks per doc  
            avg_score      = np.mean([h.get("score", 0) for h in top_hits])  
            doc_scores[fname] = avg_score  
            log.info(f"[RETRIEVER] Doc score — {fname}: {avg_score:.3f}")

        # ── Step 4: Detect relevant documents automatically ──  
        max_score = max(doc_scores.values()) if doc_scores else 0

        relevant_docs = {}  
        for fname, score in doc_scores.items():  
            is_relevant = (  
                score >= DOC_RELEVANCE_THRESHOLD           # absolute threshold  
                or score >= max_score * 0.75               # relative: within 75% of best doc; tighter than 0.60  
            )  
            if is_relevant:  
                relevant_docs[fname] = per_doc[fname]  
                log.info(f"[RETRIEVER] ✅ RELEVANT — {fname} (score: {score:.3f})")  
            else:  
                log.info(f"[RETRIEVER] ❌ EXCLUDED — {fname} (score: {score:.3f} too low)")

        # ── Step 5: If NO doc passes threshold, fall back to top-scoring doc only ──  
        if not relevant_docs:  
            best_doc = max(doc_scores, key=doc_scores.get)  
            relevant_docs[best_doc] = per_doc[best_doc]  
            log.warning(f"[RETRIEVER] No doc passed threshold — falling back to: {best_doc}")

        # ── Step 6: Retrieve chunks from EACH relevant document ──  
        # Dispatch on the intent's configured retrieval mode (see INTENT_RETRIEVAL_CONFIG):
        #   exhaustive    → entire document (enumeration / "list all")
        #   comprehensive → all chunks above the intent's threshold (synthesis)
        #   top_k         → best N chunks above threshold (precise answers)
        diverse = []
        for fname, hits in relevant_docs.items():
            if mode == "exhaustive":
                # EXHAUSTIVE MODE: enumeration queries ("list all X") and POSITIONAL
                # queries ("the last two", "the 5th item", "the one after X") must scan
                # the ENTIRE document. Threshold filtering would drop chunks that mention
                # an item only in passing, making the list incomplete. Return ALL chunks —
                # but ordered by their NATURAL POSITION in the document (page, then
                # chunk_index), NOT by similarity. Document order is what lets the model
                # resolve "first"/"last"/"Nth"; similarity order scrambles the sequence and
                # makes positional references impossible to answer.
                final = sorted(
                    hits,
                    key=lambda h: (int(h.get("page") or 0), int(h.get("chunk_index") or 0)),
                )
                log.info(f"[RETRIEVER] EXHAUSTIVE — {fname} → {len(final)} chunks (full-document scan, document order, no threshold)")
            elif mode == "comprehensive":
                # COMPREHENSIVE MODE: Get chunks above the intent's threshold, ordered by
                # similarity, then cap to the intent's top_k so we don't send an oversized
                # context to the LLM. hits are already sorted best-first.
                above = [h for h in hits if h.get("score", 0) >= chunk_threshold]
                # Apply the cap when the intent defines one OR a user override is active
                # (an exhaustive intent downgraded by override has cfg["top_k"] is None).
                apply_cap = cfg["top_k"] is not None or (top_k_override is not None and top_k_override > 0)
                selected = above[:intent_top_k] if apply_cap else above
                log.info(f"[RETRIEVER] COMPREHENSIVE — {fname} → {len(selected)} chunks (threshold {chunk_threshold}, cap {intent_top_k})")
                final = selected
            else:
                # TOP_K MODE: Get top_k chunks by similarity above threshold
                above = [h for h in hits if h.get("score", 0) >= chunk_threshold]
                selected = above[:intent_top_k]
                if len(selected) < 3:
                    selected = hits[:intent_top_k]
                log.info(f"[RETRIEVER] TOP_K — {fname} → {len(selected)} chunks (top {intent_top_k}, threshold {chunk_threshold})")
                final = selected

            diverse.extend(final)

        log.info(f"[RETRIEVER] Total returned: {len(diverse)}")  
        return diverse  
