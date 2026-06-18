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
    "factual":    {"mode": "top_k",         "top_k": 5,    "threshold": 0.20},
    # "List/enumerate all X" — must scan the whole document, no filtering
    "extraction": {"mode": "exhaustive",    "top_k": None, "threshold": 0.0},
    # Whole-topic synthesis — gather the most relevant chunks (capped to avoid huge contexts)
    "summary":    {"mode": "comprehensive", "top_k": 15,   "threshold": 0.10},
    # Reasoning over the topic — same breadth as summary
    "analysis":   {"mode": "comprehensive", "top_k": 15,   "threshold": 0.10},
    # Comparing items — broad coverage of both sides, but capped
    "comparison": {"mode": "comprehensive", "top_k": 15,   "threshold": 0.12},
    # Ambiguous — minimal retrieval (clarification usually generated instead)
    "ambiguous":  {"mode": "top_k",         "top_k": 3,    "threshold": 0.20},
}

# Fallback for unknown / missing intent
DEFAULT_INTENT_CONFIG = {"mode": "top_k", "top_k": 5, "threshold": CHUNK_RELEVANCE_THRESHOLD}


class Retriever:  
    def __init__(self, embedder: Embedder, vector_store: VectorStore):  
        self.embedder     = embedder  
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K, get_all_relevant: bool = False, retrieval_intent: str = None, top_k_override: int = None, retrieve_all: bool = False):
        log.info(f"[RETRIEVER] Query: '{query[:60]}' | top_k: {top_k} | get_all_relevant: {get_all_relevant} | intent: {retrieval_intent} | top_k_override: {top_k_override} | retrieve_all: {retrieve_all}")

        query_vector = self.embedder.embed_query(query)
        total        = self.vector_store.count

        if total == 0:
            log.warning("[RETRIEVER] Vector store is empty")
            return []

        # ── MAX / retrieve-all: return EVERY chunk in the store, ranked by similarity,
        # with NO threshold and NO document-relevance filtering. This is the explicit
        # "use all chunks" mode triggered by the frontend MAX toggle. ──
        if retrieve_all:
            all_hits = self.vector_store.search(query_vector, top_k=total)
            log.info(f"[RETRIEVER] RETRIEVE-ALL (MAX) → returning all {len(all_hits)} chunks (no threshold, no doc filter)")
            return all_hits

        # ── Resolve per-intent retrieval config ──
        cfg = INTENT_RETRIEVAL_CONFIG.get(retrieval_intent, DEFAULT_INTENT_CONFIG)
        mode            = cfg["mode"]
        intent_top_k    = cfg["top_k"] if cfg["top_k"] is not None else top_k
        chunk_threshold = cfg["threshold"]

        # ── User override: when the frontend explicitly sets Top-K, it WINS over the
        # intent's default cap for every mode. An exhaustive intent ("list all") is
        # downgraded to a capped comprehensive pass so the user's K is respected. ──
        if top_k_override is not None and top_k_override > 0:
            intent_top_k = top_k_override
            if mode == "exhaustive":
                mode = "comprehensive"
            log.info(f"[RETRIEVER] Top-K OVERRIDE active → using user top_k={intent_top_k} (mode now '{mode}')")

        log.info(
            f"[RETRIEVER] Intent '{retrieval_intent}' → mode={mode} | "
            f"top_k={intent_top_k} | chunk_threshold={chunk_threshold}"
        )
        # ── Step 1: Fetch ALL vectors ──  
        all_hits = self.vector_store.search(query_vector, top_k=total)  
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
                or score >= max_score * 0.60               # relative: within 60% of best doc  
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
                # EXHAUSTIVE MODE: enumeration queries ("list all X") must scan the
                # ENTIRE document. Threshold filtering would drop chunks that mention
                # an item only in passing, making the list incomplete. So return ALL
                # chunks from the relevant document, ordered by similarity.
                final = hits
                log.info(f"[RETRIEVER] EXHAUSTIVE — {fname} → {len(final)} chunks (full-document scan, no threshold)")
            elif mode == "comprehensive":
                # COMPREHENSIVE MODE: Get chunks above the intent's threshold, ordered by
                # similarity, then cap to the intent's top_k so we don't send an oversized
                # context to the LLM. hits are already sorted best-first.
                selected = [h for h in hits if h.get("score", 0) >= chunk_threshold]
                # Apply the cap when the intent defines one OR a user override is active
                # (an exhaustive intent downgraded by override has cfg["top_k"] is None).
                if cfg["top_k"] is not None or (top_k_override is not None and top_k_override > 0):
                    selected = selected[:intent_top_k]
                log.info(f"[RETRIEVER] COMPREHENSIVE — {fname} → {len(selected)} chunks (threshold {chunk_threshold}, cap {intent_top_k})")
                final = selected
            else:
                # TOP_K MODE: Get top_k chunks by similarity above threshold
                selected = [h for h in hits if h.get("score", 0) >= chunk_threshold]
                selected = selected[:intent_top_k]
                if len(selected) < 3:
                    selected = hits[:intent_top_k]
                log.info(f"[RETRIEVER] TOP_K — {fname} → {len(selected)} chunks (top {intent_top_k}, threshold {chunk_threshold})")
                final = selected

            diverse.extend(final)

        log.info(f"[RETRIEVER] Total returned: {len(diverse)}")  
        return diverse  
