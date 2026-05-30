from backend.ingestion.embedder import Embedder  
from backend.ingestion.vector_store import VectorStore  
from backend.config import TOP_K  
from backend.utils.logger import get_logger  
import numpy as np

log = get_logger("retriever")

# ── Thresholds ──  
DOC_RELEVANCE_THRESHOLD = 0.25   # min avg score for a doc to be considered relevant  
CHUNK_RELEVANCE_THRESHOLD = 0.20  # min score for individual chunks


class Retriever:  
    def __init__(self, embedder: Embedder, vector_store: VectorStore):  
        self.embedder     = embedder  
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K):  
        log.info(f"[RETRIEVER] Query: '{query[:60]}' | top_k: {top_k}")

        query_vector = self.embedder.embed_query(query)  
        total        = self.vector_store.count

        if total == 0:  
            log.warning("[RETRIEVER] Vector store is empty")  
            return []

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

        # ── Step 3: Score each document by avg similarity of its TOP 3 chunks ──  
        doc_scores = {}  
        for fname, hits in per_doc.items():  
            top_hits       = hits[:3]   # top 3 chunks per doc  
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

        # ── Step 6: Retrieve top_k chunks from EACH relevant document ──  
        diverse = []  
        for fname, hits in relevant_docs.items():  
            selected = [h for h in hits if h.get("score", 0) >= CHUNK_RELEVANCE_THRESHOLD]  
            selected = selected[:top_k]

            # If filtered chunks < 3, relax threshold for this doc  
            if len(selected) < 3:  
                selected = hits[:top_k]

            diverse.extend(selected)  
            log.info(f"[RETRIEVER] {fname} → {len(selected)} chunks selected")

        log.info(f"[RETRIEVER] Total returned: {len(diverse)}")  
        return diverse  
