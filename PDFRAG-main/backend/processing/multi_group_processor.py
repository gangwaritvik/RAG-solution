"""Multi-subject retrieval for comparison / multi-topic queries.

When the classifier flags a query as ``multi_group`` (e.g. "compare X and Y" or
"give me A from doc1 and B from doc2"), it splits it into per-subject sub-queries.
A single combined embedding tends to favour one subject, so this module retrieves
chunks for EACH subject independently (in parallel) and merges them — guaranteeing
balanced coverage of every subject. The merged chunks then feed the normal
generation path (and streaming).

Each sub-query carries its OWN retrieval intent (classified independently), so it is
retrieved with that intent's predefined K and threshold from the retriever's
``INTENT_RETRIEVAL_CONFIG``. There is no shared budget and no fixed per-subject cap:
every subject keeps exactly what its intent allows. A manual Top-K, when provided,
overrides every subject's intent K.
"""

from typing import Dict, List, Any, Optional

from backend.processing.parallel_executor import ParallelExecutor
from backend.utils.logger import get_logger

log = get_logger("multi_group_processor")



class MultiGroupProcessor:
    """Retrieves balanced context across the sub-queries of a multi-subject query."""

    def __init__(self, retriever, max_workers: int = 5):
        """
        Args:
            retriever: Retriever instance used for per-subject retrieval.
            max_workers: Max parallel retrieval workers.
        """
        self.retriever = retriever
        self.max_workers = max_workers
        log.info(f"[MULTI_GROUP] ✅ Initialized with max_workers={max_workers}")

    def retrieve_balanced(
        self,
        sub_queries: List[Dict[str, str]],
        retrieval_intent: str,
        per_subject_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve each sub-query INDEPENDENTLY using ITS OWN intent, then merge.

        multi_group is reserved for genuinely SEPARATE subjects that each need their own
        retrieval and answer (e.g. "give me X and Y", or subjects spanning different
        documents). Each sub-query carries its own retrieval intent (assigned by the
        classifier) and is retrieved with that intent's K and threshold from the
        retriever's ``INTENT_RETRIEVAL_CONFIG``. Retrieving the subjects separately
        (instead of one blended embedding) guarantees balanced coverage of every subject.
        A single relational comparison is NOT routed here — it is classified
        independent + comparison and uses the normal single comprehensive retrieval.

        Args:
            sub_queries: List of dicts each with a "query", optional "intent"/"filenames".
            retrieval_intent: Overall intent — fallback only for a sub-query with no intent.
            per_subject_top_k: Optional manual Top-K. When set, it overrides the intent K
                for every subject (each subject is capped at this value).

        Returns:
            Merged, de-duplicated list of chunk hits (best score kept per chunk),
            ordered by descending similarity score. Empty if < 2 valid subjects
            (caller should then fall back to normal single-query retrieval).
        """
        tasks = []
        for sq in sub_queries:
            q = (sq.get("query") or "").strip()
            if not q:
                continue
            # Each subject is retrieved with ITS OWN intent (assigned by the classifier),
            # falling back to the overall intent only when a subject has none. multi_group
            # is reserved for genuinely separate subjects that each need their own
            # retrieval+answer, so honoring per-subject intents is correct. (A single
            # relational comparison is NOT multi_group — it is classified independent +
            # comparison and goes through the normal single comprehensive retrieval.)
            sub_intent = (sq.get("intent") or "").strip() or retrieval_intent
            tasks.append({
                "subject": q,
                "intent": sub_intent,
                "top_k_override": per_subject_top_k,
                "restrict_filenames": sq.get("filenames"),
            })

        if len(tasks) < 2:
            log.info("[MULTI_GROUP] < 2 valid sub-queries — no balanced retrieval")
            return []

        log.info(
            f"[MULTI_GROUP] Independent retrieval for {len(tasks)} subjects "
            f"(per-subject intent K; manual override={per_subject_top_k})"
        )

        results = ParallelExecutor.execute_parallel(
            tasks=tasks,
            task_func=self._retrieve_one,
            max_workers=min(len(tasks), self.max_workers),
            operation_name="subject-retrieve",
        )

        # Merge + de-duplicate by (filename, chunk_index), keeping the highest score.
        # Each subject already returns its intent-capped set, so there is no shared
        # budget — a subject keeps exactly what its intent's K allows.
        merged: Dict[tuple, Dict[str, Any]] = {}
        for res in results:
            if not res or res.get("error"):
                continue
            hits = res.get("hits", [])
            log.info(
                f"[MULTI_GROUP] Subject '{res.get('subject', '')[:40]}' "
                f"(intent={res.get('intent')}) → {len(hits)} chunks"
            )
            for hit in hits:
                key = (hit.get("filename"), hit.get("chunk_index"))
                existing = merged.get(key)
                if existing is None or hit.get("score", 0) > existing.get("score", 0):
                    merged[key] = hit

        final = sorted(merged.values(), key=lambda h: h.get("score", 0), reverse=True)
        log.info(f"[MULTI_GROUP] ✅ Merged to {len(final)} unique chunks across subjects")
        return final

    def _retrieve_one(self, task: dict, index: int, total: int) -> dict:
        """Retrieve chunks for a single subject using ITS OWN intent (worker thread)."""
        subject = task["subject"]
        intent = task["intent"]
        override = task.get("top_k_override")
        restrict_filenames = task.get("restrict_filenames")
        pin = f" [pinned: {restrict_filenames}]" if restrict_filenames else ""
        log.info(f"[MULTI_GROUP] Retrieving subject {index}/{total} (intent={intent}){pin}: {subject[:60]}")
        hits = self.retriever.retrieve(
            subject,
            retrieval_intent=intent,
            top_k_override=override,
            restrict_filenames=restrict_filenames,
        )
        return {"hits": hits, "subject": subject, "intent": intent}
