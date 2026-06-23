"""LLM-based classifier for dependency, intent, and group membership detection."""

import json
from typing import Optional, Dict, Any, List
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, CHAT_MODEL
from backend.utils.logger import get_logger
from backend.prompts import (
    build_classification_context,
    build_subquery_classification_prompt,
)

log = get_logger("llm_classifier")


class LLMClassifier:
    """Uses LLM for intelligent query classification."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
        # Classification needs reliable intent/dependency discrimination. The nano model
        # collapsed analysis/comparison/extraction down to factual, so use the full 4.1.
        self.model = "gpt-4.1"
    
    def classify_query(
        self,
        query: str,
        active_group_summary: Optional[str] = None,
        active_group_topic: Optional[str] = None,
        previous_ambiguous_query: Optional[str] = None,
        available_documents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify query using LLM for dependency, intent, and group membership.
        
        Args:
            query: User's input query
            active_group_summary: Summary of active group (if any)
            active_group_topic: Topic of active group (if any)
            previous_ambiguous_query: If this is a clarification to an ambiguous query, pass the original query
            available_documents: Filenames currently loaded in the vector store (if any)
            
        Returns:
            Dict with classification results
        """
        log.info(f"[LLM_CLASSIFIER] Classifying: {query[:80]}")
        if previous_ambiguous_query:
            log.info(f"[LLM_CLASSIFIER] Clarification to ambiguous query: {previous_ambiguous_query[:80]}")
        
        # Build context for LLM
        context = build_classification_context(
            query,
            active_group_summary,
            active_group_topic,
            previous_ambiguous_query,
            available_documents
        )
        
        # Call LLM
        response = self._call_llm(context)
        
        # Parse response
        result = self._parse_llm_response(response)
        
        log.info(f"[LLM_CLASSIFIER] ✅ Classified as:")
        log.info(f"  - dependency_type: {result['dependency_type']}")
        log.info(f"  - retrieval_intent: {result['retrieval_intent']}")
        log.info(f"  - answer_source: {result.get('answer_source', 'document')}")
        log.info(f"  - standalone_query: {result.get('standalone_query', '')[:100]}")
        log.info(f"  - reasoning: {result.get('reasoning', '')[:100]}")
        return result
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with classification prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise query classifier. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temp for consistent classification
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            log.error(f"[LLM_CLASSIFIER] ❌ LLM call failed: {e}", exc_info=True)
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response)
            
            # Validate required fields
            required_fields = [
                "dependency_type",
                "retrieval_intent",
                "belongs_to_active_group",
                "reasoning",
                "standalone_query"
            ]
            
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set defaults for optional fields
            if "sub_queries" not in result:
                result["sub_queries"] = []
            if "segments" not in result:
                result["segments"] = []
            if "source_files" not in result:
                result["source_files"] = []
            if "suggested_topic" not in result:
                result["suggested_topic"] = None
            # answer_source defaults to 'document' (safe: always retrieve unless the LLM
            # explicitly says the request operates on the previous answer).
            answer_source = str(result.get("answer_source", "document")).lower().strip()
            result["answer_source"] = "previous_answer" if answer_source == "previous_answer" else "document"

            # Defensive normalization of dependency_type. The model occasionally puts an
            # operation word ('comparison', 'segments', an intent name, ...) into this field
            # instead of one of the four valid values. Rather than letting that map to
            # 'ambiguous' downstream (which would wrongly skip retrieval), INFER the right
            # value from the structured outputs the model DID produce:
            #   - 2+ segments  -> a COMPOUND query; dependency is independent (segments drive it)
            #   - 2+ sub_queries -> a MULTI_GROUP query
            #   - otherwise     -> fall back to ambiguous (genuinely unclassifiable)
            valid_dep = {"independent", "dependent", "multi_group", "ambiguous"}
            dep = str(result.get("dependency_type", "")).lower().strip()
            if dep not in valid_dep:
                segs = result.get("segments") or []
                subs = result.get("sub_queries") or []
                if isinstance(segs, list) and len(segs) >= 2:
                    inferred = "independent"
                elif isinstance(subs, list) and len(subs) >= 2:
                    inferred = "multi_group"
                else:
                    inferred = "ambiguous"
                log.warning(
                    f"[LLM_CLASSIFIER] Invalid dependency_type '{dep}' from model — "
                    f"inferred '{inferred}' from outputs (segs={len(segs)}, subs={len(subs)})"
                )
                result["dependency_type"] = inferred

            log.info(f"[LLM_CLASSIFIER] Parsed response: {result['dependency_type']} | {result['retrieval_intent']}")
            log.info(f"[LLM_CLASSIFIER] Standalone query: {result['standalone_query'][:80]}")
            return result
            
        except json.JSONDecodeError as e:
            log.error(f"[LLM_CLASSIFIER] ❌ Failed to parse JSON: {e}", exc_info=True)
            raise ValueError(f"Invalid JSON response from LLM: {response}")
    
    def recover_sub_queries(self, query: str) -> List[str]:
        """
        Extract the individual subjects from a comparison query when the main
        classification failed to provide sub_queries.

        Returns a list of subject strings (>= 2 on success, else empty list).
        """
        recover_prompt = (
            "Extract each subject being compared or related in this query.\n"
            f'Query: "{query}"\n'
            "Respond with ONLY a JSON array of the subjects, e.g. [\"A\", \"B\", \"C\"].\n"
            "Each subject must be self-contained and retrievable on its own. "
            "If there is only one subject, return an empty array []."
        )
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": recover_prompt}],
                temperature=0.0,
            ).choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            items = json.loads(raw)
            items = [str(s).strip() for s in items if str(s).strip()]
            if len(items) >= 2:
                log.info(f"[LLM_CLASSIFIER] Recovered {len(items)} sub-queries: {items}")
                return items
            log.warning(f"[LLM_CLASSIFIER] Recovery returned < 2 subjects: {items}")
            return []
        except Exception as e:
            log.warning(f"[LLM_CLASSIFIER] ⚠️ Sub-query recovery failed: {e}")
            return []
    
    def split_multi_group_query(
        self,
        query: str,
        sub_queries_from_llm: List[Any],
        available_documents: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Normalize the multi-group sub-queries from the SINGLE classification call.

        The main ``classify_query`` call already returns each sub-query WITH its own
        retrieval intent (and optional source_files), so no extra per-sub-query LLM
        calls are made here — each rich entry is validated and used directly. Only
        bare-string sub-queries (e.g. from the recovery path) are classified on demand
        as a fallback, so the normal multi_group path costs ZERO extra LLM calls.
        """
        if not sub_queries_from_llm:
            log.warning("[LLM_CLASSIFIER] ⚠️ Multi-group query but no sub-queries provided")
            return [{"query": query, "topic": "General"}]

        valid_intents = {
            "factual", "targeted_summary", "global_summary", "comparison",
            "targeted_extraction", "global_extraction", "analysis",
        }
        valid_files = {str(d).strip() for d in (available_documents or [])}

        def _clean_files(raw: Any) -> List[str]:
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                return []
            files = [f for f in (str(x).strip() for x in raw) if f in valid_files]
            return list(dict.fromkeys(files))  # de-dup, preserve order

        results: List[Optional[Dict[str, str]]] = [None] * len(sub_queries_from_llm)
        needs_llm: List[int] = []  # indices of bare/intentless sub-queries needing a fallback call

        for i, sq in enumerate(sub_queries_from_llm):
            # Rich object from the single main call: {query, intent, source_files?}.
            if isinstance(sq, dict):
                q = (sq.get("query") or "").strip()
                if not q:
                    continue
                intent = (sq.get("intent") or "").strip().lower()
                if intent in valid_intents:
                    entry: Dict[str, Any] = {"query": q, "topic": q[:40], "intent": intent}
                    files = _clean_files(sq.get("source_files"))
                    if files:
                        entry["filenames"] = files
                    results[i] = entry
                else:
                    needs_llm.append(i)  # object without a usable intent
            else:
                # Bare string (e.g. recovery path) — no intent provided.
                if str(sq).strip():
                    needs_llm.append(i)

        # Fallback ONLY for sub-queries that arrived WITHOUT an intent. In the normal
        # multi_group path this list is empty, so no extra LLM calls happen at all.
        if needs_llm:
            log.info(f"[LLM_CLASSIFIER] {len(needs_llm)} sub-query(ies) missing intent — classifying ONLY those")

            def _text(idx: int) -> str:
                s = sub_queries_from_llm[idx]
                return (s.get("query") if isinstance(s, dict) else str(s)).strip()

            with ThreadPoolExecutor(max_workers=min(len(needs_llm), 5)) as executor:
                future_to_index = {
                    executor.submit(
                        self._get_topic_and_intent_for_subquery,
                        _text(idx), n + 1, len(needs_llm), available_documents
                    ): idx
                    for n, idx in enumerate(needs_llm)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    qtext = _text(idx)
                    try:
                        meta = future.result()
                    except Exception as e:
                        log.warning(f"[LLM_CLASSIFIER] ⚠️ Failed topic+intent for sub-query {idx + 1}: {e}")
                        meta = {"topic": "General", "intent": "factual"}
                    entry = {
                        "query": qtext,
                        "topic": meta.get("topic", "General"),
                        "intent": meta.get("intent", "factual"),
                    }
                    if meta.get("source_files"):
                        entry["filenames"] = meta["source_files"]
                    results[idx] = entry

        sub_query_list = [e for e in results if e is not None]
        log.info(
            f"[LLM_CLASSIFIER] ✅ Prepared {len(sub_query_list)} sub-queries "
            f"(intent from single main call; {len(needs_llm)} fallback-classified)"
        )
        return sub_query_list
    
    def _get_topic_and_intent_for_subquery(self, sub_query: str, index: int, total: int,
                                           available_documents: List[str] = None) -> Dict[str, str]:
        """Classify a single sub-query's topic + retrieval intent (parallel worker).

        Each sub-query of a multi-group request is an independent retrieval, so it is
        classified on its own and uses that intent's predefined K in the retriever.
        When the loaded file list is provided, the LLM ALSO picks the sub-query's
        source_files SEMANTICALLY (e.g. 'the security standard' → an ISO file). It may
        return several files when the sub-query refers to several. Picks are validated
        against the real file list and used to pin retrieval to those file(s).
        """
        valid_intents = {
            "factual", "targeted_summary", "global_summary",
            "targeted_extraction", "global_extraction", "analysis",
        }

        prompt = build_subquery_classification_prompt(sub_query, available_documents)
        try:
            log.info(f"[LLM_CLASSIFIER] [PARALLEL] Classifying sub-query {index}/{total}")
            raw = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            ).choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            topic = str(data.get("topic", "General")).strip() or "General"
            intent = str(data.get("intent", "factual")).strip().lower()
            if intent not in valid_intents:
                intent = "factual"

            result = {"topic": topic, "intent": intent}
            if available_documents:
                # Validate the LLM's choices against the REAL file list (exact match);
                # unknown/hallucinated names are dropped. Accept a list or a lone string.
                valid_files = {str(d).strip() for d in available_documents}
                sf_raw = data.get("source_files", [])
                if isinstance(sf_raw, str):
                    sf_raw = [sf_raw]
                source_files = [
                    s for s in (str(x).strip() for x in sf_raw) if s in valid_files
                ]
                source_files = list(dict.fromkeys(source_files))  # de-dup, keep order
                result["source_files"] = source_files
                log.info(f"[LLM_CLASSIFIER] [PARALLEL] ✅ Sub-query {index}: topic='{topic[:30]}' intent='{intent}' source_files={source_files}")
            else:
                log.info(f"[LLM_CLASSIFIER] [PARALLEL] ✅ Sub-query {index}: topic='{topic[:30]}' intent='{intent}'")
            return result
        except Exception as e:
            log.warning(f"[LLM_CLASSIFIER] [PARALLEL] ⚠️ Failed sub-query {index}: {e}")
            return {"topic": "General", "intent": "factual"}
