"""LLM-based classifier for dependency, intent, and group membership detection."""

import json
from typing import Optional, Dict, Any, List
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, CHAT_MODEL
from backend.utils.logger import get_logger

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
        context = self._build_classification_context(
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
    
    def _build_classification_context(
        self,
        query: str,
        active_group_summary: Optional[str],
        active_group_topic: Optional[str],
        previous_ambiguous_query: Optional[str] = None,
        available_documents: Optional[List[str]] = None
    ) -> str:
        """Build prompt context for LLM classification."""
        
        context_parts = [
            "You are a query classifier for a RAG system. Given a user query and conversation "
            "context, output a single JSON object with the fields described below.",
            "Use your own judgment throughout — do NOT pattern-match on surface words; "
            "reason about what the user actually wants.",
            "",
        ]

        # Inject loaded documents so the model knows what is retrievable
        if available_documents:
            shown = available_documents[:20]
            context_parts.extend([
                "LOADED DOCUMENTS (the only sources available for retrieval):",
                *[f"  - {name}" for name in shown],
                (f"  ...and {len(available_documents) - len(shown)} more"
                 if len(available_documents) > len(shown) else ""),
                "",
                "DOCUMENT RESOLUTION RULE: A query that refers to or names one of these files "
                "is resolvable — resolve it to that file, set source_files, and do NOT mark it "
                "ambiguous. 'Refers to' includes: (a) the full or partial filename; (b) an "
                "unambiguous description of WHAT THE FILE IS, inferred from the filename's "
                "meaning — e.g. a file named like a standard/spec resolves from a description of "
                "that standard, a file named for a topic resolves from that topic. Use your "
                "knowledge of what the filename denotes. Only mark ambiguous when NO loaded file "
                "plausibly matches, or when two+ files match equally and you cannot choose.",
                "",
            ])
            if len(available_documents) == 1:
                context_parts.extend([
                    f"Only one document is loaded: {available_documents[0]}. "
                    "Any generic reference to 'the document', 'it', 'this' automatically refers to it.",
                    "",
                ])
        else:
            context_parts.extend([
                "LOADED DOCUMENTS: none. Any document-retrieval request is not yet answerable.",
                "",
            ])

        # Inject active group context if present
        if active_group_topic or active_group_summary:
            context_parts.append("ACTIVE CONVERSATION CONTEXT:")
            if active_group_topic:
                context_parts.append(f"Topic: {active_group_topic}")
            if active_group_summary:
                context_parts.append(
                    "Prior Q&A (use this to resolve any back-reference in the query — "
                    "'the above', 'it', 'that', 'explain more', etc.):\n"
                    + active_group_summary[:2500]
                )
            context_parts.append("")
        else:
            context_parts.extend(["NO PRIOR CONVERSATION — user is starting fresh.", ""])

        is_clarification = previous_ambiguous_query is not None

        if is_clarification:
            context_parts.extend([
                f'PREVIOUS AMBIGUOUS QUERY: "{previous_ambiguous_query}"',
                f'USER FOLLOW-UP: "{query}"',
                "",
                "Decide the outcome: (1) the follow-up CLARIFIES the previous query — combine "
                "them into one retrievable standalone_query; (2) it is a DIFFERENT question — "
                "classify the new message on its own; (3) still too vague — mark ambiguous again.",
                "",
            ])
        else:
            context_parts.extend([
                f'USER QUERY: "{query}"',
                "",
            ])

        context_parts.extend([
            "CLASSIFY into the following JSON fields:",
            "",
            "1. dependency_type — EXACTLY ONE of: independent | dependent | multi_group | ambiguous.",
            "   (This describes how the query relates to the conversation and how many subjects it",
            "   spans. It is NEVER an intent name like 'comparison'/'factual' — those go in",
            "   retrieval_intent, field 2.)",
            '   "dependent"   : cannot be understood without the active conversation context',
            '                   (refers to it via "it/that/the above/this", or is an implicit',
            '                   continuation like "now the magnetic version", "what about in 3D?").',
            '                   Resolve the reference and rewrite standalone_query accordingly.',
            '   "independent" : fully self-contained; names its own subject. NOTE: a query in the',
            '                   SAME broad domain as the active topic but naming its own subject is',
            '                   still independent (e.g. active="drift velocity", query="how does',
            '                   temperature affect resistance" -> independent). A query that weighs',
            '                   several subjects together into ONE combined answer is independent',
            '                   too — e.g. a comparison ("compare X and Y", "difference between X',
            '                   and Y") yields a single relational answer, so it is independent',
            '                   with retrieval_intent=comparison, NOT multi_group.',
            '   "multi_group" : use ONLY when the query asks for SEVERAL SEPARATE answers — two or',
            '                   more distinct subjects that each deserve their OWN independent',
            '                   retrieval and their own answer (e.g. "give me X and Y", "tell me',
            '                   about A, B and C", or subjects that live in different documents).',
            '                   Judge by the RESULT: would a good answer be ONE combined response',
            '                   (-> independent) or SEPARATE per-subject responses (-> multi_group)?',
            '                   When multi_group, put one entry per subject in sub_queries, each',
            '                   with its OWN retrieval_intent. This wins over "dependent" even if',
            '                   one subject is a pronoun pointing at the active topic. (If the',
            '                   query mixes DIFFERENT operations, that is a COMPOUND query — keep',
            '                   dependency_type independent/dependent and use segments.)',
            '   "ambiguous"   : LAST RESORT. Only when the query has an undefined reference AND',
            '                   there is no active context to resolve it against. If an active',
            '                   conversation exists and the reference resolves to it -> dependent,',
            '                   NOT ambiguous. An obvious typo is NOT ambiguity (fix it instead).',
            "",
            "2. retrieval_intent — judge what KIND of answer the user wants (not surface words):",
            '   "factual"   : wants ONE specific fact/definition/value stated directly.',
            '                 e.g. "what is X", "define X", "units of X".',
            '   "analysis"  : wants the REASONING / DERIVATION / PROCESS behind something — how it',
            '                 works, why it holds, or how a result is obtained — not just the end',
            '                 value. e.g. "derive X", "how does X work", "why does X happen".',
            '                 If the user wants you to EXPLAIN/SHOW/DERIVE -> analysis, not factual.',
            '   "comparison": the answer is inherently RELATIONAL — subjects must be weighed',
            '                 against each other. e.g. "difference between X and Y", "X vs Y",',
            '                 "compare X and Y", "which is better". (A plain "give X and Y" with no',
            '                 relational intent is NOT comparison — pick its action verb instead.)',
            '   "targeted_summary"    : overview of ONE specific topic/section WITHIN a document.',
            '                           e.g. "summarize the access-control section".',
            '   "global_summary"      : overview whose scope is a WHOLE document/file/corpus,',
            '                           including "what is <doc> about" / "summarize <doc>" /',
            '                           "tell me about <doc>". If the subject IS a document (not a',
            '                           section inside it) -> global_summary.',
            '   "targeted_extraction" : specific structured items — a named table/list/subset.',
            '                           e.g. "the X controls table", "extract the Y matrix".',
            '   "global_extraction"   : a COMPLETE document-wide enumeration. Triggered when the',
            '                           user asks for everything of a kind — "list all X",',
            '                           "every X", "complete list of X".',
            '   "positional"          : selects item(s) by their PLACE/ORDER in the document',
            '                           rather than by content — e.g. "the last two <items>",',
            '                           "the first three", "the 5th <item>", "the one after X",',
            '                           "the bottom of the list". Position can only be resolved',
            '                           by reading the WHOLE document in order, so use this',
            '                           whenever the selector is ordinal/positional, even though',
            '                           only a few items are ultimately wanted.',
            '   "ambiguous"           : intent cannot be determined without clarification.',
            "   Pick the NARROWEST intent that fits; reserve the GLOBAL and positional intents",
            "   for requests whose scope genuinely requires reading the whole document.",
            "",
            "3. belongs_to_active_group — true if this query continues the active conversation topic.",
            "",
            "4. standalone_query — a self-contained, retrieval-ready version of the query.",
            "   Resolve any back-references using the active context. Correct obvious typos.",
            "   For multi_group/segments, this is the combined query; per-subject wording goes in sub_queries.",
            "",
            "5. sub_queries — REQUIRED (>=2 entries) when dependency_type is multi_group; else [].",
            "   One object per subject, each with its OWN intent (using the definitions above):",
            '   {"query": "...", "intent": "<intent>", "source_files": [...]}',
            "   source_files: exact filename(s) from the loaded list if the subject clearly targets",
            "   a specific file; [] otherwise. Decide each subject's intent HERE (don't defer).",
            "",
            "6. segments — for COMPOUND queries: a single query that asks for TWO OR MORE",
            "   genuinely DIFFERENT OPERATIONS (different intents), each yielding its own answer.",
            "   The test: does the query combine operations that would NOT share one intent?",
            "   e.g. 'compare A and B, AND summarize C' = a comparison operation + a summary",
            "   operation -> TWO segments. 'define X and give me the Y table' = a factual op + an",
            "   extraction op -> TWO segments. Each segment: {title, query, intent, source_files}.",
            "   IMPORTANT: a compound query usually still has dependency_type 'independent' (or",
            "   'dependent'), NOT 'multi_group' — multi_group is for ONE operation over many",
            "   subjects, whereas segments are for MANY operations. When you emit segments (>=2),",
            "   leave sub_queries=[]. A query that is a SINGLE operation (even over many subjects,",
            "   like a pure comparison) has NO segments -> return [].",
            "",
            "7. source_files — exact filename(s) from the loaded list if the WHOLE query clearly",
            "   targets specific file(s); [] otherwise.",
            "",
            "8. answer_source — 'previous_answer' if the request can be fully satisfied from the",
            "   previous assistant answer alone (reformat it, condense, translate, tabulate,",
            "   analyse what was already said). 'document' in all other cases (default).",
            "",
            "9. suggested_topic — short topic label for a new conversation group (when needed).",
            "",
            "10. reasoning — one sentence explaining the key classification decision.",
            "",
        ])

        context_parts.extend([
            "RESPONSE FORMAT — valid JSON only, no markdown:",
            "{",
            '  "dependency_type": "...",',
            '  "retrieval_intent": "...",',
            '  "belongs_to_active_group": true/false,',
            '  "standalone_query": "...",',
            '  "sub_queries": [],',
            '  "segments": [],',
            '  "source_files": [],',
            '  "answer_source": "document",',
            '  "suggested_topic": "...",',
            '  "reasoning": "..."',
            "}",
        ])

        return "\n".join(context_parts)
    
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

        docs_block = ""
        json_fmt = '{"topic": "...", "intent": "..."}'
        if available_documents:
            listed = "\n".join(f"  - {d}" for d in available_documents[:20])
            docs_block = (
                "\n3. source_files: a JSON array of the EXACT filenames (from the loaded "
                "list below) that this sub-query SPECIFICALLY refers to. ONLY include a "
                "file when the sub-query points to it unmistakably — by its name, or by a "
                "description that clearly identifies it (e.g. 'the security standard' → an "
                "ISO file; 'the circuits doc' → an electricity file). Include EVERY file it "
                "specifically refers to (one, several, or all). If the sub-query does NOT "
                "single out any particular file, or you are at all unsure, return [] so it "
                "searches everything. Do NOT guess.\n"
                "LOADED FILES:\n" + listed + "\n"
            )
            json_fmt = '{"topic": "...", "intent": "...", "source_files": ["<exact filename>", ...]}'

        prompt = (
            "For this sub-query from a larger multi-part request, return:\n"
            "1. topic: a brief category (1-3 words)\n"
            "2. intent: ONE of [factual, targeted_summary, global_summary, "
            "targeted_extraction, global_extraction, analysis]"
            + docs_block + "\n\n"
            f'Sub-query: "{sub_query}"\n\n'
            "INTENT GUIDANCE — pick the NARROWEST intent that fits. Reserve the two GLOBAL\n"
            "(whole-document) intents for requests that EXPLICITLY ask for everything:\n"
            "  - global_extraction: ONLY when the sub-query explicitly demands a COMPLETE,\n"
            "    document-wide enumeration using words like 'all', 'every', 'complete list',\n"
            "    'list all X', 'enumerate X'. A bare topic name is NOT global.\n"
            "  - global_summary: ONLY for an explicit whole-document overview\n"
            "    ('summarize the whole document', 'overview of the entire PDF').\n"
            "  - targeted_extraction: a specific table/list/subset or a NAMED topic/section\n"
            "    (e.g. 'combination of bulbs', 'access control table', 'the controls table').\n"
            "    This is the DEFAULT for a specific subject that wants structured items.\n"
            "  - targeted_summary: a focused overview of ONE specific topic/section.\n"
            "  - factual: a single specific fact/definition ('what is X', 'SI unit of X').\n"
            "  - analysis: the sub-query wants the REASONING or WORKING behind a result — the\n"
            "    steps, causes, or justification — not just the finished artifact. Showing how a\n"
            "    result is reached or why it holds is analysis; asking only for the end value/\n"
            "    formula/table is factual/extraction.\n"
            "RULE: If the sub-query is just a topic NAME with no 'all/every/complete' wording,\n"
            "it is TARGETED (or factual), NEVER global. Example: 'combination of bulbs' →\n"
            "targeted_extraction (a specific section), NOT global_extraction.\n"
            "Respond with ONLY JSON: " + json_fmt
        )
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
