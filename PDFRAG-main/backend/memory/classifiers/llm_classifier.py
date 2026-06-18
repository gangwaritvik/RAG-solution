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
        # Use the same model configured for generation
        self.model = CHAT_MODEL or "gpt-35-turbo"
    
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
            "You are an intelligent query analyzer for a RAG (Retrieval Augmented Generation) system.",
            "Prioritize asking for clarification over using general knowledge.",
            "",
        ]

        # Tell the classifier what is actually retrievable. A request to summarize/list/
        # analyze "the document(s)" or "the content" refers to THESE loaded files and is
        # therefore resolvable — it must NOT be treated as ambiguous just because it does
        # not name a specific title. Only a missing corpus makes such a request ambiguous.
        if available_documents:
            shown = available_documents[:20]
            context_parts.extend([
                "LOADED DOCUMENTS (currently available for retrieval):",
                *[f"  - {name}" for name in shown],
                (f"  ...and {len(available_documents) - len(shown)} more"
                 if len(available_documents) > len(shown) else ""),
                "A request that refers to 'the document(s)', 'the content', 'the file',",
                "'the key topics', or the corpus as a whole resolves to these loaded",
                "documents. Such a request is NOT ambiguous — classify it by its intent",
                "(summary / extraction / analysis / etc.). Do NOT ask which document when",
                "these are the only ones loaded.",
                "",
            ])
        else:
            context_parts.extend([
                "LOADED DOCUMENTS: none currently in the store.",
                "A request to summarize/list/analyze 'the document' has no corpus to run",
                "against → treat as ambiguous and ask the user to provide a document.",
                "",
            ])
        
        # Check if this is a clarification to an ambiguous query
        is_clarification = previous_ambiguous_query is not None
        
        if not is_clarification:
            # Normal query classification
            context_parts.extend([
                "TASK: Classify the user query into:",
                "1. DEPENDENCY_TYPE: One of [independent, dependent, multi_group, ambiguous]",
                "   - independent: Specific, well-defined standalone question with clear context",
                "   - dependent: Follow-up that depends on active group context",
                "   - multi_group: Compares/relates multiple topics (needs sub-queries)",
                "   - ambiguous: Vague/undefined references OR lacks sufficient context",
                "",
                "   CLASSIFICATION PRIORITY (check in this exact order):",
                "   STEP 1: Can the query be resolved as DEPENDENT using the active group?",
                "           (an unresolved reference that points back to the active topic) → dependent",
                "   STEP 2: Else, is it INDEPENDENT? (specific, self-contained, clear on its own) → independent",
                "   STEP 3: Else, is it MULTI_GROUP? (compares/relates multiple distinct topics) → multi_group",
                "   STEP 4: ONLY IF none of the above apply → ambiguous",
                "   AMBIGUOUS IS A LAST RESORT. Never choose ambiguous if the query fits",
                "   dependent, independent, or multi_group.",
                "",
                "   AMBIGUOUS arises in exactly TWO situations:",
                "   CASE A: The query is vague/undefined AND there is NO active group to",
                "           resolve it against. (User starts fresh with an unclear query.)",
                "   CASE B: An active group EXISTS, but the query does NOT depend on it",
                "           (its reference cannot be resolved from the active topic) AND it",
                "           is also not independent or multi_group on its own.",
                "",
                "   DEPENDENT vs INDEPENDENT \u2014 how to decide the boundary:",
                "   - DEPENDENT means the query CANNOT be understood/retrieved without the",
                "     active group's topic. This includes IMPLICIT continuations that have NO",
                "     explicit pronoun but only make sense as a follow-up, e.g. (active topic",
                "     = <ACTIVE_TOPIC>): 'now the magnetic version', 'what about in 3D?',",
                "     'and the reverse case', 'same thing but for AC'. Treat these as",
                "     DEPENDENT and rewrite standalone_query to splice in <ACTIVE_TOPIC>.",
                "   - INDEPENDENT means the query is fully self-contained and names its own",
                "     subject, so it stands alone even though it may be in the SAME broader",
                "     subject area as the active topic. Being merely RELATED to the active",
                "     topic's domain is NOT enough to be dependent.",
                "     Example (active topic = <ACTIVE_TOPIC>): 'how does temperature affect",
                "     resistance?' names its own subject (resistance) and is answerable on",
                "     its own \u2192 INDEPENDENT (start a new group), even if it shares the same",
                "     overall document/domain as <ACTIVE_TOPIC>.",
                "   - RULE OF THUMB: If you must borrow words from <ACTIVE_TOPIC> to make the",
                "     query retrievable \u2192 dependent. If the query already specifies its own",
                "     subject \u2192 independent.",
                "",                "   MULTI_GROUP rules (comparisons / relating multiple topics):",
                "   - A query that COMPARES or RELATES two or more subjects is MULTI_GROUP,",
                "     even if one of those subjects is referenced by a pronoun that points to",
                "     the active topic. Comparison WINS over plain dependent.",
                "     Example (active topic = <ACTIVE_TOPIC>): 'compare it with conductivity'",
                "       → multi_group. Resolve the pronoun INSIDE the sub_queries, e.g.",
                "       sub_queries = ['<ACTIVE_TOPIC>', 'conductivity'].",
                "   - List EVERY subject being compared as its OWN entry in sub_queries.",
                "     For N-way comparisons ('compare A, B and C') include ALL N items —",
                "     never collapse or drop any (sub_queries = ['A', 'B', 'C']).",
                "   - For multi_group you MUST return at least 2 sub_queries. Never return an",
                "     empty sub_queries list when dependency_type is 'multi_group'.",
                "",                "2. RETRIEVAL_INTENT: One of [factual, summary, comparison, extraction, analysis, ambiguous]",
            ])
        else:
            # Clarification to a previously-ambiguous query.
            # The user's new message can be one of THREE things — decide which:
            context_parts.extend([
                "TASK: The previous query was vague, so we asked the user for more detail.",
                "Decide what the user's NEW message is — ONE of three outcomes:",
                "",
                "  OUTCOME 1 — CLARIFIES: The new message adds the missing detail for the",
                "    previous query. → COMBINE them into one standalone_query and classify",
                "    dependency_type as 'independent' (or 'dependent' if it now clearly",
                "    relies on the active group, or 'multi_group' if it now compares",
                "    multiple topics).",
                "",
                "  OUTCOME 2 — NEW QUESTION: The new message is a different, self-contained",
                "    question rather than a detail for the previous one. → Classify the NEW",
                "    message ON ITS OWN (independent / dependent / multi_group / ambiguous)",
                "    and base standalone_query only on the new message, setting aside the",
                "    earlier vague query.",
                "",
                "  OUTCOME 3 — STILL VAGUE: The new message is also too vague, and the two",
                "    together are still not specific enough to retrieve. → classify",
                "    dependency_type as 'ambiguous' again so we can ask for more detail.",
                "",
                "1. RETRIEVAL_INTENT for the resulting query: One of [factual, summary, comparison, extraction, analysis, ambiguous]",
            ])
        
        context_parts.extend([
            "   - factual: Direct question asking for specific information",
            "     Examples: 'What is X?', 'What are the SI units of X?', 'Define X', 'What does X mean?'",
            "     → Retrieve top_k chunks, give DIRECT CONCISE ANSWER",
            "   - summary: Asking for overview/synthesis of topic",
            "     Examples: 'Summarize X', 'Overview of X', 'Explain the concept of X'",
            "     → Retrieve all chunks, provide COMPREHENSIVE OVERVIEW",
            "   - comparison: Compare/contrast two or more items",
            "     Examples: 'Compare X and Y', 'Difference between X and Y'",
            "     → Retrieve relevant chunks, structured COMPARISON",
            "   - extraction: Explicitly asking to list/enumerate/show ALL items",
            "     Examples: 'List all X', 'Enumerate X', 'Give me all X', 'Show every X'",
            "     → Retrieve ALL chunks, COMPREHENSIVE LIST",
            "   - analysis: Asking WHY/HOW/IMPLICATIONS/RISKS",
            "     Examples: 'Why does X happen?', 'How does X work?', 'Implications of X'",
            "     → Retrieve relevant chunks, ANALYTICAL BREAKDOWN",
            "",
        ])
        
        if not is_clarification:
            context_parts.extend([
                "   - ambiguous: Query is vague or needs clarification before retrieval",
                "",
                "3. BELONGS_TO_ACTIVE_GROUP: Boolean",
                "   - true: Query relates to active group topic",
                "   - false: Query is about different topic",
                "",
            ])
        else:
            context_parts.extend([
                "2. BELONGS_TO_ACTIVE_GROUP: Boolean",
                "   - true: Combined query relates to active group topic",
                "   - false: Combined query is about different topic",
                "",
            ])
        
        # Add active group info if available
        if active_group_topic or active_group_summary:
            context_parts.append("ACTIVE GROUP CONTEXT:")
            if active_group_topic:
                context_parts.append(f"Topic: {active_group_topic}")
            if active_group_summary:
                context_parts.append(f"Summary: {active_group_summary[:200]}")
        else:
            context_parts.append("NO ACTIVE GROUP CONTEXT - User is starting fresh")
            context_parts.append("")
        
        context_parts.extend([
            "INTENT CLASSIFICATION RULES:",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "RULE 1: FACTUAL vs EXTRACTION vs SUMMARY",
            "  FACTUAL = Direct specific questions asking for single/focused information:",
            "    Examples: 'What is X?' | 'Define X' | 'What are the SI units of X?' | 'When was X discovered?'",
            "    → Use your judgment: Does the question seek ONE specific thing or ONE focused answer?",
            "",
            "  EXTRACTION = Comprehensive lists, enumerations, asking for multiple items:",
            "    Examples: 'List all X' | 'List the X' | 'Enumerate X' | 'All the X' | 'All properties of X'",
            "    Examples: 'What are all the physical quantities?' | 'Show everything about X'",
            "    → Use your judgment: Is the user asking for a COLLECTION or COMPLETE LIST of items?",
            "",
            "  SUMMARY = Overview, synthesis, explanation of concept/topic:",
            "    Examples: 'Summarize X' | 'Overview of X' | 'Explain the concept of X' | 'Describe how X works'",
            "    → Use your judgment: Does the user want a comprehensive OVERVIEW or SYNTHESIS?",
            "",
            "RULE 2: ANALYSIS vs COMPARISON",
            "  ANALYSIS = Cause/effect/implications (WHY/HOW questions):",
            "    Examples: 'Why does X happen?' | 'How does X work?' | 'What are the implications of X?'",
            "  COMPARISON = Contrasting/comparing multiple items:",
            "    Examples: 'Compare X and Y' | 'Difference between X and Y' | 'X vs Y'",
            "",
            "RULE 3: Identify AMBIGUOUS queries",
            "  Examples of ambiguity: 'both', 'that', 'it', 'those', 'they', 'these' without clear reference",
            "  Examples: 'the unit', 'the value', 'the formula' (of WHAT?)",
            "  Examples: 'What is SI unit?' without context",
            "  → Use your judgment: Does the query have undefined or unclear references?",
            "",
            "  IMPORTANT EXCEPTION — RESOLVE WITH ACTIVE GROUP FIRST:",
            "  If an ACTIVE GROUP CONTEXT exists and the query contains an unresolved",
            "  reference that clearly points back to the active group's topic, then it",
            "  is NOT ambiguous.",
            "  → Classify as 'dependent', set belongs_to_active_group=true, and",
            "    RESOLVE the reference in standalone_query using the active topic.",
            "  Example (active topic = <ACTIVE_TOPIC>):",
            "    'formula of that' → dependent → standalone: 'formula of <ACTIVE_TOPIC>'",
            "    'what is its unit' → dependent → standalone: 'unit of <ACTIVE_TOPIC>'",
            "  Only mark AMBIGUOUS when there is NO active group OR the reference",
            "  genuinely cannot be resolved from the active topic.",
            "",
        ])
        
        # If this is a clarification to an ambiguous query, include both in context
        if previous_ambiguous_query:
            context_parts.extend([
                "FOLLOW-UP CONTEXT:",
                f'Earlier vague query: "{previous_ambiguous_query}"',
                f'User follow-up message: "{query}"',
                "",
                "TASK FOR THE FOLLOW-UP — first decide the OUTCOME (see above), then:",
                "  If OUTCOME 1 (CLARIFIES): Combine the earlier query with the user's",
                "    follow-up into a single standalone_query that makes sense for",
                "    document retrieval.",
                "      Example:",
                '        Earlier: "list all quantities"',
                '        Follow-up: "from the uploaded PDF"',
                '        Standalone: "list all physical quantities mentioned in the uploaded PDF document"',
                "  If OUTCOME 2 (NEW QUESTION): Use ONLY the new message as standalone_query",
                "    and classify it on its own, setting aside the earlier vague query.",
                '      Example: Earlier "what is the unit?" + New "summarize chapter 2"',
                '        → standalone: "summarize chapter 2" (independent, summary)',
                "  If OUTCOME 3 (STILL VAGUE): Keep dependency_type 'ambiguous';",
                "    do not invent specifics that the user has not provided.",
                '      Example: Earlier "what is the value?" + Follow-up "the other one"',
                "        → still ambiguous (which value? still unclear)",
                "",
            ])
        else:
            context_parts.extend([
                "USER QUERY:",
                f'"{query}"',
                "",
                "AMBIGUITY DETECTION (FOR RAG SYSTEMS):",
                "FIRST, try to resolve using ACTIVE GROUP CONTEXT (see below).",
                "If an active group exists and the query contains an unresolved reference",
                "that points back to its topic, classify as 'dependent' (NOT ambiguous)",
                "and resolve the reference in standalone_query.",
                "",
                "Mark as AMBIGUOUS only if ANY of these apply AND it cannot be resolved from the active group:",
                "  1. Undefined pronouns without context: both, that, it, those, they, these, etc.",
                "  2. Vague/general references: 'the unit', 'the value', 'the formula', 'the type' (type of WHAT?)",
                "  3. General knowledge questions WITH NO PRIOR CONTEXT: 'What is SI unit?' needs clarification on context",
                "  4. Incomplete comparisons: 'Compare these' without specifying what",
                "  5. Questions that could be answered from general LLM knowledge but need document context",
                "",
                "CRITICAL RULE:",
                "AMBIGUOUS happens in exactly two ways:",
                "  CASE A: NO ACTIVE GROUP exists and the query is general/vague → AMBIGUOUS.",
                "  CASE B: An ACTIVE GROUP exists but the query does NOT depend on it,",
                "          and it is also not independent or multi_group → AMBIGUOUS.",
                "If an ACTIVE GROUP EXISTS and the reference resolves to its topic, mark as DEPENDENT.",
                "This ensures user clarifies their intent before retrieving from documents.",
                "Example: 'What is the SI unit?' WITHOUT prior context → AMBIGUOUS (ask what specifically)",
                "Example: 'What is current in SI units?' → Could be INDEPENDENT if specific enough",
                "",
            ])
        
        context_parts.extend([
            "4. STANDALONE_QUERY: Reformulated query for document retrieval",
            "   - Make the query self-contained and clear for the retrieval system",
            "   - If CLARIFICATION: Combine the ambiguous query + clarification naturally",
            "   - Examples of clarifications being converted to standalone queries:",
            "     'list all quantities' + 'from the chapter' → 'all physical quantities in the document'",
            "     'list all X' + 'from uploaded pdf' → 'comprehensive list of X'",
            "     'Any exemptions?' + 'for turnover requirement' → 'What are exemptions for turnover requirement?'",
            "   - Let your judgment determine the scope based on the user's intent",
            "   - If AMBIGUOUS: Keep original query, don't try to expand undefined references",
            "",
            "5. ANSWER_SOURCE: Where the answer must come from — one of [document, previous_answer]",
            "   Decide by judging what the request OPERATES ON, not by matching specific words:",
            "   - previous_answer: The request takes the assistant's PREVIOUS ANSWER as its",
            "     subject and can be fully satisfied from that answer's own content — whether by",
            "     restating it (reformat/condense/expand/translate/restyle) or by reasoning over",
            "     it (analyze, derive aspects, draw out implications). No NEW document facts are",
            "     required because the needed material is already in the previous answer.",
            "   - document: The request needs information NOT already contained in the previous",
            "     answer, so it must be retrieved from the documents. This is the DEFAULT,",
            "     including topic continuations that go beyond what was already said.",
            "   - TEST: Could the request be answered using ONLY the text of the previous answer?",
            "     Yes → 'previous_answer'. No / needs more from the source → 'document'.",
            "     When genuinely unsure, choose 'document'.",
            "",
            "RESPONSE FORMAT:",
            "Provide JSON response with exactly these fields:",
        ])
        
        if is_clarification:
            context_parts.extend([
                "{",
                '  "dependency_type": "independent" OR "dependent" OR "multi_group" OR "ambiguous",',
                '  "retrieval_intent": "factual" OR "summary" OR "extraction" OR "analysis" OR "comparison" OR "ambiguous",',
                '  "belongs_to_active_group": true or false,',
                '  "answer_source": "document" OR "previous_answer",',
                '  "reasoning": "State which OUTCOME (1 CLARIFIES / 2 NEW QUESTION / 3 STILL AMBIGUOUS) and why (1-2 sentences)",',
                '  "standalone_query": "Query ready for retrieval (combined, or new-message-only, or original if still ambiguous)",',
                '  "sub_queries": [] or ["sub_query1", "sub_query2", ...],',
                '  "suggested_topic": "Topic name if not belongs_to_active_group"',
                "}",
            ])
        else:
            context_parts.extend([
                "{",
                '  "dependency_type": "independent" OR "dependent" OR "multi_group" OR "ambiguous",',
                '  "retrieval_intent": "factual" OR "summary" OR "comparison" OR "extraction" OR "analysis" OR "ambiguous",',
                '  "belongs_to_active_group": true or false,',
                '  "answer_source": "document" OR "previous_answer",',
                '  "reasoning": "Brief explanation (1-2 sentences)",',
                '  "standalone_query": "Self-contained query for retrieval",',
                '  "sub_queries": [] or ["sub_query1", "sub_query2", ...],',
                '  "suggested_topic": "Topic name for new group if not belongs_to_active_group"',
                "}",
            ])
        
        context_parts.extend([
            "",
            "Respond with ONLY valid JSON, no markdown or additional text.",
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
            if "suggested_topic" not in result:
                result["suggested_topic"] = None
            # answer_source defaults to 'document' (safe: always retrieve unless the LLM
            # explicitly says the request operates on the previous answer).
            answer_source = str(result.get("answer_source", "document")).lower().strip()
            result["answer_source"] = "previous_answer" if answer_source == "previous_answer" else "document"
            
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
        sub_queries_from_llm: List[str]
    ) -> List[Dict[str, str]]:
        """
        Process multi-group query by splitting into sub-queries IN PARALLEL.
        
        Generates topics for all sub-queries simultaneously using thread pool.
        """
        if not sub_queries_from_llm:
            log.warning("[LLM_CLASSIFIER] ⚠️ Multi-group query but no sub-queries provided")
            return [{"query": query, "topic": "General"}]
        
        log.info(f"[LLM_CLASSIFIER] Splitting multi-group into {len(sub_queries_from_llm)} sub-queries (PARALLEL)")
        
        # Generate topics in parallel
        max_workers = min(len(sub_queries_from_llm), 5)  # Cap at 5 parallel threads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all topic generation tasks
            log.info(f"[LLM_CLASSIFIER] Submitting {len(sub_queries_from_llm)} topic generation tasks with {max_workers} workers")
            
            future_to_index = {}
            for i, sub_q in enumerate(sub_queries_from_llm):
                future = executor.submit(self._get_topic_for_subquery, sub_q, i + 1, len(sub_queries_from_llm))
                future_to_index[future] = i
            
            # Collect results as they complete
            topics_by_index = {}
            completed = 0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                
                try:
                    topic = future.result()
                    topics_by_index[index] = topic
                    log.info(f"[LLM_CLASSIFIER] Generated topic {completed}/{len(sub_queries_from_llm)}: {topic[:50]}")
                except Exception as e:
                    log.warning(f"[LLM_CLASSIFIER] ⚠️ Failed to get topic for sub-query {index + 1}: {e}")
                    topics_by_index[index] = "General"
            
            # Build result list in original order
            sub_query_list = []
            for i, sub_q in enumerate(sub_queries_from_llm):
                sub_query_list.append({
                    "query": sub_q,
                    "topic": topics_by_index.get(i, "General")
                })
        
        log.info(f"[LLM_CLASSIFIER] ✅ Generated topics for all {len(sub_queries_from_llm)} sub-queries in parallel")
        return sub_query_list
    
    def _get_topic_for_subquery(self, sub_query: str, index: int, total: int) -> str:
        """Generate a topic/category for a sub-query in parallel."""
        topic_prompt = f"""
        Given this sub-query, suggest a brief topic/category (1-3 words):
        "{sub_query}"
        
        Respond with ONLY the topic name, nothing else.
        """
        
        try:
            log.info(f"[LLM_CLASSIFIER] [PARALLEL] Generating topic for sub-query {index}/{total}")
            
            topic_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": topic_prompt}],
                temperature=0.1,
            ).choices[0].message.content.strip()
            
            log.info(f"[LLM_CLASSIFIER] [PARALLEL] ✅ Topic generated for sub-query {index}: {topic_response[:50]}")
            return topic_response
            
        except Exception as e:
            log.warning(f"[LLM_CLASSIFIER] [PARALLEL] ⚠️ Failed to get topic for sub-query {index}: {e}")
            return "General"
