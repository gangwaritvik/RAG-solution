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
                "",                "   MULTI_GROUP rules (any request spanning multiple distinct subjects):",
                "   - A query that targets two or more distinct subjects is MULTI_GROUP — whether",
                "     it COMPARES them ('compare X and Y') OR simply asks for EACH of them",
                "     ('give X and Y', 'list X and Y', 'show me X and Y'). multi_group WINS over",
                "     plain dependent even if one subject is a pronoun pointing to the active topic.",
                "   - IMPORTANT: multi_group is about DEPENDENCY (how many subjects), NOT about",
                "     intent. Do NOT assume comparison just because there are two subjects. Choose",
                "     retrieval_intent SEPARATELY from the ACTION the user wants: give/list/extract",
                "     → an extraction intent; compare/contrast/vs/difference → comparison; summarize",
                "     → a summary intent.",
                "     Example: 'compare it with conductivity' → multi_group + comparison;",
                "       sub_queries = ['<ACTIVE_TOPIC>', 'conductivity'].",
                "     Example: 'give X and Y' → multi_group + extraction (NOT comparison).",
                "   - List EVERY subject as its OWN entry in sub_queries. For N subjects",
                "     ('give A, B and C') include ALL N items — never collapse or drop any",
                "     (sub_queries = ['A', 'B', 'C']).",
                "   - For multi_group you MUST return at least 2 sub_queries. Never return an",
                "     empty sub_queries list when dependency_type is 'multi_group'.",
                "",
                "   COMPOUND REQUESTS — SEGMENTS (split into independent answers):",
                "   Split a query into 'segments' when it should yield MORE THAN ONE answer that",
                "   each stands on its own. Each segment is a self-contained sub-request with its",
                "   OWN intent and (when it clearly targets specific loaded file(s)) its OWN",
                "   source_files. Two situations call for segments:",
                "   (a) DIFFERENT OPERATIONS in one query — e.g. it COMPARES some items AND",
                "       separately SUMMARIZES or EXTRACTS something else: one segment per operation.",
                "       Example: 'differentiate A and B, and extract the C table' → TWO segments:",
                "         seg1 {query:'differentiate between A and B', intent:'comparison'},",
                "         seg2 {query:'extract the C table', intent:'targeted_extraction'}.",
                "   (b) The SAME per-item operation applied to MULTIPLE DISTINCT SUBJECTS that each",
                "       deserve their own standalone result — most clearly when the subjects are",
                "       DIFFERENT DOCUMENTS, so merging them would force unrelated material into one",
                "       answer: one segment PER subject/file.",
                "       Example: 'summarize both files' / 'summarize doc1 and doc2' → ONE segment",
                "         per file, each {query:'summarize <that file>', intent: a summary intent,",
                "         source_files:['<that file>']}.",
                "   EXCEPTION — when the operation's RESULT is inherently ABOUT THE RELATIONSHIP",
                "   between the subjects (the subjects must be weighed together to answer at all),",
                "   it is ONE combined answer, NOT one segment per subject. So asking how subjects",
                "   relate to or differ from each other does NOT split (it is a single answer about",
                "   them together), whereas asking for the SAME standalone result for each subject",
                "   DOES split (one result per subject).",
                "   A query that yields just ONE standalone answer has NO segments — return [].",
                "",                "2. RETRIEVAL_INTENT: One of [factual, targeted_summary, global_summary, comparison, targeted_extraction, global_extraction, analysis, ambiguous]",
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
                "1. RETRIEVAL_INTENT for the resulting query: One of [factual, targeted_summary, global_summary, comparison, targeted_extraction, global_extraction, analysis, ambiguous]",
            ])
        
        context_parts.extend([
            "   - factual: Direct question asking for specific information",
            "     Examples: 'What is X?', 'What are the SI units of X?', 'Define X', 'What does X mean?'",
            "     → Retrieve top_k chunks, give DIRECT CONCISE ANSWER",
            "   - targeted_summary: Focused overview/synthesis of a specific topic/section",
            "     Examples: 'Summarize access control section', 'Overview of X controls'",
            "     → Retrieve relevant subset, provide focused synthesis",
            "   - global_summary: Whole-document/corpus overview",
            "     Examples: 'Summarize the whole document', 'Give complete overview of this PDF'",
            "     → Retrieve document-wide context, provide comprehensive synthesis",
            "   - comparison: EXPLICITLY compare/contrast two or more items. Requires",
            "     comparison language (compare, contrast, versus/vs, difference between,",
            "     which is better/worse). A plain conjunction like 'give X and Y' or",
            "     'list X and Y' is NOT comparison — classify it by its action verb instead.",
            "     Examples: 'Compare X and Y', 'Difference between X and Y', 'X vs Y'",
            "     → Retrieve relevant chunks, structured COMPARISON",
            "   - targeted_extraction: Extract a specific table/list/subset",
            "     Examples: 'Give me the information security controls table', 'Extract X matrix'",
            "     → Retrieve relevant subset, return exact structured items",
            "   - global_extraction: Explicitly asking complete document-wide enumeration",
            "     Examples: 'List all X', 'Enumerate every X in the document', 'Give me complete list of X'",
            "     → Retrieve ALL chunks, COMPREHENSIVE LIST",
            "   - analysis: The user wants the REASONING or WORKING behind something — the",
            "     steps, causes, or justification that lead to a result — rather than the",
            "     finished result itself. (Asking to show how a result is reached or why",
            "     something is so is analysis; asking only for the end artifact — a value,",
            "     formula, or table — is factual/extraction.)",
            "     Examples: 'Why does X happen?', 'How does X work?', 'Derive the formula for X'",
            "     → Retrieve relevant chunks, step-by-step analytical breakdown",
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
            "RULE 1: FACTUAL vs TARGETED/GLOBAL EXTRACTION vs TARGETED/GLOBAL SUMMARY",
            "  FACTUAL = Direct specific questions asking for single/focused information:",
            "    Examples: 'What is X?' | 'Define X' | 'What are the SI units of X?' | 'When was X discovered?'",
            "    → Use your judgment: Does the question seek ONE specific thing or ONE focused answer?",
            "",
            "  TARGETED_EXTRACTION = Specific table/list/subset extraction:",
            "    Examples: 'information security controls table' | 'extract access control matrix'",
            "    → Focused structured extraction from relevant subset",
            "",
            "  GLOBAL_EXTRACTION = Comprehensive lists, enumerations, asking for all items:",
            "    Examples: 'List all X' | 'List the X' | 'Enumerate X' | 'All the X' | 'All properties of X'",
            "    Examples: 'What are all the physical quantities?' | 'Show everything about X'",
            "    → Use your judgment: Is the user asking for a COMPLETE corpus-wide list?",
            "",
            "  TARGETED_SUMMARY = Overview/synthesis of a specific topic/section:",
            "    Examples: 'Summarize section X' | 'Overview of access controls'",
            "    → Focused synthesis for the requested scope",
            "",
            "  GLOBAL_SUMMARY = Overview/synthesis of the whole document/corpus:",
            "    Examples: 'Summarize the whole document' | 'Complete overview of this PDF'",
            "    → Comprehensive synthesis across the entire corpus",
            "",
            "RULE 2: ANALYSIS vs COMPARISON vs PLAIN CONJUNCTION",
            "  ANALYSIS = the user wants the REASONING or WORKING behind a result — the steps,",
            "    causes, or justification — rather than the finished result itself. Showing HOW",
            "    a result is reached or WHY something holds is analysis; asking only for the end",
            "    artifact (a value, formula, or table) is factual/extraction.",
            "    Examples: 'Why does X happen?' | 'How does X work?' | 'Derive the formula for X'",
            "  COMPARISON = EXPLICITLY contrasting/comparing items. Needs comparison language",
            "    (compare, contrast, vs/versus, difference between, which is better):",
            "    Examples: 'Compare X and Y' | 'Difference between X and Y' | 'X vs Y'",
            "  PLAIN CONJUNCTION (NOT comparison): 'give X and Y' | 'list X and Y' | 'show X and Y'",
            "    simply asks for BOTH subjects. Keep dependency_type multi_group, but set",
            "    retrieval_intent from the ACTION (give/list/extract → an extraction intent;",
            "    summarize → a summary intent). Two subjects alone NEVER implies comparison.",
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
                "TYPO TOLERANCE (do NOT treat as ambiguous):",
                "  - A clear MISSPELLING or typo of an otherwise-recognizable word is NOT ambiguity.",
                "    Infer the intended word and classify by its MEANING; fix the spelling in",
                "    standalone_query. Never ask the user to clarify an obvious typo.",
                "    Examples: 'bublbs' → 'bulbs'; 'elecitrcity' → 'electricity'; 'controlls' → 'controls';",
                "      'combination of bublbs' → 'combination of bulbs' (clear subject, NOT ambiguous).",
                "  - A query with two or more clearly-named subjects (e.g. 'A and B') is NOT ambiguous",
                "    just because ONE word is misspelled — classify it as multi_group and correct the",
                "    spelling. Only mark ambiguous when the INTENDED meaning is genuinely unrecoverable.",
                "",
                "BARE TOPIC / KEYWORD QUERIES (do NOT treat as ambiguous):",
                "  - A query that is just a topic or noun phrase NAMING A CONCRETE SUBJECT, with",
                "    no undefined reference, is a RETRIEVAL REQUEST — the user wants that topic",
                "    from the loaded documents. A missing action verb (no 'list'/'what is'/'explain')",
                "    does NOT make it ambiguous; treat it like a search box for that subject.",
                "  - Classify it by what the subject denotes (use judgment, keep the subject in",
                "    standalone_query):",
                "      • A subject naming a CATEGORY or SET of items (a plural/collective noun",
                "        phrase) → an extraction intent: global_extraction when it reads as the",
                "        document's complete set of that category, else targeted_extraction.",
                "      • A single specific concept/term → factual (a definition/direct answer) or",
                "        targeted_summary if it is a broad section/topic.",
                "  - Ambiguity is about UNDEFINED references, NOT a missing verb. Mark such a",
                "    bare phrase ambiguous ONLY if the phrase itself is undefined — it contains a",
                "    pronoun ('it', 'that', 'those') or a dangling 'the value/the unit/the type'",
                "    of an UNSTATED thing. A phrase that fully names its own subject is resolvable.",
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
            "   - Correct obvious typos/misspellings here (e.g. 'bublbs' → 'bulbs') so retrieval",
            "     matches the intended terms",
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
            "6. SOURCE_FILES: a JSON array pinning retrieval to specific loaded file(s).",
            "   - Set it ONLY when the query unmistakably targets particular loaded file(s) —",
            "     by name (e.g. 'summarize electricity.pdf') or by a description that clearly",
            "     identifies one (e.g. 'the security standard' → an ISO file). Use the EXACT",
            "     filename(s) from the LOADED DOCUMENTS list. Include every file it targets.",
            "   - Return [] when the query does NOT single out a file, refers to the corpus as a",
            "     whole ('the documents', 'everything'), or you are at all unsure — [] searches",
            "     all files. For a multi_group/compound query, leave per-subject/segment pinning",
            "     to those fields and keep this [].",
            "",
            "RESPONSE FORMAT:",
            "Provide JSON response with exactly these fields:",
        ])
        
        if is_clarification:
            context_parts.extend([
                "{",
                '  "dependency_type": "independent" OR "dependent" OR "multi_group" OR "ambiguous",',
                '  "retrieval_intent": "factual" OR "targeted_summary" OR "global_summary" OR "comparison" OR "targeted_extraction" OR "global_extraction" OR "analysis" OR "ambiguous",',
                '  "belongs_to_active_group": true or false,',
                '  "answer_source": "document" OR "previous_answer",',
                '  "reasoning": "State which OUTCOME (1 CLARIFIES / 2 NEW QUESTION / 3 STILL AMBIGUOUS) and why (1-2 sentences)",',
                '  "standalone_query": "Query ready for retrieval (combined, or new-message-only, or original if still ambiguous)",',
                '  "sub_queries": [] or ["sub_query1", "sub_query2", ...],',
                '  "source_files": [] or ["<exact filename>", ...],',
                '  "suggested_topic": "Topic name if not belongs_to_active_group"',
                "}",
            ])
        else:
            context_parts.extend([
                "{",
                '  "dependency_type": "independent" OR "dependent" OR "multi_group" OR "ambiguous",',
                '  "retrieval_intent": "factual" OR "targeted_summary" OR "global_summary" OR "comparison" OR "targeted_extraction" OR "global_extraction" OR "analysis" OR "ambiguous",',
                '  "belongs_to_active_group": true or false,',
                '  "answer_source": "document" OR "previous_answer",',
                '  "reasoning": "Brief explanation (1-2 sentences)",',
                '  "standalone_query": "Self-contained query for retrieval",',
                '  "sub_queries": [] or ["sub_query1", "sub_query2", ...],',
                '  "segments": [] or [{"title": "short label", "query": "self-contained sub-request", "intent": "<one intent>", "source_files": ["<exact filename>", ...]}],',
                '  "source_files": [] or ["<exact filename>", ...],',
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
        sub_queries_from_llm: List[str],
        available_documents: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Process multi-group query by splitting into sub-queries IN PARALLEL.
        
        Generates topic + intent (+ optional source_file) for all sub-queries
        simultaneously using a thread pool.
        """
        if not sub_queries_from_llm:
            log.warning("[LLM_CLASSIFIER] ⚠️ Multi-group query but no sub-queries provided")
            return [{"query": query, "topic": "General"}]
        
        log.info(f"[LLM_CLASSIFIER] Splitting multi-group into {len(sub_queries_from_llm)} sub-queries (PARALLEL)")
        
        # Classify topic + retrieval intent per sub-query in parallel. Each sub-query
        # of a multi-group request is retrieved INDEPENDENTLY, so it gets its OWN
        # intent and therefore its own per-intent K (see retriever INTENT_RETRIEVAL_CONFIG).
        max_workers = min(len(sub_queries_from_llm), 5)  # Cap at 5 parallel threads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            log.info(f"[LLM_CLASSIFIER] Submitting {len(sub_queries_from_llm)} topic+intent tasks with {max_workers} workers")
            
            future_to_index = {}
            for i, sub_q in enumerate(sub_queries_from_llm):
                future = executor.submit(
                    self._get_topic_and_intent_for_subquery,
                    sub_q, i + 1, len(sub_queries_from_llm), available_documents
                )
                future_to_index[future] = i
            
            # Collect results as they complete
            meta_by_index = {}
            completed = 0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                
                try:
                    meta = future.result()
                    meta_by_index[index] = meta
                    log.info(f"[LLM_CLASSIFIER] Sub-query {completed}/{len(sub_queries_from_llm)} → topic='{meta['topic'][:30]}' intent='{meta['intent']}'")
                except Exception as e:
                    log.warning(f"[LLM_CLASSIFIER] ⚠️ Failed topic+intent for sub-query {index + 1}: {e}")
                    meta_by_index[index] = {"topic": "General", "intent": "factual"}
            
            # Build result list in original order
            sub_query_list = []
            for i, sub_q in enumerate(sub_queries_from_llm):
                meta = meta_by_index.get(i, {"topic": "General", "intent": "factual"})
                entry = {
                    "query": sub_q,
                    "topic": meta["topic"],
                    "intent": meta["intent"],
                }
                if meta.get("source_files"):
                    entry["filenames"] = meta["source_files"]
                sub_query_list.append(entry)
        
        log.info(f"[LLM_CLASSIFIER] ✅ Generated topic+intent for all {len(sub_queries_from_llm)} sub-queries in parallel")
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
