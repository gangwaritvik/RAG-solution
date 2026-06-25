"""
Classification prompts — the query-classifier prompt and the per-sub-query
topic/intent prompt used by the LLM classifier.

These are pure builders: they take the same inputs the classifier had inline and
return the prompt string, so no classifier prompt text lives inside the logic.
"""

from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────
#  Static classifier prompt sections
#
#  The field definitions and the response-format skeleton never change per call,
#  so they live here as module-level constants instead of inline in the builder.
#  Each classification field is its OWN variable (a list of lines, ending with a
#  blank separator) so it can be read and edited in isolation; the builder simply
#  EXTENDS the dynamic context (loaded docs, conversation, query) with them.
# ──────────────────────────────────────────────────────────────────────────

_FIELDS_HEADER = [
    "CLASSIFY into the following JSON fields:",
    "",
]

_FIELD_DEPENDENCY_TYPE = [

    "1. dependency_type — EXACTLY ONE of: independent | dependent | multi_group | ambiguous.",
    "   (This describes how the query relates to the conversation and how many subjects it",
    "   spans. It is NEVER an intent name like 'comparison'/'factual'/'positional' — those",
    "   go in retrieval_intent, field 2. A query that names a section/position is still",
    "   'independent' here, with retrieval_intent + position_selector carrying the rest.)",

    '   "dependent"   : cannot be understood without the active conversation context',
    '                   (refers to it via "it/that/the above/this etc.", or is an implicit',
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
]

_FIELD_RETRIEVAL_INTENT = [
    "2. retrieval_intent — judge what KIND of answer the user wants (not surface words):",

    '   "factual"   : wants ONE specific fact/definition/value stated directly.',
    '                 e.g. "what is X", "define X", "units of X".',

    '   "analysis"  : wants the REASONING / DERIVATION / PROCESS behind something — how it',
    '                 works, why it holds, or how a result is obtained — not just the end',
    '                 value. e.g. "derive X", "how does X work", "why does X happen".',
    
    '                 If the user wants you to EXPLAIN/SHOW/DERIVE -> analysis, not factual.',
    '   "comparison": the answer is inherently RELATIONAL — subjects must be wex`ighed',
    '                 against each other. e.g. "difference between X and Y", "X vs Y",',
    '                 "compare X and Y", "which is better". (A plain "give X and Y" with no',
    '                 relational intent is NOT comparison — pick its action verb instead.)',

    '   "targeted_summary"    : overview of ONE specific topic/section WITHIN a document.',
    '                           e.g. "summarize the access-control section". (If the user',
    '                           names a section by its NUMBER/heading label and wants its',
    '                           CONTENT rather than a summary -> positional.)',

    '   "global_summary"      : overview whose scope is a WHOLE document/file/corpus,',
    '                           including "what is <doc> about" / "summarize <doc>" /',
    '                           "tell me about <doc>". If the subject IS a document (not a',
    '                           section inside it) -> global_summary.',

    '   "targeted_extraction" : specific structured items identified BY CONTENT — a named',
    '                           table/list/subset. e.g. "the X controls table", "extract',
    '                           the Y matrix". (A unit identified by a STRUCTURAL LABEL —',
    '                           "section 9.2", "clause 4.3", "chapter 3" — is positional,',
    '                           NOT targeted_extraction.)',

    '   "global_extraction"   : a COMPLETE document-wide enumeration. Triggered when the',
    '                           user asks for everything of a kind — "list all X",',
    '                           "every X", "complete list of X".',

    '   "positional"          : the user wants the located content RETURNED AS-IS (verbatim),',
    '                           selected by PLACE/ORDER/STRUCTURAL LABEL rather than topic —',
    '                           e.g. "give me section 9.2", "show clause 4.3", "the last two',
    '                           questions", "the 5th item", "the one after X". Use this when',
    '                           the OPERATION is simply "return that part". (If the user wants',
    '                           to SUMMARIZE / COMPARE / EXPLAIN a positionally-named part',
    '                           instead of just see it, pick that operation intent —',
    '                           targeted_summary / comparison / analysis — and put the place',
    '                           label in position_selector. The selector is ORTHOGONAL to the',
    '                           operation: position_selector = WHICH part, retrieval_intent =',
    '                           WHAT to do with it.) Also covers asking for the ANSWER/VALUE',
    '                           of a positionally-named item ("answer of the last question").',
    '                           A structural/ordinal selector must be resolved by reading the',
    '                           WHOLE document in order (a bare number like "9.2" has almost',
    '                           no semantic signal), so ALWAYS also set position_selector.',
    '                           CRITICAL: do NOT GUESS the concrete item and bake it into',
    '                           standalone_query — keep the positional phrasing and let',
    '                           full-document retrieval resolve the actual item.',
    
    '   "ambiguous"           : intent cannot be determined without clarification.',
    "   Pick the NARROWEST intent that fits; reserve the GLOBAL and positional intents",
    "   for requests whose scope genuinely requires reading the whole document.",
    "",
]

_FIELD_BELONGS_TO_ACTIVE_GROUP = [
    "3. belongs_to_active_group — true if this query continues the active conversation topic.",
    "",
]

_FIELD_STANDALONE_QUERY = [
    "4. standalone_query — a self-contained, retrieval-ready version of the query.",
    "   Resolve any back-references using the active context. Correct obvious typos.",
    "   For a POSITIONAL reference to document content ('the last question', 'the 5th",
    "   item'), preserve the positional wording and the kind of item it scopes (e.g.",
    "   'the answer to the last long-answer-type question') — do NOT substitute a",
    "   specific item you guessed from memory, which can name the wrong one.",
    "   For multi_group/segments, this is the combined query; per-subject wording goes in sub_queries.",
    "",
]

_FIELD_SUB_QUERIES = [
    "5. sub_queries — REQUIRED (>=2 entries) when dependency_type is multi_group; else [].",
    "   One object per subject, each with its OWN intent (using the definitions above):",
    '   {"query": "...", "intent": "<intent>", "source_files": [...], "position_selector": ""}',
    "   source_files: exact filename(s) from the loaded list if the subject clearly targets",
    "   a specific file; [] otherwise. Decide each subject's intent HERE (don't defer).",
    "   position_selector: this subject's OWN place/structural label ('section 9.2', 'the",
    "   last two') if it points at a part by position/structure; \"\" otherwise.",
    "",
]

_FIELD_SEGMENTS = [
    "6. segments — for COMPOUND queries: a single query that asks for TWO OR MORE",
    "   genuinely DIFFERENT OPERATIONS (different intents), each yielding its own answer.",
    "   The test: does the query combine operations that would NOT share one intent?",
    "   e.g. 'compare A and B, AND summarize C' = a comparison operation + a summary",
    "   operation -> TWO segments. 'define X and give me the Y table' = a factual op + an",
    "   extraction op -> TWO segments. Each segment:",
    '   {"title": "...", "query": "...", "intent": "<intent>", "source_files": [...], "position_selector": ""}',
    "   position_selector: that segment's OWN place/structural label ('section 9.2',",
    "   'the last two questions') when it targets a part by position/structure; \"\"",
    "   otherwise. So 'summarize section 9.2 and compare sections 4.1 and 4.2' -> two",
    "   segments, each with its own intent AND its own position_selector.",
    "   IMPORTANT: a compound query usually still has dependency_type 'independent' (or",
    "   'dependent'), NOT 'multi_group' — multi_group is for ONE operation over many",
    "   subjects, whereas segments are for MANY operations. When you emit segments (>=2),",
    "   leave sub_queries=[]. A query that is a SINGLE operation (even over many subjects,",
    "   like a pure comparison) has NO segments -> return [].",
    "",
]

_FIELD_SOURCE_FILES = [
    "7. source_files — exact filename(s) from the loaded list if the WHOLE query clearly",
    "   targets specific file(s); [] otherwise.",
    "",
]

_FIELD_ANSWER_SOURCE = [
    "8. answer_source — 'previous_answer' if the request can be fully satisfied from the",
    "   previous assistant answer alone (reformat it, condense, translate, tabulate,",
    "   analyse what was already said). 'document' in all other cases (default).",
    "",
]

_FIELD_SUGGESTED_TOPIC = [
    "9. suggested_topic — short topic label for a new conversation group (when needed).",
    "",
]

_FIELD_REASONING = [
    "10. reasoning — one sentence explaining the key classification decision.",
    "",
]

_FIELD_POSITION_SELECTOR = [
    "11. position_selector — the PLACE/ORDER/STRUCTURAL label that scopes the query to a",
    "    specific part of the document, when present; \"\" otherwise. This is ORTHOGONAL to",
    "    retrieval_intent (which is the OPERATION). Set it whenever the user points at a",
    "    part by its position or structural label rather than (or in addition to) its topic:",
    "    a numbered/titled section, sub-section, chapter, clause, or heading ('section 9.2',",
    "    'clause 4.3', 'the Methods section', 'chapter 3'); or an ordinal/positional item",
    "    ('the last two questions', 'the 5th item', 'the one after X'). Copy the selector",
    "    phrase as the user expressed it (e.g. 'section 9.2', 'the last two questions').",
    "    ALWAYS set it when the query names a part by a label-with-number or an ordinal —",
    "    even a terse one with no other words ('section 10', 'heading 4.2', 'the last two').",
    "    A bare label+number has almost no semantic signal, so if you leave this empty the",
    "    part cannot be located; whenever you reason that the query refers to a specific",
    "    section/heading/clause/figure/item by number or order, you MUST fill this field.",
    "    Leave \"\" ONLY for ordinary topic/content queries ('what is resistance', 'the",
    "    controls table') that are NOT scoped by position. When set, the system reads the",
    "    whole document in order to locate the scope, then applies retrieval_intent to it.",
    "",
]

# The complete ordered field block, composed from the per-field variables above.
_CLASSIFICATION_FIELDS = (
    _FIELDS_HEADER
    + _FIELD_DEPENDENCY_TYPE
    + _FIELD_RETRIEVAL_INTENT
    + _FIELD_BELONGS_TO_ACTIVE_GROUP
    + _FIELD_STANDALONE_QUERY
    + _FIELD_SUB_QUERIES
    + _FIELD_SEGMENTS
    + _FIELD_SOURCE_FILES
    + _FIELD_ANSWER_SOURCE
    + _FIELD_SUGGESTED_TOPIC
    + _FIELD_REASONING
    + _FIELD_POSITION_SELECTOR
)

_RESPONSE_FORMAT = [
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
    '  "reasoning": "...",',
    '  "position_selector": ""',
    "}",
]


def build_classification_context(
    query: str,
    active_group_summary: Optional[str],
    active_group_topic: Optional[str],
    previous_ambiguous_query: Optional[str] = None,
    available_documents: Optional[List[str]] = None,
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

    # Static field definitions + response format (composed from the per-field
    # variables at module scope) extend the dynamic context built above.
    context_parts.extend(_CLASSIFICATION_FIELDS)
    context_parts.extend(_RESPONSE_FORMAT)

    return "\n".join(context_parts)


def build_subquery_classification_prompt(
    sub_query: str,
    available_documents: Optional[List[str]] = None,
) -> str:
    """Build the prompt that classifies a single sub-query's topic + retrieval intent.

    When ``available_documents`` is provided, the prompt also asks the LLM to pick the
    sub-query's ``source_files`` from that list. The caller validates the LLM's choices
    against the real file list.
    """
    docs_block = ""
    json_fmt = '{"topic": "...", "intent": "...", "position_selector": ""}'
    if available_documents:
        listed = "\n".join(f"  - {d}" for d in available_documents[:20])
        docs_block = (
            "\n4. source_files: a JSON array of the EXACT filenames (from the loaded "
            "list below) that this sub-query SPECIFICALLY refers to. ONLY include a "
            "file when the sub-query points to it unmistakably — by its name, or by a "
            "description that clearly identifies it (e.g. 'the security standard' → an "
            "ISO file; 'the circuits doc' → an electricity file). Include EVERY file it "
            "specifically refers to (one, several, or all). If the sub-query does NOT "
            "single out any particular file, or you are at all unsure, return [] so it "
            "searches everything. Do NOT guess.\n"
            "LOADED FILES:\n" + listed + "\n"
        )
        json_fmt = '{"topic": "...", "intent": "...", "position_selector": "", "source_files": ["<exact filename>", ...]}'

    return (
        "For this sub-query from a larger multi-part request, return:\n"
        "1. topic: a brief category (1-3 words)\n"
        "2. intent: ONE of [factual, targeted_summary, global_summary, "
        "targeted_extraction, global_extraction, analysis, positional]\n"
        "3. position_selector: the PLACE/ORDER/structural label this sub-query points at "
        "('section 9.2', 'the last two', 'the 5th item') if it scopes a part by position/"
        "structure; \"\" otherwise."
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
        "  - positional: the sub-query names a part by PLACE/ORDER or a structural label\n"
        "    ('section 9.2', 'the last two', 'the 5th') — also set position_selector to it.\n"
        "RULE: If the sub-query is just a topic NAME with no 'all/every/complete' wording,\n"
        "it is TARGETED (or factual), NEVER global. Example: 'combination of bulbs' →\n"
        "targeted_extraction (a specific section), NOT global_extraction.\n"
        "Respond with ONLY JSON: " + json_fmt
    )
