"""
Classification prompts — the query-classifier prompt and the per-sub-query
topic/intent prompt used by the LLM classifier.

These are pure builders: they take the same inputs the classifier had inline and
return the prompt string, so no classifier prompt text lives inside the logic.
"""

from typing import List, Optional


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
        '                           "the last question", "the bottom of the list". This',
        '                           includes asking for the ANSWER/SOLUTION/VALUE of a',
        '                           positionally-named item ("answer of the last question",',
        '                           "solve the first one") — the SELECTOR is still ordinal,',
        '                           so it is positional, NOT factual. Position can only be',
        '                           resolved by reading the WHOLE document in order, so use',
        '                           this whenever the selector is ordinal/positional, even',
        '                           though only a few items are ultimately wanted.',
        '                           CRITICAL: do NOT try to GUESS which concrete item the',
        '                           position refers to and bake that guess into',
        '                           standalone_query — the conversation memory may be',
        '                           incomplete, so naming a specific item risks picking the',
        '                           WRONG one. Keep the positional phrasing ("the last',
        '                           long-answer question and its answer") and let retrieval',
        '                           over the full document resolve the actual item.',
        '   "ambiguous"           : intent cannot be determined without clarification.',
        "   Pick the NARROWEST intent that fits; reserve the GLOBAL and positional intents",
        "   for requests whose scope genuinely requires reading the whole document.",
        "",
        "3. belongs_to_active_group — true if this query continues the active conversation topic.",
        "",
        "4. standalone_query — a self-contained, retrieval-ready version of the query.",
        "   Resolve any back-references using the active context. Correct obvious typos.",
        "   For a POSITIONAL reference to document content ('the last question', 'the 5th",
        "   item'), preserve the positional wording and the kind of item it scopes (e.g.",
        "   'the answer to the last long-answer-type question') — do NOT substitute a",
        "   specific item you guessed from memory, which can name the wrong one.",
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

    return (
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
