"""Context Resolution Layer for query classification and processing."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from backend.utils.logger import get_logger
from backend.memory.classifiers import LLMClassifier

log = get_logger("context_resolver")


class DependencyType(Enum):
    """Query dependency classification."""
    INDEPENDENT = "independent"      # Standalone query, no context needed
    DEPENDENT = "dependent"           # Follow-up, needs active group context
    MULTI_GROUP = "multi_group"       # Compares/relates multiple groups
    AMBIGUOUS = "ambiguous"           # Unclear, might need clarification


class RetrievalIntent(Enum):
    """Query intent classification for retrieval strategy."""
    FACTUAL = "factual"              # What is X? → Top relevant chunks
    TARGETED_SUMMARY = "targeted_summary"  # Summarize a specific section/topic
    GLOBAL_SUMMARY = "global_summary"      # Summarize whole document/corpus
    COMPARISON = "comparison"         # Compare X and Y → Multi-group retrieval
    TARGETED_EXTRACTION = "targeted_extraction"  # Extract specific table/list subset
    GLOBAL_EXTRACTION = "global_extraction"      # Enumerate all matching items
    POSITIONAL = "positional"         # Select item(s) by place/order (last two, 5th, ...)
    ANALYSIS = "analysis"             # Why/how/risks → Cross-section reasoning
    AMBIGUOUS = "ambiguous"           # Intent unclear, needs clarification


@dataclass
class QueryContext:
    """Processed query with all context information."""
    
    original_query: str
    standalone_query: str              # Retrieval-ready version
    dependency_type: DependencyType
    retrieval_intent: RetrievalIntent
    active_group_id: Optional[str]    # Current group if DEPENDENT
    belongs_to_active_group: bool     # From LLM classification
    relevant_groups: List[Dict[str, Any]]  # Groups found via similarity search
    memory_context: Optional[str]      # Loaded from group(s)
    needs_group_context: bool
    should_create_new_group: bool     # If independent or doesn't belong to active
    suggested_topic: Optional[str]    # Topic for new group
    is_multi_group: bool              # True if multi_group dependency type
    sub_queries: List[Dict[str, str]] = field(default_factory=list)  # For multi_group queries
    segments: List[Dict[str, Any]] = field(default_factory=list)  # Compound-query operation segments
    restrict_filenames: Optional[List[str]] = None  # Single-query file pinning (LLM-chosen)
    llm_reasoning: Optional[str] = None  # Reasoning from LLM
    answer_source: str = "document"   # 'document' or 'previous_answer' (LLM-decided)


class ContextResolver:
    """Resolves query context before retrieval and generation using LLM classification."""
    
    def __init__(self, memory_manager, embedder):
        """
        Initialize context resolver with LLM classification.
        
        Args:
            memory_manager: ConversationMemoryManager instance
            embedder: Embedder instance for generating query embeddings
        """
        self.memory_manager = memory_manager
        self.embedder = embedder
        self.llm_classifier = LLMClassifier()
        log.info("[RESOLVER] ✅ LLM classifier initialized")
    
    def resolve(
        self, 
        query: str,
        active_group_id: Optional[str] = None,
        available_documents: Optional[List[str]] = None
    ) -> QueryContext:
        """
        Resolve complete context for a query.
        
        Uses LLM classification to determine:
        - Dependency type (independent/dependent/multi_group/ambiguous)
        - Retrieval intent (factual/summary/comparison/extraction/analysis)
        - Whether query belongs to active group
        - Sub-queries if multi-group
        
        Then:
        - Creates new group if independent or doesn't belong to active
        - Splits multi-group queries into sub-queries with separate topics
        
        Args:
            query: User's input query
            active_group_id: Currently active group ID if any
            available_documents: Filenames currently loaded in the vector store (if any)
            
        Returns:
            QueryContext with all processing done
        """
        log.info(f"[RESOLVER] Processing query: {query[:80]}")
        
        # Get active group info if available
        active_group = None
        active_group_summary = None
        active_group_topic = None
        
        if active_group_id:
            log.info(f"[RESOLVER] 🔍 Looking for active group: {active_group_id}")
            active_group = self.memory_manager.get_conversation_group(active_group_id)
            if active_group:
                log.info(f"[RESOLVER] ✅ Active group found | turns: {len(active_group.all_turns)} | topic: {active_group.topic}")
                if active_group.all_turns:
                    last_turn = active_group.all_turns[-1]
                    log.info(f"[RESOLVER]    Last turn query: {last_turn.query[:60]} | dependency: {last_turn.dependency_type}")
                active_group_topic = active_group.topic

                # Build the classifier's view of this group's history. Two correctness
                # requirements:
                #  (a) RACE-SAFE: never read a half-written summary — gate on summary_ready
                #      so a query landing mid-summarization can't see a partial summary.
                #  (b) NO CONTEXT GAP: a READY summary covers only the turns up to the last
                #      roll-up; turns added AFTER it live in group.recent_turns and are NOT
                #      in the summary. So we ALWAYS append those unsummarized recent turns —
                #      otherwise a follow-up to one of them would be invisible until the next
                #      roll-up. When no summary is ready yet, send the raw Q&A of all turns
                #      (the whole "5 queries + outputs" path).
                def _turns_to_text(turns):
                    lines = []
                    for t in turns:
                        ans = (t.full_answer or t.memory_summary or "").strip()
                        if t.query:
                            lines.append(f"User: {t.query.strip()}")
                        if ans:
                            lines.append(f"Assistant: {ans[:600]}")
                    return "\n".join(lines)

                if active_group.summary_ready and (active_group.summary or "").strip():
                    parts = [f"Summary of earlier turns: {active_group.summary.strip()}"]
                    recent_text = _turns_to_text(active_group.recent_turns)
                    if recent_text:
                        parts.append("Turns since that summary:\n" + recent_text)
                    active_group_summary = "\n\n".join(parts)
                    log.info(
                        f"[RESOLVER] Classifier context = READY summary + "
                        f"{len(active_group.recent_turns)} unsummarized recent turn(s)"
                    )
                else:
                    raw_text = _turns_to_text(active_group.all_turns)
                    if raw_text:
                        active_group_summary = raw_text
                        log.info(
                            f"[RESOLVER] Summary not ready (flag={active_group.summary_ready}) — "
                            f"using {len(active_group.all_turns)} raw turn(s) of Q&A as classifier context"
                        )
            else:
                log.warning(f"[RESOLVER] ⚠️ Active group {active_group_id} NOT FOUND in memory")
        else:
            log.info(f"[RESOLVER] ℹ️ No active_group_id provided (None)")
        
        # Step 1: Classify using LLM
        log.info("[RESOLVER] Using LLM classification")
        
        llm_result = self.llm_classifier.classify_query(
            query=query,
            active_group_summary=active_group_summary,
            active_group_topic=active_group_topic,
            available_documents=available_documents
        )
        
        dependency_type = self._map_dependency_type(llm_result["dependency_type"])
        retrieval_intent = self._map_retrieval_intent(llm_result["retrieval_intent"])
        belongs_to_active = llm_result.get("belongs_to_active_group", False)
        reasoning = llm_result.get("reasoning", "")
        sub_queries_raw = llm_result.get("sub_queries", [])
        suggested_topic = llm_result.get("suggested_topic")
        answer_source = llm_result.get("answer_source", "document")
        
        # STEP 1.5: Handle ambiguous queries with follow-up clarifications
        standalone_query = llm_result.get("standalone_query", query)
        original_query = query
        
        # Check if we're in an active group with previous turns
        if active_group and active_group.all_turns and len(active_group.all_turns) > 0:
            last_turn = active_group.all_turns[-1]
            # Only process if last turn was ambiguous (dependency type)
            if last_turn.dependency_type == "ambiguous":
                log.info(f"[RESOLVER] ⚠️ Previous query was AMBIGUOUS - current query is clarification")
                log.info(f"[RESOLVER] Ambiguous Q: '{last_turn.query[:60]}'")
                log.info(f"[RESOLVER] Clarification: '{query[:60]}'")
                
                # Re-classify with both queries so LLM combines them intelligently
                log.info(f"[RESOLVER] ✅ Sending both queries to LLM for intelligent combination")
                
                try:
                    llm_result = self.llm_classifier.classify_query(
                        query=query,
                        active_group_summary=active_group_summary,
                        active_group_topic=active_group_topic,
                        previous_ambiguous_query=last_turn.query,
                        available_documents=available_documents
                    )

                    # Update from re-classification
                    dependency_type = self._map_dependency_type(llm_result["dependency_type"])
                    retrieval_intent = self._map_retrieval_intent(llm_result["retrieval_intent"])
                    belongs_to_active = llm_result.get("belongs_to_active_group", False)
                    standalone_query = llm_result.get("standalone_query", query)
                    # Carry through fields needed when the user's message was NOT a real
                    # clarification (e.g. a brand-new independent/multi_group question) or
                    # when it is still ambiguous and we must ask again.
                    suggested_topic = llm_result.get("suggested_topic")
                    sub_queries_raw = llm_result.get("sub_queries", [])
                    reasoning = llm_result.get("reasoning", reasoning)
                    answer_source = llm_result.get("answer_source", answer_source)

                    log.info(f"[RESOLVER] Clarification outcome → dependency={dependency_type.value} | intent={retrieval_intent.value}")
                    if dependency_type == DependencyType.AMBIGUOUS:
                        log.info("[RESOLVER] Still AMBIGUOUS — will ask the user to clarify again")
                    elif standalone_query.strip().lower() == query.strip().lower():
                        log.info("[RESOLVER] Treated as a NEW question (clarification ignored by user)")
                    else:
                        log.info("[RESOLVER] Combined ambiguous query + clarification")
                    log.info(f"[RESOLVER] Combined query intent: {retrieval_intent.value}")
                    log.info(f"[RESOLVER] Standalone query for retrieval: '{standalone_query[:100]}...'")
                    log.info(f"[RESOLVER] DEBUG: Reclassified - dependency_type={dependency_type.value}, retrieval_intent={retrieval_intent.value}")
                except Exception as e:
                    # Re-classification failed (e.g. content filter, network). The
                    # first-pass classification already succeeded, so keep using it
                    # instead of crashing the whole query.
                    log.warning(f"[RESOLVER] ⚠️ Clarification re-classification failed ({e}) — keeping first-pass classification: {dependency_type.value} | {retrieval_intent.value}")
        
        log.info(f"[RESOLVER] LLM: {dependency_type.value} | {retrieval_intent.value} | belongs={belongs_to_active} | answer_source={answer_source}")
        
        # Step 2: Use LLM-generated standalone query
        log.info(f"[RESOLVER] Final standalone query: {standalone_query[:80]}")
        
        # Step 3: Determine if should create new group
        should_create_new_group = (
            dependency_type == DependencyType.INDEPENDENT or
            dependency_type == DependencyType.AMBIGUOUS or
            (active_group_id and not belongs_to_active)
        )
        
        # Prepare topic for new group
        if should_create_new_group:
            if not suggested_topic:
                # LLM classifier should always provide suggested_topic
                # If not, we'll use the query itself as the topic
                suggested_topic = query[:50].title()
            log.info(f"[RESOLVER] Will create new group: {suggested_topic}")
        
        # Step 4: Handle multi-group queries
        sub_queries = []
        # Compound-query segments: distinct operations, each with its OWN intent and
        # optional file pinning, generated separately and combined. When present (>=2),
        # they REPLACE the flat sub-query split (and its extra per-subject LLM calls).
        segments = self._build_segments(llm_result.get("segments", []), available_documents)
        # Comparison requests should stay as ONE combined operation unless the LLM
        # produced true mixed-operation segments (e.g., compare A/B AND extract C).
        # If a top-level comparison was decomposed into non-comparison per-subject
        # segments, ignore those segments and keep the single comparison flow.
        if segments and retrieval_intent == RetrievalIntent.COMPARISON:
            seg_intents = {str(seg.get("intent", "")).strip().lower() for seg in segments}
            if "comparison" not in seg_intents:
                log.info(
                    "[RESOLVER] Top-level intent is comparison but segments are non-comparison "
                    "(likely per-subject split) — ignoring segments and using single comparison retrieval"
                )
                segments = []
        if segments:
            log.info(f"[RESOLVER] Compound query — {len(segments)} segments (per-segment intent; skipping flat sub-query split)")
            for seg in segments:
                log.info(f"[RESOLVER]   segment: '{seg['title']}' | intent={seg['intent']} | files={seg['files']}")
        elif dependency_type == DependencyType.MULTI_GROUP:
            log.info("[RESOLVER] Processing multi-group query")
            if self.llm_classifier and sub_queries_raw:
                sub_queries = self.llm_classifier.split_multi_group_query(
                    query, sub_queries_raw, available_documents=available_documents
                )
            else:
                # Safety net: classifier failed to provide sub-queries for a
                # multi_group query. Don't silently collapse to one entry (that loses
                # the comparison). Retry extraction once, then fall back to the raw
                # query so retrieval still runs against the full comparison.
                log.warning("[RESOLVER] ⚠️ multi_group with no sub_queries — attempting recovery")
                recovered = []
                if self.llm_classifier:
                    recovered = self.llm_classifier.recover_sub_queries(standalone_query or query)
                if recovered:
                    sub_queries = self.llm_classifier.split_multi_group_query(
                        query, recovered, available_documents=available_documents
                    )
                else:
                    log.warning("[RESOLVER] ⚠️ Recovery failed — using full query as single comparison")
                    sub_queries = [{"query": standalone_query or query, "topic": "Comparison"}]
            log.info(f"[RESOLVER] Split into {len(sub_queries)} sub-queries")

            # Per-subject document pinning: the lightweight per-sub-query LLM call
            # decides which loaded file(s) a subject targets (validated against the real
            # file list). When the user specifically refers to file(s), that subject's
            # retrieval is pinned to exactly those file(s); otherwise it stays unpinned
            # and searches all documents. No string/token heuristics — the LLM decides.
            if available_documents:
                for sq in sub_queries:
                    if sq.get("filenames"):
                        log.info(f"[RESOLVER] Sub-query '{sq.get('query', '')[:40]}' pinned to file(s): {sq['filenames']}")

            # MIXED-OPERATION PROMOTION: a multi_group request whose sub-queries carry
            # DIFFERENT intents (e.g. "compare A & B AND summarize C") is really several
            # DISTINCT operations, not one woven answer. Generating all of them under the
            # single top-level intent collapses each part (a global_summary part squeezed
            # into a comparison blurb). So when the sub-queries span >= 2 distinct intents,
            # promote them to SEGMENTS — each part then retrieves AND generates with its
            # OWN intent (its own depth + map-reduce), and the parts are combined into one
            # sectioned answer. A homogeneous multi_group (e.g. a pure comparison, all
            # sub-queries 'comparison') is NOT promoted: it stays the single balanced
            # answer. No extra LLM calls — sub-queries already carry their intent/files.
            distinct_intents = {
                (sq.get("intent") or "").strip().lower()
                for sq in sub_queries
                if (sq.get("intent") or "").strip()
            }
            if len(distinct_intents) >= 2:
                segments = [
                    {
                        "query": sq.get("query", ""),
                        "intent": (sq.get("intent") or "factual").strip().lower(),
                        "files": sq.get("filenames"),
                        "title": sq.get("topic") or sq.get("query", "")[:40],
                    }
                    for sq in sub_queries
                    if (sq.get("query") or "").strip()
                ]
                log.info(
                    f"[RESOLVER] Mixed-operation multi_group (intents={sorted(distinct_intents)}) "
                    f"— promoting {len(segments)} sub-queries to per-intent segments"
                )
                for seg in segments:
                    log.info(f"[RESOLVER]   segment: '{seg['title']}' | intent={seg['intent']} | files={seg['files']}")
        
        # Step 5: Find relevant groups based on dependency type
        # YOUR PIPELINE: For DEPENDENT queries, check if belongs to active group FIRST
        relevant_groups = []
        
        if dependency_type == DependencyType.DEPENDENT:
            if active_group_id and belongs_to_active:
                # ✅ BELONGS TO ACTIVE GROUP — use it, AND ALSO merge in any OTHER group that
                # is CLEARLY related (high threshold). A follow-up can legitimately draw on a
                # prior DIFFERENT topic at the same time as the active one; this loads that
                # topic's memory ALONGSIDE the active group instead of dropping it. The
                # active group is loaded directly; these are supplementary context only, and
                # the threshold is deliberately conservative so unrelated topics don't dilute
                # the active focus.
                log.info(f"[RESOLVER] DEPENDENT query BELONGS to active group {active_group_id}")
                merged = self._find_relevant_groups(
                    standalone_query,
                    threshold=0.6,  # higher than the 0.5 used when NOT in the active group
                    limit=2
                )
                # Never duplicate the active group — it is loaded separately.
                relevant_groups = [
                    g for g in merged
                    if g.get("group") is None or g["group"].group_id != active_group_id
                ]
                if relevant_groups:
                    log.info(f"[RESOLVER] Merging {len(relevant_groups)} additional related group(s) with the active group")
                else:
                    log.info("[RESOLVER] No other clearly-related groups to merge — active group only")
            else:
                # ❌ Does NOT belong to active group → Search for relevant ones
                log.info("[RESOLVER] DEPENDENT query — retrieving relevant groups with threshold")
                relevant_groups = self._find_relevant_groups(
                    standalone_query, 
                    threshold=0.5,  # Similarity threshold for dependent queries
                    limit=3
                )
                log.info(f"[RESOLVER] Found {len(relevant_groups)} relevant groups above threshold")
        else:
            # For other queries, find related groups without strict threshold
            relevant_groups = self._find_relevant_groups(
                standalone_query,
                threshold=0.3,  # Lower threshold for context
                limit=2
            )
        
        log.info(f"[RESOLVER] Found {len(relevant_groups)} relevant groups")
        
        # Step 5.5: Create groups for tracking
        # INDEPENDENT queries: Always create NEW group (new topic, ignore active_group_id)
        # AMBIGUOUS queries: Create group to track clarification attempts
        # DEPENDENT queries: Use active group if available, else search or create
        if dependency_type == DependencyType.INDEPENDENT:
            # An INDEPENDENT query doesn't depend on the ACTIVE group — but per the routing
            # design it may still belong to a DIFFERENT existing group in the collection
            # (the user returning to an earlier topic). So FIRST search the other groups
            # (the relevant_groups embedding match) and REUSE a strongly-matching one for
            # continuity; only create a brand-new group when nothing is a confident match.
            # This is exactly what the group-summary embeddings are for: referring back to
            # the history of previous groups across the whole collection. (Only summarized
            # groups carry an embedding, so reuse naturally targets established topics.)
            REUSE_SIMILARITY = 0.75
            best_group = None
            best_sim = 0.0
            for g in relevant_groups:
                grp = g.get("group")
                sim = g.get("similarity", 0) or 0
                if grp is not None and sim >= REUSE_SIMILARITY and sim > best_sim:
                    best_group, best_sim = grp, sim
            if best_group is not None:
                active_group_id = best_group.group_id
                should_create_new_group = False
                log.info(
                    f"[RESOLVER] INDEPENDENT query MATCHES existing group {active_group_id} "
                    f"('{best_group.topic}', similarity={best_sim:.3f}) — reusing it instead "
                    f"of creating a duplicate"
                )
            else:
                log.info("[RESOLVER] INDEPENDENT query — no strong existing-group match; creating NEW group")
                if not suggested_topic:
                    # Use query as topic if LLM didn't provide one
                    suggested_topic = query[:50].title()
                try:
                    new_group = self.memory_manager.create_conversation_group(suggested_topic)
                    active_group_id = new_group.group_id  # Override any passed active_group_id
                    should_create_new_group = False
                    log.info(f"[RESOLVER] ✅ Created new topic group {active_group_id}: {suggested_topic}")
                except Exception as e:
                    log.warning(f"[RESOLVER] ⚠️ Failed to create group: {e}")
                    should_create_new_group = True
        
        elif dependency_type == DependencyType.AMBIGUOUS:
            # Ambiguous query — track the clarification IN the existing active group
            # rather than spawning a throwaway "Clarification:" group. A dedicated group
            # would hold just this one ambiguous turn and then be orphaned the moment the
            # user clarifies and the query resolves into some other group — accumulating
            # junk one-turn groups. Reusing the active group keeps the clarification in the
            # same thread the user was already in. Only when there is NO active group to
            # attach to do we create one so the turn still has somewhere to live.
            if active_group_id and active_group is not None:
                log.info(f"[RESOLVER] AMBIGUOUS query — tracking clarification in active group {active_group_id}")
                should_create_new_group = False
            else:
                log.info("[RESOLVER] AMBIGUOUS query with no active group — creating a group for clarification")
                if not suggested_topic:
                    suggested_topic = f"Clarification: {query[:40].title()}"
                try:
                    new_group = self.memory_manager.create_conversation_group(suggested_topic)
                    active_group_id = new_group.group_id
                    should_create_new_group = False
                    log.info(f"[RESOLVER] ✅ Created clarification group {active_group_id}: {suggested_topic}")
                except Exception as e:
                    log.warning(f"[RESOLVER] ⚠️ Failed to create clarification group: {e}")
                    should_create_new_group = False  # Don't try to create again
        
        elif dependency_type == DependencyType.DEPENDENT:
            if active_group_id and belongs_to_active:
                # Already using active group ✅
                log.info("[RESOLVER] Using active group — no new group creation needed")
            elif relevant_groups:
                # The query does NOT belong to the ACTIVE group, but a DIFFERENT existing
                # group matched (the user is continuing an earlier, non-active topic). Switch
                # the active group to that match so the turn is STORED where its context is
                # loaded FROM — otherwise context comes from the matched group while the turn
                # gets filed under the stale active group, desyncing the two.
                best = max(relevant_groups, key=lambda g: g.get("similarity", 0) or 0)
                matched = best.get("group")
                if matched is not None:
                    active_group_id = matched.group_id
                    should_create_new_group = False
                    log.info(
                        f"[RESOLVER] DEPENDENT query belongs to EXISTING group {active_group_id} "
                        f"('{matched.topic}', similarity={best.get('similarity', 0):.3f}) — switching to it"
                    )
                else:
                    # Defensive: a relevant entry with no group object — create a new group.
                    if not suggested_topic:
                        suggested_topic = query[:50].title()
                    new_group = self.memory_manager.create_conversation_group(suggested_topic)
                    active_group_id = new_group.group_id
                    should_create_new_group = False
                    log.info(f"[RESOLVER] ✅ Created group {active_group_id}: {suggested_topic}")
            else:
                # No active match AND no relevant groups found — create a new one.
                log.info("[RESOLVER] DEPENDENT query with no relevant groups — auto-creating new group")
                if not suggested_topic:
                    # Use query as topic if LLM didn't provide one
                    suggested_topic = query[:50].title()
                
                # Auto-create new group
                try:
                    new_group = self.memory_manager.create_conversation_group(suggested_topic)
                    active_group_id = new_group.group_id
                    should_create_new_group = False  # Already created
                    log.info(f"[RESOLVER] ✅ Auto-created group {active_group_id}: {suggested_topic}")
                except Exception as e:
                    log.warning(f"[RESOLVER] ⚠️ Failed to auto-create group: {e}")
                    # Fall back to creating on next step
                    should_create_new_group = True

        elif dependency_type == DependencyType.MULTI_GROUP:
            # Comparison / multi-subject query — create a group so the turn is stored
            # in conversation memory (otherwise follow-ups have no context to build on).
            log.info("[RESOLVER] MULTI_GROUP query — creating group to track the comparison")
            if not suggested_topic:
                suggested_topic = query[:50].title()
            try:
                new_group = self.memory_manager.create_conversation_group(suggested_topic)
                active_group_id = new_group.group_id
                should_create_new_group = False
                log.info(f"[RESOLVER] ✅ Created multi-group topic {active_group_id}: {suggested_topic}")
            except Exception as e:
                log.warning(f"[RESOLVER] ⚠️ Failed to create multi-group group: {e}")
                should_create_new_group = True
        
        # Step 6: Load memory context
        memory_context = self._load_memory_context(
            dependency_type,
            active_group_id,
            relevant_groups,
            belongs_to_active
        )
        
        # Determine if group context is needed
        needs_group_context = dependency_type in [
            DependencyType.DEPENDENT,
            DependencyType.MULTI_GROUP
        ]
        
        # Single-query file pinning: when the user's query unmistakably targets specific
        # loaded file(s) (by name or clear description), the LLM returns source_files and
        # retrieval is restricted to exactly those file(s) so e.g. "summarize electricity.pdf"
        # never pulls chunks from other documents. Validated against the real file list;
        # an empty/unsure pick stays None (search all). Multi-group/compound queries do
        # their own per-subject/segment pinning, so this is suppressed for them.
        restrict_filenames = None
        if not segments and dependency_type != DependencyType.MULTI_GROUP:
            restrict_filenames = self._validate_filenames(
                llm_result.get("source_files", []), available_documents
            )
            if restrict_filenames:
                log.info(f"[RESOLVER] Query pinned to file(s): {restrict_filenames}")
            elif (
                dependency_type == DependencyType.DEPENDENT
                and belongs_to_active
                and active_group is not None
            ):
                # Carry-forward: a follow-up that stays within the active conversation but
                # names no file of its own inherits the group's most recent pin. This keeps
                # e.g. a follow-up to an electricity-pinned summary scoped to electricity.pdf
                # instead of letting a semantically-broad term ("safety", "precautions")
                # pull in unrelated documents. Validated against the current file list, so a
                # since-removed file falls back to searching everything.
                for turn in reversed(active_group.all_turns):
                    prev_pin = getattr(turn, "restrict_filenames", None)
                    if prev_pin:
                        restrict_filenames = self._validate_filenames(prev_pin, available_documents)
                        if restrict_filenames:
                            log.info(f"[RESOLVER] Follow-up inherited conversation pin: {restrict_filenames}")
                        break
        
        context = QueryContext(
            original_query=original_query,
            standalone_query=standalone_query,
            dependency_type=dependency_type,
            retrieval_intent=retrieval_intent,
            active_group_id=active_group_id,
            belongs_to_active_group=belongs_to_active,
            relevant_groups=relevant_groups,
            memory_context=memory_context,
            needs_group_context=needs_group_context,
            should_create_new_group=should_create_new_group,
            suggested_topic=suggested_topic,
            is_multi_group=dependency_type == DependencyType.MULTI_GROUP,
            sub_queries=sub_queries,
            segments=segments,
            restrict_filenames=restrict_filenames,
            llm_reasoning=reasoning,
            answer_source=answer_source,
        )
        
        log.info("[RESOLVER] ✅ Context resolution complete")
        return context
    
    def _build_segments(
        self,
        segments_raw: Any,
        available_documents: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Validate the LLM's compound-query ``segments`` into clean operation segments.

        Each kept segment is ``{query, intent, files, title}``. A segment's intent is
        validated against the retrieval-intent enum (defaulting to factual) and its
        source_files against the real loaded-file list (unknown names dropped). Returns
        a list ONLY when there are >= 2 valid segments; otherwise [] (single-operation
        query → normal single-intent path).
        """
        if not isinstance(segments_raw, list) or len(segments_raw) < 2:
            return []
        valid_intents = {i.value for i in RetrievalIntent if i != RetrievalIntent.AMBIGUOUS}
        out: List[Dict[str, Any]] = []
        for seg in segments_raw:
            if not isinstance(seg, dict):
                continue
            q = str(seg.get("query", "")).strip()
            if not q:
                continue
            intent = str(seg.get("intent", "")).strip().lower()
            if intent not in valid_intents:
                intent = "factual"
            files = self._validate_filenames(seg.get("source_files", []), available_documents)
            title = str(seg.get("title", "")).strip() or q[:40]
            out.append({"query": q, "intent": intent, "files": files, "title": title})
        return out if len(out) >= 2 else []

    @staticmethod
    def _validate_filenames(
        raw: Any,
        available_documents: Optional[List[str]],
    ) -> Optional[List[str]]:
        """Validate LLM-chosen filenames against the real loaded-file list.

        Accepts a list or a lone string. Drops unknown/hallucinated names and de-dups,
        preserving order. Returns the cleaned list, or ``None`` (search all files) when
        nothing valid remains — so an empty/garbage pick never over-restricts retrieval.
        """
        valid_files = {str(d).strip() for d in (available_documents or [])}
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return None
        files = [f for f in (str(x).strip() for x in raw) if f in valid_files]
        files = list(dict.fromkeys(files))  # de-dup, keep order
        return files or None
    
    def _find_relevant_groups(
        self, 
        query: str,
        threshold: float = 0.5,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find relevant conversation groups by embedding similarity with threshold.
        
        Args:
            query: The standalone query
            threshold: Minimum similarity score to include group
            limit: Maximum groups to return
            
        Returns:
            List of relevant groups with similarity scores (filtered by threshold)
        """
        try:
            # Generate query embedding. Use embed_query (single-text, cached) so this and
            # the later document-chunk retrieval of the SAME standalone query share one
            # embedding instead of paying for two identical API calls.
            query_embedding = self.embedder.embed_query(query)
            
            if not query_embedding:
                log.warning("[RESOLVER] Failed to embed query")
                return []
            
            # Search groups by embedding with threshold
            results = self.memory_manager.search_groups_by_embedding(
                query_embedding=query_embedding,
                similarity_threshold=threshold,
                top_k=limit
            )
            
            log.info(f"[RESOLVER] Found {len(results)} groups above threshold {threshold}")
            for r in results:
                log.info(f"[RESOLVER]   - {r.get('topic', 'Unknown')}: {r.get('score', 0):.3f}")
            return results
            
        except Exception as e:
            log.error(f"[RESOLVER] Failed to find groups: {e}", exc_info=True)
            return []
    
    def _load_memory_context(
        self,
        dependency_type: DependencyType,
        active_group_id: Optional[str],
        relevant_groups: List[Dict[str, Any]],
        belongs_to_active: bool = True
    ) -> Optional[str]:
        """
        Load memory context from groups based on dependency type.
        
        INDEPENDENT:   No context needed
        DEPENDENT:     Load active group summary + recent turns
        MULTI_GROUP:   Load relevant group summaries
        AMBIGUOUS:     Load active group if available
        """
        context_parts = []
        
        if dependency_type == DependencyType.INDEPENDENT:
            return None
        
        elif dependency_type == DependencyType.DEPENDENT:
            # Load active group context
            if active_group_id:
                group_context = self.memory_manager.get_group_context(active_group_id)
                if group_context:
                    context_parts.append(self._format_group_context(group_context))
            # MERGE: also include any OTHER clearly-related group's summary (the active
            # group is already loaded above and is skipped here to avoid duplication), so a
            # follow-up that also draws on a different prior topic gets that context too.
            # Only ready summaries are used.
            for group_info in relevant_groups:
                group = group_info.get("group")
                if not group or group.group_id == active_group_id:
                    continue
                if group.summary and group.summary_ready:
                    context_parts.append(f"## {group.topic}\n{group.summary}")
        
        elif dependency_type == DependencyType.MULTI_GROUP:
            # Load all relevant group summaries (only if ready)
            for group_info in relevant_groups:
                group = group_info.get("group")
                # Only use summary if it's ready, otherwise skip this group's summary
                if group and group.summary and group.summary_ready:
                    context_parts.append(
                        f"## {group.topic}\n{group.summary}"
                    )
                elif group:
                    # Summary not ready, include group info without summary
                    log.info(f"[RESOLVER] Summary not ready for {group.topic}, skipping")

        
        elif dependency_type == DependencyType.AMBIGUOUS:
            # Try active group, fall back to relevant groups
            if active_group_id:
                group_context = self.memory_manager.get_group_context(active_group_id)
                if group_context:
                    context_parts.append(self._format_group_context(group_context))
            
            # Also include relevant group summaries (only if ready)
            for group_info in relevant_groups:
                group = group_info.get("group")
                if group and group.summary and group.summary_ready:
                    context_parts.append(
                        f"## {group.topic}\n{group.summary}"
                    )
        
        if context_parts:
            return "\n\n".join(context_parts)
        
        return None
    
    def _format_group_context(self, group_context: Dict[str, Any]) -> str:
        """
        Format group context for LLM.
        
        If summary is ready, use it. Otherwise, use all recent turns
        (to provide full context while summary is still being generated).
        """
        parts = [f"## {group_context['topic']}"]
        
        # Check if summary is ready (not still being generated)
        summary_ready = group_context.get("summary_ready", False)
        
        recent = group_context.get("recent_turns", [])
        if summary_ready and group_context.get("summary"):
            # The summary covers turns up to the last roll-up. Include it AND the turns
            # added since (recent_turns) — the summary does NOT contain those, so without
            # them a follow-up to a post-summary turn would lose that turn's context.
            parts.append(f"\n**Summary:** {group_context['summary']}")
            if recent:
                parts.append("\n**Since that summary:**")
                for turn in recent:
                    parts.append(f"- Q: {turn.query}")
                    parts.append(f"  A: {turn.memory_summary}")
            log.info(f"[RESOLVER] Context = summary + {len(recent)} recent turn(s)")
        elif recent:
            # No ready summary yet — use the recent (unsummarized) turns directly.
            log.info(f"[RESOLVER] Summary not ready, using {len(recent)} recent turns for context")
            parts.append("\n**Recent Discussion:**")
            for turn in recent:  # All recent turns, not just last 3
                parts.append(f"- Q: {turn.query}")
                parts.append(f"  A: {turn.memory_summary}")
        else:
            # No recent turns and no ready summary — fall back to any stored summary.
            if group_context.get("summary"):
                parts.append(f"\n**Summary:** {group_context['summary']}")
            else:
                parts.append("\n(No context available)")
        
        return "\n".join(parts)
    
    def _map_dependency_type(self, llm_result: str) -> DependencyType:
        """Map LLM string result to DependencyType enum."""
        mapping = {
            "independent": DependencyType.INDEPENDENT,
            "dependent": DependencyType.DEPENDENT,
            "multi_group": DependencyType.MULTI_GROUP,
            "ambiguous": DependencyType.AMBIGUOUS,
        }
        return mapping.get(llm_result.lower(), DependencyType.AMBIGUOUS)
    
    def _map_retrieval_intent(self, llm_result: str) -> RetrievalIntent:
        """Map LLM string result to RetrievalIntent enum."""
        mapping = {
            "factual": RetrievalIntent.FACTUAL,
            "targeted_summary": RetrievalIntent.TARGETED_SUMMARY,
            "global_summary": RetrievalIntent.GLOBAL_SUMMARY,
            "comparison": RetrievalIntent.COMPARISON,
            "targeted_extraction": RetrievalIntent.TARGETED_EXTRACTION,
            "global_extraction": RetrievalIntent.GLOBAL_EXTRACTION,
            "positional": RetrievalIntent.POSITIONAL,
            "analysis": RetrievalIntent.ANALYSIS,
            "ambiguous": RetrievalIntent.AMBIGUOUS,
        }
        return mapping.get(llm_result.lower(), RetrievalIntent.FACTUAL)
    

