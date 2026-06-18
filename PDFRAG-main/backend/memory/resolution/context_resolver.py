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
    SUMMARY = "summary"               # Summarize X → Broader coverage
    COMPARISON = "comparison"         # Compare X and Y → Multi-group retrieval
    EXTRACTION = "extraction"         # List all X → Aggregate information
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
                active_group_summary = active_group.summary
                active_group_topic = active_group.topic
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
        if dependency_type == DependencyType.MULTI_GROUP:
            log.info("[RESOLVER] Processing multi-group query")
            if self.llm_classifier and sub_queries_raw:
                sub_queries = self.llm_classifier.split_multi_group_query(query, sub_queries_raw)
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
                    sub_queries = self.llm_classifier.split_multi_group_query(query, recovered)
                else:
                    log.warning("[RESOLVER] ⚠️ Recovery failed — using full query as single comparison")
                    sub_queries = [{"query": standalone_query or query, "topic": "Comparison"}]
            log.info(f"[RESOLVER] Split into {len(sub_queries)} sub-queries")
        
        # Step 5: Find relevant groups based on dependency type
        # YOUR PIPELINE: For DEPENDENT queries, check if belongs to active group FIRST
        relevant_groups = []
        
        if dependency_type == DependencyType.DEPENDENT:
            if active_group_id and belongs_to_active:
                # ✅ BELONGS TO ACTIVE GROUP → Use it directly
                log.info(f"[RESOLVER] DEPENDENT query BELONGS to active group {active_group_id}")
                log.info(f"[RESOLVER] Using active group context directly (no new group)")
                relevant_groups = []  # Don't search, use active group
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
            # New conversation topic - ALWAYS create a new group, ignoring any passed active_group_id
            log.info("[RESOLVER] INDEPENDENT query — creating NEW group (new topic)")
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
            # Ambiguous query - create group to track clarification conversation
            log.info("[RESOLVER] AMBIGUOUS query — creating group for clarification attempts")
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
            elif not relevant_groups:
                # No relevant groups found — create new one
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
            llm_reasoning=reasoning,
            answer_source=answer_source,
        )
        
        log.info("[RESOLVER] ✅ Context resolution complete")
        return context
    
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
            # Generate query embedding
            embeddings = self.embedder.embed_texts([query])
            query_embedding = embeddings[0] if embeddings else None
            
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
        
        if summary_ready and group_context.get("summary"):
            # Summary is complete, use it (more concise)
            parts.append(f"\n**Summary:** {group_context['summary']}")
            log.info("[RESOLVER] Using prepared summary for context")
        else:
            # Summary not ready (being generated) or doesn't exist
            # Use all recent turns to provide complete context
            recent = group_context.get("recent_turns", [])
            if recent:
                log.info(f"[RESOLVER] Summary not ready, using {len(recent)} recent turns for context")
                parts.append("\n**Recent Discussion:**")
                for turn in recent:  # All recent turns, not just last 3
                    parts.append(f"- Q: {turn.query}")
                    parts.append(f"  A: {turn.memory_summary}")
            else:
                # Fallback: if no recent turns and no summary, provide minimal context
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
            "summary": RetrievalIntent.SUMMARY,
            "comparison": RetrievalIntent.COMPARISON,
            "extraction": RetrievalIntent.EXTRACTION,
            "analysis": RetrievalIntent.ANALYSIS,
            "ambiguous": RetrievalIntent.AMBIGUOUS,
        }
        return mapping.get(llm_result.lower(), RetrievalIntent.FACTUAL)
    

