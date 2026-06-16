"""Context Resolution Layer for query classification and processing."""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from backend.utils.logger import get_logger

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


@dataclass
class QueryContext:
    """Processed query with all context information."""
    
    original_query: str
    standalone_query: str              # Retrieval-ready version
    dependency_type: DependencyType
    retrieval_intent: RetrievalIntent
    active_group_id: Optional[str]    # Current group if DEPENDENT
    relevant_groups: List[Dict[str, Any]]  # Groups found via similarity search
    memory_context: Optional[str]      # Loaded from group(s)
    needs_group_context: bool


class ContextResolver:
    """Resolves query context before retrieval and generation."""
    
    def __init__(self, memory_manager, embedder):
        """
        Initialize context resolver.
        
        Args:
            memory_manager: ConversationMemoryManager instance
            embedder: Embedder instance for generating query embeddings
        """
        self.memory_manager = memory_manager
        self.embedder = embedder
    
    def resolve(
        self, 
        query: str,
        active_group_id: Optional[str] = None
    ) -> QueryContext:
        """
        Resolve complete context for a query.
        
        Args:
            query: User's input query
            active_group_id: Currently active group ID if any
            
        Returns:
            QueryContext with all processing done
        """
        log.info(f"[RESOLVER] Processing query: {query[:80]}")
        
        # Step 1: Classify dependency type
        dependency_type = self._classify_dependency(query, active_group_id)
        log.info(f"[RESOLVER] Dependency type: {dependency_type.value}")
        
        # Step 2: Generate standalone query
        standalone_query = self._generate_standalone_query(
            query, 
            dependency_type, 
            active_group_id
        )
        log.info(f"[RESOLVER] Standalone query: {standalone_query[:80]}")
        
        # Step 3: Classify retrieval intent
        retrieval_intent = self._classify_intent(query)
        log.info(f"[RESOLVER] Retrieval intent: {retrieval_intent.value}")
        
        # Step 4: Find relevant groups
        relevant_groups = self._find_relevant_groups(standalone_query)
        log.info(f"[RESOLVER] Found {len(relevant_groups)} relevant groups")
        
        # Step 5: Load memory context
        memory_context = self._load_memory_context(
            dependency_type,
            active_group_id,
            relevant_groups
        )
        
        # Determine if group context is needed
        needs_group_context = dependency_type in [
            DependencyType.DEPENDENT,
            DependencyType.MULTI_GROUP
        ]
        
        context = QueryContext(
            original_query=query,
            standalone_query=standalone_query,
            dependency_type=dependency_type,
            retrieval_intent=retrieval_intent,
            active_group_id=active_group_id,
            relevant_groups=relevant_groups,
            memory_context=memory_context,
            needs_group_context=needs_group_context,
        )
        
        log.info("[RESOLVER] ✅ Context resolution complete")
        return context
    
    def _classify_dependency(
        self, 
        query: str, 
        active_group_id: Optional[str]
    ) -> DependencyType:
        """
        Classify query dependency type.
        
        Uses heuristics:
        - Contains pronouns (this, that, it) → likely DEPENDENT
        - References previous topic → DEPENDENT
        - Compares/relates concepts → MULTI_GROUP
        - Clear standalone question → INDEPENDENT
        """
        query_lower = query.lower()
        
        # Check for dependency indicators
        dependent_keywords = [
            "this", "that", "these", "those", "it", "its",
            "above", "following", "previous", "earlier",
            "same", "another", "other", "also",
            "too", "again", "as mentioned", "mentioned above"
        ]
        
        comparison_keywords = [
            "compare", "versus", "vs", "difference", "similar",
            "between", "both", "either", "neither",
            "more than", "less than", "while"
        ]
        
        # Check for comparison (multi-group)
        if any(kw in query_lower for kw in comparison_keywords):
            return DependencyType.MULTI_GROUP
        
        # Check for dependency on previous context
        if active_group_id and any(kw in query_lower for kw in dependent_keywords):
            return DependencyType.DEPENDENT
        
        # Check if query is standalone (starts with question words)
        question_words = ["what", "when", "where", "who", "why", "how"]
        if query_lower.startswith(tuple(question_words)):
            return DependencyType.INDEPENDENT
        
        # Default based on active group
        if active_group_id:
            return DependencyType.AMBIGUOUS
        
        return DependencyType.INDEPENDENT
    
    def _generate_standalone_query(
        self,
        query: str,
        dependency_type: DependencyType,
        active_group_id: Optional[str]
    ) -> str:
        """
        Generate retrieval-ready standalone query.
        
        Converts ambiguous follow-ups like "Any exemptions?" into
        "Are there startup or MSME exemptions under eligibility requirements?"
        """
        if dependency_type == DependencyType.INDEPENDENT:
            # Already standalone
            return query
        
        if dependency_type == DependencyType.DEPENDENT and active_group_id:
            # Augment with context from active group
            group = self.memory_manager.get_conversation_group(active_group_id)
            if group:
                # Expand query with group topic context
                expanded = f"{query} (regarding {group.topic})"
                log.info(f"[RESOLVER] Expanded query: {expanded}")
                return expanded
        
        # For MULTI_GROUP and AMBIGUOUS, return as-is
        # Could use LLM to expand in future
        return query
    
    def _classify_intent(self, query: str) -> RetrievalIntent:
        """
        Classify query retrieval intent.
        
        Uses keywords to determine retrieval strategy:
        - "What/Where/Which" → FACTUAL
        - "Summarize/Overview" → SUMMARY
        - "Compare/Difference" → COMPARISON
        - "List/All/Every" → EXTRACTION
        - "Why/How/Risk/Impact" → ANALYSIS
        """
        query_lower = query.lower()
        
        # Check for EXTRACTION
        extraction_keywords = ["list", "all", "every", "each", "enumerate", "provide all"]
        if any(kw in query_lower for kw in extraction_keywords):
            return RetrievalIntent.EXTRACTION
        
        # Check for SUMMARY
        summary_keywords = ["summarize", "overview", "summary", "outline", "brief"]
        if any(kw in query_lower for kw in summary_keywords):
            return RetrievalIntent.SUMMARY
        
        # Check for COMPARISON
        comparison_keywords = ["compare", "versus", "vs", "difference", "similar", "between"]
        if any(kw in query_lower for kw in comparison_keywords):
            return RetrievalIntent.COMPARISON
        
        # Check for ANALYSIS
        analysis_keywords = ["why", "how", "risk", "impact", "cause", "effect", "implication"]
        if any(kw in query_lower for kw in analysis_keywords):
            return RetrievalIntent.ANALYSIS
        
        # Default to FACTUAL
        return RetrievalIntent.FACTUAL
    
    def _find_relevant_groups(
        self, 
        query: str,
        similarity_threshold: float = 0.5,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find relevant conversation groups by embedding similarity.
        
        Args:
            query: The standalone query
            similarity_threshold: Minimum similarity score
            top_k: Maximum groups to return
            
        Returns:
            List of relevant groups with similarity scores
        """
        try:
            # Generate query embedding
            embeddings = self.embedder.embed_texts([query])
            query_embedding = embeddings[0] if embeddings else None
            
            if not query_embedding:
                log.warning("[RESOLVER] Failed to embed query")
                return []
            
            # Search groups by embedding
            results = self.memory_manager.search_groups_by_embedding(
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
            
            log.info(f"[RESOLVER] Found {len(results)} relevant groups")
            return results
            
        except Exception as e:
            log.error(f"[RESOLVER] Failed to find groups: {e}", exc_info=True)
            return []
    
    def _load_memory_context(
        self,
        dependency_type: DependencyType,
        active_group_id: Optional[str],
        relevant_groups: List[Dict[str, Any]]
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
            # Load all relevant group summaries
            for group_info in relevant_groups:
                group = group_info.get("group")
                if group and group.summary:
                    context_parts.append(
                        f"## {group.topic}\n{group.summary}"
                    )
        
        elif dependency_type == DependencyType.AMBIGUOUS:
            # Try active group, fall back to relevant groups
            if active_group_id:
                group_context = self.memory_manager.get_group_context(active_group_id)
                if group_context:
                    context_parts.append(self._format_group_context(group_context))
            
            # Also include relevant groups
            for group_info in relevant_groups:
                group = group_info.get("group")
                if group and group.summary:
                    context_parts.append(
                        f"## {group.topic}\n{group.summary}"
                    )
        
        if context_parts:
            return "\n\n".join(context_parts)
        
        return None
    
    def _format_group_context(self, group_context: Dict[str, Any]) -> str:
        """Format group context for LLM."""
        parts = [f"## {group_context['topic']}"]
        
        if group_context.get("summary"):
            parts.append(f"\n**Summary:** {group_context['summary']}")
        
        recent = group_context.get("recent_turns", [])
        if recent:
            parts.append("\n**Recent Discussion:**")
            for turn in recent[-3:]:  # Last 3 turns
                parts.append(f"- Q: {turn.query}")
                parts.append(f"  A: {turn.memory_summary}")
        
        return "\n".join(parts)
