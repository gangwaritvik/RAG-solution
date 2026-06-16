"""Dependency type classification for queries."""

from enum import Enum
from typing import Optional

from backend.utils.logger import get_logger

log = get_logger("dependency_classifier")


class DependencyType(Enum):
    """Query dependency classification."""
    INDEPENDENT = "independent"      # Standalone query, no context needed
    DEPENDENT = "dependent"           # Follow-up, needs active group context
    MULTI_GROUP = "multi_group"       # Compares/relates multiple groups
    AMBIGUOUS = "ambiguous"           # Unclear, might need clarification


class DependencyClassifier:
    """Classifies query dependency type."""
    
    # Keywords indicating dependency on previous context
    DEPENDENT_KEYWORDS = [
        "this", "that", "these", "those", "it", "its",
        "above", "following", "previous", "earlier",
        "same", "another", "other", "also",
        "too", "again", "as mentioned", "mentioned above",
        "the mentioned", "that said", "furthermore",
    ]
    
    # Keywords indicating multi-group comparison
    COMPARISON_KEYWORDS = [
        "compare", "versus", "vs", "difference", "similar",
        "between", "both", "either", "neither",
        "more than", "less than", "while", "unlike",
        "differ", "comparison", "contrast", "relation",
    ]
    
    # Keywords indicating standalone questions
    QUESTION_WORDS = [
        "what", "when", "where", "who", "why", "how",
        "is", "are", "can", "could", "will", "would",
        "should", "may", "might", "does", "did", "do"
    ]
    
    @classmethod
    def classify(
        cls,
        query: str,
        active_group_id: Optional[str] = None
    ) -> DependencyType:
        """
        Classify query dependency type.
        
        Args:
            query: User's input query
            active_group_id: Currently active group ID if any
            
        Returns:
            DependencyType classification
        """
        query_lower = query.lower().strip()
        
        log.info(f"[DEPENDENCY] Classifying: {query[:80]}")
        
        # Priority 1: Check for multi-group comparison
        if cls._is_comparison(query_lower):
            log.info("[DEPENDENCY] ✅ Classified as: MULTI_GROUP")
            return DependencyType.MULTI_GROUP
        
        # Priority 2: Check for dependency indicators
        if active_group_id and cls._is_dependent(query_lower):
            log.info("[DEPENDENCY] ✅ Classified as: DEPENDENT")
            return DependencyType.DEPENDENT
        
        # Priority 3: Check if standalone question
        if cls._is_independent(query_lower):
            log.info("[DEPENDENCY] ✅ Classified as: INDEPENDENT")
            return DependencyType.INDEPENDENT
        
        # Default based on active group
        if active_group_id:
            log.info("[DEPENDENCY] ✅ Classified as: AMBIGUOUS (active group exists)")
            return DependencyType.AMBIGUOUS
        
        log.info("[DEPENDENCY] ✅ Classified as: INDEPENDENT (no active group)")
        return DependencyType.INDEPENDENT
    
    @classmethod
    def _is_comparison(cls, query_lower: str) -> bool:
        """Check if query is a comparison (multi-group)."""
        return any(kw in query_lower for kw in cls.COMPARISON_KEYWORDS)
    
    @classmethod
    def _is_dependent(cls, query_lower: str) -> bool:
        """Check if query depends on previous context."""
        return any(kw in query_lower for kw in cls.DEPENDENT_KEYWORDS)
    
    @classmethod
    def _is_independent(cls, query_lower: str) -> bool:
        """Check if query is standalone."""
        # Check if starts with question word
        for word in cls.QUESTION_WORDS:
            if query_lower.startswith(word):
                return True
        return False
