"""Retrieval intent classification for queries."""

from enum import Enum

from backend.utils.logger import get_logger

log = get_logger("intent_classifier")


class RetrievalIntent(Enum):
    """Query intent classification for retrieval strategy."""
    FACTUAL = "factual"              # What is X? → Top relevant chunks
    SUMMARY = "summary"               # Summarize X → Broader coverage
    COMPARISON = "comparison"         # Compare X and Y → Multi-group retrieval
    EXTRACTION = "extraction"         # List all X → Aggregate information
    ANALYSIS = "analysis"             # Why/how/risks → Cross-section reasoning


class IntentClassifier:
    """Classifies query retrieval intent."""
    
    # Keywords for EXTRACTION intent
    EXTRACTION_KEYWORDS = [
        "list", "all", "every", "each", "enumerate",
        "provide all", "give all", "show all", "include all",
        "complete list", "full list", "all types", "all items",
        "entire", "comprehensive", "everything", "whole"
    ]
    
    # Keywords for SUMMARY intent
    SUMMARY_KEYWORDS = [
        "summarize", "overview", "summary", "outline",
        "brief", "general", "overall", "gist",
        "main points", "key points", "highlights",
        "condensed", "abridged", "short version",
        "tell me about", "explain", "describe"
    ]
    
    # Keywords for COMPARISON intent
    COMPARISON_KEYWORDS = [
        "compare", "versus", "vs", "difference", "similar",
        "between", "both", "either", "neither",
        "more than", "less than", "while", "unlike",
        "differ", "comparison", "contrast", "relation",
        "same as", "different from"
    ]
    
    # Keywords for ANALYSIS intent
    ANALYSIS_KEYWORDS = [
        "why", "how", "risk", "impact", "cause",
        "effect", "implication", "reason", "purpose",
        "mechanism", "process", "consequence", "benefit",
        "drawback", "advantage", "disadvantage",
        "analysis", "analyze", "examine", "investigate"
    ]
    
    # Keywords for FACTUAL intent (default)
    FACTUAL_KEYWORDS = [
        "what", "which", "where", "when", "who",
        "is", "are", "does", "do", "can", "could",
        "define", "definition", "specify", "requirement"
    ]
    
    @classmethod
    def classify(cls, query: str) -> RetrievalIntent:
        """
        Classify query retrieval intent.
        
        Args:
            query: User's input query
            
        Returns:
            RetrievalIntent classification
        """
        query_lower = query.lower().strip()
        
        log.info(f"[INTENT] Classifying: {query[:80]}")
        
        # Priority order: more specific intents first
        
        # Check for EXTRACTION (highest priority - very specific)
        if cls._is_extraction(query_lower):
            log.info("[INTENT] ✅ Classified as: EXTRACTION")
            return RetrievalIntent.EXTRACTION
        
        # Check for SUMMARY
        if cls._is_summary(query_lower):
            log.info("[INTENT] ✅ Classified as: SUMMARY")
            return RetrievalIntent.SUMMARY
        
        # Check for COMPARISON
        if cls._is_comparison(query_lower):
            log.info("[INTENT] ✅ Classified as: COMPARISON")
            return RetrievalIntent.COMPARISON
        
        # Check for ANALYSIS
        if cls._is_analysis(query_lower):
            log.info("[INTENT] ✅ Classified as: ANALYSIS")
            return RetrievalIntent.ANALYSIS
        
        # Default to FACTUAL
        log.info("[INTENT] ✅ Classified as: FACTUAL (default)")
        return RetrievalIntent.FACTUAL
    
    @classmethod
    def _is_extraction(cls, query_lower: str) -> bool:
        """Check if query is extraction intent."""
        return any(kw in query_lower for kw in cls.EXTRACTION_KEYWORDS)
    
    @classmethod
    def _is_summary(cls, query_lower: str) -> bool:
        """Check if query is summary intent."""
        return any(kw in query_lower for kw in cls.SUMMARY_KEYWORDS)
    
    @classmethod
    def _is_comparison(cls, query_lower: str) -> bool:
        """Check if query is comparison intent."""
        return any(kw in query_lower for kw in cls.COMPARISON_KEYWORDS)
    
    @classmethod
    def _is_analysis(cls, query_lower: str) -> bool:
        """Check if query is analysis intent."""
        return any(kw in query_lower for kw in cls.ANALYSIS_KEYWORDS)
    
    @classmethod
    def get_strategy(cls, intent: RetrievalIntent) -> dict:
        """
        Get retrieval strategy for intent.
        
        Returns dict with parameters for retrieval optimization.
        """
        strategies = {
            RetrievalIntent.FACTUAL: {
                "top_k": 5,
                "description": "Return top relevant chunks",
            },
            RetrievalIntent.SUMMARY: {
                "top_k": 10,
                "description": "Broader coverage for overview",
            },
            RetrievalIntent.COMPARISON: {
                "top_k": 8,
                "description": "Multiple groups and perspectives",
            },
            RetrievalIntent.EXTRACTION: {
                "top_k": 15,
                "description": "Comprehensive results for list",
            },
            RetrievalIntent.ANALYSIS: {
                "top_k": 10,
                "description": "Cross-section reasoning chunks",
            },
        }
        return strategies.get(intent, strategies[RetrievalIntent.FACTUAL])
