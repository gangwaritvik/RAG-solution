"""Memory module - Group-based conversational memory system."""

# Storage layer (persistence)
from backend.memory.storage import (
    Group,
    ConversationTurn,
    GroupMemory,
    MemoryStore
)

# Management layer (high-level orchestration)
from backend.memory.conversation_manager import ConversationMemoryManager

# Resolution layer (query context resolution)
from backend.memory.context_resolver import (
    ContextResolver,
    QueryContext,
    DependencyType,
    RetrievalIntent
)

# Classification layer (LLM-based classification)
from backend.memory.llm_classifier import LLMClassifier

__all__ = [
    # Storage
    "Group",
    "ConversationTurn",
    "GroupMemory",
    "MemoryStore",
    # Management
    "ConversationMemoryManager",
    # Resolution
    "ContextResolver",
    "QueryContext",
    "DependencyType",
    "RetrievalIntent",
    # Classification
    "LLMClassifier",
]
