"""Memory module - Group-based conversational memory system."""

# Storage layer
from backend.memory.storage import (
    Group,
    ConversationTurn,
    GroupMemory,
    MemoryStore
)

# Management layer
from backend.memory.management import ConversationMemoryManager

# Resolution layer
from backend.memory.resolution import (
    ContextResolver,
    QueryContext,
    DependencyType,
    RetrievalIntent
)

# Classification layer
from backend.memory.classifiers import LLMClassifier

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
