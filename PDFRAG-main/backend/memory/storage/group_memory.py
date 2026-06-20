"""Core group memory data structures and operations."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class ConversationTurn:
    """Single turn in a conversation group."""
    
    turn_id: str
    query: str
    memory_summary: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    dependency_type: Optional[str] = None  # independent, dependent, multi_group, ambiguous
    retrieval_intent: Optional[str] = None  # factual, summary, comparison, extraction, analysis, ambiguous
    full_answer: str = ""  # Complete answer text (used to resolve follow-ups like "summarize the above")
    restrict_filenames: Optional[List[str]] = None  # File pin used for this turn (for follow-up carry-forward)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            turn_id=data["turn_id"],
            query=data["query"],
            memory_summary=data["memory_summary"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            dependency_type=data.get("dependency_type"),
            retrieval_intent=data.get("retrieval_intent"),
            full_answer=data.get("full_answer", ""),
            restrict_filenames=data.get("restrict_filenames"),
        )


@dataclass
class Group:
    """Memory group representing a semantic topic in conversation."""
    
    group_id: str
    topic: str
    summary: str = ""
    summary_embedding: Optional[List[float]] = None
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    all_turns: List[ConversationTurn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    summary_ready: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "group_id": self.group_id,
            "topic": self.topic,
            "summary": self.summary,
            "summary_embedding": self.summary_embedding,
            "recent_turns": [t.to_dict() for t in self.recent_turns],
            "all_turns": [t.to_dict() for t in self.all_turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary_ready": self.summary_ready,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Group":
        """Create from dictionary."""
        recent_turns = [ConversationTurn.from_dict(t) for t in data.get("recent_turns", [])]
        all_turns = [ConversationTurn.from_dict(t) for t in data.get("all_turns", [])]
        
        return cls(
            group_id=data["group_id"],
            topic=data["topic"],
            summary=data.get("summary", ""),
            summary_embedding=data.get("summary_embedding"),
            recent_turns=recent_turns,
            all_turns=all_turns,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            summary_ready=data.get("summary_ready", False),
        )
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add turn to both recent and all turns."""
        self.recent_turns.append(turn)
        self.all_turns.append(turn)
        self.updated_at = datetime.utcnow().isoformat()
    
    def unsummarized_turn_count(self) -> int:
        """Count turns since last summary (or all if never summarized)."""
        return len(self.recent_turns)
    
    def clear_recent_turns(self) -> None:
        """Clear recent turns after summarization."""
        self.recent_turns.clear()


class GroupMemory:
    """In-memory group memory manager."""
    
    def __init__(self):
        """Initialize group memory."""
        self.groups: Dict[str, Group] = {}
        self.active_group_id: Optional[str] = None
    
    def create_group(self, group_id: str, topic: str) -> Group:
        """Create a new conversation group."""
        group = Group(group_id=group_id, topic=topic)
        self.groups[group_id] = group
        if self.active_group_id is None:
            self.active_group_id = group_id
        return group
    
    def get_group(self, group_id: str) -> Optional[Group]:
        """Retrieve a group by ID."""
        return self.groups.get(group_id)
    
    def get_active_group(self) -> Optional[Group]:
        """Get the currently active group."""
        if self.active_group_id:
            return self.groups.get(self.active_group_id)
        return None
    
    def set_active_group(self, group_id: str) -> bool:
        """Set active group if it exists."""
        if group_id in self.groups:
            self.active_group_id = group_id
            return True
        return False
    
    def list_groups(self) -> List[Group]:
        """Get all groups."""
        return list(self.groups.values())
    
    def add_turn_to_group(self, group_id: str, turn: ConversationTurn) -> bool:
        """Add a turn to a specific group."""
        group = self.get_group(group_id)
        if group:
            group.add_turn(turn)
            return True
        return False
    
    def update_group_summary(
        self, 
        group_id: str, 
        summary: str, 
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update group summary and optionally its embedding."""
        group = self.get_group(group_id)
        if group:
            group.summary = summary
            if embedding:
                group.summary_embedding = embedding
            group.summary_ready = True
            group.clear_recent_turns()
            group.updated_at = datetime.utcnow().isoformat()
            return True
        return False
    
    def should_summarize_group(self, group_id: str, threshold: int = 5) -> bool:
        """Check if group should be summarized."""
        group = self.get_group(group_id)
        if group:
            return group.unsummarized_turn_count() >= threshold
        return False
    
    def get_groups_by_topic(self, topic_substring: str) -> List[Group]:
        """Find groups by topic substring."""
        return [
            g for g in self.groups.values() 
            if topic_substring.lower() in g.topic.lower()
        ]
    
    def clear_all(self) -> None:
        """Clear all groups (for testing/reset)."""
        self.groups.clear()
        self.active_group_id = None
