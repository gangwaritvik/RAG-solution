"""High-level manager for conversation memory."""

from typing import Optional, List, Dict, Any
import uuid

from backend.utils.logger import get_logger
from .group_memory import GroupMemory, Group, ConversationTurn
from .memory_store import MemoryStore

log = get_logger("memory_manager")


class ConversationMemoryManager:
    """
    Manages conversational memory with:
    - In-memory GroupMemory for fast access
    - Chroma DB backed MemoryStore for persistence
    """
    
    def __init__(self, chroma_db_path: str):
        """
        Initialize memory manager.
        
        Args:
            chroma_db_path: Path to Chroma DB directory
        """
        log.info("[MEMORY_MGR] Initializing conversation memory manager")
        
        self.group_memory = GroupMemory()
        self.memory_store = MemoryStore(chroma_db_path)
        
        # Load existing groups from storage
        self._load_groups_from_storage()
        
        log.info("[MEMORY_MGR] ✅ Manager initialized")
    
    def _load_groups_from_storage(self) -> None:
        """Load all groups from persistent storage into memory."""
        try:
            groups = self.memory_store.list_groups()
            for group in groups:
                self.group_memory.groups[group.group_id] = group
            
            log.info(f"[MEMORY_MGR] ✅ Loaded {len(groups)} groups from storage")
        except Exception as e:
            log.error(f"[MEMORY_MGR] ❌ Failed to load groups: {e}", exc_info=True)
    
    def create_conversation_group(self, topic: str) -> Group:
        """
        Create a new conversation group.
        
        Args:
            topic: The topic/semantic group name
            
        Returns:
            The created Group object
        """
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        group = self.group_memory.create_group(group_id, topic)
        
        # Persist immediately
        self.memory_store.save_group(group)
        
        log.info(f"[MEMORY_MGR] ✅ Created group {group_id}: {topic}")
        return group
    
    def get_conversation_group(self, group_id: str) -> Optional[Group]:
        """
        Get a conversation group by ID.
        
        Args:
            group_id: The group ID
            
        Returns:
            Group object or None
        """
        return self.group_memory.get_group(group_id)
    
    def list_conversation_groups(self) -> List[Group]:
        """
        List all conversation groups.
        
        Returns:
            List of Group objects
        """
        return self.group_memory.list_groups()
    
    def add_conversation_turn(
        self, 
        group_id: str, 
        query: str, 
        memory_summary: str
    ) -> Optional[ConversationTurn]:
        """
        Add a turn to a conversation group.
        
        Args:
            group_id: The group ID
            query: The user's query
            memory_summary: Compressed summary for memory (not full answer)
            
        Returns:
            The created ConversationTurn or None if group not found
        """
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        turn = ConversationTurn(
            turn_id=turn_id,
            query=query,
            memory_summary=memory_summary,
        )
        
        if self.group_memory.add_turn_to_group(group_id, turn):
            # Persist the updated group
            group = self.group_memory.get_group(group_id)
            if group:
                self.memory_store.save_group(group)
            
            log.info(f"[MEMORY_MGR] ✅ Added turn {turn_id} to group {group_id}")
            return turn
        else:
            log.warning(f"[MEMORY_MGR] ⚠️ Group not found: {group_id}")
            return None
    
    def should_summarize_group(self, group_id: str, threshold: int = 5) -> bool:
        """
        Check if a group should be summarized.
        
        Args:
            group_id: The group ID
            threshold: Number of unsummarized turns before summarizing
            
        Returns:
            True if should summarize, False otherwise
        """
        return self.group_memory.should_summarize_group(group_id, threshold)
    
    def update_group_summary(
        self, 
        group_id: str, 
        summary: str,
        summary_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update a group's summary and optional embedding.
        
        Args:
            group_id: The group ID
            summary: The new summary text
            summary_embedding: Optional embedding vector for semantic search
            
        Returns:
            True if successful, False otherwise
        """
        if self.group_memory.update_group_summary(group_id, summary, summary_embedding):
            group = self.group_memory.get_group(group_id)
            if group:
                self.memory_store.save_group(group)
            
            log.info(f"[MEMORY_MGR] ✅ Updated summary for group {group_id}")
            return True
        else:
            log.warning(f"[MEMORY_MGR] ⚠️ Group not found: {group_id}")
            return False
    
    def search_groups_by_embedding(
        self, 
        query_embedding: List[float],
        similarity_threshold: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant groups by embedding similarity.
        
        Used to find which group a new query belongs to.
        
        Args:
            query_embedding: Embedding vector to search with
            similarity_threshold: Minimum similarity to include
            top_k: Maximum results to return
            
        Returns:
            List of dicts with group info and similarity scores
        """
        results = self.memory_store.search_groups_by_embedding(query_embedding, top_k)
        
        # Filter by threshold
        filtered = [r for r in results if r["similarity"] >= similarity_threshold]
        
        log.info(f"[MEMORY_MGR] ✅ Found {len(filtered)} relevant groups")
        return filtered
    
    def get_active_group(self) -> Optional[Group]:
        """
        Get the currently active conversation group.
        
        Returns:
            Group object or None
        """
        return self.group_memory.get_active_group()
    
    def set_active_group(self, group_id: str) -> bool:
        """
        Set the active conversation group.
        
        Args:
            group_id: The group ID to activate
            
        Returns:
            True if successful, False otherwise
        """
        return self.group_memory.set_active_group(group_id)
    
    def get_group_context(
        self, 
        group_id: str,
        include_all_turns: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get group context for answer generation.
        
        Args:
            group_id: The group ID
            include_all_turns: If True, include all historical turns
            
        Returns:
            Dict with summary and recent/all turns, or None
        """
        group = self.group_memory.get_group(group_id)
        if not group:
            return None
        
        return {
            "group_id": group.group_id,
            "topic": group.topic,
            "summary": group.summary,
            "summary_ready": group.summary_ready,
            "recent_turns": group.recent_turns,
            "all_turns": group.all_turns if include_all_turns else [],
        }
    
    def delete_group(self, group_id: str) -> bool:
        """
        Delete a conversation group.
        
        Args:
            group_id: The group ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Remove from memory
        if group_id in self.group_memory.groups:
            del self.group_memory.groups[group_id]
        
        # Remove from storage
        result = self.memory_store.delete_group(group_id)
        
        if result:
            log.info(f"[MEMORY_MGR] ✅ Deleted group {group_id}")
        
        return result
    
    def clear_all(self) -> bool:
        """
        Clear all conversation memory (testing/reset).
        
        Returns:
            True if successful, False otherwise
        """
        self.group_memory.clear_all()
        result = self.memory_store.clear_all()
        
        if result:
            log.info("[MEMORY_MGR] ✅ All conversation memory cleared")
        
        return result
