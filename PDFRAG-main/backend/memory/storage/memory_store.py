"""Chroma DB backed persistent storage for group memory."""

import chromadb
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from backend.utils.logger import get_logger
from .group_memory import Group, ConversationTurn

log = get_logger("memory_store")


class MemoryStore:
    """Persistent storage for conversation groups using Chroma DB."""
    
    METADATA_COLLECTION = "conversation_memory"
    SUMMARY_COLLECTION = "group_summaries"
    
    def __init__(self, chroma_db_path: str):
        """
        Initialize memory store with Chroma DB.
        
        Args:
            chroma_db_path: Path to Chroma DB directory (reuses existing)
        """
        log.info(f"[MEMORY_STORE] Initializing with Chroma DB at {chroma_db_path}")
        
        try:
            self.client = chromadb.PersistentClient(path=chroma_db_path)
            
            # Collection for group metadata (serialized JSON)
            self.metadata_collection = self.client.get_or_create_collection(
                name=self.METADATA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            
            # Collection for group summary embeddings (for semantic search)
            self.summary_collection = self.client.get_or_create_collection(
                name=self.SUMMARY_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            
            log.info(
                f"[MEMORY_STORE] ✅ Collections loaded — "
                f"metadata: {self.metadata_collection.count()}, "
                f"summaries: {self.summary_collection.count()}"
            )
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to initialize — "
                f"{type(e).__name__}: {e}", 
                exc_info=True
            )
            raise
    
    def save_group(self, group: Group) -> bool:
        """
        Save or update a group in storage.
        
        Args:
            group: Group to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group_id = group.group_id
            group_data = group.to_dict()
            
            # Serialize group data as JSON
            group_json = json.dumps(group_data, default=str)
            
            # Check if already exists
            existing = self.metadata_collection.get(
                ids=[group_id],
                include=[]
            )
            
            if existing["ids"]:
                # Update existing
                self.metadata_collection.update(
                    ids=[group_id],
                    documents=[group_json],
                    metadatas=[{
                        "group_id": group_id,
                        "topic": group.topic,
                        "updated_at": group.updated_at,
                    }],
                )
                log.info(f"[MEMORY_STORE] ✅ Updated group: {group_id}")
            else:
                # Create new
                self.metadata_collection.add(
                    ids=[group_id],
                    documents=[group_json],
                    metadatas=[{
                        "group_id": group_id,
                        "topic": group.topic,
                        "updated_at": group.updated_at,
                    }],
                )
                log.info(f"[MEMORY_STORE] ✅ Created group: {group_id}")
            
            # If group has summary embedding, save it separately
            if group.summary_embedding:
                self._save_summary_embedding(group)
            
            return True
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to save group {group.group_id} — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return False
    
    def get_group(self, group_id: str) -> Optional[Group]:
        """
        Retrieve a group by ID.
        
        Args:
            group_id: The group ID
            
        Returns:
            Group object if found, None otherwise
        """
        try:
            result = self.metadata_collection.get(
                ids=[group_id],
                include=["documents"]
            )
            
            if not result["ids"]:
                log.warning(f"[MEMORY_STORE] Group not found: {group_id}")
                return None
            
            group_json = result["documents"][0]
            group_data = json.loads(group_json)
            group = Group.from_dict(group_data)
            
            log.info(f"[MEMORY_STORE] ✅ Retrieved group: {group_id}")
            return group
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to get group {group_id} — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return None
    
    def list_groups(self) -> List[Group]:
        """
        Get all groups from storage.
        
        Returns:
            List of Group objects
        """
        try:
            result = self.metadata_collection.get(include=["documents"])
            
            groups = []
            for group_json in result["documents"]:
                try:
                    group_data = json.loads(group_json)
                    group = Group.from_dict(group_data)
                    groups.append(group)
                except Exception as e:
                    log.warning(f"[MEMORY_STORE] Failed to parse group data: {e}")
                    continue
            
            log.info(f"[MEMORY_STORE] ✅ Retrieved {len(groups)} groups")
            return groups
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to list groups — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return []
    
    def delete_group(self, group_id: str) -> bool:
        """
        Delete a group from storage.
        
        Args:
            group_id: The group ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete metadata
            self.metadata_collection.delete(ids=[group_id])
            
            # Delete summary embedding if exists
            self.summary_collection.delete(ids=[group_id])
            
            log.info(f"[MEMORY_STORE] ✅ Deleted group: {group_id}")
            return True
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to delete group {group_id} — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return False
    
    def search_groups_by_embedding(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for groups by summary embedding similarity.
        
        Args:
            query_embedding: Embedding vector to search with
            top_k: Number of top results to return
            
        Returns:
            List of dicts with group info and similarity scores
        """
        try:
            if self.summary_collection.count() == 0:
                log.warning("[MEMORY_STORE] No group summaries indexed yet")
                return []
            
            results = self.summary_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.summary_collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            
            hits = []
            for i in range(len(results["ids"][0])):
                group_id = results["ids"][0][i]
                meta = results["metadatas"][0][i]
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                
                # Retrieve full group
                group = self.get_group(group_id)
                if group:
                    hits.append({
                        "group": group,
                        "group_id": group_id,
                        "topic": group.topic,
                        "similarity": round(score, 4),
                    })
            
            log.info(
                f"[MEMORY_STORE] ✅ Search returned {len(hits)} results | "
                f"top score: {hits[0]['similarity'] if hits else 'N/A'}"
            )
            return hits
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Search failed — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return []
    
    def _save_summary_embedding(self, group: Group) -> bool:
        """
        Save a group's summary embedding to the summary collection.
        
        Args:
            group: Group with summary embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            group_id = group.group_id
            
            # Check if already exists
            existing = self.summary_collection.get(
                ids=[group_id],
                include=[]
            )
            
            if existing["ids"]:
                # Update
                self.summary_collection.update(
                    ids=[group_id],
                    embeddings=[group.summary_embedding],
                    documents=[group.summary],
                    metadatas=[{
                        "group_id": group_id,
                        "topic": group.topic,
                    }],
                )
            else:
                # Add new
                self.summary_collection.add(
                    ids=[group_id],
                    embeddings=[group.summary_embedding],
                    documents=[group.summary],
                    metadatas=[{
                        "group_id": group_id,
                        "topic": group.topic,
                    }],
                )
            
            log.info(f"[MEMORY_STORE] ✅ Saved summary embedding for: {group_id}")
            return True
            
        except Exception as e:
            log.error(
                f"[MEMORY_STORE] ❌ Failed to save summary embedding — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return False
