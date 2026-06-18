"""Background summarization for conversation groups."""

import threading
from typing import Optional, Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.utils.logger import get_logger

log = get_logger("background_summarizer")


class BackgroundSummarizer:
    """
    Handles background summarization of conversation groups.
    
    Runs summarization in background threads without blocking query responses.
    Triggered when groups reach 5+ unsummarized turns.
    """
    
    def __init__(self, memory_manager, generator, embedder, max_workers: int = 3):
        """
        Initialize background summarizer.
        
        Args:
            memory_manager: ConversationMemoryManager instance
            generator: Generator instance for LLM summarization
            embedder: Embedder instance for generating embeddings
            max_workers: Max concurrent summarization threads (default 3)
        """
        self.memory_manager = memory_manager
        self.generator = generator
        self.embedder = embedder
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_summarizations: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        
        log.info(f"[BG_SUMMARIZER] Initialized with {max_workers} workers")
    
    def summarize_if_needed(
        self,
        group_id: str,
        threshold: int = 5,
        callback: Optional[Callable[[str, bool], None]] = None
    ) -> bool:
        """
        Check if group needs summarization and trigger if needed.
        
        Non-blocking: Returns immediately after submitting task.
        
        Args:
            group_id: The group to check
            threshold: Number of unsummarized turns before summarizing (default 5)
            callback: Optional callback(group_id, success) called after summarization
            
        Returns:
            True if summarization was triggered, False if not needed
        """
        # Check if group needs summarization
        if not self.memory_manager.should_summarize_group(group_id, threshold):
            return False
        
        # Check if already summarizing this group
        with self._lock:
            if group_id in self.active_summarizations:
                log.info(f"[BG_SUMMARIZER] Group {group_id} already being summarized")
                return False
        
        # Trigger background summarization
        log.info(f"[BG_SUMMARIZER] Triggering summarization for group {group_id}")
        self._submit_summarization(group_id, callback)
        return True
    
    def _submit_summarization(
        self,
        group_id: str,
        callback: Optional[Callable[[str, bool], None]] = None
    ) -> None:
        """
        Submit summarization task to background executor.
        
        Args:
            group_id: The group to summarize
            callback: Optional callback function
        """
        future = self.executor.submit(
            self._summarize_group_worker,
            group_id,
            callback
        )
        
        # Track active summarization
        with self._lock:
            self.active_summarizations[group_id] = threading.current_thread()
        
        log.info(f"[BG_SUMMARIZER] [BACKGROUND] Submitted summarization task for {group_id}")
    
    def _summarize_group_worker(
        self,
        group_id: str,
        callback: Optional[Callable[[str, bool], None]] = None
    ) -> bool:
        """
        Worker function that runs in background thread.
        
        Args:
            group_id: The group to summarize
            callback: Optional callback function
            
        Returns:
            True if successful, False otherwise
        """
        success = False
        try:
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] ▶ Summarizing group {group_id}")
            
            # Get group
            group = self.memory_manager.get_conversation_group(group_id)
            if not group:
                log.warning(f"[BG_SUMMARIZER] [BACKGROUND] ⚠️ Group not found: {group_id}")
                return False
            
            # Get unsummarized turns
            recent_turns = group.recent_turns
            if not recent_turns:
                log.warning(f"[BG_SUMMARIZER] [BACKGROUND] ⚠️ No recent turns to summarize")
                return False
            
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Found {len(recent_turns)} unsummarized turns")
            
            # Build summarization context
            turns_text = self._format_turns_for_summarization(recent_turns)
            
            # Generate summary via LLM
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Calling LLM to summarize...")
            summary = self._generate_summary_from_turns(
                group.topic,
                group.summary,  # Existing summary for context
                turns_text
            )
            
            if not summary:
                log.error(f"[BG_SUMMARIZER] [BACKGROUND] ❌ Failed to generate summary")
                return False
            
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Generated summary ({len(summary)} chars)")
            
            # Generate embedding for new summary
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Generating embedding...")
            embeddings = self.embedder.embed_texts([summary])
            summary_embedding = embeddings[0] if embeddings else None
            
            if not summary_embedding:
                log.error(f"[BG_SUMMARIZER] [BACKGROUND] ❌ Failed to generate embedding")
                return False
            
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Generated embedding vector")
            
            # Update group with new summary and embedding
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Updating group {group_id}...")
            success = self.memory_manager.update_group_summary(
                group_id=group_id,
                summary=summary,
                summary_embedding=summary_embedding
            )
            
            if success:
                log.info(f"[BG_SUMMARIZER] [BACKGROUND] ✅ Group {group_id} summarized successfully")
                log.info(f"[BG_SUMMARIZER] [BACKGROUND]   Summary: {summary[:100]}...")
            else:
                log.error(f"[BG_SUMMARIZER] [BACKGROUND] ❌ Failed to update group")
            
            return success
            
        except Exception as e:
            log.error(
                f"[BG_SUMMARIZER] [BACKGROUND] ❌ Error summarizing {group_id} — "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            return False
        
        finally:
            # Remove from active summarizations
            with self._lock:
                if group_id in self.active_summarizations:
                    del self.active_summarizations[group_id]
            
            # Call callback if provided
            if callback:
                try:
                    callback(group_id, success)
                except Exception as e:
                    log.error(f"[BG_SUMMARIZER] ❌ Callback error: {e}", exc_info=True)
    
    def _format_turns_for_summarization(self, turns: List[Any]) -> str:
        """
        Format conversation turns for summarization prompt.
        
        Args:
            turns: List of ConversationTurn objects
            
        Returns:
            Formatted string for LLM
        """
        lines = []
        for i, turn in enumerate(turns, 1):
            lines.append(f"Q{i}: {turn.query}")
            lines.append(f"A{i}: {turn.memory_summary}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_summary_from_turns(
        self,
        topic: str,
        existing_summary: Optional[str],
        turns_text: str
    ) -> Optional[str]:
        """
        Generate a comprehensive summary from recent turns.
        
        Args:
            topic: Group topic
            existing_summary: Previous summary if any
            turns_text: Formatted recent turns
            
        Returns:
            Generated summary or None if failed
        """
        try:
            # Build prompt
            prompt_parts = [
                f"You are summarizing a conversation about: {topic}",
                "",
                "Recent conversation turns:",
                turns_text,
            ]
            
            if existing_summary:
                prompt_parts.insert(2, f"Previous summary: {existing_summary}")
            
            prompt = "\n".join(prompt_parts)
            prompt += "\n\nCreate a concise summary (2-3 sentences) of this conversation, combining the previous summary with new information from recent turns."
            
            # Call LLM for summary
            from openai import AzureOpenAI
            from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION
            
            client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
            )
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise summarization expert. Create clear, factual summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Low temp for consistency
            )
            
            summary = response.choices[0].message.content.strip()
            return summary if summary else None
            
        except Exception as e:
            log.error(f"[BG_SUMMARIZER] Failed to generate summary: {e}", exc_info=True)
            return None
    
    def get_active_summarizations(self) -> List[str]:
        """
        Get list of groups currently being summarized.
        
        Returns:
            List of group IDs
        """
        with self._lock:
            return list(self.active_summarizations.keys())
    
    def is_summarizing(self, group_id: str) -> bool:
        """
        Check if a specific group is being summarized.
        
        Args:
            group_id: The group ID to check
            
        Returns:
            True if currently summarizing, False otherwise
        """
        with self._lock:
            return group_id in self.active_summarizations
    
    def shutdown(self) -> None:
        """
        Shutdown the executor and wait for pending tasks.
        
        Call this on server shutdown to ensure pending summarizations complete.
        """
        log.info("[BG_SUMMARIZER] Shutting down executor...")
        self.executor.shutdown(wait=True)
        log.info("[BG_SUMMARIZER] ✅ Executor shutdown complete")
