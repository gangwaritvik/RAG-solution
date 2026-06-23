"""Background summarization for conversation groups."""

import threading
from typing import Optional, Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.config import CHAT_MODEL
from backend.utils.logger import get_logger
from backend.prompts import SUMMARIZER_SYSTEM_PROMPT, build_summary_user_prompt

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
        self.active_summarizations: Dict[str, Any] = {}  # group_id -> claimed sentinel (presence = in-progress)
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
        
        # ATOMIC check-and-claim under ONE lock: decide AND mark the group in-progress in
        # the same critical section. This prevents two callers from both passing the check,
        # and guarantees the key EXISTS before the worker's finally-clause runs — otherwise
        # a fast worker could clear a key that is only added afterwards, wedging the group
        # as "summarizing" forever and blocking all future roll-ups.
        with self._lock:
            if group_id in self.active_summarizations:
                log.info(f"[BG_SUMMARIZER] Group {group_id} already being summarized")
                return False
            self.active_summarizations[group_id] = True  # claim slot (presence is what matters)
        
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
        # The slot in active_summarizations was already claimed atomically by the caller
        # (summarize_if_needed) BEFORE this submit, so we must NOT re-add it here. The
        # worker's finally-clause is the single place that clears it.
        self.executor.submit(
            self._summarize_group_worker,
            group_id,
            callback
        )
        
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
            
            # Atomically snapshot the unsummarized turns + their count. We summarize THIS
            # snapshot and later clear exactly this many, so a query that arrives while the
            # summary is generating is never lost from the recent buffer.
            recent_turns, num_summarized = self.memory_manager.snapshot_recent_turns(group_id)
            if not recent_turns:
                log.warning(f"[BG_SUMMARIZER] [BACKGROUND] ⚠️ No recent turns to summarize")
                return False
            
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Found {num_summarized} unsummarized turns")
            
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

            # Persist the summary FIRST so it is never lost — even if embedding fails.
            # (Previously a failed embedding discarded the whole summary, wasting the LLM call.)
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Saving summary for group {group_id}...")
            success = self.memory_manager.update_group_summary(
                group_id=group_id,
                summary=summary,
                summarized_count=num_summarized,
            )
            if not success:
                log.error(f"[BG_SUMMARIZER] [BACKGROUND] ❌ Failed to save summary")
                return False

            # Generate the embedding and attach it to the already-saved summary. If the
            # embedding step fails, the summary still stands; the group just won't be
            # matchable by embedding similarity until the next summarization run.
            log.info(f"[BG_SUMMARIZER] [BACKGROUND] Generating embedding...")
            embeddings = self.embedder.embed_texts([summary])
            summary_embedding = embeddings[0] if embeddings else None

            if summary_embedding:
                self.memory_manager.update_group_summary_with_embedding(
                    group_id=group_id,
                    embedding=summary_embedding,
                )
                log.info(f"[BG_SUMMARIZER] [BACKGROUND] ✅ Group {group_id} summarized (with embedding)")
            else:
                log.warning(f"[BG_SUMMARIZER] [BACKGROUND] ⚠️ Embedding failed — summary saved without it")

            log.info(f"[BG_SUMMARIZER] [BACKGROUND]   Summary: {summary[:100]}...")
            return True
            
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

        The group summary is built STRICTLY from each turn's ORIGINAL full answer
        (turn.full_answer) — never the compressed per-turn memory_summary — so the
        summary distills from the rich source and never compounds compression loss.
        The memory_summary is used ONLY as a last-resort safety net for a turn that
        somehow has no stored full answer, and that fallback is logged as a warning so
        it never happens silently.

        Args:
            turns: List of ConversationTurn objects

        Returns:
            Formatted string for LLM
        """
        lines = []
        idx = 0
        for turn in turns:
            # Skip AMBIGUOUS turns: their "answer" is just a request for clarification
            # ('could you specify which X?'), which would pollute the group summary. The
            # user's real, clarified question is a SEPARATE non-ambiguous turn that IS
            # included — so the summary reflects actual content, not the clarifying detour.
            if getattr(turn, "dependency_type", None) == "ambiguous":
                continue
            idx += 1
            full_answer = (getattr(turn, "full_answer", "") or "").strip()
            if full_answer:
                answer_text = full_answer
            else:
                # Safety net only — should not normally happen since _store_turn always
                # persists the full answer. Surface it so a missing answer is visible.
                answer_text = (turn.memory_summary or "").strip()
                log.warning(
                    f"[BG_SUMMARIZER] Turn {getattr(turn, 'turn_id', '?')} has no full_answer — "
                    f"falling back to memory_summary for the group summary"
                )
            lines.append(f"Q{idx}: {turn.query}")
            lines.append(f"A{idx}: {answer_text}")
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
            prompt = build_summary_user_prompt(topic, turns_text, existing_summary)

            # Call LLM for summary (use the generator's configured client + deployment
            # so this matches the rest of the app and doesn't hit a non-existent model).
            response = self.generator.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SUMMARIZER_SYSTEM_PROMPT
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
