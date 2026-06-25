from openai import AzureOpenAI  
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, CHAT_MODEL
from backend.utils.logger import get_logger
from backend.prompts import (
    get_system_prompt,
    build_map_system_prompt,
    build_reduce_system_prompt,
)
from backend.processing.parallel_executor import ParallelExecutor

log = get_logger("generator")

# ── Parallel map-reduce generation ──
# Broad intents that gather many chunks are processed by splitting the chunks into
# batches, running the "map" LLM calls in parallel, then merging via a "reduce" call.
# This avoids one oversized LLM request and is faster than a single giant call.
MAP_REDUCE_INTENTS = {
    "targeted_extraction",
    "global_extraction",
    "targeted_summary",
    "global_summary",
    "analysis",
}

MAP_REDUCE_BATCH_SIZE = 10   # chunks per parallel map call
# Only split into parallel map/reduce when the retrieval is genuinely large (3+ batches).
# Splitting a SMALL retrieval and then merging the partials is where trailing items get
# dropped from enumerations ("list all X" losing the last entry): the reduce LLM silently
# omits an item that sat at the end of one partial. A document that fits comfortably in a
# single call must therefore be answered in ONE call, which sees every item together and
# enumerates them completely. Map-reduce stays only for large multi-batch retrievals where
# a single oversized call isn't practical.
MAP_REDUCE_MIN_CHUNKS = 24   # only parallelize when chunk count exceeds this
# Map batches run concurrently, but in MODERATE waves rather than all at once. Bursting
# many simultaneous GPT requests at one Azure deployment blows its per-minute token
# burst, so the deployment throttles/queues the excess — the queued calls then exceed
# the per-call timeout and FAIL, silently dropping those chunks from the answer. A
# smaller ceiling sends batches in waves that each clear quickly, which is both more
# RELIABLE (far fewer timeouts) and usually no slower overall (no throttle backoff).
MAP_REDUCE_MAX_WORKERS = 6

# Per-request timeouts (seconds) so a single hung HTTP call can't stall a query.
LLM_REQUEST_TIMEOUT = 60      # default for normal single-call generation
MAP_BATCH_TIMEOUT = 60        # per-batch timeout in the parallel map step (room for queued waves)
LLM_MAX_RETRIES = 2           # SDK-level retries on transient failures/timeouts


class Generator:  
    def __init__(self):  
        self.client = AzureOpenAI(  
            azure_endpoint=AZURE_ENDPOINT,  
            api_key=AZURE_API_KEY,  
            api_version=AZURE_API_VERSION,  
            timeout=LLM_REQUEST_TIMEOUT,  
            max_retries=LLM_MAX_RETRIES,  
        )

    def generate(self, query: str, context_chunks: list, temperature: float = 0.2, memory_context: str = None, retrieval_intent: str = None, turn_count: int = 1, position_selector: str = "") -> tuple:
        """
        Generate answer and memory summary from query and context.
        
        DYNAMIC CONTEXT STRATEGY:
        - Turns 1-5:   Full input + memory summary (build understanding)
        - Turn 6+:     Memory summary only (if available) → if not, use input (compress context)
        
        Args:
            query: User query
            context_chunks: Retrieved document chunks
            temperature: LLM temperature
            memory_context: Optional conversation memory context from previous turns
            retrieval_intent: Intent classification from LLM (factual, summary, comparison, extraction, analysis, ambiguous)
            turn_count: Number of turns in current group (for dynamic context optimization)
            position_selector: When set (e.g. "section 9.2"), the answer is scoped to that
                located part of the document — generation runs as a SINGLE call (the whole
                document is present in order) so the scope can be located then operated on.
            
        Returns:
            tuple: (answer, memory_summary)
                - answer: Full detailed response for user
                - memory_summary: Compressed key points for conversation memory
        """
        # PARALLEL MAP-REDUCE: for broad intents (extraction/summary/analysis) that
        # retrieve many chunks, process chunk batches in parallel instead of sending
        # one giant context to the LLM. Falls through to the single-call path below
        # for focused intents or small chunk sets. SKIPPED for position-scoped queries:
        # locating "section 9.2" needs the whole document together in one call (a batch
        # that lacks the section would just return NONE), so those run single-call.
        if (not position_selector) and retrieval_intent in MAP_REDUCE_INTENTS and len(context_chunks) > MAP_REDUCE_MIN_CHUNKS:
            return self._generate_map_reduce(
                query=query,
                context_chunks=context_chunks,
                temperature=temperature,
                memory_context=memory_context,
                retrieval_intent=retrieval_intent,
            )

        context_sections = []
        
        # DYNAMIC CONTEXT: Choose what to send based on turn count
        log.info(f"[GENERATOR] Dynamic context strategy: turn_count={turn_count}")
        
        messages = self._build_single_messages(
            query, context_chunks, memory_context, retrieval_intent, turn_count, position_selector
        )

        response = self.client.chat.completions.create(  
            model=CHAT_MODEL,  
            messages=messages,  
            temperature=temperature,  
        )

        full_response = response.choices[0].message.content  
        answer, memory_summary = self._parse_answer_and_summary(full_response)
        
        log.info("[GENERATOR] ✅ Answer generated with memory summary")  
        return answer, memory_summary

    def _build_single_messages(self, query: str, context_chunks: list, memory_context: str,
                               retrieval_intent: str, turn_count: int, position_selector: str = "") -> list:
        """Build the chat messages for the single-call generation path (shared by the
        streaming and non-streaming paths so they send identical context)."""
        context_sections = []

        if memory_context:
            if turn_count >= 6:
                # Turn 6+: Use summary only (save tokens)
                log.info("[GENERATOR] Turn 6+: Using memory summary only (context optimization)")
                context_sections.append("## Conversation Memory:\n" + memory_context)
            else:
                # Turns 1-5: Send full context for clarity
                log.info("[GENERATOR] Turns 1-5: Using full memory context (build understanding)")
                context_sections.append("## Previous Conversation Context:\n" + memory_context)

        # Add retrieved document chunks
        context_sections.append("## Document Context:")
        context_sections.append("\n\n".join([
            f"[Source: {c['filename']} | Page: {c['page']} | Chunk: {c['chunk_index']}]\n{c['text']}"
            for c in context_chunks
        ]))

        full_context = "\n\n".join(context_sections)
        system_prompt = self._get_system_prompt(retrieval_intent)

        # POSITION SCOPE: when the query targets a specific located part (e.g. "section 9.2"),
        # the whole document is present in ORDER. Tell the model to first locate that exact
        # part, then apply its task ONLY to it — so the operation (return / summarize /
        # compare / explain) is scoped correctly instead of run over the entire document.
        if position_selector:
            log.info(f"[GENERATOR] Position-scoped generation → scope='{position_selector[:60]}'")
            user_content = (
                f"Context (the full document, in its original order):\n{full_context}\n\n"
                f"POSITION SCOPE: First locate this exact part of the document: "
                f"\"{position_selector}\". Match it by its heading/number/position, not by "
                f"topic similarity. Then carry out the task ONLY on that located part "
                f"(ignore unrelated parts of the document). If that part cannot be found, "
                f"say so plainly.\n\nTask: {query}"
            )
        else:
            user_content = f"Context:\n{full_context}\n\nQuestion: {query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    
    def _generate_map_reduce(self, query: str, context_chunks: list, temperature: float,
                             memory_context: str, retrieval_intent: str) -> tuple:
        """
        Generate an answer by processing chunk batches in parallel (map) and merging
        the partial results into one final answer (reduce).

        Used for broad intents (extraction/summary/analysis) over many chunks so the
        whole document is covered without a single oversized LLM call.

        Returns:
            tuple: (answer, memory_summary)
        """
        partials, system_prompt = self._map_phase(query, context_chunks, temperature, retrieval_intent)

        if not partials:
            log.warning("[GENERATOR] Map-reduce: no relevant content found in any batch")
            answer = "The provided document chunks do not contain information relevant to this question."
            return answer, answer[:150]

        log.info(f"[GENERATOR] Map-reduce: {len(partials)} batches returned content — reducing")

        # ── REDUCE: merge partial findings into one final answer ──
        return self._reduce_partials(query, partials, system_prompt, memory_context, temperature)

    def _map_phase(self, query: str, context_chunks: list, temperature: float,
                   retrieval_intent: str) -> tuple:
        """
        MAP phase: split chunks into batches and extract relevant content from each
        batch in parallel. Shared by the streaming and non-streaming generation paths.

        Returns:
            tuple: (partials, system_prompt)
                - partials: list of non-empty partial extraction strings (NONE filtered out)
                - system_prompt: the intent-specific system prompt (reused by reduce)
        """
        system_prompt = self._get_system_prompt(retrieval_intent)
        batches = [
            context_chunks[i:i + MAP_REDUCE_BATCH_SIZE]
            for i in range(0, len(context_chunks), MAP_REDUCE_BATCH_SIZE)
        ]
        log.info(
            f"[GENERATOR] Map-reduce ({retrieval_intent}): {len(context_chunks)} chunks "
            f"→ {len(batches)} parallel batches (size {MAP_REDUCE_BATCH_SIZE})"
        )

        # ── MAP: extract relevant content from each batch in parallel ──
        # One worker per batch so ALL batches fire concurrently (capped at the
        # ceiling). ParallelExecutor already uses min(num_tasks, max_workers).
        map_workers = min(len(batches), MAP_REDUCE_MAX_WORKERS)
        tasks = [
            {"batch": batch, "query": query, "system_prompt": system_prompt, "temperature": temperature}
            for batch in batches
        ]
        map_results = ParallelExecutor.execute_parallel(
            tasks=tasks,
            task_func=self._map_batch,
            max_workers=map_workers,
            operation_name="map-extract",
        )

        # COMPLETENESS SAFETY NET: a batch that ERRORED (e.g. timed out under throttling)
        # contributed NONE of its chunks to the answer. Don't silently drop them — retry
        # just the failed batches once, sequentially (max_workers=1) so the retry itself
        # can't re-trigger the throttling that caused the failure.
        failed_idx = [i for i, r in enumerate(map_results) if not r or r.get("error")]
        if failed_idx:
            log.warning(
                f"[GENERATOR] Map-reduce: {len(failed_idx)} batch(es) failed on first pass "
                f"— retrying them sequentially so no chunks are dropped"
            )
            retry_results = ParallelExecutor.execute_parallel(
                tasks=[tasks[i] for i in failed_idx],
                task_func=self._map_batch,
                max_workers=1,
                operation_name="map-extract-retry",
            )
            for slot, res in zip(failed_idx, retry_results):
                map_results[slot] = res
            still_failed = sum(1 for i in failed_idx if not map_results[i] or map_results[i].get("error"))
            if still_failed:
                log.error(f"[GENERATOR] Map-reduce: {still_failed} batch(es) STILL failed after retry — their chunks are missing from the answer")

        partials = []
        for r in map_results:
            if not r or r.get("error"):
                continue
            text = (r.get("partial") or "").strip()
            if text and text.upper() != "NONE":
                partials.append(text)

        return partials, system_prompt

    def _map_batch(self, task: dict, index: int, total: int) -> dict:
        """MAP worker: extract content relevant to the query from a single chunk batch."""
        batch = task["batch"]
        context = "\n\n".join([
            f"[Source: {c['filename']} | Page: {c['page']} | Chunk: {c['chunk_index']}]\n{c['text']}"
            for c in batch
        ])
        map_system = build_map_system_prompt(task["system_prompt"], index, total)
        messages = [
            {"role": "system", "content": map_system},
            {"role": "user", "content": f"Context (batch {index}/{total}):\n{context}\n\nQuestion: {task['query']}"},
        ]
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=task["temperature"],
            timeout=MAP_BATCH_TIMEOUT,
        )
        return {"partial": response.choices[0].message.content}

    def _reduce_partials(self, query: str, partials: list, system_prompt: str,
                         memory_context: str, temperature: float) -> tuple:
        """REDUCE step: merge parallel partial findings into one final answer + summary."""
        messages = self._build_reduce_messages(query, partials, system_prompt, memory_context)
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer, memory_summary = self._parse_answer_and_summary(response.choices[0].message.content)
        log.info("[GENERATOR] ✅ Map-reduce answer generated with memory summary")
        return answer, memory_summary

    def _build_reduce_messages(self, query: str, partials: list, system_prompt: str,
                               memory_context: str) -> list:
        """Build the chat messages for the REDUCE step (shared by streaming and
        non-streaming paths so they merge partials identically)."""
        combined = "\n\n---\n\n".join([f"[Partial {i + 1}]\n{p}" for i, p in enumerate(partials)])

        reduce_sections = []
        if memory_context:
            reduce_sections.append("## Previous Conversation Context:\n" + memory_context)
        reduce_sections.append("## Extracted partial findings from across the document:\n" + combined)
        full_context = "\n\n".join(reduce_sections)

        reduce_system = build_reduce_system_prompt(system_prompt)
        return [
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query}"},
        ]

    # ──────────────────────────────────────────────────────────────────────
    #  STREAMING GENERATION
    # ──────────────────────────────────────────────────────────────────────
    def generate_stream(self, query: str, context_chunks: list, temperature: float = 0.2,
                        memory_context: str = None, retrieval_intent: str = None, turn_count: int = 1,
                        position_selector: str = ""):
        """
        Streaming variant of generate(). A Python generator that yields event dicts:
            {"type": "token", "text": <delta>}                      # ANSWER-section deltas
            {"type": "done", "answer": <full>, "memory_summary": <s>}  # terminal event

        For broad intents over many chunks the MAP phase runs first (not streamable),
        then the REDUCE call is streamed. Otherwise the single generation call streams.
        Only the ANSWER section is streamed to the client; MEMORY_SUMMARY is buffered
        for conversation memory and returned in the terminal event.

        Position-scoped queries (position_selector set) always stream a SINGLE call so the
        whole document is present in order and the located part can be operated on.
        """
        if (not position_selector) and retrieval_intent in MAP_REDUCE_INTENTS and len(context_chunks) > MAP_REDUCE_MIN_CHUNKS:
            partials, system_prompt = self._map_phase(query, context_chunks, temperature, retrieval_intent)
            if not partials:
                log.warning("[GENERATOR] Stream map-reduce: no relevant content in any batch")
                answer = "The provided document chunks do not contain information relevant to this question."
                yield {"type": "token", "text": answer}
                yield {"type": "done", "answer": answer, "memory_summary": answer[:150]}
                return
            log.info(f"[GENERATOR] Stream map-reduce: {len(partials)} batches returned content — streaming reduce")
            messages = self._build_reduce_messages(query, partials, system_prompt, memory_context)
        else:
            messages = self._build_single_messages(
                query, context_chunks, memory_context, retrieval_intent, turn_count, position_selector
            )

        yield from self._stream_chat(messages, temperature)

    # ──────────────────────────────────────────────────────────────────────
    #  COMPOUND (MULTI-SEGMENT) GENERATION
    # ──────────────────────────────────────────────────────────────────────
    def generate_segments(self, segments: list, temperature: float = 0.2,
                          memory_context: str = None, turn_count: int = 1) -> tuple:
        """Answer a COMPOUND query by generating each segment with its OWN intent and
        its OWN retrieved chunks, then combining into one sectioned answer.

        Each segment is just a normal generation (single-call or map-reduce, decided by
        that segment's intent), so a 'compare A and B' segment gets the comparison format
        and an 'extract C' segment gets the extraction/table format.
        """
        answer_parts = []
        summaries = []
        for i, seg in enumerate(segments):
            title = seg.get("title") or seg.get("query", "")
            seg_answer, seg_summary = self.generate(
                query=seg.get("query", ""),
                context_chunks=seg.get("hits", []),
                temperature=temperature,
                memory_context=memory_context if i == 0 else None,
                retrieval_intent=seg.get("intent"),
                turn_count=turn_count,
                position_selector=seg.get("position_selector", ""),
            )
            answer_parts.append(f"## {title}\n\n{(seg_answer or '').strip()}".strip())
            if seg_summary:
                summaries.append(seg_summary)
        full_answer = "\n\n".join(p for p in answer_parts if p).strip()
        full_summary = " ".join(summaries).strip()[:300]
        log.info(f"[GENERATOR] ✅ Compound answer generated from {len(segments)} segment(s)")
        return full_answer, full_summary

    def generate_segments_stream(self, segments: list, temperature: float = 0.2,
                                 memory_context: str = None, turn_count: int = 1):
        """Streaming variant of generate_segments: stream each segment's answer under its
        own heading, then emit ONE terminal 'done' with the combined answer + summary."""
        answer_parts = []
        summaries = []
        for i, seg in enumerate(segments):
            title = seg.get("title") or seg.get("query", "")
            heading = f"## {title}\n\n"
            yield {"type": "token", "text": (("\n\n" if i > 0 else "") + heading)}
            seg_answer = ""
            seg_summary = ""
            for ev in self.generate_stream(
                query=seg.get("query", ""),
                context_chunks=seg.get("hits", []),
                temperature=temperature,
                memory_context=memory_context if i == 0 else None,
                retrieval_intent=seg.get("intent"),
                turn_count=turn_count,
                position_selector=seg.get("position_selector", ""),
            ):
                etype = ev.get("type")
                if etype == "token":
                    yield ev
                elif etype == "done":
                    seg_answer = ev.get("answer", "")
                    seg_summary = ev.get("memory_summary", "")
                elif etype == "error":
                    yield ev
            answer_parts.append(f"## {title}\n\n{(seg_answer or '').strip()}".strip())
            if seg_summary:
                summaries.append(seg_summary)
        full_answer = "\n\n".join(p for p in answer_parts if p).strip()
        full_summary = " ".join(summaries).strip()[:300]
        log.info(f"[GENERATOR] ✅ Compound stream generated from {len(segments)} segment(s)")
        yield {"type": "done", "answer": full_answer, "memory_summary": full_summary}

    @staticmethod
    def _answer_portion(raw: str) -> str:
        """Extract the ANSWER-section text seen so far from a partial raw stream:
        strips a leading 'ANSWER:' label and cuts anything from 'MEMORY_SUMMARY:' on."""
        text = raw
        if "MEMORY_SUMMARY:" in text:
            text = text.split("MEMORY_SUMMARY:")[0]
        if "ANSWER:" in text:
            text = text.split("ANSWER:", 1)[1]
        return text.lstrip()

    def _stream_chat(self, messages: list, temperature: float):
        """
        Stream a single chat completion, yielding ANSWER-section text deltas and a
        terminal 'done' event with the parsed (answer, memory_summary).

        A small tail of the answer text is held back on each delta so a partially
        arrived 'MEMORY_SUMMARY:' marker is never emitted to the client.
        """
        raw = ""
        emitted_len = 0
        HOLD = 24  # > len("MEMORY_SUMMARY:") so a partial marker is never emitted

        stream = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
            timeout=LLM_REQUEST_TIMEOUT,
        )
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            raw += delta
            answer_so_far = self._answer_portion(raw)
            safe_upto = max(emitted_len, len(answer_so_far) - HOLD)
            if safe_upto > emitted_len:
                yield {"type": "token", "text": answer_so_far[emitted_len:safe_upto]}
                emitted_len = safe_upto

        # Stream finished — flush any remaining answer text, then the terminal event.
        answer_final, memory_summary = self._parse_answer_and_summary(raw)
        if len(answer_final) > emitted_len:
            yield {"type": "token", "text": answer_final[emitted_len:]}
        log.info("[GENERATOR] ✅ Streamed answer generated with memory summary")
        yield {"type": "done", "answer": answer_final, "memory_summary": memory_summary}
    
    def _parse_answer_and_summary(self, response: str) -> tuple:
        """
        Parse response to extract answer and memory summary.
        
        Robust to the model omitting the leading ``ANSWER:`` label: the split on
        ``MEMORY_SUMMARY:`` happens INDEPENDENTLY of whether ``ANSWER:`` is present, so a
        response that dives straight into content but still appends a ``MEMORY_SUMMARY:``
        section never leaks that section into the visible answer.
        
        Args:
            response: Full response from LLM with optional ANSWER and MEMORY_SUMMARY sections
            
        Returns:
            tuple: (answer, memory_summary)
        """
        answer = response
        memory_summary = ""
        
        # Split off the memory summary first — do NOT gate this on the ANSWER: label,
        # which the model sometimes drops while still emitting MEMORY_SUMMARY:.
        if "MEMORY_SUMMARY:" in answer:
            answer, _, memory_summary = answer.partition("MEMORY_SUMMARY:")
            memory_summary = memory_summary.strip()
        
        # Strip a leading ANSWER: label if the model included one.
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:", 1)[1]
        
        return answer.strip(), memory_summary
    
    def _get_system_prompt(self, retrieval_intent: str) -> str:
        """
        Get intent-specific system prompt for generation.

        Prompts are centralized in the backend.prompts package for easy modification.
        
        Args:
            retrieval_intent: The classified intent (factual, summary, comparison, extraction, analysis, ambiguous)
            
        Returns:
            System prompt string tailored to the intent
        """
        if not retrieval_intent:
            log.warning("[GENERATOR] ⚠️ No retrieval_intent provided, defaulting to factual")
            retrieval_intent = "factual"
        
        return get_system_prompt(retrieval_intent)


