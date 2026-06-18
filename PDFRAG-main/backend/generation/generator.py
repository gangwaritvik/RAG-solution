from openai import AzureOpenAI  
from backend.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, CHAT_MODEL  
from backend.utils.logger import get_logger
from backend.generation.prompts_config import get_system_prompt
from backend.processing.parallel_executor import ParallelExecutor

log = get_logger("generator")

# ── Parallel map-reduce generation ──
# Broad intents that gather many chunks are processed by splitting the chunks into
# batches, running the "map" LLM calls in parallel, then merging via a "reduce" call.
# This avoids one oversized LLM request and is faster than a single giant call.
MAP_REDUCE_INTENTS = {"extraction", "summary", "analysis"}
MAP_REDUCE_BATCH_SIZE = 10   # chunks per parallel map call
MAP_REDUCE_MIN_CHUNKS = 12   # only parallelize when chunk count exceeds this
# Run ALL map batches concurrently (one worker per batch) up to this ceiling.
# The ceiling protects against Azure's request rate limit (250 req/min) and
# thread explosion on pathological chunk counts.
MAP_REDUCE_MAX_WORKERS = 30

# Per-request timeouts (seconds) so a single hung HTTP call can't stall a query.
LLM_REQUEST_TIMEOUT = 60      # default for normal single-call generation
MAP_BATCH_TIMEOUT = 30        # tighter per-batch timeout in the parallel map step
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

    def generate(self, query: str, context_chunks: list, temperature: float = 0.2, memory_context: str = None, retrieval_intent: str = None, turn_count: int = 1) -> tuple:
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
            
        Returns:
            tuple: (answer, memory_summary)
                - answer: Full detailed response for user
                - memory_summary: Compressed key points for conversation memory
        """
        # PARALLEL MAP-REDUCE: for broad intents (extraction/summary/analysis) that
        # retrieve many chunks, process chunk batches in parallel instead of sending
        # one giant context to the LLM. Falls through to the single-call path below
        # for focused intents or small chunk sets.
        if retrieval_intent in MAP_REDUCE_INTENTS and len(context_chunks) > MAP_REDUCE_MIN_CHUNKS:
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
        
        # Get intent-specific system prompt
        system_prompt = self._get_system_prompt(retrieval_intent)

        messages = [  
            {
                "role": "system",  
                "content": system_prompt,
            },  
            {
                "role": "user",  
                "content": f"Context:\n{full_context}\n\nQuestion: {query}",  
            },  
        ]

        response = self.client.chat.completions.create(  
            model=CHAT_MODEL,  
            messages=messages,  
            temperature=temperature,  
        )

        full_response = response.choices[0].message.content  
        answer, memory_summary = self._parse_answer_and_summary(full_response)
        
        log.info("[GENERATOR] ✅ Answer generated with memory summary")  
        return answer, memory_summary
    
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

        partials = []
        for r in map_results:
            if not r or r.get("error"):
                continue
            text = (r.get("partial") or "").strip()
            if text and text.upper() != "NONE":
                partials.append(text)

        if not partials:
            log.warning("[GENERATOR] Map-reduce: no relevant content found in any batch")
            answer = "The provided document chunks do not contain information relevant to this question."
            return answer, answer[:150]

        log.info(f"[GENERATOR] Map-reduce: {len(partials)}/{len(batches)} batches returned content — reducing")

        # ── REDUCE: merge partial findings into one final answer ──
        return self._reduce_partials(query, partials, system_prompt, memory_context, temperature)

    def _map_batch(self, task: dict, index: int, total: int) -> dict:
        """MAP worker: extract content relevant to the query from a single chunk batch."""
        batch = task["batch"]
        context = "\n\n".join([
            f"[Source: {c['filename']} | Page: {c['page']} | Chunk: {c['chunk_index']}]\n{c['text']}"
            for c in batch
        ])
        map_system = (
            task["system_prompt"]
            + f"\n\nMAP STEP (batch {index}/{total}): You are seeing only PART of the "
              "document chunks. Extract every detail from THESE chunks that is relevant "
              "to the question. Output ONLY the relevant extracted facts/points (with "
              "their Source/Page/Chunk labels). "
              "If the request targets items the source demarcates with a specific label, "
              "heading, or marker, include ONLY passages in these chunks that actually "
              "carry that marker; do NOT treat ordinary body text, formulas, or general "
              "statements that merely resemble them as matches. When in doubt, exclude it. "
              "Do NOT write a preamble, do NOT say "
              "information is missing or incomplete (other batches cover the rest), and "
              "do NOT add a MEMORY_SUMMARY. If NONE of these chunks are relevant, reply "
              "with exactly: NONE"
        )
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
        combined = "\n\n---\n\n".join([f"[Partial {i + 1}]\n{p}" for i, p in enumerate(partials)])

        reduce_sections = []
        if memory_context:
            reduce_sections.append("## Previous Conversation Context:\n" + memory_context)
        reduce_sections.append("## Extracted partial findings from across the document:\n" + combined)
        full_context = "\n\n".join(reduce_sections)

        reduce_system = (
            system_prompt
            + "\n\nREDUCE STEP: The context below contains partial findings that were "
              "extracted IN PARALLEL from different sections of the SAME document. Merge "
              "them into ONE coherent answer: de-duplicate repeated items, resolve "
              "overlaps, and present a single unified response following your formatting "
              "rules. Use ONLY the information in these partial findings."
        )
        messages = [
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query}"},
        ]
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        answer, memory_summary = self._parse_answer_and_summary(response.choices[0].message.content)
        log.info("[GENERATOR] ✅ Map-reduce answer generated with memory summary")
        return answer, memory_summary
    
    def _parse_answer_and_summary(self, response: str) -> tuple:
        """
        Parse response to extract answer and memory summary.
        
        Args:
            response: Full response from LLM with ANSWER and MEMORY_SUMMARY sections
            
        Returns:
            tuple: (answer, memory_summary)
        """
        answer = response
        memory_summary = ""
        
        # Try to extract sections if they exist
        if "ANSWER:" in response:
            parts = response.split("MEMORY_SUMMARY:")
            if len(parts) == 2:
                answer = parts[0].replace("ANSWER:", "").strip()
                memory_summary = parts[1].strip()
            else:
                answer = parts[0].replace("ANSWER:", "").strip()
        
        return answer, memory_summary
    
    def create_summary_and_embedding_parallel(self, memory_summary: str, embedder) -> tuple:
        """
        Create summary and embedding in parallel with race condition handling.
        
        Args:
            memory_summary: The summary text to embed
            embedder: Embedder instance for creating embeddings
            
        Returns:
            tuple: (memory_summary, embedding)
                - memory_summary: The input summary (unchanged)
                - embedding: Generated embedding vector
        """
        import threading
        
        log.info("[GENERATOR] Creating summary and embedding in parallel")
        
        embedding_result = {"result": None, "error": None}
        lock = threading.Lock()
        
        def generate_embedding():
            """Thread worker for embedding generation with lock protection."""
            try:
                embeddings = embedder.embed_texts([memory_summary])
                if embeddings:
                    with lock:
                        embedding_result["result"] = embeddings[0]
                    log.info("[GENERATOR] ✅ Embedding created successfully")
                else:
                    with lock:
                        embedding_result["error"] = "Failed to generate embedding"
                    log.warning("[GENERATOR] ⚠️ No embedding returned")
            except Exception as e:
                with lock:
                    embedding_result["error"] = str(e)
                log.error(f"[GENERATOR] ❌ Embedding failed: {e}", exc_info=True)
        
        # Start embedding thread
        embedding_thread = threading.Thread(target=generate_embedding, daemon=False)
        embedding_thread.start()
        
        # Wait for embedding to complete with timeout
        embedding_thread.join(timeout=30)
        
        # Check race conditions and errors
        if embedding_thread.is_alive():
            log.warning("[GENERATOR] ⚠️ Embedding generation timed out after 30s")
            # Thread still running, but we can't wait forever
            # Return with None embedding and let it complete in background
            return memory_summary, None
        
        if embedding_result["error"]:
            log.error(f"[GENERATOR] ❌ Embedding creation failed: {embedding_result['error']}")
            return memory_summary, None
        
        if embedding_result["result"] is None:
            log.warning("[GENERATOR] ⚠️ Embedding result is None")
            return memory_summary, None
        
        log.info("[GENERATOR] ✅ Summary and embedding created in parallel")
        return memory_summary, embedding_result["result"]
    
    def _get_system_prompt(self, retrieval_intent: str) -> str:
        """
        Get intent-specific system prompt for generation.
        
        Prompts are generalized and stored in prompts_config.py for easy modification.
        
        Args:
            retrieval_intent: The classified intent (factual, summary, comparison, extraction, analysis, ambiguous)
            
        Returns:
            System prompt string tailored to the intent
        """
        if not retrieval_intent:
            log.warning("[GENERATOR] ⚠️ No retrieval_intent provided, defaulting to factual")
            retrieval_intent = "factual"
        
        return get_system_prompt(retrieval_intent)
    
    def _parse_llm_response(self, response: str) -> tuple:
        try:
            # Split by MEMORY_SUMMARY marker
            if "MEMORY_SUMMARY:" in response:
                parts = response.split("MEMORY_SUMMARY:")
                answer = parts[0].replace("ANSWER:", "").strip()
                memory_summary = parts[1].strip()
            else:
                # Fallback: use entire response as answer, create summary from it
                answer = response.replace("ANSWER:", "").strip()
                memory_summary = self._extract_key_points(answer)
            
            # Validate lengths
            if not answer:
                answer = "Unable to generate answer from context."
            if not memory_summary:
                memory_summary = answer[:150]
            
            return answer, memory_summary
            
        except Exception as e:
            log.error(f"[GENERATOR] ❌ Failed to parse response: {e}", exc_info=True)
            # Return full response as answer, first 150 chars as summary
            return response, response[:150]
    
    def _extract_key_points(self, text: str, max_length: int = 150) -> str:
        """
        Extract key points from text as fallback summary method.
        
        Args:
            text: The text to extract from
            max_length: Maximum length of summary
            
        Returns:
            Extracted key points
        """
        import re
        
        # Find bullet points
        lines = text.split('\n')
        bullets = [line.strip() for line in lines if line.strip().startswith('-') or line.strip().startswith('•')]
        
        if bullets:
            # Use first 2-3 bullets
            summary = "; ".join(bullets[:3])
        else:
            # Use first 2 sentences
            sentences = re.split(r'[.!?]+', text)
            summary = ". ".join([s.strip() for s in sentences[:2] if s.strip()]) + "."
        
        # Limit length
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
        
        return summary  

