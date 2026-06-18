"""Multi-group query processor for handling comparison queries."""

from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.utils.logger import get_logger

log = get_logger("multi_group_processor")


class MultiGroupProcessor:
    """
    Processes multi-group queries by splitting into sub-queries
    and handling them in parallel.
    """
    
    def __init__(self, memory_manager, context_resolver, retriever, generator, embedder, max_workers: int = 5):
        """
        Initialize multi-group processor with required components.
        
        Args:
            memory_manager: ConversationMemoryManager instance
            context_resolver: ContextResolver instance
            retriever: Retriever instance
            generator: Generator instance
            embedder: Embedder instance
            max_workers: Max parallel workers (default 5)
        """
        self.memory_manager = memory_manager
        self.context_resolver = context_resolver
        self.retriever = retriever
        self.generator = generator
        self.embedder = embedder
        self.max_workers = max_workers
        
        # Intent-based retrieval strategy
        self.intent_strategy = {
            "factual": 5,
            "summary": 50,
            "comparison": 8,
            "extraction": 15,
            "analysis": 10,
            "ambiguous": 7,
        }
        
        log.info(f"[MULTI_GROUP_PROCESSOR] ✅ Initialized with max_workers={max_workers}")
    
    def process(self, original_query: str, sub_queries: List[Dict[str, str]], temperature: float = 0.2) -> List[Dict[str, Any]]:
        """
        Process multi-group query with parallel sub-query handling.
        
        Args:
            original_query: The original multi-group query
            sub_queries: List of dicts with {query, topic}
            temperature: Temperature for generation
            
        Returns:
            List of result dicts in original order: {query, answer, group_id, memory_summary, ...}
        """
        # Depth limiting: cap sub-queries to prevent runaway comparisons
        MAX_SUB_QUERIES = 5
        if len(sub_queries) > MAX_SUB_QUERIES:
            original_count = len(sub_queries)
            sub_queries = sub_queries[:MAX_SUB_QUERIES]
            log.warning(f"[MULTI_GROUP_PROCESSOR] ⚠️ Depth limiting: reduced {original_count} sub-queries to {MAX_SUB_QUERIES}")
        
        log.info(f"[MULTI_GROUP_PROCESSOR] Processing {len(sub_queries)} sub-queries IN PARALLEL")
        
        sub_query_results = []
        num_workers = min(len(sub_queries), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all sub-queries
            log.info(f"[MULTI_GROUP_PROCESSOR] Submitting {len(sub_queries)} sub-queries with {num_workers} workers")
            
            future_to_index = {}
            for i, sub_q_info in enumerate(sub_queries):
                future = executor.submit(
                    self._process_single_sub_query,
                    sub_q_info=sub_q_info,
                    sub_index=i + 1,
                    total_subs=len(sub_queries),
                    temperature=temperature
                )
                future_to_index[future] = i
            
            # Collect results as they complete
            results_by_index = {}
            completed = 0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                
                try:
                    result = future.result()
                    results_by_index[index] = result
                    log.info(f"[MULTI_GROUP_PROCESSOR] Completed {completed}/{len(sub_queries)} sub-queries")
                except Exception as e:
                    log.error(f"[MULTI_GROUP_PROCESSOR] ❌ Sub-query {index + 1} failed: {e}", exc_info=True)
                    results_by_index[index] = {
                        "query": sub_queries[index].get("query", ""),
                        "answer": f"Error: {str(e)}",
                        "group_id": None,
                        "memory_summary": "Error",
                        "retrieval_intent": "error",
                        "chunks_retrieved": 0,
                    }
            
            # Return results in original order
            for i in range(len(sub_queries)):
                sub_query_results.append(results_by_index[i])
        
        log.info(f"[MULTI_GROUP_PROCESSOR] ✅ All {len(sub_queries)} sub-queries processed in parallel")
        return sub_query_results
    
    def _process_single_sub_query(self, sub_q_info: dict, sub_index: int, total_subs: int, temperature: float) -> dict:
        """
        Process a single sub-query in parallel.
        
        Args:
            sub_q_info: Dict with {query, topic}
            sub_index: Index of this sub-query (1-based)
            total_subs: Total number of sub-queries
            temperature: Temperature for generation
            
        Returns:
            Dict with {query, answer, group_id, memory_summary, retrieval_intent, chunks_retrieved}
        """
        sub_query = sub_q_info.get("query", "")
        suggested_topic = sub_q_info.get("topic", f"Sub-Query {sub_index}")
        
        log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Processing sub-query {sub_index}/{total_subs}: {sub_query[:80]}")
        
        try:
            # Create a new group for this sub-query
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Creating group for: {suggested_topic}")
            new_group = self.memory_manager.create_conversation_group(suggested_topic)
            sub_group_id = new_group.group_id
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] ✅ Created group {sub_group_id}")
            
            # Process sub-query through context resolution
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Resolving context for sub-query {sub_index}")
            sub_context = self.context_resolver.resolve(sub_query, active_group_id=sub_group_id)
            
            # Retrieve with sub-query using intent-based top_k
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Retrieving chunks for sub-query {sub_index}")
            intent_top_k = self.intent_strategy.get(sub_context.retrieval_intent.value, 5)
            
            hits = self.retriever.retrieve(
                sub_context.standalone_query, 
                top_k=intent_top_k,
                retrieval_intent=sub_context.retrieval_intent.value
            )
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Retrieved {len(hits)} chunks for sub-query {sub_index}")
            
            # Generate answer for sub-query
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Generating answer for sub-query {sub_index}")
            sub_answer, sub_memory_summary = self.generator.generate(
                query=sub_query,
                context_chunks=hits,
                temperature=temperature,
                memory_context=sub_context.memory_context,
                retrieval_intent=sub_context.retrieval_intent.value
            )
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] ✅ Answer generated for sub-query {sub_index}")
            
            # Store in sub-query group with parallel embedding
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] Storing turn in group {sub_group_id}")
            sub_memory_summary, sub_embedding = self.generator.create_summary_and_embedding_parallel(
                memory_summary=sub_memory_summary,
                embedder=self.embedder
            )
            
            turn = self.memory_manager.add_conversation_turn(
                group_id=sub_group_id,
                query=sub_query,
                memory_summary=sub_memory_summary
            )
            
            if turn and sub_embedding is not None:
                self.memory_manager.update_group_summary_with_embedding(
                    group_id=sub_group_id,
                    embedding=sub_embedding
                )
            
            # Collect result
            result = {
                "query": sub_query,
                "answer": sub_answer,
                "group_id": sub_group_id,
                "memory_summary": sub_memory_summary,
                "retrieval_intent": sub_context.retrieval_intent.value,
                "chunks_retrieved": len(hits),
            }
            
            log.info(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] ✅ Sub-query {sub_index} complete")
            return result
            
        except Exception as e:
            log.error(f"[MULTI_GROUP_PROCESSOR] [PARALLEL] ❌ Failed to process sub-query {sub_index}: {e}", exc_info=True)
            return {
                "query": sub_query,
                "answer": f"Error processing this sub-query: {str(e)}",
                "group_id": None,
                "memory_summary": "Error",
                "retrieval_intent": "error",
                "chunks_retrieved": 0,
            }
