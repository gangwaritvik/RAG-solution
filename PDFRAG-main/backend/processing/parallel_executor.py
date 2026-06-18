"""Parallel execution utilities for LLM operations."""

from typing import Callable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.utils.logger import get_logger

log = get_logger("parallel_executor")


class ParallelExecutor:
    """
    Utility class for executing operations in parallel with proper error handling.
    """
    
    @staticmethod
    def execute_parallel(
        tasks: List[Dict[str, Any]],
        task_func: Callable,
        max_workers: int = 5,
        operation_name: str = "Operation"
    ) -> List[Dict[str, Any]]:
        """
        Execute a list of tasks in parallel.
        
        Args:
            tasks: List of task dicts with data to pass to task_func
            task_func: Function that takes task dict and returns result
            max_workers: Max parallel workers (default 5)
            operation_name: Name for logging
            
        Returns:
            List of results in original order
        """
        log.info(f"[PARALLEL] Executing {len(tasks)} {operation_name} tasks in parallel with {max_workers} workers")
        
        results = []
        num_workers = min(len(tasks), max_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, task in enumerate(tasks):
                future = executor.submit(task_func, task, i + 1, len(tasks))
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
                    log.info(f"[PARALLEL] Completed {completed}/{len(tasks)} {operation_name} tasks")
                except Exception as e:
                    log.error(f"[PARALLEL] ❌ Task {index + 1} failed: {e}", exc_info=True)
                    results_by_index[index] = {
                        "error": str(e),
                        "index": index
                    }
            
            # Return results in original order
            for i in range(len(tasks)):
                results.append(results_by_index[i])
        
        log.info(f"[PARALLEL] ✅ All {len(tasks)} {operation_name} tasks completed")
        return results
    
    @staticmethod
    def execute_parallel_with_fallback(
        tasks: List[Dict[str, Any]],
        task_func: Callable,
        fallback_func: Optional[Callable] = None,
        max_workers: int = 5,
        operation_name: str = "Operation"
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks in parallel with fallback for errors.
        
        Args:
            tasks: List of task dicts
            task_func: Primary function to execute
            fallback_func: Optional fallback function if task_func fails
            max_workers: Max parallel workers
            operation_name: Name for logging
            
        Returns:
            List of results in original order
        """
        log.info(f"[PARALLEL] Executing {len(tasks)} {operation_name} tasks with fallback")
        
        results = []
        num_workers = min(len(tasks), max_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, task in enumerate(tasks):
                future = executor.submit(
                    ParallelExecutor._execute_with_fallback,
                    task=task,
                    index=i + 1,
                    total=len(tasks),
                    task_func=task_func,
                    fallback_func=fallback_func
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
                    log.info(f"[PARALLEL] Completed {completed}/{len(tasks)} {operation_name} tasks")
                except Exception as e:
                    log.error(f"[PARALLEL] ❌ Task {index + 1} failed: {e}", exc_info=True)
                    results_by_index[index] = {"error": str(e), "index": index}
            
            # Return results in original order
            for i in range(len(tasks)):
                results.append(results_by_index[i])
        
        log.info(f"[PARALLEL] ✅ All {len(tasks)} {operation_name} tasks completed")
        return results
    
    @staticmethod
    def _execute_with_fallback(task, index, total, task_func, fallback_func):
        """Helper to execute task with fallback."""
        try:
            return task_func(task, index, total)
        except Exception as e:
            log.warning(f"[PARALLEL] Task {index} failed, trying fallback: {e}")
            if fallback_func:
                return fallback_func(task, index, total)
            raise
