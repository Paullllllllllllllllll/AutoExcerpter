"""Task runner utilities for concurrent processing with progress tracking.

This module provides reusable utilities for running concurrent tasks with
progress bars, ETA calculation, and error handling.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar

from tqdm import tqdm

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Type variables for generic task runner
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

# ============================================================================
# Constants
# ============================================================================
MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3


# ============================================================================
# Task Runner
# ============================================================================
class ConcurrentTaskRunner(Generic[T, R]):
    """
    Generic concurrent task runner with progress tracking and ETA calculation.
    
    This class handles:
    - Concurrent execution of tasks using ThreadPoolExecutor
    - Progress bar display with tqdm
    - ETA calculation based on processing times
    - Error handling and logging
    - Statistics collection
    """
    
    def __init__(
        self,
        max_workers: int,
        description: str = "Processing",
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the task runner.
        
        Args:
            max_workers: Maximum number of concurrent workers.
            description: Description for progress bar.
            show_progress: Whether to show progress bar.
        """
        self.max_workers = max_workers
        self.description = description
        self.show_progress = show_progress
        self.processing_times: deque[float] = deque(maxlen=50)
        self.start_time: Optional[float] = None
    
    def run(
        self,
        task_func: Callable[[T], R],
        items: Iterable[T],
        total: Optional[int] = None,
    ) -> List[R]:
        """
        Run tasks concurrently with progress tracking.
        
        Args:
            task_func: Function to execute for each item.
            items: Iterable of items to process.
            total: Total number of items (for progress bar).
        
        Returns:
            List of results in the same order as input items.
        """
        self.start_time = time.time()
        results: List[R] = []
        
        items_list = list(items)
        if total is None:
            total = len(items_list)
        
        # Adjust workers if fewer items than workers
        actual_workers = min(self.max_workers, max(1, total))
        
        logger.info(f"Starting concurrent processing with {actual_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            if self.show_progress:
                # Use tqdm for progress tracking
                futures = [executor.submit(task_func, item) for item in items_list]
                future_to_index = {future: idx for idx, future in enumerate(futures)}
                results = [None] * len(items_list)  # type: ignore
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total,
                    desc=self.description,
                ):
                    try:
                        result = future.result()
                        idx = future_to_index[future]
                        results[idx] = result  # type: ignore
                    except Exception as e:
                        logger.exception(f"Task failed with exception: {e}")
                        # Append error result if task_func returns a result type that can handle errors
                        idx = future_to_index[future]
                        results[idx] = None  # type: ignore
            else:
                # No progress bar
                results = list(executor.map(task_func, items_list))
        
        return results
    
    def calculate_eta(self, processed_count: int, total_items: int) -> str:
        """
        Calculate estimated time of arrival for remaining items.
        
        Args:
            processed_count: Number of items processed so far.
            total_items: Total number of items to process.
        
        Returns:
            Formatted ETA string.
        """
        if processed_count <= MIN_SAMPLES_FOR_ETA or not self.start_time:
            return "ETA: N/A"
        
        elapsed_total = time.time() - self.start_time
        items_per_sec_overall = processed_count / elapsed_total
        
        if items_per_sec_overall <= 0:
            return "ETA: N/A"
        
        # Blend overall and recent rates for more accurate estimates
        blended_rate = self._calculate_blended_rate(items_per_sec_overall)
        
        if blended_rate <= 0:
            return "ETA: N/A"
        
        remaining_items = total_items - processed_count
        eta_seconds = remaining_items / blended_rate
        return f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"
    
    def _calculate_blended_rate(self, overall_rate: float) -> float:
        """Calculate blended processing rate from overall and recent samples."""
        recent_samples = list(self.processing_times)[-RECENT_SAMPLES_FOR_ETA:]
        if not recent_samples:
            return overall_rate
        
        recent_avg_time = sum(recent_samples) / len(recent_samples)
        recent_rate = 1.0 / recent_avg_time if recent_avg_time > 0 else overall_rate
        
        return (ETA_BLEND_WEIGHT_OVERALL * overall_rate + 
                ETA_BLEND_WEIGHT_RECENT * recent_rate)
    
    def record_processing_time(self, processing_time: float) -> None:
        """Record a processing time for ETA calculation."""
        self.processing_times.append(processing_time)


# ============================================================================
# Batch Processing Utilities
# ============================================================================
def process_in_batches(
    items: List[T],
    batch_size: int,
    process_func: Callable[[List[T]], List[R]],
) -> List[R]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process.
        batch_size: Size of each batch.
        process_func: Function to process each batch.
    
    Returns:
        Flattened list of all results.
    """
    results: List[R] = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ConcurrentTaskRunner",
    "process_in_batches",
    "MIN_SAMPLES_FOR_ETA",
    "RECENT_SAMPLES_FOR_ETA",
]
