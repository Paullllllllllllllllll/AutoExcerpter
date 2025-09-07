# modules/multiprocessing_utils.py

from multiprocessing import Pool, cpu_count
from typing import Callable, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def wrapper(func: Callable[..., Any], args: Tuple[Any, ...]) -> Any:
    """
    Unpack arguments and call the target function.
    """
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"Error in multiprocessing task with args {args}: {e}")
        return None


def run_multiprocessing_tasks(
    func: Callable[..., Any],
    args_list: List[Tuple[Any, ...]],
    processes: int | None = None,
) -> List[Any]:
    """
    Run the given function over a list of argument tuples using multiprocessing.

    Returns:
        List[Any]: List of results.
    """
    if processes is None:
        processes = max(1, cpu_count() - 1)

    with Pool(processes=processes) as pool:
        results = pool.starmap(wrapper, [(func, args) for args in args_list])

    return results
