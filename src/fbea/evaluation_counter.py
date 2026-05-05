"""
Evaluation counter management in line with the reproduction checklist.

The counter is incremented whenever a solution is fully decoded and all three
objective values (makespan, energy, agreement) are computed.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solution import Solution  # pragma: no cover

MAX_EVALUATIONS = 150_000
_evaluation_count = 0
_parallel_workers = 1
_EVAL_CACHE_MAX_SIZE = 20_000
_evaluation_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_process_pool: ProcessPoolExecutor | None = None
_pool_workers = 1


def _make_solution_cache_key(solution: "Solution") -> tuple:
    return (
        id(solution.instance),
        tuple(solution.scheduling_string),
        tuple(solution.machine_assignment_string),
    )


def _apply_cached_evaluation(solution: "Solution", cached: tuple) -> None:
    makespan, energy, agreement, schedule, objective_vector = cached
    if schedule is not None and getattr(schedule, "instance", None) is not solution.instance:
        schedule.instance = solution.instance
    solution.makespan = makespan
    solution.energy = energy
    solution.agreement = agreement
    solution.schedule = schedule
    solution.evaluated = True
    solution.objective_key = None
    solution.objective_vector = objective_vector


def _store_cached_evaluation(solution: "Solution") -> None:
    key = _make_solution_cache_key(solution)
    _evaluation_cache[key] = (
        solution.makespan,
        solution.energy,
        solution.agreement,
        solution.schedule,
        solution.objective_vector,
    )
    _evaluation_cache.move_to_end(key)
    if len(_evaluation_cache) > _EVAL_CACHE_MAX_SIZE:
        _evaluation_cache.popitem(last=False)


def _shutdown_process_pool() -> None:
    global _process_pool, _pool_workers
    if _process_pool is not None:
        _process_pool.shutdown(wait=True, cancel_futures=True)
        _process_pool = None
    _pool_workers = 1


def _get_process_pool() -> ProcessPoolExecutor | None:
    global _process_pool, _pool_workers
    if _parallel_workers <= 1:
        return None
    if _process_pool is not None and _pool_workers == _parallel_workers:
        return _process_pool
    _shutdown_process_pool()
    _process_pool = ProcessPoolExecutor(max_workers=_parallel_workers)
    _pool_workers = _parallel_workers
    return _process_pool


def _evaluate_solution_payload(payload: tuple) -> tuple:
    instance, scheduling, machines = payload
    from .decoder import evaluate_solution_strings_fast

    return evaluate_solution_strings_fast(instance, list(scheduling), list(machines))


def _evaluate_pending_in_parallel(pending: list["Solution"]) -> list[tuple]:
    executor = _get_process_pool()
    if executor is None:
        return []
    payloads = [
        (
            solution.instance,
            tuple(solution.scheduling_string),
            tuple(solution.machine_assignment_string),
        )
        for solution in pending
    ]
    return list(executor.map(_evaluate_solution_payload, payloads))


atexit.register(_shutdown_process_pool)


def set_parallel_workers(workers: int | None) -> None:
    """
    Compatibility hook for configuring parallel evaluations.

    Current implementation always evaluates sequentially, but we keep the hook so
    callers do not break if they supply ``--eval-workers``.
    """
    global _parallel_workers
    new_workers = 1 if workers is None or workers <= 1 else int(workers)
    if new_workers == _parallel_workers:
        return
    _parallel_workers = new_workers
    _shutdown_process_pool()


def reset_global_counter() -> None:
    """
    Reset the global evaluation counter to zero.
    """
    global _evaluation_count
    _evaluation_count = 0
    _evaluation_cache.clear()


def get_evaluation_count() -> int:
    """
    Return the number of complete evaluations performed so far.
    """
    return _evaluation_count


def set_max_evaluations(value: int) -> None:
    """
    Update the global evaluation limit.
    """
    global MAX_EVALUATIONS
    MAX_EVALUATIONS = int(value)


def increment_evaluation_counter(amount: int = 1) -> None:
    """
    Increment the global evaluation counter by ``amount``.
    """
    global _evaluation_count
    _evaluation_count += amount


def is_termination_reached() -> bool:
    """
    Check whether the termination criterion based on ``MAX_EVALUATIONS`` is met.
    """
    return _evaluation_count >= MAX_EVALUATIONS


def calculate_objectives_with_count(solution: "Solution") -> None:
    """
    Decode and evaluate the provided solution, updating the evaluation counter.

    Parameters
    ----------
    solution:
        Solution instance that provides a ``calculate_objectives`` method. The
        method must compute makespan, energy, and agreement, and update the
        solution in-place.
    """
    if solution.evaluated:
        return
    key = _make_solution_cache_key(solution)
    cached = _evaluation_cache.get(key)
    if cached is not None:
        _evaluation_cache.move_to_end(key)
        _apply_cached_evaluation(solution, cached)
        # Keep evaluation-budget semantics aligned with "one attempted
        # objective evaluation per unevaluated solution", even on cache hits.
        increment_evaluation_counter()
        return
    solution.calculate_objectives()
    _store_cached_evaluation(solution)
    increment_evaluation_counter()


def evaluate_solutions_with_count(solutions):
    """
    Evaluate a sequence of solutions and count each unevaluated solution once.
    """
    pending = [sol for sol in solutions if sol is not None and not getattr(sol, "evaluated", False)]
    if not pending:
        return
    to_compute: list["Solution"] = []
    for sol in pending:
        key = _make_solution_cache_key(sol)
        cached = _evaluation_cache.get(key)
        if cached is not None:
            _evaluation_cache.move_to_end(key)
            _apply_cached_evaluation(sol, cached)
            continue
        to_compute.append(sol)
    if to_compute:
        from .decoder import evaluate_solution_strings_fast

        sample_ops = getattr(to_compute[0].instance, "total_operations", 0)
        estimated_work = len(to_compute) * max(int(sample_ops), 1)
        min_parallel_batch = max(128, _parallel_workers * 32)
        should_try_parallel = (
            _parallel_workers > 1
            and len(to_compute) >= min_parallel_batch
            and estimated_work >= 50_000
        )
        if should_try_parallel:
            try:
                results = _evaluate_pending_in_parallel(to_compute)
            except Exception:
                results = []
            if results:
                for sol, cached in zip(to_compute, results):
                    _apply_cached_evaluation(sol, cached)
                    _store_cached_evaluation(sol)
            else:
                for sol in to_compute:
                    cached = evaluate_solution_strings_fast(
                        sol.instance,
                        sol.scheduling_string,
                        sol.machine_assignment_string,
                    )
                    _apply_cached_evaluation(sol, cached)
                    _store_cached_evaluation(sol)
        else:
            for sol in to_compute:
                cached = evaluate_solution_strings_fast(
                    sol.instance,
                    sol.scheduling_string,
                    sol.machine_assignment_string,
                )
                _apply_cached_evaluation(sol, cached)
                _store_cached_evaluation(sol)
    # Count all pending evaluations, including cache hits.
    increment_evaluation_counter(len(pending))
