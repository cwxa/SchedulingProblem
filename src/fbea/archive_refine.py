"""
Archive refine component (plan C).
"""

from __future__ import annotations

from typing import Callable

from . import evaluation_counter as eval_counter
from .archive import (
    Archive,
    _ensure_objective_vector,
    calculate_crowding_distance,
    dominates,
    fast_non_dominated_sort,
)
from .evaluation_counter import get_evaluation_count
from .random_manager import get_rng
from .solution import Solution


def _objective_tuple(solution: Solution) -> tuple[float, float, float]:
    vector = _ensure_objective_vector(solution)
    return vector[0], vector[3], vector[6]


def _solution_edit_distance(base: Solution, candidate: Solution) -> int:
    scheduling_changes = sum(
        1
        for old, new in zip(base.scheduling_string, candidate.scheduling_string)
        if old != new
    )
    machine_changes = sum(
        1
        for old, new in zip(base.machine_assignment_string, candidate.machine_assignment_string)
        if old != new
    )
    return scheduling_changes + machine_changes


def _compare_refine_candidates(
    base: Solution,
    left: Solution,
    right: Solution,
) -> int:
    left_ms, left_en, left_ag = _objective_tuple(left)
    right_ms, right_en, right_ag = _objective_tuple(right)
    if left_ms < right_ms - 1e-12:
        return -1
    if left_ms > right_ms + 1e-12:
        return 1
    if left_en < right_en - 1e-12:
        return -1
    if left_en > right_en + 1e-12:
        return 1

    base_ag = _objective_tuple(base)[2]
    left_ag_penalty = 0 if left_ag + 1e-12 >= base_ag else 1
    right_ag_penalty = 0 if right_ag + 1e-12 >= base_ag else 1
    if left_ag_penalty < right_ag_penalty:
        return -1
    if left_ag_penalty > right_ag_penalty:
        return 1

    if left_ag > right_ag + 1e-12:
        return -1
    if left_ag < right_ag - 1e-12:
        return 1

    left_edit = _solution_edit_distance(base, left)
    right_edit = _solution_edit_distance(base, right)
    if left_edit < right_edit:
        return -1
    if left_edit > right_edit:
        return 1
    return 0


def _best_candidate_for_base(
    base: Solution,
    candidates: list[Solution],
) -> Solution | None:
    if not candidates:
        return None
    best = candidates[0]
    for candidate in candidates[1:]:
        if _compare_refine_candidates(base, candidate, best) < 0:
            best = candidate
    return best


def collect_archive_refine_candidates(
    instance,
    solution: Solution,
    *,
    candidate_topk: int,
    build_neighbor: Callable[[object, Solution], Solution | None],
) -> list[Solution]:
    if candidate_topk <= 0:
        return []
    candidates: list[Solution] = []
    seen_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    attempts = max(candidate_topk * 3, candidate_topk)
    for _ in range(attempts):
        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break
        neighbor = build_neighbor(instance, solution)
        if neighbor is None:
            continue
        try:
            neighbor.evaluate()
        except Exception:
            continue
        key = (
            tuple(neighbor.scheduling_string),
            tuple(neighbor.machine_assignment_string),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(neighbor)
        if len(candidates) >= candidate_topk:
            break
    return candidates


def build_archive_refine_neighbor(
    instance,
    solution: Solution,
    *,
    candidate_topk: int,
    build_neighbor: Callable[[object, Solution], Solution | None],
) -> Solution | None:
    candidates = collect_archive_refine_candidates(
        instance,
        solution,
        candidate_topk=candidate_topk,
        build_neighbor=build_neighbor,
    )
    return _best_candidate_for_base(solution, candidates)


def _select_archive_samples(archive: Archive, sample_ratio: float) -> list[Solution]:
    if not archive.solutions or sample_ratio <= 0:
        return []
    snapshot = list(archive.solutions)
    sample_size = int(len(snapshot) * sample_ratio + 0.5)
    sample_size = min(max(sample_size, 0), len(snapshot))
    if sample_size <= 0:
        return []

    fronts = fast_non_dominated_sort(snapshot, deduplicate=False)
    ranked: list[Solution] = []
    for front in fronts:
        distances = calculate_crowding_distance(front)
        ranked.extend(
            sorted(front, key=lambda sol: distances.get(id(sol), 0.0), reverse=True)
        )
    return ranked[:sample_size]


def apply_archive_refine(
    instance,
    archive: Archive,
    population_a,
    population_b,
    capacity_a: int,
    capacity_b: int,
    *,
    sample_ratio: float = 0.2,
    candidate_topk: int = 3,
    truncation_strategy: str = "original",
    build_neighbor: Callable[[object, Solution], Solution | None] | None = None,
    inject_solution: Callable[[object, Solution, int, str], None] | None = None,
) -> None:
    if build_neighbor is None or inject_solution is None:
        return
    selected = _select_archive_samples(archive, sample_ratio)
    if not selected:
        return
    rng = get_rng()
    for original in selected:
        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break
        best_neighbor = build_archive_refine_neighbor(
            instance,
            original,
            candidate_topk=candidate_topk,
            build_neighbor=build_neighbor,
        )
        if best_neighbor is None:
            continue

        source_population_id = archive.solution_sources.get(id(original), 0)
        dominates_old = dominates(best_neighbor, original)
        dominated_by_old = dominates(original, best_neighbor)
        should_inject = False

        if dominates_old:
            if original in archive.solutions:
                archive.solutions.remove(original)
                archive.solution_sources.pop(id(original), None)
            should_inject = archive.add_solution(best_neighbor, source_population_id)
        elif not dominated_by_old:
            should_inject = archive.add_solution(best_neighbor, source_population_id)

        if not should_inject:
            continue

        if source_population_id == 1:
            inject_solution(population_a, best_neighbor, capacity_a, truncation_strategy)
        elif source_population_id == 2:
            inject_solution(population_b, best_neighbor, capacity_b, truncation_strategy)
        elif rng.random() < 0.5:
            inject_solution(population_a, best_neighbor, capacity_a, truncation_strategy)
        else:
            inject_solution(population_b, best_neighbor, capacity_b, truncation_strategy)
