"""
Archive management utilities providing non-dominated sorting and crowding distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import EPS
from .solution import Solution

ROUND_DECIMALS = 4


def _ensure_objective_vector(sol: Solution) -> tuple:
    if not sol.evaluated:
        sol.evaluate()
    vector = sol.objective_vector
    if vector is not None:
        return vector
    vector = (
        sol.makespan._c1(),
        sol.makespan._c2(),
        sol.makespan._c3(),
        sol.energy._c1(),
        sol.energy._c2(),
        sol.energy._c3(),
        sol.agreement._c1(),
        sol.agreement._c2(),
        sol.agreement._c3(),
    )
    sol.objective_vector = vector
    return vector


def _compare_objective_vectors(v1: tuple, v2: tuple) -> tuple[int, int, int]:
    if v1[0] > v2[0] + EPS:
        ms_cmp = 1
    elif v1[0] < v2[0] - EPS:
        ms_cmp = -1
    elif v1[1] > v2[1] + EPS:
        ms_cmp = 1
    elif v1[1] < v2[1] - EPS:
        ms_cmp = -1
    elif v1[2] > v2[2] + EPS:
        ms_cmp = 1
    elif v1[2] < v2[2] - EPS:
        ms_cmp = -1
    else:
        ms_cmp = 0

    if v1[3] > v2[3] + EPS:
        en_cmp = 1
    elif v1[3] < v2[3] - EPS:
        en_cmp = -1
    elif v1[4] > v2[4] + EPS:
        en_cmp = 1
    elif v1[4] < v2[4] - EPS:
        en_cmp = -1
    elif v1[5] > v2[5] + EPS:
        en_cmp = 1
    elif v1[5] < v2[5] - EPS:
        en_cmp = -1
    else:
        en_cmp = 0

    if v1[6] > v2[6] + EPS:
        ag_cmp = 1
    elif v1[6] < v2[6] - EPS:
        ag_cmp = -1
    elif v1[7] > v2[7] + EPS:
        ag_cmp = 1
    elif v1[7] < v2[7] - EPS:
        ag_cmp = -1
    elif v1[8] > v2[8] + EPS:
        ag_cmp = 1
    elif v1[8] < v2[8] - EPS:
        ag_cmp = -1
    else:
        ag_cmp = 0
    return ms_cmp, en_cmp, ag_cmp


def _dominates_vectors(v1: tuple, v2: tuple) -> bool:
    ms_cmp, en_cmp, ag_cmp = _compare_objective_vectors(v1, v2)

    makespan_better_or_equal = ms_cmp <= 0
    energy_better_or_equal = en_cmp <= 0
    agreement_better_or_equal = ag_cmp <= 0

    makespan_strict = ms_cmp < 0
    energy_strict = en_cmp < 0
    agreement_strict = ag_cmp < 0

    return (
        makespan_better_or_equal
        and energy_better_or_equal
        and agreement_better_or_equal
        and (makespan_strict or energy_strict or agreement_strict)
    )


def _solution_key(sol: Solution) -> tuple:
    sol.evaluate()
    cached = getattr(sol, "objective_key", None)
    if cached is not None:
        return cached
    makespan_tuple = tuple(round(value, ROUND_DECIMALS) for value in sol.makespan.to_tuple())
    energy_tuple = tuple(round(value, ROUND_DECIMALS) for value in sol.energy.to_tuple())
    agreement_tuple = tuple(round(value, ROUND_DECIMALS) for value in sol.agreement.to_tuple())
    key = (makespan_tuple, energy_tuple, agreement_tuple)
    sol.objective_key = key
    return key


def dominates(sol1: Solution, sol2: Solution) -> bool:
    v1 = _ensure_objective_vector(sol1)
    v2 = _ensure_objective_vector(sol2)
    return _dominates_vectors(v1, v2)


def is_duplicate(sol1: Solution, sol2: Solution) -> bool:
    return _solution_key(sol1) == _solution_key(sol2)


def fast_non_dominated_sort(solutions: List[Solution], deduplicate: bool = True) -> List[List[Solution]]:
    # Optionally drop exact duplicates first to shrink comparison set
    if deduplicate:
        seen = set()
        unique_solutions: List[Solution] = []
        for sol in solutions:
            key = _solution_key(sol)
            if key in seen:
                continue
            seen.add(key)
            unique_solutions.append(sol)
    else:
        unique_solutions = solutions

    size = len(unique_solutions)
    if size == 0:
        return []
    vectors = [_ensure_objective_vector(sol) for sol in unique_solutions]

    domination_counts: List[int] = [0] * size
    dominated_sets: List[List[int]] = [[] for _ in range(size)]

    for i in range(size):
        vector_i = vectors[i]
        for j in range(i + 1, size):
            vector_j = vectors[j]
            ms_cmp, en_cmp, ag_cmp = _compare_objective_vectors(vector_i, vector_j)
            i_dominates_j = (
                ms_cmp <= 0 and en_cmp <= 0 and ag_cmp <= 0 and (ms_cmp < 0 or en_cmp < 0 or ag_cmp < 0)
            )
            if i_dominates_j:
                dominated_sets[i].append(j)
                domination_counts[j] += 1
                continue
            j_dominates_i = (
                ms_cmp >= 0 and en_cmp >= 0 and ag_cmp >= 0 and (ms_cmp > 0 or en_cmp > 0 or ag_cmp > 0)
            )
            if j_dominates_i:
                dominated_sets[j].append(i)
                domination_counts[i] += 1

    current_front_indices = [idx for idx, count in enumerate(domination_counts) if count == 0]
    fronts: List[List[Solution]] = []

    while current_front_indices:
        current_front = [unique_solutions[idx] for idx in current_front_indices]
        fronts.append(current_front)
        next_front_indices: List[int] = []
        for idx in current_front_indices:
            for dominated_idx in dominated_sets[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front_indices.append(dominated_idx)
        current_front_indices = next_front_indices

    return fronts


def calculate_crowding_distance(front: List[Solution]) -> Dict[int, float]:
    if not front:
        return {}
    size = len(front)
    distances = [0.0] * size
    ids = [id(sol) for sol in front]

    def update(values: list[float], minimize: bool) -> None:
        order = sorted(range(size), key=values.__getitem__, reverse=not minimize)
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        min_value = values[order[0]]
        max_value = values[order[-1]]
        if abs(max_value - min_value) <= EPS:
            return
        denom = max_value - min_value
        for idx in range(1, size - 1):
            current = order[idx]
            prev_value = values[order[idx - 1]]
            next_value = values[order[idx + 1]]
            distances[current] += (next_value - prev_value) / denom

    update([sol.makespan._c1() for sol in front], minimize=True)
    update([sol.energy._c1() for sol in front], minimize=True)
    update([sol.agreement._c1() for sol in front], minimize=True)

    return {ids[idx]: distances[idx] for idx in range(size)}


@dataclass
class Archive:
    solutions: List[Solution] = field(default_factory=list)
    opt_count_p1: int = 0
    opt_count_p2: int = 0
    solution_sources: Dict[int, int] = field(default_factory=dict)

    def _register_solution(self, solution: Solution, source_population_id: Optional[int]) -> None:
        if source_population_id == 1:
            self.opt_count_p1 += 1
            self.solution_sources[id(solution)] = 1
        elif source_population_id == 2:
            self.opt_count_p2 += 1
            self.solution_sources[id(solution)] = 2

    def _unregister_solution(self, solution: Solution) -> None:
        source = self.solution_sources.pop(id(solution), None)
        if source == 1:
            self.opt_count_p1 = max(self.opt_count_p1 - 1, 0)
        elif source == 2:
            self.opt_count_p2 = max(self.opt_count_p2 - 1, 0)

    def add_solution(self, solution: Solution, source_population_id: Optional[int]) -> bool:
        solution.evaluate()
        incoming_key = _solution_key(solution)
        incoming_vector = _ensure_objective_vector(solution)
        for existing in self.solutions:
            if _solution_key(existing) == incoming_key:
                return False
            if _dominates_vectors(_ensure_objective_vector(existing), incoming_vector):
                return False

        retained: List[Solution] = []
        for sol in self.solutions:
            if _dominates_vectors(incoming_vector, _ensure_objective_vector(sol)):
                self._unregister_solution(sol)
            else:
                retained.append(sol)
        self.solutions = retained
        self.solutions.append(solution)
        self._register_solution(solution, source_population_id)
        return True

    def remove_weakest_if_needed(self, max_size: int) -> None:
        if len(self.solutions) <= max_size:
            return
        fronts = fast_non_dominated_sort(self.solutions)
        new_solutions: List[Solution] = []
        for front in fronts:
            if len(new_solutions) + len(front) <= max_size:
                new_solutions.extend(front)
            else:
                distances = calculate_crowding_distance(front)
                front_sorted = sorted(front, key=lambda sol: distances[id(sol)], reverse=True)
                remaining = max_size - len(new_solutions)
                new_solutions.extend(front_sorted[:remaining])
                break
        kept_ids = {id(sol) for sol in new_solutions}
        for sol in self.solutions:
            if id(sol) not in kept_ids:
                self._unregister_solution(sol)
        self.solutions = new_solutions
