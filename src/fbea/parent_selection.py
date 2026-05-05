"""
Parent selection operators implementing Algorithm 2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .archive import calculate_crowding_distance, fast_non_dominated_sort
from .constants import EPS
from .random_manager import get_rng
from .solution import Solution


def calculate_F_values(solution: Solution) -> Tuple[float, float, float]:
    solution.evaluate()
    c1_makespan = max(solution.makespan._c1(), EPS)
    c1_energy = max(solution.energy._c1(), EPS)
    c1_dissatisfaction = max(solution.agreement._c1(), EPS)
    return (1.0 / c1_makespan, 1.0 / c1_dissatisfaction, 1.0 / c1_energy)


def _compute_ranks_and_crowding(solutions: Sequence[Solution]) -> Tuple[Dict[int, int], Dict[int, float]]:
    # Keep duplicates so every individual receives rank/crowding values.
    fronts = fast_non_dominated_sort(list(solutions), deduplicate=False)
    ranks: Dict[int, int] = {}
    crowding: Dict[int, float] = {}
    for idx, front in enumerate(fronts):
        distances = calculate_crowding_distance(front)
        for sol in front:
            ranks[id(sol)] = idx
            crowding[id(sol)] = distances.get(id(sol), float("inf"))
    return ranks, crowding


def binary_tournament_selection(
    population,
    ranks: Dict[int, int],
    crowding: Dict[int, float],
) -> Solution:
    # 二元锦标赛逻辑已移除防御，直接假定调用方提供足够的个体
    rng = get_rng()
    contenders = rng.sample(population.solutions, k=2)
    a, b = contenders[0], contenders[1]
    rank_a = ranks.get(id(a), float("inf"))
    rank_b = ranks.get(id(b), float("inf"))
    if rank_a < rank_b:
        return a
    if rank_b < rank_a:
        return b
    crowd_a = crowding.get(id(a), 0.0)
    crowd_b = crowding.get(id(b), 0.0)
    if crowd_a > crowd_b + EPS:
        return a
    if crowd_b > crowd_a + EPS:
        return b
    return a if rng.random() < 0.5 else b


def select_from_archive_by_objectives(archive, worse_objectives: Sequence[int]) -> Solution:
    rng = get_rng()
    if not archive or not archive.solutions:
        raise ValueError("Archive is empty.")

    objective_functions = {
        0: lambda sol: sol.makespan._c1(),
        1: lambda sol: sol.agreement._c1(),
        2: lambda sol: sol.energy._c1(),
    }

    candidates: List[Solution] = []
    for objective in worse_objectives:
        best_solution = min(
            archive.solutions,
            key=objective_functions[objective],
        )
        candidates.append(best_solution)
    return rng.choice(candidates)


def select_random_solution_from_archive(archive) -> Solution:
    rng = get_rng()
    if not archive or not archive.solutions:
        raise ValueError("Archive is empty.")
    return rng.choice(archive.solutions)


def algorithm2_parent_selection(
    population,
    archive,
    delta_i: float,
    target_size: int,
    beta_override: Optional[float] = None,
) -> List[Tuple[Solution, Solution]]:
    rng = get_rng()
    parent_pairs: List[Tuple[Solution, Solution]] = []
    ranks, crowding = _compute_ranks_and_crowding(population.solutions)
    beta = beta_override if beta_override is not None else 0.5 * (1.0 - delta_i)
    # Use the target size to control how many parent pairs are generated so reproduction
    # produces the intended next-generation population size.
    pairs_needed = max(target_size // 2, 0)
    while len(parent_pairs) < pairs_needed:
        alpha = rng.random()
        parent1 = binary_tournament_selection(population, ranks, crowding)
        if archive and archive.solutions and alpha <= beta:
            F_values = calculate_F_values(parent1)
            worse_indices = []
            for idx, (value, avg) in enumerate(zip(F_values, population.u_values)):
                if value < avg - EPS:
                    worse_indices.append(idx)
            if not worse_indices:
                parent2 = select_random_solution_from_archive(archive)
            elif len(worse_indices) == len(F_values):
                parent2 = select_random_solution_from_archive(archive)
            else:
                parent2 = select_from_archive_by_objectives(archive, worse_indices)
        else:
            parent2 = binary_tournament_selection(population, ranks, crowding)
        parent_pairs.append((parent1, parent2))
    return parent_pairs
