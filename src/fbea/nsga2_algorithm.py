"""
Classic NSGA-II implementation adapted to the fuzzy multi-objective setting.

The baseline follows the same operator family as FBEA (GPX/POX crossover,
insert/swap/inverse + change1/2/3 mutations) but uses fixed probabilities so
that no feedback mechanism is involved. This keeps the comparison fair while
retaining a straightforward NSGA-II workflow (single population, elitist
environmental selection).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .archive import (
    ROUND_DECIMALS,
    _solution_key,
    calculate_crowding_distance,
    fast_non_dominated_sort,
)
from .algorithm3_crossover_mutation import (
    OperatorUsageInfo,
    apply_machine_mutation,
    apply_scheduling_mutation,
    gpx_crossover,
    pox_crossover,
    uniform_machine_crossover,
)
from .population import generate_random_initialized_solution
from .evaluation_counter import MAX_EVALUATIONS, evaluate_solutions_with_count, get_evaluation_count, reset_global_counter
from .random_manager import get_rng, set_global_seed
from .solution import Solution

FIXED_MODE_PROBS = (1 / 3, 1 / 3, 1 / 3)
FIXED_SCHED_MUTATION = (1 / 3, 1 / 3, 1 / 3)
FIXED_MACHINE_MUTATION = (1 / 3, 1 / 3, 1 / 3)


def _initialise_population(instance, population_size: int) -> List[Solution]:
    population: List[Solution] = []
    for _ in range(population_size):
        solution = generate_random_initialized_solution(instance)
        population.append(solution)
    return population


def _choose_mode(rng) -> str:
    r = rng.random()
    if r < FIXED_MODE_PROBS[0]:
        return "scheduling"
    if r < FIXED_MODE_PROBS[0] + FIXED_MODE_PROBS[1]:
        return "machine"
    return "both"


def _compute_ranks_and_crowding(population: Sequence[Solution]) -> Tuple[Dict[int, int], Dict[int, float], List[List[Solution]]]:
    # Do not deduplicate here; ranks must cover every individual in the population.
    fronts = fast_non_dominated_sort(list(population), deduplicate=False)
    ranks: Dict[int, int] = {}
    crowding: Dict[int, float] = {}
    for idx, front in enumerate(fronts):
        distances = calculate_crowding_distance(front)
        for sol in front:
            ranks[id(sol)] = idx
            crowding[id(sol)] = distances.get(id(sol), float("inf"))
    return ranks, crowding, fronts


def _binary_tournament(population: Sequence[Solution], ranks: Dict[int, int], crowding: Dict[int, float]) -> Solution:
    rng = get_rng()
    a, b = rng.sample(population, 2)
    rank_a = ranks[id(a)]
    rank_b = ranks[id(b)]
    if rank_a < rank_b:
        return a
    if rank_b < rank_a:
        return b
    crowd_a = crowding[id(a)]
    crowd_b = crowding[id(b)]
    if crowd_a > crowd_b:
        return a
    if crowd_b > crowd_a:
        return b
    return a if rng.random() < 0.5 else b


def _create_offspring_population(
    instance,
    population: Sequence[Solution],
    ranks: Dict[int, int],
    crowding: Dict[int, float],
    crossover_probability: float,
    mutation_probability: float,
) -> List[Solution]:
    rng = get_rng()
    offspring: List[Solution] = []
    population_size = len(population)
    while len(offspring) < population_size:
        parent1 = _binary_tournament(population, ranks, crowding)
        parent2 = _binary_tournament(population, ranks, crowding)

        sched1 = list(parent1.scheduling_string)
        sched2 = list(parent2.scheduling_string)
        mach1 = list(parent1.machine_assignment_string)
        mach2 = list(parent2.machine_assignment_string)

        if rng.random() < crossover_probability:
            mode = _choose_mode(rng)
            if mode in ("scheduling", "both"):
                if rng.random() < 0.5:
                    sched1, sched2, _ = gpx_crossover(instance, parent1, parent2)
                else:
                    sched1, sched2, _ = pox_crossover(instance, parent1, parent2)
            if mode in ("machine", "both"):
                mach1, mach2, _ = uniform_machine_crossover(parent1, parent2)

        if rng.random() < mutation_probability:
            mode = _choose_mode(rng)
            op_info = OperatorUsageInfo()
            if mode in ("scheduling", "both"):
                sched1 = apply_scheduling_mutation(sched1, FIXED_SCHED_MUTATION, op_info)
            if mode in ("machine", "both"):
                mach1 = apply_machine_mutation(instance, mach1, sched1, FIXED_MACHINE_MUTATION, op_info)
        if rng.random() < mutation_probability:
            mode = _choose_mode(rng)
            op_info = OperatorUsageInfo()
            if mode in ("scheduling", "both"):
                sched2 = apply_scheduling_mutation(sched2, FIXED_SCHED_MUTATION, op_info)
            if mode in ("machine", "both"):
                mach2 = apply_machine_mutation(instance, mach2, sched2, FIXED_MACHINE_MUTATION, op_info)

        child1 = Solution(
            instance=instance,
            scheduling_string=sched1,
            machine_assignment_string=mach1,
            skip_validation=True,
        )
        children = [child1]

        if len(offspring) + len(children) < population_size:
            child2 = Solution(
                instance=instance,
                scheduling_string=sched2,
                machine_assignment_string=mach2,
                skip_validation=True,
            )
            children.append(child2)

        offspring.extend(children)
    evaluate_solutions_with_count(offspring)
    return offspring


def _environmental_selection(population: List[Solution], population_size: int) -> List[Solution]:
    ranks, crowding, fronts = _compute_ranks_and_crowding(population)
    new_population: List[Solution] = []
    for front in fronts:
        if len(new_population) + len(front) <= population_size:
            new_population.extend(front)
        else:
            distances = {id(sol): crowding[id(sol)] for sol in front}
            sorted_front = sorted(front, key=lambda sol: distances.get(id(sol), 0.0), reverse=True)
            remaining = population_size - len(new_population)
            new_population.extend(sorted_front[:remaining])
            break
    return new_population


def _deduplicate_front(front: List[Solution]) -> List[Solution]:
    seen = set()
    unique: List[Solution] = []
    for sol in front:
        key = _solution_key(sol)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sol)
    return unique


def run_nsga2(
    instance,
    population_size: int = 50,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: Optional[int] = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    max_evaluations: Optional[int] = None,
    snapshot_callback: Optional[Callable[[int, List[Solution]], None]] = None,
) -> List[Solution]:
    if random_seed is not None:
        set_global_seed(random_seed)
    reset_global_counter()
    population = _initialise_population(instance, population_size)

    generation = 0
    evaluation_limit = max_evaluations or MAX_EVALUATIONS
    while get_evaluation_count() < evaluation_limit:
        generation += 1
        ranks, crowding, _ = _compute_ranks_and_crowding(population)
        offspring = _create_offspring_population(
            instance,
            population,
            ranks,
            crowding,
            crossover_probability,
            mutation_probability,
        )
        combined = population + offspring
        population = _environmental_selection(combined, population_size)
        if generation_callback is not None:
            generation_callback(generation, get_evaluation_count())
        if snapshot_callback is not None:
            snapshot_callback(generation, population)

    final_fronts = fast_non_dominated_sort(population)
    if not final_fronts:
        return []
    return _deduplicate_front(final_fronts[0])


__all__ = ["run_nsga2"]
