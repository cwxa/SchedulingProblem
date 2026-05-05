"""
Population management utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .archive import calculate_crowding_distance, fast_non_dominated_sort
from .evaluation_counter import calculate_objectives_with_count
from .instance_loader import Instance
from .random_manager import get_rng
from .solution import Solution


def generate_initial_solution(instance: Instance) -> Solution:
    solution = Solution(instance=instance)
    calculate_objectives_with_count(solution)
    return solution


def generate_random_initialized_solution(instance: Instance) -> Solution:
    """
    Generate a solution whose two strings are sampled purely at random.
    """
    from .heuristics import random_machine_assignment, random_scheduling_string

    scheduling = random_scheduling_string(instance)
    machines = random_machine_assignment(instance)
    solution = Solution(
        instance=instance,
        scheduling_string=scheduling,
        machine_assignment_string=machines,
    )
    calculate_objectives_with_count(solution)
    return solution


def generate_initial_solution_idle_mix_25(instance: Instance) -> Solution:
    """
    Generate a solution with machine-assignment strategy sampled uniformly from
    four rules: H1, H2, idle-first, and random.
    """
    from .heuristics import (
        heuristic_1_machine_assignment,
        heuristic_2_machine_assignment,
        heuristic_3_scheduling_string,
        heuristic_4_scheduling_string,
        heuristic_idle_first_machine_assignment,
        random_machine_assignment,
        random_scheduling_string,
    )

    rng = get_rng()
    alpha_machine = rng.random()
    if alpha_machine < 0.25:
        machine_assignment = heuristic_1_machine_assignment(instance)
    elif alpha_machine < 0.50:
        machine_assignment = heuristic_2_machine_assignment(instance)
    elif alpha_machine < 0.75:
        machine_assignment = heuristic_idle_first_machine_assignment(instance)
    else:
        machine_assignment = random_machine_assignment(instance)

    alpha_schedule = rng.random()
    if alpha_schedule < 0.4:
        scheduling = heuristic_3_scheduling_string(instance, machine_assignment)
    elif alpha_schedule < 0.8:
        scheduling = heuristic_4_scheduling_string(instance)
    else:
        scheduling = random_scheduling_string(instance)

    solution = Solution(
        instance=instance,
        scheduling_string=scheduling,
        machine_assignment_string=machine_assignment,
    )
    calculate_objectives_with_count(solution)
    return solution


@dataclass
class Population:
    population_id: int
    instance: Instance
    solutions: List[Solution] = field(default_factory=list)
    renew_i: int = 0
    u_values: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def add_solution(self, solution: Solution) -> None:
        if not solution.evaluated:
            calculate_objectives_with_count(solution)
        self.solutions.append(solution)

    def clear(self) -> None:
        self.solutions.clear()

    @property
    def size(self) -> int:
        return len(self.solutions)

    def elite_resize(self, new_size: int, archive, rng=None) -> None:
        rng = rng or get_rng()
        current_size = self.size
        if current_size == new_size:
            return
        if current_size > new_size:
            fronts = fast_non_dominated_sort(self.solutions)
            resized: List[Solution] = []
            for front in fronts:
                if len(resized) + len(front) <= new_size:
                    resized.extend(front)
                else:
                    distances = calculate_crowding_distance(front)
                    sorted_front = sorted(front, key=lambda sol: distances.get(id(sol), 0.0), reverse=True)
                    remaining = new_size - len(resized)
                    resized.extend(sorted_front[:remaining])
                    break
            self.solutions = resized
        else:
            additional_needed = new_size - current_size
            archive_candidates = list(archive.solutions) if archive else []
            archive_distances = calculate_crowding_distance(archive_candidates) if archive_candidates else {}
            archive_sorted = sorted(
                archive_candidates,
                key=lambda sol: archive_distances.get(id(sol), 0.0),
                reverse=True,
            )
            idx = 0
            while additional_needed > 0 and idx < len(archive_sorted):
                candidate = archive_sorted[idx]
                idx += 1
                if any(candidate is sol for sol in self.solutions):
                    continue
                self.solutions.append(candidate.clone())
                additional_needed -= 1

            while additional_needed > 0:
                new_solution = generate_initial_solution(self.instance)
                self.solutions.append(new_solution)
                additional_needed -= 1


def initialize_populations(
    instance: Instance,
    total_size: int,
    initializer: Callable[[Instance], Solution] | None = None,
) -> Tuple[Population, Population]:
    if total_size % 2 != 0:
        raise ValueError("Total population size must be even.")
    initializer = initializer or generate_initial_solution
    solutions = [initializer(instance) for _ in range(total_size)]
    half = total_size // 2
    p1 = Population(population_id=1, instance=instance, solutions=solutions[:half])
    p2 = Population(population_id=2, instance=instance, solutions=solutions[half:])
    return p1, p2
