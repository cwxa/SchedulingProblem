"""
Main evolutionary loop (Algorithm 4) including enhanced local search.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .algorithm3_crossover_mutation import (
    OperatorUsageInfo,
    algorithm3_crossover_mutation,
    apply_machine_mutation,
    apply_scheduling_mutation,
)
from .archive import (
    Archive,
    calculate_crowding_distance,
    _ensure_objective_vector,
    _solution_key,
    dominates,
    fast_non_dominated_sort,
)
from .archive_refine import apply_archive_refine
from . import evaluation_counter as eval_counter
from .evaluation_counter import (
    evaluate_solutions_with_count,
    get_evaluation_count,
    is_termination_reached,
    reset_global_counter,
)
from .constants import EPS
from .feedback_statistics import (
    FeedbackStatistics,
    update_population_u_values,
)
from .local_search_q import LocalSearchQAgent, LocalSearchQConfig
from .struct_local_search import (
    StructLocalSearchQAgent,
    enhanced_local_search_struct,
)
from .parent_selection import (
    algorithm2_parent_selection,
    calculate_F_values,
    select_from_archive_by_objectives,
    select_random_solution_from_archive,
)
from .population import (
    Population,
    generate_initial_solution,
    generate_initial_solution_idle_mix_25,
    generate_random_initialized_solution,
    initialize_populations,
)
from .random_manager import get_rng, set_global_seed
from .solution import Solution
from .fuzzy_operations import TriangularFuzzyNumber


UNIFORM_OPERATOR_PROBABILITIES = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def _resolve_initialization_strategy(
    initialization_strategy: str,
    *,
    allow_unity_v2: bool = False,
):
    init_mode = initialization_strategy.lower()
    valid_modes = {"heuristic", "random", "idle_mix_25"}
    if allow_unity_v2:
        valid_modes.add("unity_v2")
    if init_mode not in valid_modes:
        raise ValueError(
            "initialization_strategy must be one of: "
            + ", ".join(f"'{mode}'" for mode in sorted(valid_modes))
            + "."
        )
    if init_mode == "random":
        return init_mode, generate_random_initialized_solution
    if init_mode == "idle_mix_25":
        return init_mode, generate_initial_solution_idle_mix_25
    if init_mode == "unity_v2":
        return init_mode, None
    return init_mode, generate_initial_solution


def _resolve_population_management_strategy(population_management_strategy: str) -> str:
    strategy = population_management_strategy.lower()
    valid_strategies = {"original", "v2_parent_selection_truncation"}
    if strategy not in valid_strategies:
        raise ValueError(
            "population_management_strategy must be one of: "
            + ", ".join(f"'{mode}'" for mode in sorted(valid_strategies))
            + "."
        )
    return strategy


def _resolve_replacement_strategy(replacement_strategy: str) -> str:
    strategy = replacement_strategy.lower()
    valid_strategies = {"original", "elitist_replacement"}
    if strategy not in valid_strategies:
        raise ValueError(
            "replacement_strategy must be one of: "
            + ", ".join(f"'{mode}'" for mode in sorted(valid_strategies))
            + "."
        )
    return strategy


def _select_parent_pairs_for_population(
    population: Population,
    archive: Archive,
    delta_i: float,
    target_size: int,
    beta_override: float | None,
    parent_selection_strategy: str,
):
    if parent_selection_strategy == "original":
        return algorithm2_parent_selection(
            population,
            archive,
            delta_i,
            target_size,
            beta_override=beta_override,
        )

    from fbea_unity_v2 import select_parents_p1, select_parents_p2

    pairs_needed = max(target_size // 2, 0)
    if population.population_id == 1:
        return select_parents_p1(population, pairs_needed)
    if population.population_id == 2:
        return select_parents_p2(population, archive, pairs_needed)
    raise ValueError("population_id must be either 1 or 2.")


def _select_ms_ego_best_donor(
    guide_solution: Solution,
    population: Population,
    archive: Archive,
) -> Solution:
    if archive and archive.solutions:
        f_values = calculate_F_values(guide_solution)
        worse_indices: list[int] = []
        for idx, (value, avg) in enumerate(zip(f_values, population.u_values)):
            if value < avg - EPS:
                worse_indices.append(idx)
        if not worse_indices or len(worse_indices) == len(f_values):
            return select_random_solution_from_archive(archive)
        return select_from_archive_by_objectives(archive, worse_indices)
    return guide_solution


def _select_ms_ego_random_donor(population: Population, fallback_solution: Solution) -> Solution:
    rng = get_rng()
    if population.solutions:
        return rng.choice(population.solutions)
    return fallback_solution


def _apply_ms_ego_lite(
    offspring: Solution,
    worst_solution: Solution | None,
    best_donor: Solution,
    random_donor: Solution,
    *,
    random_fill_probability: float,
) -> None:
    if worst_solution is None:
        return

    current_ms = list(offspring.machine_assignment_string)
    worst_ms = worst_solution.machine_assignment_string
    best_ms = best_donor.machine_assignment_string
    random_ms = random_donor.machine_assignment_string
    rng = get_rng()

    replaced = 0
    for idx, machine_id in enumerate(current_ms):
        if machine_id != worst_ms[idx]:
            continue
        donor_ms = random_ms if rng.random() < random_fill_probability else best_ms
        current_ms[idx] = donor_ms[idx]
        replaced += 1

    if replaced == 0:
        return

    offspring.machine_assignment_string = current_ms
    offspring.evaluated = False
    offspring.makespan = None
    offspring.energy = None
    offspring.agreement = None
    offspring.schedule = None
    offspring.objective_key = None
    offspring.objective_vector = None


def one_generation_reproduction_with_feedback_single(
    population: Population,
    other_population: Population,
    archive: Archive,
    delta_i: float,
    crossover_probability: float,
    mutation_probability: float,
    feedback_stats: FeedbackStatistics,
    target_size: int,
    beta_override: float | None = None,
    use_feedback_statistics: bool = True,
    parent_selection_strategy: str = "original",
    truncation_strategy: str = "original",
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
) -> Population:
    new_population = Population(population.population_id, population.instance)
    parents = _select_parent_pairs_for_population(
        population,
        archive,
        delta_i,
        target_size,
        beta_override=beta_override,
        parent_selection_strategy=parent_selection_strategy,
    )
    offspring_records = []
    offspring_batch: list[Solution] = []
    target_batch_size = max(target_size, 0)
    if target_batch_size == 0:
        new_population.renew_i = 0
        return new_population
    improved_total = 0
    rng = get_rng()
    worst_index = _find_worst_solution_index(population) if population.solutions else None
    worst_solution = population.solutions[worst_index] if worst_index is not None else None
    if use_feedback_statistics:
        q_bar_1 = feedback_stats.q_bar_1
        q_bar_2 = feedback_stats.q_bar_2
        w_bar_1 = feedback_stats.w_bar_1
        w_bar_2 = feedback_stats.w_bar_2
    else:
        q_bar_1 = 0.5
        q_bar_2 = 0.5
        w_bar_1 = UNIFORM_OPERATOR_PROBABILITIES
        w_bar_2 = UNIFORM_OPERATOR_PROBABILITIES
    for parent_x, parent_y in parents:
        offspring_x, offspring_y, operator_info_x, operator_info_y = algorithm3_crossover_mutation(
            population.instance,
            parent_x,
            parent_y,
            crossover_probability,
            mutation_probability,
            q_bar_1,
            q_bar_2,
            w_bar_1,
            w_bar_2,
            evaluate_offspring=False,
        )
        if ms_ego_enabled:
            if rng.random() < ms_ego_probability:
                _apply_ms_ego_lite(
                    offspring_x,
                    worst_solution,
                    _select_ms_ego_best_donor(parent_x, population, archive),
                    _select_ms_ego_random_donor(population, parent_x),
                    random_fill_probability=ms_ego_random_fill_probability,
                )
            if rng.random() < ms_ego_probability:
                _apply_ms_ego_lite(
                    offspring_y,
                    worst_solution,
                    _select_ms_ego_best_donor(parent_y, population, archive),
                    _select_ms_ego_random_donor(population, parent_y),
                    random_fill_probability=ms_ego_random_fill_probability,
                )
        offspring_records.append(
            (parent_x, parent_y, offspring_x, offspring_y, operator_info_x, operator_info_y)
        )
        offspring_batch.append(offspring_x)
        offspring_batch.append(offspring_y)
        if len(offspring_batch) >= target_batch_size:
            break

    evaluate_solutions_with_count(offspring_batch)

    for parent_x, parent_y, offspring_x, offspring_y, operator_info_x, operator_info_y in offspring_records:
        improved = 0
        offspring_x_dominates = dominates(offspring_x, parent_x)
        parent_x_dominates = dominates(parent_x, offspring_x)
        if offspring_x_dominates:
            if use_feedback_statistics:
                feedback_stats.update_crossover_stats(operator_info_x)
                feedback_stats.update_mutation_stats(operator_info_x)
            improved += 1
        if not parent_x_dominates:
            archive.add_solution(offspring_x, population.population_id)

        offspring_y_dominates = dominates(offspring_y, parent_y)
        parent_y_dominates = dominates(parent_y, offspring_y)
        if offspring_y_dominates:
            if use_feedback_statistics:
                feedback_stats.update_crossover_stats(operator_info_y)
                feedback_stats.update_mutation_stats(operator_info_y)
            improved += 1
        if not parent_y_dominates:
            archive.add_solution(offspring_y, population.population_id)

        improved_total += improved
        new_population.add_solution(offspring_x)
        new_population.add_solution(offspring_y)
        if len(new_population.solutions) >= target_size:
            break
    _truncate_population_with_strategy(new_population, target_size, truncation_strategy)
    new_population.renew_i = improved_total
    return new_population


def _compute_injection_probabilities(
    delta_values,
    eta_values,
    population_a: Population,
    population_b: Population,
) -> tuple[float, float]:
    delta_1, delta_2 = delta_values
    eta_1, eta_2 = eta_values
    u_a = population_a.u_values
    u_b = population_b.u_values

    def compute_u_term(u_i, u_j):
        total = 0.0
        for val_i, val_j in zip(u_i, u_j):
            denom = val_i + val_j
            if denom > 0:
                total += val_i / denom
        return total / 3.0 if total > 0 else 0.0

    term_p1 = compute_u_term(u_a, u_b)
    term_p2 = compute_u_term(u_b, u_a)

    u1 = (delta_1 + eta_1 + term_p1) / 3.0
    u2 = (delta_2 + eta_2 + term_p2) / 3.0

    total = u1 + u2
    if total <= 0:
        return 0.5, 0.5
    return u1 / total, u2 / total


def _calculate_delta_from_capacity(
    archive: Archive,
    population_id: int,
    capacity_i: int,
    capacity_other: int,
) -> float:
    if archive is None:
        return 0.5
    ratio_i = (
        archive.opt_count_p1 / max(capacity_i, EPS)
        if population_id == 1
        else archive.opt_count_p2 / max(capacity_i, EPS)
    )
    ratio_other = (
        archive.opt_count_p2 / max(capacity_other, EPS)
        if population_id == 1
        else archive.opt_count_p1 / max(capacity_other, EPS)
    )
    denom = ratio_i + ratio_other
    if denom <= EPS:
        return 0.5
    return ratio_i / denom


def _calculate_eta_from_capacity(
    renew_i: int,
    renew_other: int,
    capacity_i: int,
    capacity_other: int,
) -> float:
    rate_i = renew_i / max(capacity_i, EPS)
    rate_other = renew_other / max(capacity_other, EPS)
    denom = rate_i + rate_other
    if denom <= EPS:
        return 0.5
    return rate_i / denom


def _calculate_new_population_sizes_from_capacity(
    total_size: int,
    capacity_p1: int,
    capacity_p2: int,
    delta_values: tuple[float, float],
    eta_values: tuple[float, float],
    u_values_p1: tuple[float, float, float],
    u_values_p2: tuple[float, float, float],
) -> tuple[int, int]:
    if total_size == 0:
        return 0, 0
    delta_1, delta_2 = delta_values
    eta_1, eta_2 = eta_values
    u11, u12, u13 = u_values_p1
    u21, u22, u23 = u_values_p2

    def safe_frac(num: float, denom: float) -> float:
        return num / max(denom, EPS)

    term_p1 = (
        safe_frac(u11, u11 + u21)
        + safe_frac(u12, u12 + u22)
        + safe_frac(u13, u13 + u23)
    ) / 3

    n1_real = total_size / 4 + (total_size / 8) * (
        delta_1 + eta_1 + term_p1 + capacity_p1 / max(total_size, EPS)
    )
    n1_new = math.ceil(n1_real / 2.0) * 2
    n1_new = max(n1_new, total_size // 4)
    n1_new = min(n1_new, total_size - 2)
    if n1_new < 2:
        n1_new = 2
    if n1_new % 2 != 0:
        if n1_new + 1 <= total_size - 2:
            n1_new += 1
        else:
            n1_new -= 1
    n2_new = total_size - n1_new
    if n2_new < 2:
        n2_new = 2
        n1_new = total_size - n2_new
    if n2_new % 2 != 0:
        n2_new += 1
        n1_new -= 1
    return n1_new, n2_new


def _find_worst_solution_index(population: Population) -> int | None:
    if population.size == 0:
        return None
    solutions = population.solutions
    size = len(solutions)
    vectors = [_ensure_objective_vector(sol) for sol in solutions]

    domination_counts: list[int] = [0] * size
    dominated_sets: list[list[int]] = [[] for _ in range(size)]

    eps = EPS

    def compare_pair(v1: tuple, v2: tuple) -> tuple[int, int, int]:
        if v1[0] > v2[0] + eps:
            ms_cmp = 1
        elif v1[0] < v2[0] - eps:
            ms_cmp = -1
        elif v1[1] > v2[1] + eps:
            ms_cmp = 1
        elif v1[1] < v2[1] - eps:
            ms_cmp = -1
        elif v1[2] > v2[2] + eps:
            ms_cmp = 1
        elif v1[2] < v2[2] - eps:
            ms_cmp = -1
        else:
            ms_cmp = 0

        if v1[3] > v2[3] + eps:
            en_cmp = 1
        elif v1[3] < v2[3] - eps:
            en_cmp = -1
        elif v1[4] > v2[4] + eps:
            en_cmp = 1
        elif v1[4] < v2[4] - eps:
            en_cmp = -1
        elif v1[5] > v2[5] + eps:
            en_cmp = 1
        elif v1[5] < v2[5] - eps:
            en_cmp = -1
        else:
            en_cmp = 0

        if v1[6] > v2[6] + eps:
            ag_cmp = 1
        elif v1[6] < v2[6] - eps:
            ag_cmp = -1
        elif v1[7] > v2[7] + eps:
            ag_cmp = 1
        elif v1[7] < v2[7] - eps:
            ag_cmp = -1
        elif v1[8] > v2[8] + eps:
            ag_cmp = 1
        elif v1[8] < v2[8] - eps:
            ag_cmp = -1
        else:
            ag_cmp = 0
        return ms_cmp, en_cmp, ag_cmp

    for i in range(size):
        vector_i = vectors[i]
        dominated_i = dominated_sets[i]
        for j in range(i + 1, size):
            vector_j = vectors[j]
            ms_cmp, en_cmp, ag_cmp = compare_pair(vector_i, vector_j)
            i_dominates_j = (
                ms_cmp <= 0 and en_cmp <= 0 and ag_cmp <= 0 and (ms_cmp < 0 or en_cmp < 0 or ag_cmp < 0)
            )
            if i_dominates_j:
                dominated_i.append(j)
                domination_counts[j] += 1
                continue
            j_dominates_i = (
                ms_cmp >= 0 and en_cmp >= 0 and ag_cmp >= 0 and (ms_cmp > 0 or en_cmp > 0 or ag_cmp > 0)
            )
            if j_dominates_i:
                dominated_sets[j].append(i)
                domination_counts[i] += 1

    current_front: list[int] = [idx for idx, count in enumerate(domination_counts) if count == 0]
    worst_front: list[int] = current_front
    while current_front:
        worst_front = current_front
        next_front: list[int] = []
        for idx in current_front:
            for dominated_idx in dominated_sets[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current_front = next_front

    return _select_worst_index_by_crowding(vectors, worst_front)


def _select_worst_index_by_crowding(vectors: list[tuple], front_indices: list[int]) -> int | None:
    size = len(front_indices)
    if size == 0:
        return None
    if size == 1:
        return front_indices[0]

    distances = [0.0] * size

    def update_distances(values: list[float], minimize: bool) -> None:
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

    update_distances([vectors[idx][0] for idx in front_indices], minimize=True)
    update_distances([vectors[idx][3] for idx in front_indices], minimize=True)
    update_distances([vectors[idx][6] for idx in front_indices], minimize=True)

    local_worst = min(range(size), key=distances.__getitem__)
    return front_indices[local_worst]


def _truncate_population_by_rank_and_crowding(population: Population, max_size: int) -> None:
    if population.size <= max_size:
        return
    fronts = fast_non_dominated_sort(population.solutions, deduplicate=False)
    kept: list[Solution] = []
    for front in fronts:
        if len(kept) + len(front) <= max_size:
            kept.extend(front)
            continue
        distances = calculate_crowding_distance(front)
        sorted_front = sorted(front, key=lambda sol: distances.get(id(sol), 0.0), reverse=True)
        remaining = max_size - len(kept)
        kept.extend(sorted_front[:remaining])
        break
    population.solutions = kept


def _truncate_population_with_strategy(
    population: Population,
    max_size: int,
    truncation_strategy: str,
) -> None:
    if truncation_strategy == "original":
        _truncate_population_by_rank_and_crowding(population, max_size)
        return

    from fbea_unity_v2 import truncate_population_p1, truncate_population_p2

    if population.population_id == 1:
        truncate_population_p1(population, max_size)
        return
    if population.population_id == 2:
        truncate_population_p2(population, max_size)
        return
    raise ValueError("population_id must be either 1 or 2.")


def _select_next_population(
    old_population: Population,
    offspring_population: Population,
    target_size: int,
    replacement_strategy: str,
    truncation_strategy: str,
) -> Population:
    if replacement_strategy == "original":
        return offspring_population

    selected_population = Population(
        old_population.population_id,
        old_population.instance,
        solutions=list(old_population.solutions) + list(offspring_population.solutions),
    )
    _truncate_population_with_strategy(
        selected_population,
        target_size,
        truncation_strategy,
    )
    selected_population.renew_i = offspring_population.renew_i
    return selected_population


def _inject_solution_into_population(
    population: Population,
    solution: Solution,
    max_size: int,
    truncation_strategy: str = "original",
) -> None:
    solution.evaluate()
    clone = solution.clone()
    population.add_solution(clone)
    _truncate_population_with_strategy(population, max_size, truncation_strategy)


def _build_local_search_q_agent(
    local_search_policy: str,
    local_search_q_config: Optional[dict[str, object]] = None,
) -> LocalSearchQAgent | None:
    policy = local_search_policy.lower()
    if policy not in {"random", "q"}:
        raise ValueError("local_search_policy must be either 'random' or 'q'.")
    if policy == "random":
        return None
    config = LocalSearchQConfig()
    if local_search_q_config:
        try:
            config = LocalSearchQConfig(**local_search_q_config)
        except TypeError as exc:
            raise ValueError(f"Invalid local_search_q_config: {exc}") from exc
    return LocalSearchQAgent(config)


def _deterministic_operator_probabilities(operator_idx: int) -> tuple[float, float, float]:
    if operator_idx == 0:
        return (1.0, 0.0, 0.0)
    if operator_idx == 1:
        return (0.0, 1.0, 0.0)
    if operator_idx == 2:
        return (0.0, 0.0, 1.0)
    raise ValueError("operator_idx must be 0, 1, or 2.")


def enhanced_local_search(
    instance,
    archive: Archive,
    population_a: Population,
    population_b: Population,
    capacity_a: int,
    capacity_b: int,
    feedback_stats: FeedbackStatistics,
    delta_values,
    eta_values,
    use_feedback_probabilities: bool = True,
    use_feedback_statistics: bool = True,
    q_agent: LocalSearchQAgent | None = None,
    truncation_strategy: str = "original",
) -> None:
    if not archive.solutions:
        return
    rng = get_rng()
    if use_feedback_probabilities:
        prob_p1, _ = _compute_injection_probabilities(
            delta_values,
            eta_values,
            population_a,
            population_b,
        )
    else:
        prob_p1 = 0.5
    solutions_snapshot = list(archive.solutions)
    progress = get_evaluation_count() / max(eval_counter.MAX_EVALUATIONS, 1)
    pending_pairs: list[
        tuple[Solution, Solution, OperatorUsageInfo, int | None, int | None, int]
    ] = []
    pending_neighbors: list[Solution] = []
    for solution in solutions_snapshot:
        if eval_counter.get_evaluation_count() + len(pending_neighbors) >= eval_counter.MAX_EVALUATIONS:
            break
        scheduling = list(solution.scheduling_string)
        machines = list(solution.machine_assignment_string)
        source_population_id = archive.solution_sources.get(id(solution), 0)
        state_index: int | None = None
        action_index: int | None = None
        mode = rng.choice(["scheduling", "machine", "both"])
        scheduling_operator_idx: int | None = None
        machine_operator_idx: int | None = None
        if q_agent is not None:
            state_index = q_agent.current_state()
            action_index = q_agent.select_action(state_index, progress, rng)
            mode, scheduling_operator_idx, machine_operator_idx = q_agent.action_from_index(action_index)
        operator_info = OperatorUsageInfo()

        if mode in ("scheduling", "both"):
            default_sched_probs = (
                feedback_stats.w_bar_1 if use_feedback_statistics else UNIFORM_OPERATOR_PROBABILITIES
            )
            probs_sched = (
                default_sched_probs
                if scheduling_operator_idx is None
                else _deterministic_operator_probabilities(scheduling_operator_idx)
            )
            scheduling = apply_scheduling_mutation(
                scheduling,
                probs_sched,
                operator_info,
            )
        if mode in ("machine", "both"):
            default_machine_probs = (
                feedback_stats.w_bar_2 if use_feedback_statistics else UNIFORM_OPERATOR_PROBABILITIES
            )
            probs_machine = (
                default_machine_probs
                if machine_operator_idx is None
                else _deterministic_operator_probabilities(machine_operator_idx)
            )
            machines = apply_machine_mutation(
                instance,
                machines,
                scheduling,
                probs_machine,
                operator_info,
            )

        neighbor = Solution(
            instance=instance,
            scheduling_string=scheduling,
            machine_assignment_string=machines,
        )
        pending_pairs.append((solution, neighbor, operator_info, state_index, action_index, source_population_id))
        pending_neighbors.append(neighbor)

    evaluate_solutions_with_count(pending_neighbors)

    for solution, neighbor, operator_info, state_index, action_index, source_population_id in pending_pairs:
        dominates_old = dominates(neighbor, solution)
        dominated_by_old = dominates(solution, neighbor)
        archive_added = False

        if dominates_old:
            if solution in archive.solutions:
                archive.solutions.remove(solution)
            archive_added = archive.add_solution(neighbor, archive.solution_sources.get(id(solution)))
            should_inject = True
        elif not dominated_by_old:
            archive_added = archive.add_solution(neighbor, archive.solution_sources.get(id(solution)))
            should_inject = True
        else:
            should_inject = False

        if should_inject:
            if rng.random() < prob_p1:
                _inject_solution_into_population(
                    population_a,
                    neighbor,
                    capacity_a,
                    truncation_strategy=truncation_strategy,
                )
            else:
                _inject_solution_into_population(
                    population_b,
                    neighbor,
                    capacity_b,
                    truncation_strategy=truncation_strategy,
                )

        if q_agent is not None and action_index is not None:
            reward = q_agent.compute_reward(
                dominates_old=dominates_old,
                dominated_by_old=dominated_by_old,
                archive_added=archive_added,
            )
            if state_index is not None:
                q_agent.update(state_index, action_index, reward, next_state=action_index)
            q_agent.set_previous_action(action_index)


def _validate_solution_strings_for_gap_strategy(
    instance,
    scheduling_string: list[int],
    machine_assignment_string: list[int],
) -> bool:
    if len(scheduling_string) != instance.total_operations:
        return False
    if len(machine_assignment_string) != instance.total_operations:
        return False

    counts = {job_id: 0 for job_id in range(1, instance.num_jobs + 1)}
    for job_id in scheduling_string:
        if job_id not in counts:
            return False
        counts[job_id] += 1

    for job_id, expected in instance.job_operation_counts.items():
        if counts.get(job_id, 0) != expected:
            return False

    valid_machine_options = instance.operation_machine_options
    for pos, machine_id in enumerate(machine_assignment_string):
        if machine_id not in valid_machine_options[pos]:
            return False
    return True


def _move_job_occurrence_earlier(
    scheduling_string: list[int],
    job_id: int,
    operation_idx: int,
    desired_position: int,
) -> list[int] | None:
    positions: list[int] = [idx for idx, value in enumerate(scheduling_string) if value == job_id]
    if operation_idx >= len(positions):
        return None
    current_pos = positions[operation_idx]
    if desired_position >= current_pos:
        return None

    lower_bound = positions[operation_idx - 1] + 1 if operation_idx > 0 else 0
    target_pos = max(lower_bound, desired_position)
    if target_pos >= current_pos:
        return None

    updated = list(scheduling_string)
    token = updated.pop(current_pos)
    updated.insert(target_pos, token)
    return updated


def _extract_machine_gaps_for_schedule(schedule) -> list[tuple[int, TriangularFuzzyNumber, TriangularFuzzyNumber, int]]:
    # Returns tuples: (machine_id, gap_start, gap_end, insert_before_step_idx)
    gaps: list[tuple[int, TriangularFuzzyNumber, TriangularFuzzyNumber, int]] = []
    zero = TriangularFuzzyNumber(0.0, 0.0, 0.0)
    step_index_by_operation = {
        (op.job_id, op.operation_idx): idx for idx, op in enumerate(schedule.operations)
    }

    for machine_id, machine_schedule in schedule.machines.items():
        operations = sorted(machine_schedule.operations, key=lambda op: op.start_time.m)
        if not operations:
            continue

        first = operations[0]
        if first.start_time.m > EPS:
            insert_before = step_index_by_operation[(first.job_id, first.operation_idx)]
            gaps.append((machine_id, zero, first.start_time, insert_before))

        prev_completion = first.completion_time
        for op in operations[1:]:
            if op.start_time.m > prev_completion.m + EPS:
                insert_before = step_index_by_operation[(op.job_id, op.operation_idx)]
                gaps.append((machine_id, prev_completion, op.start_time, insert_before))
            prev_completion = op.completion_time
    return gaps


def _try_build_gap_strategy_neighbor(instance, solution: Solution) -> Solution | None:
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None or not schedule.operations:
        return None

    rng = get_rng()
    zero = TriangularFuzzyNumber(0.0, 0.0, 0.0)
    operation_map = {(op.job_id, op.operation_idx): op for op in schedule.operations}
    gaps = _extract_machine_gaps_for_schedule(schedule)
    if not gaps:
        return None
    gaps.sort(key=lambda item: item[2].m - item[1].m, reverse=True)

    operation_indices = list(range(len(schedule.operations)))
    rng.shuffle(operation_indices)
    base_scheduling = list(solution.scheduling_string)
    base_machines = list(solution.machine_assignment_string)

    for machine_id, gap_start, gap_end, insert_before in gaps:
        for op_idx_in_list in operation_indices:
            op = schedule.operations[op_idx_in_list]
            if op.start_time.m < gap_end.m - EPS:
                continue

            position = instance.get_operation_position(op.job_id, op.operation_idx)
            processing_map = instance.operation_processing_map[position]
            if machine_id not in processing_map:
                continue

            predecessor_completion = (
                zero
                if op.operation_idx == 0
                else operation_map[(op.job_id, op.operation_idx - 1)].completion_time
            )
            candidate_start = predecessor_completion.max_with(gap_start)
            candidate_processing = processing_map[machine_id]
            candidate_completion = candidate_start.add(candidate_processing)
            if not candidate_completion.leq(gap_end):
                continue

            updated_scheduling = _move_job_occurrence_earlier(
                base_scheduling,
                op.job_id,
                op.operation_idx,
                insert_before,
            )
            if updated_scheduling is None:
                continue

            updated_machines = list(base_machines)
            updated_machines[position] = machine_id
            if not _validate_solution_strings_for_gap_strategy(
                instance, updated_scheduling, updated_machines
            ):
                continue

            neighbor = Solution(
                instance=instance,
                scheduling_string=updated_scheduling,
                machine_assignment_string=updated_machines,
                skip_validation=True,
            )
            try:
                neighbor.evaluate()
            except Exception:
                continue
            return neighbor
    return None


def apply_gap_strategy(
    instance,
    archive: Archive,
    population_a: Population,
    population_b: Population,
    capacity_a: int,
    capacity_b: int,
    sample_ratio: float = 0.2,
    truncation_strategy: str = "original",
) -> None:
    if not archive.solutions or sample_ratio <= 0:
        return
    rng = get_rng()
    snapshot = list(archive.solutions)
    sample_size = int(len(snapshot) * sample_ratio + 0.5)
    sample_size = min(max(sample_size, 0), len(snapshot))
    if sample_size <= 0:
        return

    selected = rng.sample(snapshot, sample_size)
    for original in selected:
        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break

        neighbor = _try_build_gap_strategy_neighbor(instance, original)
        if neighbor is None:
            continue

        source_population_id = archive.solution_sources.get(id(original), 0)
        dominates_old = dominates(neighbor, original)
        dominated_by_old = dominates(original, neighbor)

        if dominates_old:
            if original in archive.solutions:
                archive.solutions.remove(original)
            should_inject = archive.add_solution(neighbor, source_population_id)
        elif not dominated_by_old:
            should_inject = archive.add_solution(neighbor, source_population_id)
        else:
            should_inject = False

        if should_inject:
            if source_population_id == 1:
                _inject_solution_into_population(
                    population_a,
                    neighbor,
                    capacity_a,
                    truncation_strategy=truncation_strategy,
                )
            elif source_population_id == 2:
                _inject_solution_into_population(
                    population_b,
                    neighbor,
                    capacity_b,
                    truncation_strategy=truncation_strategy,
                )
            elif rng.random() < 0.5:
                _inject_solution_into_population(
                    population_a,
                    neighbor,
                    capacity_a,
                    truncation_strategy=truncation_strategy,
                )
            else:
                _inject_solution_into_population(
                    population_b,
                    neighbor,
                    capacity_b,
                    truncation_strategy=truncation_strategy,
                )


def _initialize_populations_for_main_loop(
    instance,
    total_population_size: int,
    *,
    initialization_strategy: str,
    initial_dedup: bool = False,
) -> tuple[Population, Population]:
    init_mode, initializer = _resolve_initialization_strategy(
        initialization_strategy,
        allow_unity_v2=True,
    )
    if not initial_dedup:
        if init_mode == "unity_v2":
            from fbea_unity_v2 import initialize_populations_unity

            return initialize_populations_unity(instance, total_population_size)
        return initialize_populations(instance, total_population_size, initializer=initializer)

    if total_population_size % 2 != 0:
        raise ValueError("Total population size must be even.")

    if init_mode == "unity_v2":
        from fbea_unity_v2 import initialize_populations_unity

        base_p1, base_p2 = initialize_populations_unity(instance, total_population_size)
        solutions = list(base_p1.solutions) + list(base_p2.solutions)
    else:
        initializer = initializer or generate_initial_solution
        solutions = [initializer(instance) for _ in range(total_population_size)]

    seen_keys = set()
    unique: list[Solution] = []
    for sol in solutions:
        key = _solution_key(sol)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(sol)

    max_backfill_attempts = max(total_population_size * 50, 100)
    attempts = 0
    while len(unique) < total_population_size and attempts < max_backfill_attempts:
        attempts += 1
        candidate = generate_random_initialized_solution(instance)
        key = _solution_key(candidate)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(candidate)

    while len(unique) < total_population_size:
        # Tiny instances may not have enough unique initial solutions. In that
        # case, keep the requested population size by allowing duplicate backfill.
        unique.append(generate_random_initialized_solution(instance))

    half = total_population_size // 2
    p1 = Population(population_id=1, instance=instance, solutions=unique[:half])
    p2 = Population(population_id=2, instance=instance, solutions=unique[half:])
    return p1, p2


def _initialize_populations_with_initial_dedup(
    instance,
    total_population_size: int,
    initializer: Callable[[object], Solution] | None = None,
) -> tuple[Population, Population]:
    """
    Backward-compatible helper for the old init-dedup path.
    """
    strategy = "heuristic"
    if initializer is generate_random_initialized_solution:
        strategy = "random"
    elif initializer is generate_initial_solution_idle_mix_25:
        strategy = "idle_mix_25"
    return _initialize_populations_for_main_loop(
        instance,
        total_population_size,
        initialization_strategy=strategy,
        initial_dedup=True,
    )


def fbea_main_algorithm(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    initial_dedup: bool = False,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
    population_management_strategy: str = "original",
    replacement_strategy: str = "original",
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
    struct_local_search_enabled: bool = False,
    struct_local_search_action_mode: str = "addon19",
    archive_refine_enabled: bool = False,
    archive_refine_interval: int = 50,
    archive_refine_sample_ratio: float = 0.2,
    archive_refine_candidate_topk: int = 3,
):
    if random_seed is not None:
        set_global_seed(random_seed)
    reset_global_counter()
    if not 0.0 <= ms_ego_probability <= 1.0:
        raise ValueError("ms_ego_probability must be in [0, 1].")
    if not 0.0 <= ms_ego_random_fill_probability <= 1.0:
        raise ValueError("ms_ego_random_fill_probability must be in [0, 1].")
    if archive_refine_interval < 0:
        raise ValueError("archive_refine_interval must be >= 0.")
    if not 0.0 <= archive_refine_sample_ratio <= 1.0:
        raise ValueError("archive_refine_sample_ratio must be in [0, 1].")
    if archive_refine_candidate_topk < 1:
        raise ValueError("archive_refine_candidate_topk must be >= 1.")

    population_management_mode = _resolve_population_management_strategy(
        population_management_strategy
    )
    replacement_mode = _resolve_replacement_strategy(replacement_strategy)

    feedback_mode_normalized = feedback_mode.lower()
    if feedback_mode_normalized not in {"feedback", "fixed"}:
        raise ValueError("feedback_mode must be either 'feedback' or 'fixed'.")
    feedback_enabled = feedback_mode_normalized == "feedback"
    local_search_q_agent = _build_local_search_q_agent(local_search_policy, local_search_q_config)
    struct_local_search_q_agent = (
        StructLocalSearchQAgent() if struct_local_search_enabled else None
    )

    p1, p2 = _initialize_populations_for_main_loop(
        instance,
        total_population_size,
        initialization_strategy=initialization_strategy,
        initial_dedup=initial_dedup,
    )
    archive = Archive()
    for solution in p1.solutions:
        archive.add_solution(solution, 1)
    for solution in p2.solutions:
        archive.add_solution(solution, 2)

    feedback_stats = FeedbackStatistics()
    if beta_override_value is not None:
        beta_override = float(beta_override_value)
    else:
        beta_override = None if feedback_enabled else 0.0

    # Keep the initial split as N1 = N2 = N/2 for the first generation.
    # Dynamic resizing is applied after the first reproduction cycle.
    update_population_u_values(p1, p2)
    target_size_p1 = p1.size
    target_size_p2 = p2.size
    delta_state_1 = delta_state_2 = 0.5
    eta_state_1 = eta_state_2 = 0.5

    generation = 0

    def _snapshot_view():
        # Snapshot uses the current generation's non-dominated front (p1 + p2), not cumulative archive.
        combined = p1.solutions + p2.solutions
        fronts = fast_non_dominated_sort(combined)
        class Snap:
            def __init__(self, sols):
                self.solutions = sols
        return Snap(fronts[0] if fronts else [])
    while not is_termination_reached():
        generation += 1
        feedback_stats.reset_per_generation()
        if feedback_enabled:
            delta_1 = delta_state_1
            delta_2 = delta_state_2
            eta_1 = eta_state_1
            eta_2 = eta_state_2
        else:
            delta_1 = delta_2 = 0.5
            eta_1 = eta_2 = 0.5

        new_p1 = one_generation_reproduction_with_feedback_single(
            p1,
            p2,
            archive,
            delta_1,
            crossover_probability,
            mutation_probability,
            feedback_stats,
            target_size_p1,
            beta_override=beta_override,
            use_feedback_statistics=feedback_enabled,
            parent_selection_strategy=population_management_mode,
            truncation_strategy=population_management_mode,
            ms_ego_enabled=ms_ego_enabled,
            ms_ego_probability=ms_ego_probability,
            ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        )
        new_p2 = one_generation_reproduction_with_feedback_single(
            p2,
            p1,
            archive,
            delta_2,
            crossover_probability,
            mutation_probability,
            feedback_stats,
            target_size_p2,
            beta_override=beta_override,
            use_feedback_statistics=feedback_enabled,
            parent_selection_strategy=population_management_mode,
            truncation_strategy=population_management_mode,
            ms_ego_enabled=ms_ego_enabled,
            ms_ego_probability=ms_ego_probability,
            ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        )

        p1 = _select_next_population(
            p1,
            new_p1,
            target_size_p1,
            replacement_mode,
            population_management_mode,
        )
        p2 = _select_next_population(
            p2,
            new_p2,
            target_size_p2,
            replacement_mode,
            population_management_mode,
        )

        update_population_u_values(p1, p2)
        if feedback_enabled:
            delta_current_1 = _calculate_delta_from_capacity(
                archive, population_id=1, capacity_i=target_size_p1, capacity_other=target_size_p2
            )
            delta_current_2 = _calculate_delta_from_capacity(
                archive, population_id=2, capacity_i=target_size_p2, capacity_other=target_size_p1
            )
            eta_current_1 = _calculate_eta_from_capacity(
                renew_i=p1.renew_i,
                renew_other=p2.renew_i,
                capacity_i=target_size_p1,
                capacity_other=target_size_p2,
            )
            eta_current_2 = _calculate_eta_from_capacity(
                renew_i=p2.renew_i,
                renew_other=p1.renew_i,
                capacity_i=target_size_p2,
                capacity_other=target_size_p1,
            )
            feedback_stats.refresh_probabilities()

            next_target_p1, next_target_p2 = _calculate_new_population_sizes_from_capacity(
                total_size=total_population_size,
                capacity_p1=target_size_p1,
                capacity_p2=target_size_p2,
                delta_values=(delta_current_1, delta_current_2),
                eta_values=(eta_current_1, eta_current_2),
                u_values_p1=p1.u_values,
                u_values_p2=p2.u_values,
            )
            delta_new_1, delta_new_2 = delta_current_1, delta_current_2
            eta_new_1, eta_new_2 = eta_current_1, eta_current_2
            delta_state_1, delta_state_2 = delta_current_1, delta_current_2
            eta_state_1, eta_state_2 = eta_current_1, eta_current_2
        else:
            delta_new_1 = delta_new_2 = 0.5
            eta_new_1 = eta_new_2 = 0.5
            next_target_p1 = target_size_p1
            next_target_p2 = target_size_p2

        if struct_local_search_enabled:
            enhanced_local_search_struct(
                instance,
                archive,
                p1,
                p2,
                target_size_p1,
                target_size_p2,
                feedback_stats,
                (delta_new_1, delta_new_2),
                (eta_new_1, eta_new_2),
                use_feedback_probabilities=feedback_enabled,
                use_feedback_statistics=feedback_enabled,
                q_agent=struct_local_search_q_agent,
                action_mode=struct_local_search_action_mode,
                truncation_strategy=population_management_mode,
                inject_solution=_inject_solution_into_population,
            )
        else:
            enhanced_local_search(
                instance,
                archive,
                p1,
                p2,
                target_size_p1,
                target_size_p2,
                feedback_stats,
                (delta_new_1, delta_new_2),
                (eta_new_1, eta_new_2),
                use_feedback_probabilities=feedback_enabled,
                use_feedback_statistics=feedback_enabled,
                q_agent=local_search_q_agent,
                truncation_strategy=population_management_mode,
            )

        if (
            gap_strategy_enabled
            and gap_trigger_interval > 0
            and generation % gap_trigger_interval == 0
        ):
            apply_gap_strategy(
                instance,
                archive,
                p1,
                p2,
                target_size_p1,
                target_size_p2,
                sample_ratio=gap_archive_sample_ratio,
                truncation_strategy=population_management_mode,
            )

        if (
            archive_refine_enabled
            and archive_refine_interval > 0
            and generation % archive_refine_interval == 0
        ):
            apply_archive_refine(
                instance,
                archive,
                p1,
                p2,
                target_size_p1,
                target_size_p2,
                sample_ratio=archive_refine_sample_ratio,
                candidate_topk=archive_refine_candidate_topk,
                truncation_strategy=population_management_mode,
                build_neighbor=_try_build_gap_strategy_neighbor,
                inject_solution=_inject_solution_into_population,
            )

        # Do not trim/fill populations at generation end; next generation reproduction
        # directly uses next_target sizes to control offspring counts.
        target_size_p1 = next_target_p1
        target_size_p2 = next_target_p2

        if generation_callback is not None:
            generation_callback(generation, get_evaluation_count())
        elif generation % 50 == 0:
            print(
                f"Generation {generation} | evaluations "
                f"{get_evaluation_count()}/{eval_counter.MAX_EVALUATIONS}"
            )

        if snapshot_callback is not None:
            snapshot_callback(generation, _snapshot_view())

        if is_termination_reached():
            break
        if max_generations is not None and generation >= max_generations:
            break

    return archive


def fbea_main_algorithm_q_local_search(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    feedback_mode: str = "feedback",
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_q_config: Optional[dict[str, object]] = None,
):
    """
    Convenience wrapper that enables two-step Q-learning local-search mode.
    """
    q_config = dict(local_search_q_config or {})
    q_config["state_design"] = "two_step_eff"
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        feedback_mode=feedback_mode,
        snapshot_callback=snapshot_callback,
        local_search_policy="q",
        local_search_q_config=q_config,
    )


def fbea_main_algorithm_idle_mix_25(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
):
    """
    FBEA variant that uses a 25/25/25/25 machine-initialisation mix:
    H1 / H2 / idle-first / random.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy="idle_mix_25",
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
    )


def fbea_main_algorithm_gap_strategy(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
):
    """
    FBEA variant that adds only one extra component:
    periodic archive gap-handling strategy (every 50 generations, sample 20%).
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=True,
        gap_trigger_interval=50,
        gap_archive_sample_ratio=0.2,
    )


def fbea_main_algorithm_idle25_gap50(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
):
    """
    FBEA variant with exactly two extra components:
    1) idle_mix_25 initialisation
    2) periodic gap strategy (every 50 generations, 20% archive sample)
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy="idle_mix_25",
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=True,
        gap_trigger_interval=50,
        gap_archive_sample_ratio=0.2,
    )


def fbea_main_algorithm_unity_init(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
):
    """
    Original FBEA loop with only the initialization replaced by the V2
    heterogeneous population initializer.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy="unity_v2",
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
    )


def fbea_main_algorithm_v2_parent_selection_truncation(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
):
    """
    Original FBEA loop with only the V2 dual-population parent selection and
    over-capacity truncation rules enabled.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        population_management_strategy="v2_parent_selection_truncation",
    )


def fbea_main_algorithm_elitist_replacement(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
):
    """
    Original FBEA loop with only the generation update changed to
    parent-offspring elitist replacement: P union P' -> truncate.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        replacement_strategy="elitist_replacement",
    )


def fbea_main_algorithm_init_dedup(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
    population_management_strategy: str = "original",
    replacement_strategy: str = "original",
):
    """
    Thin wrapper that enables initial population de-duplication while keeping
    the main loop otherwise identical to fbea_main_algorithm.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        initial_dedup=True,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        population_management_strategy=population_management_strategy,
        replacement_strategy=replacement_strategy,
    )


def fbea_main_algorithm_init_dedup_q_local_search(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
    population_management_strategy: str = "original",
    replacement_strategy: str = "original",
):
    """
    Convenience wrapper for init-dedup + two-step Q local search.
    """
    q_config = dict(local_search_q_config or {})
    q_config["state_design"] = "two_step_eff"
    return fbea_main_algorithm_init_dedup(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy="q",
        local_search_q_config=q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        population_management_strategy=population_management_strategy,
        replacement_strategy=replacement_strategy,
    )


def fbea_main_algorithm_idle25_init_dedup_gap50_q_local_search(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
):
    """
    Convenience wrapper for:
    idle_mix_25 + initial_dedup + GAP50 + two-step Q local search.
    """
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy="idle_mix_25",
        initial_dedup=True,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy="q",
        local_search_q_config={"state_design": "two_step_eff"},
        gap_strategy_enabled=True,
        gap_trigger_interval=50,
        gap_archive_sample_ratio=0.2,
        ms_ego_enabled=ms_ego_enabled,
        ms_ego_probability=ms_ego_probability,
        ms_ego_random_fill_probability=ms_ego_random_fill_probability,
    )


def fbea_main_algorithm_struct_local_search(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    initial_dedup: bool = False,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
    population_management_strategy: str = "original",
    replacement_strategy: str = "original",
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
    struct_local_search_action_mode: str = "addon19",
):
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        initial_dedup=initial_dedup,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        population_management_strategy=population_management_strategy,
        replacement_strategy=replacement_strategy,
        ms_ego_enabled=ms_ego_enabled,
        ms_ego_probability=ms_ego_probability,
        ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        struct_local_search_enabled=True,
        struct_local_search_action_mode=struct_local_search_action_mode,
    )


def fbea_main_algorithm_archive_refine(
    instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: int | None = None,
    max_generations: int | None = None,
    generation_callback: Optional[Callable[[int, int], None]] = None,
    initialization_strategy: str = "heuristic",
    initial_dedup: bool = False,
    feedback_mode: str = "feedback",
    beta_override_value: float | None = None,
    snapshot_callback: Optional[Callable[[int, Archive], None]] = None,
    local_search_policy: str = "random",
    local_search_q_config: Optional[dict[str, object]] = None,
    gap_strategy_enabled: bool = False,
    gap_trigger_interval: int = 50,
    gap_archive_sample_ratio: float = 0.2,
    population_management_strategy: str = "original",
    replacement_strategy: str = "original",
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
    archive_refine_interval: int = 50,
    archive_refine_sample_ratio: float = 0.2,
    archive_refine_candidate_topk: int = 3,
):
    return fbea_main_algorithm(
        instance=instance,
        total_population_size=total_population_size,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        random_seed=random_seed,
        max_generations=max_generations,
        generation_callback=generation_callback,
        initialization_strategy=initialization_strategy,
        initial_dedup=initial_dedup,
        feedback_mode=feedback_mode,
        beta_override_value=beta_override_value,
        snapshot_callback=snapshot_callback,
        local_search_policy=local_search_policy,
        local_search_q_config=local_search_q_config,
        gap_strategy_enabled=gap_strategy_enabled,
        gap_trigger_interval=gap_trigger_interval,
        gap_archive_sample_ratio=gap_archive_sample_ratio,
        population_management_strategy=population_management_strategy,
        replacement_strategy=replacement_strategy,
        ms_ego_enabled=ms_ego_enabled,
        ms_ego_probability=ms_ego_probability,
        ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        archive_refine_enabled=True,
        archive_refine_interval=archive_refine_interval,
        archive_refine_sample_ratio=archive_refine_sample_ratio,
        archive_refine_candidate_topk=archive_refine_candidate_topk,
    )



