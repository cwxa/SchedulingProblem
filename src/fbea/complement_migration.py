"""
Standalone FBEA variant with bidirectional complement migration.

This module intentionally leaves the original FBEA implementation untouched and
reuses its internal helpers to keep the baseline behavior aligned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable, Optional, Sequence

from . import evaluation_counter as eval_counter
from .algorithm4_main_loop import (
    _build_local_search_q_agent,
    _calculate_delta_from_capacity,
    _calculate_eta_from_capacity,
    _calculate_new_population_sizes_from_capacity,
    _resolve_initialization_strategy,
    _resolve_population_management_strategy,
    _resolve_replacement_strategy,
    _select_next_population,
    _truncate_population_by_rank_and_crowding,
    apply_gap_strategy,
    enhanced_local_search,
    one_generation_reproduction_with_feedback_single,
)
from .archive import Archive, fast_non_dominated_sort
from .evaluation_counter import get_evaluation_count, is_termination_reached, reset_global_counter
from .feedback_statistics import FeedbackStatistics, update_population_u_values
from .population import Population, initialize_populations
from .random_manager import set_global_seed
from .solution import Solution


@dataclass(slots=True)
class ComplementMigrationConfig:
    interval: int = 50
    migrants_per_direction: int = 2
    quality_front_cap: int = 2
    cover_threshold: float = 0.04
    migrant_pair_threshold: float = 0.02


@dataclass(slots=True)
class MigrationRoundStats:
    generation: int
    a_to_b_selected: int = 0
    a_to_b_survived: int = 0
    b_to_a_selected: int = 0
    b_to_a_survived: int = 0


class MigrationStatsTracker:
    def __init__(self, config: ComplementMigrationConfig) -> None:
        self.config = config
        self.rounds: list[MigrationRoundStats] = []

    def record(self, round_stats: MigrationRoundStats) -> None:
        self.rounds.append(round_stats)

    def summary(self) -> dict:
        return {
            "config": asdict(self.config),
            "n_rounds": len(self.rounds),
            "a_to_b_selected_total": sum(item.a_to_b_selected for item in self.rounds),
            "a_to_b_survived_total": sum(item.a_to_b_survived for item in self.rounds),
            "b_to_a_selected_total": sum(item.b_to_a_selected for item in self.rounds),
            "b_to_a_survived_total": sum(item.b_to_a_survived for item in self.rounds),
            "rounds": [asdict(item) for item in self.rounds],
        }


def _objective_triplet(solution: Solution) -> tuple[float, float, float]:
    solution.evaluate()
    return (
        float(solution.makespan._c1()),
        float(solution.energy._c1()),
        float(solution.agreement._c1()),
    )


def _build_normalization_bounds(solutions: Sequence[Solution]) -> tuple[tuple[float, float], ...]:
    vectors = [_objective_triplet(sol) for sol in solutions]
    if not vectors:
        return ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bounds = []
    for dim in range(3):
        values = [vec[dim] for vec in vectors]
        low = min(values)
        high = max(values)
        if abs(high - low) <= 1e-12:
            high = low + 1.0
        bounds.append((low, high))
    return tuple(bounds)


def _normalized_vector(solution: Solution, bounds: tuple[tuple[float, float], ...]) -> tuple[float, float, float]:
    raw = _objective_triplet(solution)
    normalized = []
    for value, (low, high) in zip(raw, bounds):
        normalized.append((value - low) / (high - low))
    return tuple(normalized)


def _distance_between(
    sol_a: Solution,
    sol_b: Solution,
    bounds: tuple[tuple[float, float], ...],
) -> float:
    vec_a = _normalized_vector(sol_a, bounds)
    vec_b = _normalized_vector(sol_b, bounds)
    return math.dist(vec_a, vec_b)


def _min_distance_to_population(
    solution: Solution,
    population: Sequence[Solution],
    bounds: tuple[tuple[float, float], ...],
) -> float:
    if not population:
        return float("inf")
    return min(_distance_between(solution, other, bounds) for other in population)


def _front_rank_map(solutions: Sequence[Solution]) -> dict[int, int]:
    rank_map: dict[int, int] = {}
    for rank, front in enumerate(fast_non_dominated_sort(list(solutions), deduplicate=False), start=1):
        for sol in front:
            rank_map[id(sol)] = rank
    return rank_map


def _select_complement_migrants(
    sender_snapshot: Sequence[Solution],
    receiver_snapshot: Sequence[Solution],
    config: ComplementMigrationConfig,
) -> list[Solution]:
    if not sender_snapshot or config.migrants_per_direction <= 0:
        return []

    rank_map = _front_rank_map(sender_snapshot)
    bounds = _build_normalization_bounds(list(sender_snapshot) + list(receiver_snapshot))

    candidates: list[tuple[Solution, float, int]] = []
    for solution in sender_snapshot:
        rank = rank_map.get(id(solution), math.inf)
        if rank > config.quality_front_cap:
            continue
        novelty = _min_distance_to_population(solution, receiver_snapshot, bounds)
        if novelty <= config.cover_threshold:
            continue
        candidates.append((solution, novelty, rank))

    candidates.sort(
        key=lambda item: (
            -item[1],
            item[2],
            _objective_triplet(item[0])[0],
            _objective_triplet(item[0])[1],
            _objective_triplet(item[0])[2],
        )
    )

    selected: list[Solution] = []
    for solution, _, _ in candidates:
        if any(
            _distance_between(solution, existing, bounds) <= config.migrant_pair_threshold
            for existing in selected
        ):
            continue
        selected.append(solution)
        if len(selected) >= config.migrants_per_direction:
            break
    return selected


def _merge_migrants_into_population(
    receiver: Population,
    migrants: Sequence[Solution],
    target_size: int,
) -> int:
    if not migrants:
        return 0

    incoming = [solution.clone() for solution in migrants]
    incoming_ids = {id(solution) for solution in incoming}
    merged = Population(
        population_id=receiver.population_id,
        instance=receiver.instance,
        solutions=list(receiver.solutions) + incoming,
        renew_i=receiver.renew_i,
        u_values=receiver.u_values,
    )
    _truncate_population_by_rank_and_crowding(merged, target_size)
    receiver.solutions = merged.solutions
    receiver.renew_i = merged.renew_i
    receiver.u_values = merged.u_values
    return sum(1 for solution in receiver.solutions if id(solution) in incoming_ids)


def _apply_complement_migration(
    population_a: Population,
    population_b: Population,
    target_size_a: int,
    target_size_b: int,
    config: ComplementMigrationConfig,
    generation: int,
) -> MigrationRoundStats:
    snapshot_a = list(population_a.solutions)
    snapshot_b = list(population_b.solutions)

    migrants_a_to_b = _select_complement_migrants(snapshot_a, snapshot_b, config)
    migrants_b_to_a = _select_complement_migrants(snapshot_b, snapshot_a, config)

    stats = MigrationRoundStats(
        generation=generation,
        a_to_b_selected=len(migrants_a_to_b),
        b_to_a_selected=len(migrants_b_to_a),
    )
    stats.a_to_b_survived = _merge_migrants_into_population(
        population_b,
        migrants_a_to_b,
        target_size_b,
    )
    stats.b_to_a_survived = _merge_migrants_into_population(
        population_a,
        migrants_b_to_a,
        target_size_a,
    )
    return stats


def fbea_main_algorithm_complement_migration(
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
    ms_ego_enabled: bool = False,
    ms_ego_probability: float = 0.2,
    ms_ego_random_fill_probability: float = 0.2,
    migration_config: Optional[ComplementMigrationConfig] = None,
) -> Archive:
    if random_seed is not None:
        set_global_seed(random_seed)
    reset_global_counter()
    if not 0.0 <= ms_ego_probability <= 1.0:
        raise ValueError("ms_ego_probability must be in [0, 1].")
    if not 0.0 <= ms_ego_random_fill_probability <= 1.0:
        raise ValueError("ms_ego_random_fill_probability must be in [0, 1].")

    config = migration_config or ComplementMigrationConfig()

    init_mode, initializer = _resolve_initialization_strategy(
        initialization_strategy,
        allow_unity_v2=True,
    )
    population_management_mode = _resolve_population_management_strategy(
        population_management_strategy
    )
    replacement_mode = _resolve_replacement_strategy(replacement_strategy)

    feedback_mode_normalized = feedback_mode.lower()
    if feedback_mode_normalized not in {"feedback", "fixed"}:
        raise ValueError("feedback_mode must be either 'feedback' or 'fixed'.")
    feedback_enabled = feedback_mode_normalized == "feedback"
    local_search_q_agent = _build_local_search_q_agent(local_search_policy, local_search_q_config)

    if init_mode == "unity_v2":
        from fbea_unity_v2 import initialize_populations_unity

        population_a, population_b = initialize_populations_unity(instance, total_population_size)
    else:
        population_a, population_b = initialize_populations(
            instance,
            total_population_size,
            initializer=initializer,
        )

    archive = Archive()
    for solution in population_a.solutions:
        archive.add_solution(solution, 1)
    for solution in population_b.solutions:
        archive.add_solution(solution, 2)

    feedback_stats = FeedbackStatistics()
    if beta_override_value is not None:
        beta_override = float(beta_override_value)
    else:
        beta_override = None if feedback_enabled else 0.0

    update_population_u_values(population_a, population_b)
    target_size_a = population_a.size
    target_size_b = population_b.size
    delta_state_a = delta_state_b = 0.5
    eta_state_a = eta_state_b = 0.5
    generation = 0
    migration_tracker = MigrationStatsTracker(config)

    def _snapshot_view():
        combined = population_a.solutions + population_b.solutions
        fronts = fast_non_dominated_sort(combined)

        class Snap:
            def __init__(self, sols):
                self.solutions = sols

        return Snap(fronts[0] if fronts else [])

    while not is_termination_reached():
        generation += 1
        feedback_stats.reset_per_generation()
        if feedback_enabled:
            delta_a = delta_state_a
            delta_b = delta_state_b
            eta_a = eta_state_a
            eta_b = eta_state_b
        else:
            delta_a = delta_b = 0.5
            eta_a = eta_b = 0.5

        new_population_a = one_generation_reproduction_with_feedback_single(
            population_a,
            population_b,
            archive,
            delta_a,
            crossover_probability,
            mutation_probability,
            feedback_stats,
            target_size_a,
            beta_override=beta_override,
            use_feedback_statistics=feedback_enabled,
            parent_selection_strategy=population_management_mode,
            truncation_strategy=population_management_mode,
            ms_ego_enabled=ms_ego_enabled,
            ms_ego_probability=ms_ego_probability,
            ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        )
        new_population_b = one_generation_reproduction_with_feedback_single(
            population_b,
            population_a,
            archive,
            delta_b,
            crossover_probability,
            mutation_probability,
            feedback_stats,
            target_size_b,
            beta_override=beta_override,
            use_feedback_statistics=feedback_enabled,
            parent_selection_strategy=population_management_mode,
            truncation_strategy=population_management_mode,
            ms_ego_enabled=ms_ego_enabled,
            ms_ego_probability=ms_ego_probability,
            ms_ego_random_fill_probability=ms_ego_random_fill_probability,
        )

        population_a = _select_next_population(
            population_a,
            new_population_a,
            target_size_a,
            replacement_mode,
            population_management_mode,
        )
        population_b = _select_next_population(
            population_b,
            new_population_b,
            target_size_b,
            replacement_mode,
            population_management_mode,
        )

        if config.interval > 0 and generation % config.interval == 0:
            migration_tracker.record(
                _apply_complement_migration(
                    population_a,
                    population_b,
                    target_size_a,
                    target_size_b,
                    config,
                    generation,
                )
            )

        update_population_u_values(population_a, population_b)
        if feedback_enabled:
            delta_current_a = _calculate_delta_from_capacity(
                archive, population_id=1, capacity_i=target_size_a, capacity_other=target_size_b
            )
            delta_current_b = _calculate_delta_from_capacity(
                archive, population_id=2, capacity_i=target_size_b, capacity_other=target_size_a
            )
            eta_current_a = _calculate_eta_from_capacity(
                renew_i=population_a.renew_i,
                renew_other=population_b.renew_i,
                capacity_i=target_size_a,
                capacity_other=target_size_b,
            )
            eta_current_b = _calculate_eta_from_capacity(
                renew_i=population_b.renew_i,
                renew_other=population_a.renew_i,
                capacity_i=target_size_b,
                capacity_other=target_size_a,
            )
            feedback_stats.refresh_probabilities()

            next_target_a, next_target_b = _calculate_new_population_sizes_from_capacity(
                total_size=total_population_size,
                capacity_p1=target_size_a,
                capacity_p2=target_size_b,
                delta_values=(delta_current_a, delta_current_b),
                eta_values=(eta_current_a, eta_current_b),
                u_values_p1=population_a.u_values,
                u_values_p2=population_b.u_values,
            )
            delta_new_a, delta_new_b = delta_current_a, delta_current_b
            eta_new_a, eta_new_b = eta_current_a, eta_current_b
            delta_state_a, delta_state_b = delta_current_a, delta_current_b
            eta_state_a, eta_state_b = eta_current_a, eta_current_b
        else:
            delta_new_a = delta_new_b = 0.5
            eta_new_a = eta_new_b = 0.5
            next_target_a = target_size_a
            next_target_b = target_size_b

        enhanced_local_search(
            instance,
            archive,
            population_a,
            population_b,
            target_size_a,
            target_size_b,
            feedback_stats,
            (delta_new_a, delta_new_b),
            (eta_new_a, eta_new_b),
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
                population_a,
                population_b,
                target_size_a,
                target_size_b,
                sample_ratio=gap_archive_sample_ratio,
                truncation_strategy=population_management_mode,
            )

        target_size_a = next_target_a
        target_size_b = next_target_b

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

    archive.complement_migration_stats = migration_tracker.summary()
    return archive
