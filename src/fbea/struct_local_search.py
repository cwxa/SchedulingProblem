"""
Structure-aware local search component (plan A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from . import evaluation_counter as eval_counter
from .algorithm3_crossover_mutation import (
    OperatorUsageInfo,
    apply_machine_mutation,
    apply_scheduling_mutation,
)
from .archive import Archive, dominates
from .constants import EPS
from .evaluation_counter import evaluate_solutions_with_count, get_evaluation_count
from .feedback_statistics import FeedbackStatistics
from .population import Population
from .random_manager import get_rng
from .solution import Solution

UNIFORM_OPERATOR_PROBABILITIES = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

StructMacroAction = tuple[str, int | None, int | None, str | None]

# (mode, scheduling_operator_idx, machine_operator_idx, struct_action_name)
STRUCT_MACRO_ACTIONS: tuple[StructMacroAction, ...] = (
    ("scheduling", 0, None, None),
    ("scheduling", 1, None, None),
    ("scheduling", 2, None, None),
    ("machine", None, 0, None),
    ("machine", None, 1, None),
    ("machine", None, 2, None),
    ("both", 0, 0, None),
    ("both", 0, 1, None),
    ("both", 0, 2, None),
    ("both", 1, 0, None),
    ("both", 1, 1, None),
    ("both", 1, 2, None),
    ("both", 2, 0, None),
    ("both", 2, 1, None),
    ("both", 2, 2, None),
    ("struct_os", None, None, "critical_tail_pull"),
    ("struct_os", None, None, "cross_bottleneck_swap"),
    ("struct_ms", None, None, "last_completion_reassign"),
    ("struct_ms", None, None, "max_load_machine_migrate"),
)


@dataclass
class StructLocalSearchQConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    greedy_start: float = 0.60
    greedy_end: float = 0.85
    greedy_power: float = 1.0
    reward_dominate: float = 2.0
    reward_nondominated: float = 1.0
    reward_dominated: float = 0.0


class StructLocalSearchQAgent:
    ACTION_COUNT = len(STRUCT_MACRO_ACTIONS)

    def __init__(self, config: StructLocalSearchQConfig | None = None) -> None:
        self.config = config or StructLocalSearchQConfig()
        self.q_table: list[list[float]] = [
            [0.0 for _ in range(self.ACTION_COUNT)] for _ in range(self.ACTION_COUNT)
        ]
        self.visits: list[list[int]] = [
            [0 for _ in range(self.ACTION_COUNT)] for _ in range(self.ACTION_COUNT)
        ]
        self.action_use_counts: list[int] = [0 for _ in range(self.ACTION_COUNT)]
        self.previous_action_idx: int | None = None

    @classmethod
    def action_from_index(cls, action_idx: int) -> StructMacroAction:
        if action_idx < 0 or action_idx >= cls.ACTION_COUNT:
            return STRUCT_MACRO_ACTIONS[0]
        return STRUCT_MACRO_ACTIONS[action_idx]

    def current_state(self) -> int | None:
        return self.previous_action_idx

    def set_previous_action(self, action_idx: int) -> None:
        if 0 <= action_idx < self.ACTION_COUNT:
            self.previous_action_idx = action_idx

    def greedy_factor(self, progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        return self.config.greedy_start + (
            self.config.greedy_end - self.config.greedy_start
        ) * (p ** self.config.greedy_power)

    def select_action(self, state_idx: int | None, progress: float, rng) -> int:
        if state_idx is None:
            unvisited = [
                idx for idx, count in enumerate(self.action_use_counts) if count == 0
            ]
            if unvisited:
                return int(rng.choice(unvisited))
            return int(rng.randrange(self.ACTION_COUNT))

        row_visits = self.visits[state_idx]
        unvisited = [idx for idx, count in enumerate(row_visits) if count == 0]
        if unvisited:
            return int(rng.choice(unvisited))

        q_values = self.q_table[state_idx]
        if rng.random() < self.greedy_factor(progress):
            best = max(q_values)
            candidates = [idx for idx, value in enumerate(q_values) if abs(value - best) <= 1e-12]
            return int(rng.choice(candidates))
        return int(rng.randrange(self.ACTION_COUNT))

    def compute_reward(
        self,
        dominates_old: bool,
        dominated_by_old: bool,
        archive_added: bool,
    ) -> float:
        if dominates_old:
            return float(self.config.reward_dominate)
        if dominated_by_old:
            return float(self.config.reward_dominated)
        if archive_added:
            return float(self.config.reward_nondominated)
        return float(self.config.reward_dominated)

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state: int | None = None,
    ) -> None:
        if not (0 <= state_idx < self.ACTION_COUNT and 0 <= action_idx < self.ACTION_COUNT):
            return
        self.visits[state_idx][action_idx] += 1
        self.action_use_counts[action_idx] += 1
        visits = self.visits[state_idx][action_idx]
        alpha = self.config.alpha / (1.0 + 0.01 * visits)
        target = reward
        if next_state is not None and 0 <= next_state < self.ACTION_COUNT:
            target += self.config.gamma * max(self.q_table[next_state])
        current = self.q_table[state_idx][action_idx]
        self.q_table[state_idx][action_idx] = current + alpha * (target - current)

    def mark_first_action(self, action_idx: int) -> None:
        if 0 <= action_idx < self.ACTION_COUNT:
            self.action_use_counts[action_idx] += 1


def _deterministic_operator_probabilities(operator_idx: int) -> tuple[float, float, float]:
    if operator_idx == 0:
        return (1.0, 0.0, 0.0)
    if operator_idx == 1:
        return (0.0, 1.0, 0.0)
    if operator_idx == 2:
        return (0.0, 0.0, 1.0)
    raise ValueError("operator_idx must be 0, 1, or 2.")


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


def _schedule_index_map(solution: Solution) -> dict[tuple[int, int], int]:
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None:
        return {}
    return {
        (op.job_id, op.operation_idx): idx for idx, op in enumerate(schedule.operations)
    }


def _job_occurrence_positions(scheduling_string: list[int], job_id: int) -> list[int]:
    return [idx for idx, value in enumerate(scheduling_string) if value == job_id]


def _move_job_occurrence_earlier(
    scheduling_string: list[int],
    job_id: int,
    operation_idx: int,
    desired_position: int,
) -> list[int] | None:
    positions = _job_occurrence_positions(scheduling_string, job_id)
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


def _machine_loads(solution: Solution) -> dict[int, float]:
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None:
        return {}
    loads: dict[int, float] = {}
    for machine_id, machine_schedule in schedule.machines.items():
        loads[machine_id] = sum(op.processing_time._c1() for op in machine_schedule.operations)
    return loads


def _find_last_completion_operation(solution: Solution):
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None or not schedule.operations:
        return None
    return max(
        schedule.operations,
        key=lambda op: (
            op.completion_time._c1(),
            op.completion_time._c2(),
            op.completion_time._c3(),
        ),
    )


def _find_bottleneck_machine(solution: Solution) -> int | None:
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None or not schedule.machines:
        return None
    machine_schedule = max(
        schedule.machines.values(),
        key=lambda machine: (
            machine.completion_time._c1(),
            machine.completion_time._c2(),
            machine.completion_time._c3(),
        ),
    )
    return machine_schedule.machine_id


def _critical_tail_pull_neighbor(instance, solution: Solution) -> Solution | None:
    solution.evaluate()
    schedule = solution.schedule
    target = _find_last_completion_operation(solution)
    if schedule is None or target is None:
        return None
    step_index_map = _schedule_index_map(solution)
    current_pos = step_index_map.get((target.job_id, target.operation_idx))
    if current_pos is None or current_pos <= 0:
        return None
    machine_schedule = schedule.machines.get(target.machine_id)
    desired_position = max(0, current_pos - 1)
    if machine_schedule is not None and machine_schedule.operations:
        ordered = sorted(machine_schedule.operations, key=lambda op: op.start_time._c1())
        for idx, op in enumerate(ordered):
            if op.job_id == target.job_id and op.operation_idx == target.operation_idx and idx > 0:
                predecessor = ordered[idx - 1]
                desired_position = step_index_map.get(
                    (predecessor.job_id, predecessor.operation_idx),
                    desired_position,
                )
                break
    updated_scheduling = _move_job_occurrence_earlier(
        list(solution.scheduling_string),
        target.job_id,
        target.operation_idx,
        desired_position,
    )
    if updated_scheduling is None:
        return None
    return Solution(
        instance=instance,
        scheduling_string=updated_scheduling,
        machine_assignment_string=list(solution.machine_assignment_string),
        skip_validation=True,
    )


def _cross_bottleneck_swap_neighbor(instance, solution: Solution) -> Solution | None:
    solution.evaluate()
    schedule = solution.schedule
    bottleneck_machine_id = _find_bottleneck_machine(solution)
    if schedule is None or bottleneck_machine_id is None:
        return None
    step_index_map = _schedule_index_map(solution)
    bottleneck_ops = [
        op for op in schedule.operations if op.machine_id == bottleneck_machine_id
    ]
    if not bottleneck_ops:
        return None
    target = max(
        bottleneck_ops,
        key=lambda op: (
            op.completion_time._c1(),
            op.processing_time._c1(),
        ),
    )
    current_pos = step_index_map.get((target.job_id, target.operation_idx))
    if current_pos is None:
        return None
    partner_pos: int | None = None
    best_distance: int | None = None
    for idx, op in enumerate(schedule.operations):
        if idx == current_pos:
            continue
        if op.machine_id == bottleneck_machine_id:
            continue
        if schedule.operations[idx].job_id == target.job_id:
            continue
        distance = abs(current_pos - idx)
        if partner_pos is None:
            partner_pos = idx
            best_distance = distance
            continue
        if idx < current_pos <= partner_pos:
            partner_pos = idx
            best_distance = distance
            continue
        if best_distance is not None and distance < best_distance:
            partner_pos = idx
            best_distance = distance
    if partner_pos is None:
        return None
    updated_scheduling = list(solution.scheduling_string)
    updated_scheduling[current_pos], updated_scheduling[partner_pos] = (
        updated_scheduling[partner_pos],
        updated_scheduling[current_pos],
    )
    return Solution(
        instance=instance,
        scheduling_string=updated_scheduling,
        machine_assignment_string=list(solution.machine_assignment_string),
        skip_validation=True,
    )


def _last_completion_reassign_neighbor(instance, solution: Solution) -> Solution | None:
    target = _find_last_completion_operation(solution)
    if target is None:
        return None
    pos = instance.get_operation_position(target.job_id, target.operation_idx)
    operation = instance.jobs[target.job_id - 1].operations[target.operation_idx]
    current_machine = solution.machine_assignment_string[pos]
    alternatives = [opt for opt in operation.options if opt.machine_id != current_machine]
    if not alternatives:
        return None
    best_option = min(
        alternatives,
        key=lambda opt: (
            opt.processing_time._c1(),
            opt.processing_time._c2(),
            opt.processing_time._c3(),
        ),
    )
    updated_machines = list(solution.machine_assignment_string)
    updated_machines[pos] = best_option.machine_id
    return Solution(
        instance=instance,
        scheduling_string=list(solution.scheduling_string),
        machine_assignment_string=updated_machines,
        skip_validation=True,
    )


def _max_load_machine_migrate_neighbor(instance, solution: Solution) -> Solution | None:
    solution.evaluate()
    schedule = solution.schedule
    if schedule is None or not schedule.machines:
        return None
    loads = _machine_loads(solution)
    if not loads:
        return None
    max_machine_id = max(loads, key=loads.get)
    machine_schedule = schedule.machines.get(max_machine_id)
    if machine_schedule is None or not machine_schedule.operations:
        return None

    candidate = None
    for op in sorted(
        machine_schedule.operations,
        key=lambda item: (
            -item.processing_time._c1(),
            -item.completion_time._c1(),
        ),
    ):
        pos = instance.get_operation_position(op.job_id, op.operation_idx)
        current_machine = solution.machine_assignment_string[pos]
        operation = instance.jobs[op.job_id - 1].operations[op.operation_idx]
        alternatives = [opt for opt in operation.options if opt.machine_id != current_machine]
        if not alternatives:
            continue
        best_option = min(
            alternatives,
            key=lambda opt: (
                loads.get(opt.machine_id, 0.0),
                opt.processing_time._c1(),
                opt.processing_time._c2(),
                opt.processing_time._c3(),
            ),
        )
        candidate = (pos, best_option.machine_id)
        break
    if candidate is None:
        return None
    updated_machines = list(solution.machine_assignment_string)
    updated_machines[candidate[0]] = candidate[1]
    return Solution(
        instance=instance,
        scheduling_string=list(solution.scheduling_string),
        machine_assignment_string=updated_machines,
        skip_validation=True,
    )


def build_struct_local_search_neighbor(
    instance,
    solution: Solution,
    action_name: str,
) -> Solution | None:
    if action_name == "critical_tail_pull":
        return _critical_tail_pull_neighbor(instance, solution)
    if action_name == "cross_bottleneck_swap":
        return _cross_bottleneck_swap_neighbor(instance, solution)
    if action_name == "last_completion_reassign":
        return _last_completion_reassign_neighbor(instance, solution)
    if action_name == "max_load_machine_migrate":
        return _max_load_machine_migrate_neighbor(instance, solution)
    raise ValueError(f"Unknown struct local-search action: {action_name}")


def _apply_action(
    instance,
    base_solution: Solution,
    action_idx: int,
    feedback_stats: FeedbackStatistics,
    *,
    use_feedback_statistics: bool,
) -> tuple[Solution, OperatorUsageInfo]:
    mode, scheduling_operator_idx, machine_operator_idx, struct_action_name = (
        StructLocalSearchQAgent.action_from_index(action_idx)
    )
    operator_info = OperatorUsageInfo()
    scheduling = list(base_solution.scheduling_string)
    machines = list(base_solution.machine_assignment_string)

    if mode in ("scheduling", "both"):
        default_sched_probs = (
            feedback_stats.w_bar_1 if use_feedback_statistics else UNIFORM_OPERATOR_PROBABILITIES
        )
        probs_sched = (
            default_sched_probs
            if scheduling_operator_idx is None
            else _deterministic_operator_probabilities(scheduling_operator_idx)
        )
        scheduling = apply_scheduling_mutation(scheduling, probs_sched, operator_info)
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
    if mode in {"scheduling", "machine", "both"}:
        return (
            Solution(
                instance=instance,
                scheduling_string=scheduling,
                machine_assignment_string=machines,
                skip_validation=True,
            ),
            operator_info,
        )

    if struct_action_name is not None:
        neighbor = build_struct_local_search_neighbor(instance, base_solution, struct_action_name)
        if neighbor is not None:
            return neighbor, operator_info

        if mode == "struct_os":
            fallback_sched = apply_scheduling_mutation(
                list(base_solution.scheduling_string),
                _deterministic_operator_probabilities(0),
                operator_info,
            )
            return (
                Solution(
                    instance=instance,
                    scheduling_string=fallback_sched,
                    machine_assignment_string=list(base_solution.machine_assignment_string),
                    skip_validation=True,
                ),
                operator_info,
            )
        fallback_machines = apply_machine_mutation(
            instance,
            list(base_solution.machine_assignment_string),
            list(base_solution.scheduling_string),
            _deterministic_operator_probabilities(1),
            operator_info,
        )
        return (
            Solution(
                instance=instance,
                scheduling_string=list(base_solution.scheduling_string),
                machine_assignment_string=fallback_machines,
                skip_validation=True,
            ),
            operator_info,
        )

    return base_solution.clone(), operator_info


def enhanced_local_search_struct(
    instance,
    archive: Archive,
    population_a: Population,
    population_b: Population,
    capacity_a: int,
    capacity_b: int,
    feedback_stats: FeedbackStatistics,
    delta_values,
    eta_values,
    *,
    use_feedback_probabilities: bool = True,
    use_feedback_statistics: bool = True,
    q_agent: StructLocalSearchQAgent | None = None,
    action_mode: str = "addon19",
    truncation_strategy: str = "original",
    inject_solution: Callable[[Population, Solution, int, str], None] | None = None,
) -> None:
    if action_mode.lower() != "addon19":
        raise ValueError("struct_local_search_action_mode currently only supports 'addon19'.")
    if not archive.solutions:
        return
    q_agent = q_agent or StructLocalSearchQAgent()
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
        tuple[Solution, Solution, int | None, int, int]
    ] = []
    pending_neighbors: list[Solution] = []

    for solution in solutions_snapshot:
        if eval_counter.get_evaluation_count() + len(pending_neighbors) >= eval_counter.MAX_EVALUATIONS:
            break
        source_population_id = archive.solution_sources.get(id(solution), 0)
        state_index = q_agent.current_state()
        action_index = q_agent.select_action(state_index, progress, rng)
        neighbor, _operator_info = _apply_action(
            instance,
            solution,
            action_index,
            feedback_stats,
            use_feedback_statistics=use_feedback_statistics,
        )
        pending_pairs.append((solution, neighbor, state_index, action_index, source_population_id))
        pending_neighbors.append(neighbor)

    evaluate_solutions_with_count(pending_neighbors)

    for solution, neighbor, state_index, action_index, source_population_id in pending_pairs:
        dominates_old = dominates(neighbor, solution)
        dominated_by_old = dominates(solution, neighbor)
        archive_added = False

        if dominates_old:
            if solution in archive.solutions:
                archive.solutions.remove(solution)
                archive.solution_sources.pop(id(solution), None)
            archive_added = archive.add_solution(neighbor, source_population_id)
            should_inject = True
        elif not dominated_by_old:
            archive_added = archive.add_solution(neighbor, source_population_id)
            should_inject = True
        else:
            should_inject = False

        if should_inject and inject_solution is not None:
            if rng.random() < prob_p1:
                inject_solution(
                    population_a,
                    neighbor,
                    capacity_a,
                    truncation_strategy,
                )
            else:
                inject_solution(
                    population_b,
                    neighbor,
                    capacity_b,
                    truncation_strategy,
                )

        reward = q_agent.compute_reward(
            dominates_old=dominates_old,
            dominated_by_old=dominated_by_old,
            archive_added=archive_added,
        )
        if state_index is not None:
            q_agent.update(state_index, action_index, reward, next_state=action_index)
        else:
            q_agent.mark_first_action(action_index)
        q_agent.set_previous_action(action_index)
