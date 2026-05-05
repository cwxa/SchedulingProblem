"""
Heuristic constructors and scheduling state utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .constants import EPS
from .fuzzy_operations import TriangularFuzzyNumber
from .instance_loader import Instance, Operation
from .random_manager import get_rng


def _zero_tfn() -> TriangularFuzzyNumber:
    return TriangularFuzzyNumber(0.0, 0.0, 0.0)


def _tfn_rank_equal(a: TriangularFuzzyNumber, b: TriangularFuzzyNumber) -> bool:
    # Treat two TFNs as equal under the same lexicographic ranking used by lt/gt.
    return (not a.lt(b)) and (not b.lt(a))


@dataclass
class ScheduleState:
    machine_last_completion_time: Dict[int, TriangularFuzzyNumber]
    job_last_completion_time: Dict[int, TriangularFuzzyNumber]
    job_current_operation: Dict[int, int]


def initialise_schedule_state(instance: Instance) -> ScheduleState:
    zero = _zero_tfn()
    machine_completion = {machine_id: zero for machine_id in instance.machines}
    job_completion = {job_idx: zero for job_idx in range(1, instance.num_jobs + 1)}
    job_current = {job_idx: 0 for job_idx in range(1, instance.num_jobs + 1)}
    return ScheduleState(machine_completion, job_completion, job_current)


def calculate_operation_start_and_completion(
    job_id: int,
    operation_idx: int,
    machine_id: int,
    state: ScheduleState,
    processing_time: TriangularFuzzyNumber,
) -> Tuple[TriangularFuzzyNumber, TriangularFuzzyNumber]:
    job_ready = state.job_last_completion_time[job_id]
    machine_ready = state.machine_last_completion_time[machine_id]
    start = job_ready.max_with(machine_ready)
    completion = start.add(processing_time)
    return start, completion


def update_schedule_state(
    job_id: int,
    operation_idx: int,
    machine_id: int,
    start: TriangularFuzzyNumber,
    completion: TriangularFuzzyNumber,
    state: ScheduleState,
) -> None:
    state.job_last_completion_time[job_id] = completion
    state.machine_last_completion_time[machine_id] = completion
    state.job_current_operation[job_id] = operation_idx + 1


def random_machine_assignment(instance: Instance) -> List[int]:
    rng = get_rng()
    assignment: List[int] = []
    for job_id, op_idx in instance.linearized_operations:
        operation = instance.jobs[job_id - 1].operations[op_idx]
        machine_id = rng.choice([option.machine_id for option in operation.options])
        assignment.append(machine_id)
    return assignment


def random_scheduling_string(instance: Instance) -> List[int]:
    rng = get_rng()
    schedule: List[int] = []
    for job_id, job in enumerate(instance.jobs, start=1):
        schedule.extend([job_id] * job.num_operations)
    rng.shuffle(schedule)
    return schedule


def heuristic_1_machine_assignment(instance: Instance) -> List[int]:
    rng = get_rng()
    zero = _zero_tfn()
    machine_completion: Dict[int, TriangularFuzzyNumber] = {
        machine_id: zero for machine_id in instance.machines
    }
    job_completion: Dict[int, TriangularFuzzyNumber] = {
        job_idx: zero for job_idx in range(1, instance.num_jobs + 1)
    }
    assignment: List[int] = [0] * instance.total_operations
    job_order = list(range(1, instance.num_jobs + 1))
    rng.shuffle(job_order)
    for job_id in job_order:
        job = instance.jobs[job_id - 1]
        for operation_idx in range(job.num_operations):
            operation = job.operations[operation_idx]
            candidates: List[Tuple[int, TriangularFuzzyNumber]] = []
            best_completion: TriangularFuzzyNumber | None = None
            for option in operation.options:
                machine_id = option.machine_id
                start = job_completion[job_id].max_with(machine_completion[machine_id])
                completion = start.add(option.processing_time)
                if best_completion is None or completion.lt(best_completion):
                    candidates = [(machine_id, completion)]
                    best_completion = completion
                elif _tfn_rank_equal(completion, best_completion):
                    candidates.append((machine_id, completion))
            assert candidates
            machine_id, completion = rng.choice(candidates)
            pos = instance.get_operation_position(job_id, operation_idx)
            assignment[pos] = machine_id
            job_completion[job_id] = completion
            machine_completion[machine_id] = completion
    return assignment


def heuristic_2_machine_assignment(instance: Instance) -> List[int]:
    rng = get_rng()
    assignment: List[int] = []
    for job_id, operation_idx in instance.linearized_operations:
        operation = instance.jobs[job_id - 1].operations[operation_idx]
        best_machines: List[int] = []
        best_time: TriangularFuzzyNumber | None = None
        for option in operation.options:
            processing_time = option.processing_time
            if best_time is None or processing_time.lt(best_time):
                best_time = processing_time
                best_machines = [option.machine_id]
            elif _tfn_rank_equal(processing_time, best_time):
                best_machines.append(option.machine_id)
        assignment.append(rng.choice(best_machines))
    return assignment


def heuristic_idle_first_machine_assignment(instance: Instance) -> List[int]:
    """
    Idle-first machine assignment rule.

    For each operation, if one or more feasible machines are already idle at the
    job-ready time, choose randomly among those idle machines. Otherwise choose
    the machine with the earliest ready time (ties broken randomly).
    """
    rng = get_rng()
    zero = _zero_tfn()
    machine_completion: Dict[int, TriangularFuzzyNumber] = {
        machine_id: zero for machine_id in instance.machines
    }
    job_completion: Dict[int, TriangularFuzzyNumber] = {
        job_idx: zero for job_idx in range(1, instance.num_jobs + 1)
    }
    assignment: List[int] = [0] * instance.total_operations
    job_order = list(range(1, instance.num_jobs + 1))
    rng.shuffle(job_order)

    for job_id in job_order:
        job = instance.jobs[job_id - 1]
        for operation_idx in range(job.num_operations):
            operation = job.operations[operation_idx]
            job_ready = job_completion[job_id]

            idle_candidates: List[int] = []
            earliest_ready: TriangularFuzzyNumber | None = None
            earliest_candidates: List[int] = []
            option_by_machine: Dict[int, TriangularFuzzyNumber] = {}

            for option in operation.options:
                machine_id = option.machine_id
                option_by_machine[machine_id] = option.processing_time
                machine_ready = machine_completion[machine_id]
                if machine_ready.leq(job_ready):
                    idle_candidates.append(machine_id)
                if earliest_ready is None or machine_ready.lt(earliest_ready):
                    earliest_ready = machine_ready
                    earliest_candidates = [machine_id]
                elif _tfn_rank_equal(machine_ready, earliest_ready):
                    earliest_candidates.append(machine_id)

            if idle_candidates:
                chosen_machine = rng.choice(idle_candidates)
            else:
                chosen_machine = rng.choice(earliest_candidates)

            processing_time = option_by_machine[chosen_machine]
            start = job_ready.max_with(machine_completion[chosen_machine])
            completion = start.add(processing_time)

            pos = instance.get_operation_position(job_id, operation_idx)
            assignment[pos] = chosen_machine
            job_completion[job_id] = completion
            machine_completion[chosen_machine] = completion

    return assignment


def heuristic_3_scheduling_string(instance: Instance, machine_assignment: List[int]) -> List[int]:
    rng = get_rng()
    total_operations = instance.total_operations
    schedule: List[int] = []
    job_operation_processing: Dict[int, List[TriangularFuzzyNumber]] = {}
    for job_idx, job in enumerate(instance.jobs, start=1):
        durations: List[TriangularFuzzyNumber] = []
        for op_idx in range(job.num_operations):
            pos = instance.get_operation_position(job_idx, op_idx)
            machine_id = machine_assignment[pos]
            operation = job.operations[op_idx]
            proc = next(
                option.processing_time for option in operation.options if option.machine_id == machine_id
            )
            durations.append(proc)
        job_operation_processing[job_idx] = durations

    # Precompute suffix sums of TFN processing times for each job.
    suffix_sums: Dict[int, List[TriangularFuzzyNumber]] = {}
    for job_idx, durations in job_operation_processing.items():
        suffix: List[TriangularFuzzyNumber] = [_zero_tfn() for _ in range(len(durations) + 1)]
        for idx in range(len(durations) - 1, -1, -1):
            suffix[idx] = suffix[idx + 1].add(durations[idx])
        suffix_sums[job_idx] = suffix

    job_next_op: Dict[int, int] = {job_idx: 0 for job_idx in range(1, instance.num_jobs + 1)}
    for _ in range(total_operations):
        candidates = []
        max_value: TriangularFuzzyNumber | None = None
        for job_idx in range(1, instance.num_jobs + 1):
            if job_next_op[job_idx] >= len(job_operation_processing[job_idx]):
                continue
            remaining = suffix_sums[job_idx][job_next_op[job_idx]]
            if max_value is None or remaining.gt(max_value):
                candidates = [job_idx]
                max_value = remaining
            elif _tfn_rank_equal(remaining, max_value):
                candidates.append(job_idx)
        chosen = rng.choice(candidates)
        schedule.append(chosen)
        job_next_op[chosen] += 1
    return schedule


def heuristic_4_scheduling_string(instance: Instance) -> List[int]:
    rng = get_rng()
    total_operations = instance.total_operations
    schedule: List[int] = []
    remaining_counts: Dict[int, int] = {
        job_idx: job.num_operations for job_idx, job in enumerate(instance.jobs, start=1)
    }
    for _ in range(total_operations):
        active = {job_id: count for job_id, count in remaining_counts.items() if count > 0}
        if not active:
            raise ValueError("Heuristic 4 encountered no remaining operations to schedule.")
        max_count = max(active.values())
        candidates = [job_id for job_id, count in active.items() if count == max_count]
        chosen = rng.choice(candidates)
        schedule.append(chosen)
        remaining_counts[chosen] -= 1
    return schedule


def test_all_heuristics(instance: Instance) -> bool:
    """
    Basic validation to ensure heuristic outputs satisfy structural constraints.
    """
    machine_assignment = heuristic_1_machine_assignment(instance)
    return len(machine_assignment) == instance.total_operations
