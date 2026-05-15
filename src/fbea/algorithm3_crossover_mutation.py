"""
Crossover and mutation operators corresponding to Algorithm 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .evaluation_counter import calculate_objectives_with_count
from .instance_loader import Instance
from .random_manager import get_rng
from .solution import Solution


@dataclass
class OperatorUsageInfo:
    used_gpx: bool = False
    used_pox: bool = False
    used_uniform_machine_crossover: bool = False
    used_insert_mutation: bool = False
    used_swap_mutation: bool = False
    used_inverse_mutation: bool = False
    used_change1_mutation: bool = False
    used_change2_mutation: bool = False
    used_change3_mutation: bool = False

    def any_scheduling_mutation(self) -> bool:
        return self.used_insert_mutation or self.used_swap_mutation or self.used_inverse_mutation

    def any_machine_mutation(self) -> bool:
        return self.used_change1_mutation or self.used_change2_mutation or self.used_change3_mutation


def _job_operation_counts(instance: Instance) -> Dict[int, int]:
    return instance.job_operation_counts


def _clone_solution_with_strings(solution: Solution, scheduling: List[int], machines: List[int]) -> Solution:
    clone = Solution(
        instance=solution.instance,
        scheduling_string=list(scheduling),
        machine_assignment_string=list(machines),
        random_seed=solution.random_seed,
        skip_validation=True,
    )
    return clone


def uniform_machine_crossover(
    parent_a: Solution,
    parent_b: Solution,
) -> Tuple[List[int], List[int], OperatorUsageInfo]:
    rng = get_rng()
    instance = parent_a.instance
    machines_a = list(parent_a.machine_assignment_string)
    machines_b = list(parent_b.machine_assignment_string)
    child_a = machines_a[:]
    child_b = machines_b[:]
    
    for idx in range(len(machines_a)):
        if rng.random() < 0.5:
            child_a[idx], child_b[idx] = child_b[idx], child_a[idx]
    
    # 修复无效的机器分配：确保每个操作的机器分配都是有效的
    def repair_invalid_assignment(child: List[int]) -> None:
        for idx in range(len(child)):
            job_id, op_idx = instance.linearized_operations[idx]
            operation = instance.jobs[job_id - 1].operations[op_idx]
            current_machine = child[idx]
            valid_machines = {opt.machine_id for opt in operation.options}
            if current_machine not in valid_machines:
                # 如果当前分配无效，随机选择一个有效的机器
                child[idx] = rng.choice(list(valid_machines))
    
    repair_invalid_assignment(child_a)
    repair_invalid_assignment(child_b)
    
    info = OperatorUsageInfo(used_uniform_machine_crossover=True)
    return child_a, child_b, info


def gpx_crossover(instance: Instance, parent_a: Solution, parent_b: Solution) -> Tuple[List[int], List[int], OperatorUsageInfo]:
    rng = get_rng()
    length = len(parent_a.scheduling_string)
    job_counts = _job_operation_counts(instance)

    def build_child(primary: List[int], secondary: List[int]) -> List[int]:
        segment_start = rng.randrange(length)
        segment_end = rng.randrange(segment_start, length)
        child = [-1] * length
        used_counts = {job_id: 0 for job_id in job_counts}
        for idx in range(segment_start, segment_end + 1):
            job_id = primary[idx]
            child[idx] = job_id
            used_counts[job_id] += 1
        empty_positions = [idx for idx, value in enumerate(child) if value == -1]
        insert_index = 0
        for job_id in secondary:
            if used_counts[job_id] < job_counts[job_id]:
                position = empty_positions[insert_index]
                child[position] = job_id
                used_counts[job_id] += 1
                insert_index += 1
                if insert_index >= len(empty_positions):
                    break
        return child

    child_a = build_child(parent_a.scheduling_string, parent_b.scheduling_string)
    child_b = build_child(parent_b.scheduling_string, parent_a.scheduling_string)
    info = OperatorUsageInfo(used_gpx=True)
    return child_a, child_b, info


def pox_crossover(instance: Instance, parent_a: Solution, parent_b: Solution) -> Tuple[List[int], List[int], OperatorUsageInfo]:
    rng = get_rng()
    job_counts = _job_operation_counts(instance)
    job_ids = list(job_counts.keys())
    rng.shuffle(job_ids)
    half = len(job_ids) // 2
    group_a = set(job_ids[:half])
    group_b = set(job_ids[half:])
    length = len(parent_a.scheduling_string)

    def build_child(primary: List[int], secondary: List[int], primary_group: set[int]) -> List[int]:
        child = [-1] * length
        used_counts = {job_id: 0 for job_id in job_counts}
        # Copy the selected job group from the primary parent, preserving positions
        for idx, job_id in enumerate(primary):
            if job_id in primary_group and used_counts[job_id] < job_counts[job_id]:
                child[idx] = job_id
                used_counts[job_id] += 1
        empty_positions = [idx for idx, value in enumerate(child) if value == -1]
        insert_index = 0
        # Fill remaining slots using the secondary parent's order
        for job_id in secondary:
            if used_counts[job_id] < job_counts[job_id]:
                position = empty_positions[insert_index]
                child[position] = job_id
                used_counts[job_id] += 1
                insert_index += 1
                if insert_index >= len(empty_positions):
                    break
        return child

    child_a = build_child(parent_a.scheduling_string, parent_b.scheduling_string, group_a)
    child_b = build_child(parent_b.scheduling_string, parent_a.scheduling_string, group_b)
    info = OperatorUsageInfo(used_pox=True)
    return child_a, child_b, info


def insert_mutation(sequence: List[int]) -> List[int]:
    rng = get_rng()
    if len(sequence) < 2:
        return list(sequence)
    i, j = rng.sample(range(len(sequence)), 2)
    seq = list(sequence)
    value = seq.pop(i)
    seq.insert(j, value)
    return seq


def swap_mutation(sequence: List[int]) -> List[int]:
    rng = get_rng()
    if len(sequence) < 2:
        return list(sequence)
    i, j = rng.sample(range(len(sequence)), 2)
    seq = list(sequence)
    seq[i], seq[j] = seq[j], seq[i]
    return seq


def inverse_mutation(sequence: List[int]) -> List[int]:
    rng = get_rng()
    if len(sequence) < 2:
        return list(sequence)
    i, j = sorted(rng.sample(range(len(sequence)), 2))
    seq = list(sequence)
    seq[i : j + 1] = reversed(seq[i : j + 1])
    return seq


def change1_mutation(instance: Instance, machines: List[int]) -> List[int]:
    rng = get_rng()
    pos = rng.randrange(len(machines))
    job_id, op_idx = instance.linearized_operations[pos]
    operation = instance.jobs[job_id - 1].operations[op_idx]
    current_machine = machines[pos]
    best_machine = current_machine
    best_value = None
    for option in operation.options:
        delta_power = (
            instance.get_machine_by_id(option.machine_id).processing_power
            - instance.get_machine_by_id(option.machine_id).idle_power
        )
        energy_tfn = option.processing_time.mul(delta_power)
        c1_value = energy_tfn._c1()
        if best_value is None or c1_value < best_value:
            best_value = c1_value
            best_machine = option.machine_id
        elif abs(c1_value - best_value) <= 1e-12 and rng.random() < 0.5:
            best_machine = option.machine_id
            best_value = c1_value
    if best_machine == current_machine:
        alternatives = [opt.machine_id for opt in operation.options if opt.machine_id != current_machine]
        if alternatives:
            best_machine = rng.choice(alternatives)
    new_assignment = list(machines)
    new_assignment[pos] = best_machine
    return new_assignment


def change2_mutation(instance: Instance, machines: List[int]) -> List[int]:
    rng = get_rng()
    pos = rng.randrange(len(machines))
    job_id, op_idx = instance.linearized_operations[pos]
    operation = instance.jobs[job_id - 1].operations[op_idx]
    candidates = []
    for option in operation.options:
        candidates.append((option.processing_time._c1(), option.machine_id))
    min_value = min(candidates, key=lambda item: item[0])[0]
    best_machines = [machine_id for value, machine_id in candidates if abs(value - min_value) <= 1e-12]
    new_assignment = list(machines)
    new_assignment[pos] = rng.choice(best_machines)
    return new_assignment


def change3_mutation(instance: Instance, machines: List[int], scheduling: List[int]) -> List[int]:
    rng = get_rng()
    from .decoder import decode_solution
    from .evaluation_counter import increment_evaluation_counter

    schedule = decode_solution(
        instance=instance,
        scheduling_string=list(scheduling),
        machine_assignment_string=list(machines),
    )
    # Preserve the existing termination behaviour: historically this operator
    # performed a full evaluation (and incremented the global counter) just to
    # obtain a decoded schedule for bottleneck analysis.
    increment_evaluation_counter()
    if not schedule.machines:
        return list(machines)
    bottleneck_machine = max(
        schedule.machines.values(),
        key=lambda ms: ms.completion_time._c1(),
    )
    if not bottleneck_machine.operations:
        return list(machines)
    # Select operation with longest processing time on bottleneck machine
    # Randomly select an operation on the bottleneck machine as described in the paper
    target_operation = rng.choice(list(bottleneck_machine.operations))
    job_id = target_operation.job_id
    op_idx = target_operation.operation_idx
    pos = instance.get_operation_position(job_id, op_idx)
    operation = instance.jobs[job_id - 1].operations[op_idx]
    current_machine = machines[pos]
    alternatives = [opt for opt in operation.options if opt.machine_id != current_machine]
    if not alternatives:
        return list(machines)
    best_option = rng.choice(alternatives)
    new_assignment = list(machines)
    new_assignment[pos] = best_option.machine_id
    return new_assignment


def apply_scheduling_mutation(sequence: List[int], probabilities: Tuple[float, float, float], info: OperatorUsageInfo) -> List[int]:
    rng = get_rng()
    cumulative = [
        probabilities[0],
        probabilities[0] + probabilities[1],
        probabilities[0] + probabilities[1] + probabilities[2],
    ]
    r = rng.random()
    if r < cumulative[0]:
        info.used_insert_mutation = True
        return insert_mutation(sequence)
    if r < cumulative[1]:
        info.used_swap_mutation = True
        return swap_mutation(sequence)
    info.used_inverse_mutation = True
    return inverse_mutation(sequence)


def apply_machine_mutation(
    instance: Instance,
    machines: List[int],
    scheduling: List[int],
    probabilities: Tuple[float, float, float],
    info: OperatorUsageInfo,
) -> List[int]:
    rng = get_rng()
    cumulative = [
        probabilities[0],
        probabilities[0] + probabilities[1],
        probabilities[0] + probabilities[1] + probabilities[2],
    ]
    r = rng.random()
    if r < cumulative[0]:
        info.used_change1_mutation = True
        return change1_mutation(instance, machines)
    if r < cumulative[1]:
        info.used_change2_mutation = True
        return change2_mutation(instance, machines)
    info.used_change3_mutation = True
    return change3_mutation(instance, machines, scheduling)


def algorithm3_crossover_mutation(
    instance: Instance,
    parent_x: Solution,
    parent_y: Solution,
    crossover_probability: float,
    mutation_probability: float,
    q_bar_1: float,
    q_bar_2: float,
    w_bar_1: Tuple[float, float, float],
    w_bar_2: Tuple[float, float, float],
    evaluate_offspring: bool = True,
) -> Tuple[Solution, Solution, OperatorUsageInfo, OperatorUsageInfo]:
    rng = get_rng()
    info_x = OperatorUsageInfo()
    info_y = OperatorUsageInfo()
    scheduling_x = list(parent_x.scheduling_string)
    scheduling_y = list(parent_y.scheduling_string)
    machine_x = list(parent_x.machine_assignment_string)
    machine_y = list(parent_y.machine_assignment_string)

    if rng.random() < crossover_probability:
        mode = rng.choice(["scheduling", "machine", "both"])
        if mode in ("scheduling", "both"):
            if rng.random() < q_bar_1:
                scheduling_x, scheduling_y, gpx_info = gpx_crossover(instance, parent_x, parent_y)
                if gpx_info.used_gpx:
                    info_x.used_gpx = True
                    info_y.used_gpx = True
            else:
                scheduling_x, scheduling_y, pox_info = pox_crossover(instance, parent_x, parent_y)
                if pox_info.used_pox:
                    info_x.used_pox = True
                    info_y.used_pox = True
        if mode in ("machine", "both"):
            machine_x, machine_y, machine_info = uniform_machine_crossover(parent_x, parent_y)
            if machine_info.used_uniform_machine_crossover:
                info_x.used_uniform_machine_crossover = True
                info_y.used_uniform_machine_crossover = True
    else:
        scheduling_x = list(parent_x.scheduling_string)
        scheduling_y = list(parent_y.scheduling_string)
        machine_x = list(parent_x.machine_assignment_string)
        machine_y = list(parent_y.machine_assignment_string)

    def mutate_child(scheduling, machines, info: OperatorUsageInfo):
        if rng.random() < mutation_probability:
            mode = rng.choice(["scheduling", "machine", "both"])
            if mode in ("scheduling", "both"):
                scheduling = apply_scheduling_mutation(scheduling, w_bar_1, info)
            if mode in ("machine", "both"):
                machines = apply_machine_mutation(instance, machines, scheduling, w_bar_2, info)
        return scheduling, machines

    scheduling_x, machine_x = mutate_child(scheduling_x, machine_x, info_x)
    scheduling_y, machine_y = mutate_child(scheduling_y, machine_y, info_y)

    offspring_x = _clone_solution_with_strings(parent_x, scheduling_x, machine_x)
    offspring_y = _clone_solution_with_strings(parent_y, scheduling_y, machine_y)

    if evaluate_offspring:
        calculate_objectives_with_count(offspring_x)
        calculate_objectives_with_count(offspring_y)

    return offspring_x, offspring_y, info_x, info_y
