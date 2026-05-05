"""
Decoder for converting solution strings into a detailed schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .constants import EPS
from .fuzzy_operations import TriangularFuzzyNumber
from .heuristics import initialise_schedule_state
from .instance_loader import Instance

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None

_NUMBA_AVAILABLE = np is not None and njit is not None

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _agreement_index_numba(
        due_date_1: float,
        due_date_2: float,
        l: float,
        m: float,
        r: float,
        eps: float,
    ) -> tuple[float, float, float]:
        width = due_date_2 - due_date_1
        if width <= eps:
            sat_l = 1.0 if r <= due_date_1 else 0.0
            sat_m = 1.0 if m <= due_date_1 else 0.0
            sat_r = 1.0 if l <= due_date_1 else 0.0
            return sat_l, sat_m, sat_r
        sat_l = 1.0 - (r - due_date_1) / width
        sat_m = 1.0 - (m - due_date_1) / width
        sat_r = 1.0 - (l - due_date_1) / width
        if sat_l < 0.0:
            sat_l = 0.0
        elif sat_l > 1.0:
            sat_l = 1.0
        if sat_m < 0.0:
            sat_m = 0.0
        elif sat_m > 1.0:
            sat_m = 1.0
        if sat_r < 0.0:
            sat_r = 0.0
        elif sat_r > 1.0:
            sat_r = 1.0
        return sat_l, sat_m, sat_r


    @njit(cache=True)
    def _fuzzy_lt_numba(
        l1: float,
        m1: float,
        r1: float,
        l2: float,
        m2: float,
        r2: float,
        eps: float,
    ) -> bool:
        c1_1 = (l1 + 2.0 * m1 + r1) / 4.0
        c1_2 = (l2 + 2.0 * m2 + r2) / 4.0
        if c1_1 < c1_2 - eps:
            return True
        if c1_1 > c1_2 + eps:
            return False
        if m1 < m2 - eps:
            return True
        if m1 > m2 + eps:
            return False
        return (r1 - l1) < (r2 - l2) - eps


    @njit(cache=True)
    def _evaluate_solution_numba_kernel(
        scheduling: "np.ndarray",
        machine_assignment: "np.ndarray",
        op_offsets: "np.ndarray",
        processing_l_table: "np.ndarray",
        processing_m_table: "np.ndarray",
        processing_r_table: "np.ndarray",
        valid_machine_mask: "np.ndarray",
        machine_delta_power: "np.ndarray",
        machine_idle_power: "np.ndarray",
        job_due_1: "np.ndarray",
        job_due_2: "np.ndarray",
        eps: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        num_jobs = job_due_1.shape[0] - 1
        max_machine_id = machine_idle_power.shape[0] - 1

        job_current_op = np.zeros(num_jobs + 1, dtype=np.int64)
        job_l = np.zeros(num_jobs + 1, dtype=np.float64)
        job_m = np.zeros(num_jobs + 1, dtype=np.float64)
        job_r = np.zeros(num_jobs + 1, dtype=np.float64)

        machine_l = np.zeros(max_machine_id + 1, dtype=np.float64)
        machine_m = np.zeros(max_machine_id + 1, dtype=np.float64)
        machine_r = np.zeros(max_machine_id + 1, dtype=np.float64)
        processing_l = np.zeros(max_machine_id + 1, dtype=np.float64)
        processing_m = np.zeros(max_machine_id + 1, dtype=np.float64)
        processing_r = np.zeros(max_machine_id + 1, dtype=np.float64)

        for idx in range(scheduling.shape[0]):
            job_id = int(scheduling[idx])
            operation_idx = job_current_op[job_id]
            op_pos = op_offsets[job_id] + operation_idx
            machine_id = int(machine_assignment[op_pos])
            if machine_id < 0 or machine_id > max_machine_id or not valid_machine_mask[op_pos, machine_id]:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            pt_l = processing_l_table[op_pos, machine_id]
            pt_m = processing_m_table[op_pos, machine_id]
            pt_r = processing_r_table[op_pos, machine_id]

            ready_l = job_l[job_id]
            ready_m = job_m[job_id]
            ready_r = job_r[job_id]
            machine_ready_l = machine_l[machine_id]
            machine_ready_m = machine_m[machine_id]
            machine_ready_r = machine_r[machine_id]
            start_l = ready_l if ready_l >= machine_ready_l else machine_ready_l
            start_m = ready_m if ready_m >= machine_ready_m else machine_ready_m
            start_r = ready_r if ready_r >= machine_ready_r else machine_ready_r
            completion_l = start_l + pt_l
            completion_m = start_m + pt_m
            completion_r = start_r + pt_r

            job_l[job_id] = completion_l
            job_m[job_id] = completion_m
            job_r[job_id] = completion_r
            machine_l[machine_id] = completion_l
            machine_m[machine_id] = completion_m
            machine_r[machine_id] = completion_r
            job_current_op[job_id] = operation_idx + 1

            delta = machine_delta_power[machine_id]
            if delta >= 0.0:
                processing_l[machine_id] += pt_l * delta
                processing_m[machine_id] += pt_m * delta
                processing_r[machine_id] += pt_r * delta
            else:
                processing_l[machine_id] += pt_r * delta
                processing_m[machine_id] += pt_m * delta
                processing_r[machine_id] += pt_l * delta

        makespan_l = 0.0
        makespan_m = 0.0
        makespan_r = 0.0
        for job_id in range(1, num_jobs + 1):
            if job_l[job_id] > makespan_l:
                makespan_l = job_l[job_id]
            if job_m[job_id] > makespan_m:
                makespan_m = job_m[job_id]
            if job_r[job_id] > makespan_r:
                makespan_r = job_r[job_id]

        energy_l = 0.0
        energy_m = 0.0
        energy_r = 0.0
        for machine_id in range(1, max_machine_id + 1):
            idle = machine_idle_power[machine_id]
            completion_l = machine_l[machine_id]
            completion_m = machine_m[machine_id]
            completion_r = machine_r[machine_id]
            if idle >= 0.0:
                idle_l = completion_l * idle
                idle_m = completion_m * idle
                idle_r = completion_r * idle
            else:
                idle_l = completion_r * idle
                idle_m = completion_m * idle
                idle_r = completion_l * idle
            energy_l += processing_l[machine_id] + idle_l
            energy_m += processing_m[machine_id] + idle_m
            energy_r += processing_r[machine_id] + idle_r

        dissatisfaction_sum_l = 0.0
        dissatisfaction_sum_m = 0.0
        dissatisfaction_sum_r = 0.0
        for job_id in range(1, num_jobs + 1):
            agreement_l, agreement_m, agreement_r = _agreement_index_numba(
                job_due_1[job_id],
                job_due_2[job_id],
                job_l[job_id],
                job_m[job_id],
                job_r[job_id],
                eps,
            )
            dissatisfaction_sum_l += 1.0 - agreement_r
            dissatisfaction_sum_m += 1.0 - agreement_m
            dissatisfaction_sum_r += 1.0 - agreement_l

        inv_num_jobs = 1.0 / num_jobs

        return (
            makespan_l,
            makespan_m,
            makespan_r,
            energy_l,
            energy_m,
            energy_r,
            dissatisfaction_sum_l * inv_num_jobs,
            dissatisfaction_sum_m * inv_num_jobs,
            dissatisfaction_sum_r * inv_num_jobs,
        )


_ZERO_TFN = TriangularFuzzyNumber(0.0, 0.0, 0.0)


@dataclass(slots=True)
class OperationSchedule:
    job_id: int
    operation_idx: int
    machine_id: int
    start_time: TriangularFuzzyNumber
    completion_time: TriangularFuzzyNumber
    processing_time: TriangularFuzzyNumber


@dataclass(slots=True)
class MachineSchedule:
    machine_id: int
    operations: List[OperationSchedule] = field(default_factory=list)
    completion_time: TriangularFuzzyNumber = _ZERO_TFN

    def add_operation(self, operation_schedule: OperationSchedule) -> None:
        self.operations.append(operation_schedule)
        self.completion_time = self.completion_time.max_with(operation_schedule.completion_time)


@dataclass(slots=True)
class Schedule:
    instance: Instance
    operations: List[OperationSchedule] = field(default_factory=list)
    machines: Dict[int, MachineSchedule] = field(default_factory=dict)
    makespan: Optional[TriangularFuzzyNumber] = None
    total_energy_consumption: Optional[TriangularFuzzyNumber] = None
    minimum_agreement_index: Optional[float] = None

    def add_operation_schedule(self, operation_schedule: OperationSchedule) -> None:
        self.operations.append(operation_schedule)
        machine_schedule = self.machines.setdefault(
            operation_schedule.machine_id,
            MachineSchedule(machine_id=operation_schedule.machine_id),
        )
        machine_schedule.add_operation(operation_schedule)

    def get_operation_schedule(self, job_id: int, operation_idx: int) -> OperationSchedule:
        for operation in self.operations:
            if operation.job_id == job_id and operation.operation_idx == operation_idx:
                return operation
        raise ValueError(f"Operation schedule not found for job {job_id} operation {operation_idx}")

    def calculate_all_metrics(self) -> None:
        from .metrics_calculator import calculate_all_metrics

        makespan, energy, agreement = calculate_all_metrics(self)
        self.makespan = makespan
        self.total_energy_consumption = energy
        self.minimum_agreement_index = agreement


def decode_solution(
    instance: Instance,
    scheduling_string: List[int],
    machine_assignment_string: List[int],
) -> Schedule:
    state = initialise_schedule_state(instance)
    schedule = Schedule(instance=instance)

    job_current_op = state.job_current_operation
    job_last_completion = state.job_last_completion_time
    machine_last_completion = state.machine_last_completion_time

    op_offsets = instance.job_operation_offsets
    processing_maps = instance.operation_processing_map
    assignments = machine_assignment_string

    operations = schedule.operations
    machine_schedules = schedule.machines
    machine_schedules_get = machine_schedules.get

    for job_id in scheduling_string:
        operation_idx = job_current_op[job_id]
        pos = op_offsets[job_id] + operation_idx
        machine_id = assignments[pos]
        processing_map = processing_maps[pos]
        try:
            processing_time = processing_map[machine_id]
        except KeyError as exc:
            raise ValueError(
                f"Machine {machine_id} is not a valid option for job {job_id} operation {operation_idx}"
            ) from exc

        job_ready = job_last_completion[job_id]
        machine_ready = machine_last_completion[machine_id]
        start = job_ready.max_with(machine_ready)
        completion = start.add(processing_time)

        operation_schedule = OperationSchedule(
            job_id=job_id,
            operation_idx=operation_idx,
            machine_id=machine_id,
            start_time=start,
            completion_time=completion,
            processing_time=processing_time,
        )
        operations.append(operation_schedule)

        machine_schedule = machine_schedules_get(machine_id)
        if machine_schedule is None:
            machine_schedule = MachineSchedule(machine_id=machine_id)
            machine_schedules[machine_id] = machine_schedule
        machine_schedule.operations.append(operation_schedule)
        machine_schedule.completion_time = machine_schedule.completion_time.max_with(completion)

        job_last_completion[job_id] = completion
        machine_last_completion[machine_id] = completion
        job_current_op[job_id] = operation_idx + 1

    return schedule


def _evaluate_solution_strings_python(
    instance: Instance,
    scheduling_string: List[int],
    machine_assignment_string: List[int],
) -> tuple[TriangularFuzzyNumber, TriangularFuzzyNumber, TriangularFuzzyNumber, None, tuple]:
    from .metrics_calculator import calculate_dissatisfaction_degree_for_job

    num_jobs = instance.num_jobs
    max_machine_id = instance.max_machine_id

    job_current_op = [0] * (num_jobs + 1)
    job_l = [0.0] * (num_jobs + 1)
    job_m = [0.0] * (num_jobs + 1)
    job_r = [0.0] * (num_jobs + 1)

    machine_l = [0.0] * (max_machine_id + 1)
    machine_m = [0.0] * (max_machine_id + 1)
    machine_r = [0.0] * (max_machine_id + 1)
    processing_l = [0.0] * (max_machine_id + 1)
    processing_m = [0.0] * (max_machine_id + 1)
    processing_r = [0.0] * (max_machine_id + 1)
    machine_seen = [False] * (max_machine_id + 1)
    machine_order: list[int] = []

    op_offsets = instance.job_operation_offsets
    processing_maps = instance.operation_processing_map
    assignments = machine_assignment_string
    delta_power = instance.machine_delta_power
    idle_power = instance.machine_idle_power

    for job_id in scheduling_string:
        operation_idx = job_current_op[job_id]
        pos = op_offsets[job_id] + operation_idx
        machine_id = assignments[pos]
        if not machine_seen[machine_id]:
            machine_seen[machine_id] = True
            machine_order.append(machine_id)
        processing_map = processing_maps[pos]
        try:
            processing_time = processing_map[machine_id]
        except KeyError as exc:
            raise ValueError(
                f"Machine {machine_id} is not a valid option for job {job_id} operation {operation_idx}"
            ) from exc

        ready_l = job_l[job_id]
        ready_m = job_m[job_id]
        ready_r = job_r[job_id]
        machine_ready_l = machine_l[machine_id]
        machine_ready_m = machine_m[machine_id]
        machine_ready_r = machine_r[machine_id]
        start_l = ready_l if ready_l >= machine_ready_l else machine_ready_l
        start_m = ready_m if ready_m >= machine_ready_m else machine_ready_m
        start_r = ready_r if ready_r >= machine_ready_r else machine_ready_r
        completion_l = start_l + processing_time.l
        completion_m = start_m + processing_time.m
        completion_r = start_r + processing_time.r

        job_l[job_id] = completion_l
        job_m[job_id] = completion_m
        job_r[job_id] = completion_r
        machine_l[machine_id] = completion_l
        machine_m[machine_id] = completion_m
        machine_r[machine_id] = completion_r
        job_current_op[job_id] = operation_idx + 1

        delta = delta_power[machine_id]
        if delta >= 0:
            processing_l[machine_id] += processing_time.l * delta
            processing_m[machine_id] += processing_time.m * delta
            processing_r[machine_id] += processing_time.r * delta
        else:
            processing_l[machine_id] += processing_time.r * delta
            processing_m[machine_id] += processing_time.m * delta
            processing_r[machine_id] += processing_time.l * delta

    makespan_l = makespan_m = makespan_r = 0.0
    for job_id in range(1, num_jobs + 1):
        if job_l[job_id] > makespan_l:
            makespan_l = job_l[job_id]
        if job_m[job_id] > makespan_m:
            makespan_m = job_m[job_id]
        if job_r[job_id] > makespan_r:
            makespan_r = job_r[job_id]
    makespan = TriangularFuzzyNumber._new_unchecked(makespan_l, makespan_m, makespan_r)

    energy_l = energy_m = energy_r = 0.0
    for machine_id in machine_order:
        idle = idle_power[machine_id]
        completion_l = machine_l[machine_id]
        completion_m = machine_m[machine_id]
        completion_r = machine_r[machine_id]
        if idle >= 0:
            idle_l = completion_l * idle
            idle_m = completion_m * idle
            idle_r = completion_r * idle
        else:
            idle_l = completion_r * idle
            idle_m = completion_m * idle
            idle_r = completion_l * idle
        energy_l += processing_l[machine_id] + idle_l
        energy_m += processing_m[machine_id] + idle_m
        energy_r += processing_r[machine_id] + idle_r
    energy = TriangularFuzzyNumber._new_unchecked(energy_l, energy_m, energy_r)

    total_dissatisfaction_l = total_dissatisfaction_m = total_dissatisfaction_r = 0.0
    jobs = instance.jobs
    for job_idx in range(1, num_jobs + 1):
        completion = TriangularFuzzyNumber._new_unchecked(job_l[job_idx], job_m[job_idx], job_r[job_idx])
        dissatisfaction = calculate_dissatisfaction_degree_for_job(jobs[job_idx - 1], completion)
        total_dissatisfaction_l += dissatisfaction.l
        total_dissatisfaction_m += dissatisfaction.m
        total_dissatisfaction_r += dissatisfaction.r
    dissatisfaction = TriangularFuzzyNumber._new_unchecked(
        total_dissatisfaction_l / num_jobs,
        total_dissatisfaction_m / num_jobs,
        total_dissatisfaction_r / num_jobs,
    )

    objective_vector = (
        makespan._c1(),
        makespan._c2(),
        makespan._c3(),
        energy._c1(),
        energy._c2(),
        energy._c3(),
        dissatisfaction._c1(),
        dissatisfaction._c2(),
        dissatisfaction._c3(),
    )
    return makespan, energy, dissatisfaction, None, objective_vector


def _evaluate_solution_strings_numba(
    instance: Instance,
    scheduling_string: List[int],
    machine_assignment_string: List[int],
) -> tuple[TriangularFuzzyNumber, TriangularFuzzyNumber, TriangularFuzzyNumber, None, tuple] | None:
    if not _NUMBA_AVAILABLE:
        return None
    if instance.job_operation_offsets_array is None:
        return None

    scheduling = np.asarray(scheduling_string, dtype=np.int64)
    machine_assignment = np.asarray(machine_assignment_string, dtype=np.int64)
    result = _evaluate_solution_numba_kernel(
        scheduling,
        machine_assignment,
        instance.job_operation_offsets_array,
        instance.operation_processing_l_array,
        instance.operation_processing_m_array,
        instance.operation_processing_r_array,
        instance.operation_valid_machine_mask,
        instance.machine_delta_power_array,
        instance.machine_idle_power_array,
        instance.job_due_date_1,
        instance.job_due_date_2,
        EPS,
    )
    (
        makespan_l,
        makespan_m,
        makespan_r,
        energy_l,
        energy_m,
        energy_r,
        agreement_l,
        agreement_m,
        agreement_r,
    ) = result
    if np.isnan(makespan_l):
        return None

    makespan = TriangularFuzzyNumber._new_unchecked(makespan_l, makespan_m, makespan_r)
    energy = TriangularFuzzyNumber._new_unchecked(energy_l, energy_m, energy_r)
    agreement = TriangularFuzzyNumber._new_unchecked(agreement_l, agreement_m, agreement_r)
    objective_vector = (
        makespan._c1(),
        makespan._c2(),
        makespan._c3(),
        energy._c1(),
        energy._c2(),
        energy._c3(),
        agreement._c1(),
        agreement._c2(),
        agreement._c3(),
    )
    return makespan, energy, agreement, None, objective_vector


def evaluate_solution_strings_fast(
    instance: Instance,
    scheduling_string: List[int],
    machine_assignment_string: List[int],
) -> tuple[TriangularFuzzyNumber, TriangularFuzzyNumber, TriangularFuzzyNumber, None, tuple]:
    """
    Fast objective evaluation used by batched algorithm paths.

    This keeps objective semantics unchanged while avoiding per-operation
    schedule object construction in hot loops. If Numba is available, a
    compiled kernel is used; otherwise it falls back to the pure-Python path.
    """
    numba_result = _evaluate_solution_strings_numba(
        instance,
        scheduling_string,
        machine_assignment_string,
    )
    if numba_result is not None:
        return numba_result
    return _evaluate_solution_strings_python(
        instance,
        scheduling_string,
        machine_assignment_string,
    )
