"""
Objective metric calculations for schedules.
"""

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

from .constants import EPS
from .fuzzy_operations import TriangularFuzzyNumber

if TYPE_CHECKING:  # pragma: no cover
    from .decoder import Schedule
    from .instance_loader import Job


def _zero_tfn() -> TriangularFuzzyNumber:
    return TriangularFuzzyNumber(0.0, 0.0, 0.0)


def calculate_makespan(schedule: "Schedule") -> TriangularFuzzyNumber:
    job_completion: Dict[int, TriangularFuzzyNumber] = {}
    for operation in schedule.operations:
        job_completion[operation.job_id] = operation.completion_time
    max_l = max_m = max_r = 0.0
    for completion in job_completion.values():
        if completion.l > max_l:
            max_l = completion.l
        if completion.m > max_m:
            max_m = completion.m
        if completion.r > max_r:
            max_r = completion.r
    return TriangularFuzzyNumber(max_l, max_m, max_r)


def calculate_total_energy_consumption(schedule: "Schedule") -> TriangularFuzzyNumber:
    total_l = total_m = total_r = 0.0
    for machine_id, machine_schedule in schedule.machines.items():
        machine = schedule.instance.get_machine_by_id(machine_id)
        processing_l = processing_m = processing_r = 0.0
        delta_power = machine.processing_power - machine.idle_power
        for operation in machine_schedule.operations:
            processing_time = operation.processing_time
            if delta_power >= 0:
                processing_l += processing_time.l * delta_power
                processing_m += processing_time.m * delta_power
                processing_r += processing_time.r * delta_power
            else:
                processing_l += processing_time.r * delta_power
                processing_m += processing_time.m * delta_power
                processing_r += processing_time.l * delta_power
        completion_time = machine_schedule.completion_time
        idle_power = machine.idle_power
        if idle_power >= 0:
            idle_l = completion_time.l * idle_power
            idle_m = completion_time.m * idle_power
            idle_r = completion_time.r * idle_power
        else:
            idle_l = completion_time.r * idle_power
            idle_m = completion_time.m * idle_power
            idle_r = completion_time.l * idle_power
        total_l += processing_l + idle_l
        total_m += processing_m + idle_m
        total_r += processing_r + idle_r
    return TriangularFuzzyNumber(total_l, total_m, total_r)


def _triangle_area(l: float, r: float) -> float:
    return max(r - l, 0.0) / 2.0


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _deadline_satisfaction(value: float, due_date_1: float, due_date_2: float) -> float:
    width = due_date_2 - due_date_1
    if width <= EPS:
        return 1.0 if value <= due_date_1 else 0.0
    return _clamp_unit(1.0 - (value - due_date_1) / width)


def _integrate_triangle_segment(
    l: float,
    m: float,
    r: float,
    start: float,
    end: float,
) -> float:
    start = max(start, l)
    end = min(end, r)
    if start >= end:
        return 0.0
    area = 0.0
    if start < m:
        upper = min(end, m)
        denom = max(m - l, EPS)
        area += ((upper - l) ** 2 - (start - l) ** 2) / (2.0 * denom)
    if end > m:
        lower = max(start, m)
        denom = max(r - m, EPS)
        area += ((r - lower) ** 2 - (r - end) ** 2) / (2.0 * denom)
    return max(area, 0.0)


def calculate_agreement_index_for_job(
    job: "Job",
    completion_time: TriangularFuzzyNumber,
) -> TriangularFuzzyNumber:
    due_date_1, due_date_2 = job.due_date
    l, m, r = completion_time.l, completion_time.m, completion_time.r

    satisfaction_l = _deadline_satisfaction(r, due_date_1, due_date_2)
    satisfaction_m = _deadline_satisfaction(m, due_date_1, due_date_2)
    satisfaction_r = _deadline_satisfaction(l, due_date_1, due_date_2)
    return TriangularFuzzyNumber._new_unchecked(satisfaction_l, satisfaction_m, satisfaction_r)


def calculate_dissatisfaction_degree_for_job(
    job: "Job",
    completion_time: TriangularFuzzyNumber,
) -> TriangularFuzzyNumber:
    satisfaction = calculate_agreement_index_for_job(job, completion_time)
    return TriangularFuzzyNumber._new_unchecked(
        1.0 - satisfaction.r,
        1.0 - satisfaction.m,
        1.0 - satisfaction.l,
    )


def calculate_average_dissatisfaction_degree(schedule: "Schedule") -> TriangularFuzzyNumber:
    job_completion: Dict[int, TriangularFuzzyNumber] = {}
    for operation in schedule.operations:
        job_completion[operation.job_id] = operation.completion_time
    total_l = total_m = total_r = 0.0
    for idx, job in enumerate(schedule.instance.jobs, start=1):
        dd = calculate_dissatisfaction_degree_for_job(job, job_completion[idx])
        total_l += dd.l
        total_m += dd.m
        total_r += dd.r
    num_jobs = max(schedule.instance.num_jobs, 1)
    return TriangularFuzzyNumber._new_unchecked(
        total_l / num_jobs,
        total_m / num_jobs,
        total_r / num_jobs,
    )


def calculate_all_metrics(schedule: "Schedule") -> Tuple[TriangularFuzzyNumber, TriangularFuzzyNumber, TriangularFuzzyNumber]:
    makespan = calculate_makespan(schedule)
    energy = calculate_total_energy_consumption(schedule)
    agreement = calculate_average_dissatisfaction_degree(schedule)
    schedule.makespan = makespan
    schedule.total_energy_consumption = energy
    schedule.minimum_agreement_index = agreement
    return makespan, energy, agreement
