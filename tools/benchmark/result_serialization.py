"""
Helpers for serialising algorithm run results for later analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from fbea.evaluation_counter import get_evaluation_count
from fbea.solution import Solution


def _tf_to_tuple(tfn):
    try:
        return tuple(tfn.to_tuple())
    except AttributeError:
        return tuple(tfn)


def _solution_to_dict(solution: Solution) -> dict:
    """
    Serialize a Solution,并附带调度信息（供甘特图使用）。
    兼容：原有字段保持不变，额外字段为 scheduling_string、machine_assignment_string、
    gantt_operations（每道工序的模糊起止时间、加工时间）。
    """
    solution.evaluate()
    makespan = list(solution.makespan.to_tuple())
    energy = list(solution.energy.to_tuple())
    dissatisfaction = list(solution.agreement.to_tuple())

    # 构造甘特所需数据
    gantt_ops = []
    if solution.schedule is not None:
        for op in solution.schedule.operations:
            gantt_ops.append(
                {
                    "job_id": op.job_id,
                    "operation_idx": op.operation_idx,
                    "machine_id": op.machine_id,
                    "start_time": _tf_to_tuple(op.start_time),
                    "completion_time": _tf_to_tuple(op.completion_time),
                    "processing_time": _tf_to_tuple(op.processing_time),
                }
            )

    return {
        "makespan": makespan,
        "energy": energy,
        "dissatisfaction": dissatisfaction,
        "dissatisfaction_c1": solution.agreement._c1(),
        "dissatisfaction_c2": solution.agreement._c2(),
        "dissatisfaction_c3": solution.agreement._c3(),
        "agreement": dissatisfaction,
        "makespan_c1": solution.makespan._c1(),
        "energy_c1": solution.energy._c1(),
        "agreement_c1": solution.agreement._c1(),
        "agreement_c2": solution.agreement._c2(),
        "agreement_c3": solution.agreement._c3(),
        # 额外信息
        "scheduling_string": solution.scheduling_string,
        "machine_assignment_string": solution.machine_assignment_string,
        "gantt_operations": gantt_ops,
    }


def serialise_solutions(solutions: Iterable[Solution]) -> List[dict]:
    return [_solution_to_dict(sol) for sol in solutions]


def write_run_output(
    output_path: str | Path,
    *,
    algorithm: str,
    dataset_id: str,
    instance_path: str,
    instance_id: str,
    algorithm_seed: Optional[int],
    instance_seed: Optional[int],
    solutions: Iterable[Solution],
    runtime_seconds: Optional[float] = None,
    extra_metadata: Optional[dict] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": algorithm,
        "dataset_id": dataset_id,
        "instance_path": instance_path,
        "instance_id": instance_id,
        "algorithm_seed": algorithm_seed,
        "instance_seed": instance_seed,
        "evaluations": get_evaluation_count(),
        "runtime_seconds": runtime_seconds,
        "solutions": serialise_solutions(solutions),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
