"""
Problem instance loading utilities.

Supports loading from structured text or dictionary inputs while relying on the
global RNG for reproducibility.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from .constants import EPS
from .fuzzy_operations import TriangularFuzzyNumber
from .random_manager import get_rng


@dataclass
class Machine:
    machine_id: int
    idle_power: float
    processing_power: float


@dataclass
class OperationOption:
    machine_id: int
    processing_time: TriangularFuzzyNumber


@dataclass
class Operation:
    options: List[OperationOption]


@dataclass
class Job:
    operations: List[Operation]
    due_date: Tuple[float, float]

    @property
    def num_operations(self) -> int:
        return len(self.operations)


@dataclass
class Instance:
    num_jobs: int
    num_machines: int
    avg_options: int
    jobs: List[Job]
    machines: Dict[int, Machine]
    job_random_factors: List[float]
    total_operations: int = field(init=False)
    operation_position_map: Dict[Tuple[int, int], int] = field(init=False)
    linearized_operations: List[Tuple[int, int]] = field(init=False)
    operation_machine_options: List[List[int]] = field(init=False)
    job_operation_counts: Dict[int, int] = field(init=False)
    operation_processing_map: List[Dict[int, TriangularFuzzyNumber]] = field(init=False)
    job_operation_offsets: List[int] = field(init=False)
    max_machine_id: int = field(init=False)
    machine_idle_power: List[float] = field(init=False)
    machine_delta_power: List[float] = field(init=False)
    job_operation_offsets_array: Any = field(init=False, repr=False, default=None)
    machine_idle_power_array: Any = field(init=False, repr=False, default=None)
    machine_delta_power_array: Any = field(init=False, repr=False, default=None)
    operation_processing_l_array: Any = field(init=False, repr=False, default=None)
    operation_processing_m_array: Any = field(init=False, repr=False, default=None)
    operation_processing_r_array: Any = field(init=False, repr=False, default=None)
    operation_valid_machine_mask: Any = field(init=False, repr=False, default=None)
    job_due_date_1: Any = field(init=False, repr=False, default=None)
    job_due_date_2: Any = field(init=False, repr=False, default=None)
    objective_key_set: set = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        self._build_operation_map()

    def get_machine_by_id(self, machine_id: int) -> Machine:
        try:
            return self.machines[machine_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Machine id {machine_id} not found") from exc

    def _build_operation_map(self) -> None:
        position = 0
        mapping: Dict[Tuple[int, int], int] = {}
        linear: List[Tuple[int, int]] = []
        machine_options: List[List[int]] = []
        job_counts: Dict[int, int] = {}
        processing_map: List[Dict[int, TriangularFuzzyNumber]] = []
        job_offsets: List[int] = [0] * (self.num_jobs + 1)
        for job_idx, job in enumerate(self.jobs, start=1):
            job_offsets[job_idx] = position
            job_counts[job_idx] = job.num_operations
            for op_idx in range(job.num_operations):
                mapping[(job_idx, op_idx)] = position
                linear.append((job_idx, op_idx))
                options = job.operations[op_idx].options
                machine_options.append([option.machine_id for option in options])
                processing_map.append({option.machine_id: option.processing_time for option in options})
                position += 1
        self.total_operations = position
        self.operation_position_map = mapping
        self.linearized_operations = linear
        self.operation_machine_options = machine_options
        self.job_operation_counts = job_counts
        self.operation_processing_map = processing_map
        self.job_operation_offsets = job_offsets

        max_machine_id = max(self.machines.keys()) if self.machines else 0
        idle_power = [0.0] * (max_machine_id + 1)
        delta_power = [0.0] * (max_machine_id + 1)
        for machine_id, machine in self.machines.items():
            idle_power[machine_id] = machine.idle_power
            delta_power[machine_id] = machine.processing_power - machine.idle_power
        self.max_machine_id = max_machine_id
        self.machine_idle_power = idle_power
        self.machine_delta_power = delta_power

        if np is None:
            self.job_operation_offsets_array = None
            self.machine_idle_power_array = None
            self.machine_delta_power_array = None
            self.operation_processing_l_array = None
            self.operation_processing_m_array = None
            self.operation_processing_r_array = None
            self.operation_valid_machine_mask = None
            self.job_due_date_1 = None
            self.job_due_date_2 = None
            return

        self.job_operation_offsets_array = np.asarray(job_offsets, dtype=np.int64)
        self.machine_idle_power_array = np.asarray(idle_power, dtype=np.float64)
        self.machine_delta_power_array = np.asarray(delta_power, dtype=np.float64)

        shape = (self.total_operations, max_machine_id + 1)
        processing_l = np.zeros(shape, dtype=np.float64)
        processing_m = np.zeros(shape, dtype=np.float64)
        processing_r = np.zeros(shape, dtype=np.float64)
        valid_mask = np.zeros(shape, dtype=np.bool_)
        for op_pos, proc_map in enumerate(processing_map):
            for machine_id, tfn in proc_map.items():
                valid_mask[op_pos, machine_id] = True
                processing_l[op_pos, machine_id] = tfn.l
                processing_m[op_pos, machine_id] = tfn.m
                processing_r[op_pos, machine_id] = tfn.r
        self.operation_processing_l_array = processing_l
        self.operation_processing_m_array = processing_m
        self.operation_processing_r_array = processing_r
        self.operation_valid_machine_mask = valid_mask

        due_1 = np.zeros(self.num_jobs + 1, dtype=np.float64)
        due_2 = np.zeros(self.num_jobs + 1, dtype=np.float64)
        for job_idx, job in enumerate(self.jobs, start=1):
            due_1[job_idx] = float(job.due_date[0])
            due_2[job_idx] = float(job.due_date[1])
        self.job_due_date_1 = due_1
        self.job_due_date_2 = due_2

    def get_operation_position(self, job_id: int, operation_idx: int) -> int:
        """
        Return the linearised position of the given job operation.
        """
        if 1 <= job_id <= self.num_jobs:
            max_ops = self.job_operation_counts.get(job_id)
            if max_ops is not None and 0 <= operation_idx < max_ops:
                return self.job_operation_offsets[job_id] + operation_idx
        raise ValueError(f"Unknown operation reference ({job_id}, {operation_idx})")

    @property
    def random_factor_a(self) -> float:
        """
        Backwards-compatible aggregate random factor (mean of per-job factors).
        """
        if not self.job_random_factors:
            return 0.0
        return sum(self.job_random_factors) / len(self.job_random_factors)


def _round_two_decimals(value: float) -> float:
    return round(value, 2)


def _to_tfn_from_scalar(t: float) -> TriangularFuzzyNumber:
    rng = get_rng()
    l = _round_two_decimals(rng.uniform(0.85 * t, t))
    r = _round_two_decimals(rng.uniform(t, 1.2 * t))
    return TriangularFuzzyNumber(l, t, r)


def _parse_processing_time(value) -> TriangularFuzzyNumber:
    """
    Parse processing time from either scalar or tuple/list format.
    """
    if isinstance(value, (list, tuple)) and len(value) == 3:
        # Already a TFN tuple (l, m, r)
        return TriangularFuzzyNumber(value[0], value[1], value[2])
    elif isinstance(value, (int, float)):
        # Scalar value - convert to TFN
        return _to_tfn_from_scalar(value)
    else:
        raise ValueError(f"Invalid processing time format: {value}")


def _parse_machine_powers(
    num_machines: int, data: Optional[Sequence[Tuple[float, float]]] = None
) -> Dict[int, Machine]:
    rng = get_rng()
    machines: Dict[int, Machine] = {}
    for idx in range(1, num_machines + 1):
        if data is not None and idx - 1 < len(data):
            idle, proc = data[idx - 1]
        else:
            idle = _round_two_decimals(rng.uniform(1.0, 4.0))
            proc = _round_two_decimals(rng.uniform(4.0, 16.0))
        machines[idx] = Machine(machine_id=idx, idle_power=idle, processing_power=proc)
    return machines


def _calculate_due_dates(jobs: List[Job]) -> Tuple[List[Job], List[float]]:
    rng = get_rng()
    job_random_factors: List[float] = []
    for job in jobs:
        # 调整 due date 随机因子范围至 [2.0, 2.5]
        random_factor = _round_two_decimals(rng.uniform(0.5, 0.8))
        sum_max = TriangularFuzzyNumber(0.0, 0.0, 0.0)
        for operation in job.operations:
            max_tfn = operation.options[0].processing_time
            for option in operation.options[1:]:
                candidate = option.processing_time
                if candidate._c1() > max_tfn._c1():
                    max_tfn = candidate
                elif abs(candidate._c1() - max_tfn._c1()) <= EPS:
                    max_tfn = max_tfn.max_with(candidate)
            sum_max = sum_max.add(max_tfn)
        scaled = sum_max.mul(random_factor)
        due_date = (scaled.m, scaled.r)
        job.due_date = due_date
        job_random_factors.append(random_factor)
    return jobs, job_random_factors


def load_instance_from_memory(
    data: Dict,
    random_seed: Optional[int] = None,
) -> Instance:
    if random_seed is not None:
        from .random_manager import set_global_seed

        set_global_seed(random_seed)
    num_jobs = data["num_jobs"]
    num_machines = data["num_machines"]
    avg_options = data.get("avg_options", 0)
    machine_powers = data.get("machine_powers")
    machines = _parse_machine_powers(num_machines, machine_powers)

    jobs: List[Job] = []
    for job_data in data["jobs"]:
        operations: List[Operation] = []
        for op in job_data["operations"]:
            options = [
                OperationOption(machine_id=option["machine_id"], processing_time=_parse_processing_time(option["processing_time"]))
                for option in op["options"]
            ]
            operations.append(Operation(options=options))
        jobs.append(Job(operations=operations, due_date=(0.0, 0.0)))

    jobs, job_random_factors = _calculate_due_dates(jobs)
    instance = Instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        avg_options=avg_options,
        jobs=jobs,
        machines=machines,
        job_random_factors=job_random_factors,
    )
    return instance


def load_instance(path: str | pathlib.Path, random_seed: Optional[int] = None) -> Instance:
    if random_seed is not None:
        from .random_manager import set_global_seed

        set_global_seed(random_seed)

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return load_instance_from_memory(data)
    except json.JSONDecodeError:
        return _load_instance_from_plain_text(content)


def _load_instance_from_plain_text(content: str) -> Instance:
    tokens = [float(x) for x in content.split()]
    if len(tokens) < 3:
        raise ValueError("Plain text instance must start with num_jobs num_machines avg_options")
    num_jobs = int(tokens[0])
    num_machines = int(tokens[1])
    avg_options = int(tokens[2])

    machines = _parse_machine_powers(num_machines)

    idx = 3
    jobs: List[Job] = []
    for _ in range(num_jobs):
        if idx >= len(tokens):
            raise ValueError("Unexpected end of instance description")
        num_operations = int(tokens[idx])
        idx += 1
        operations: List[Operation] = []
        for _ in range(num_operations):
            num_options = int(tokens[idx])
            idx += 1
            options: List[OperationOption] = []
            for _ in range(num_options):
                machine_id = int(tokens[idx])
                duration = tokens[idx + 1]
                idx += 2
                options.append(
                    OperationOption(machine_id=machine_id, processing_time=_to_tfn_from_scalar(duration))
                )
            operations.append(Operation(options=options))
        jobs.append(Job(operations=operations, due_date=(0.0, 0.0)))

    jobs, job_random_factors = _calculate_due_dates(jobs)
    instance = Instance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        avg_options=avg_options,
        jobs=jobs,
        machines=machines,
        job_random_factors=job_random_factors,
    )
    return instance


def instance_to_dict(instance: Instance) -> Dict[str, Any]:
    return {
        "num_jobs": instance.num_jobs,
        "num_machines": instance.num_machines,
        "avg_options": instance.avg_options,
        "random_factor_a": instance.random_factor_a,
        "job_random_factors_a": list(instance.job_random_factors),
        "machines": [
            {
                "machine_id": machine.machine_id,
                "idle_power": machine.idle_power,
                "processing_power": machine.processing_power,
            }
            for machine in sorted(instance.machines.values(), key=lambda m: m.machine_id)
        ],
        "jobs": [
            {
                "due_date": list(job.due_date),
                "operations": [
                    {
                        "options": [
                            {
                                "machine_id": option.machine_id,
                                "processing_time": option.processing_time.to_tuple(),
                            }
                            for option in operation.options
                        ]
                    }
                    for operation in job.operations
                ],
            }
            for job in instance.jobs
        ],
    }


def load_instance_from_serialized(data: Dict[str, Any]) -> Instance:
    machines = {
        machine_data["machine_id"]: Machine(
            machine_id=machine_data["machine_id"],
            idle_power=machine_data["idle_power"],
            processing_power=machine_data["processing_power"],
        )
        for machine_data in data["machines"]
    }
    jobs: List[Job] = []
    for job_data in data["jobs"]:
        operations: List[Operation] = []
        for op in job_data["operations"]:
            options = [
                OperationOption(
                    machine_id=option["machine_id"],
                    processing_time=TriangularFuzzyNumber(*option["processing_time"]),
                )
                for option in op["options"]
            ]
            operations.append(Operation(options=options))
        jobs.append(Job(operations=operations, due_date=tuple(job_data["due_date"])))
    job_random_factors = data.get("job_random_factors_a")
    if isinstance(job_random_factors, list):
        job_random_factors = [float(value) for value in job_random_factors]
    else:
        legacy_factor = data.get("random_factor_a")
        if isinstance(legacy_factor, list):
            job_random_factors = [float(value) for value in legacy_factor]
        elif legacy_factor is not None:
            job_random_factors = [float(legacy_factor)] * data["num_jobs"]
        else:
            job_random_factors = []
    instance = Instance(
        num_jobs=data["num_jobs"],
        num_machines=data["num_machines"],
        avg_options=data.get("avg_options", 0),
        jobs=jobs,
        machines=machines,
        job_random_factors=job_random_factors,
    )
    return instance


def save_instance_to_json(instance: Instance, path: str | pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = instance_to_dict(instance)
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def load_instance_from_json(path: str | pathlib.Path) -> Instance:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Serialized instance file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return load_instance_from_serialized(data)
