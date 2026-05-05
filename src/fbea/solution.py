"""
Solution representation for the FBEA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from .evaluation_counter import calculate_objectives_with_count
from .instance_loader import Instance
from .random_manager import get_rng

if TYPE_CHECKING:  # pragma: no cover
    from .decoder import Schedule
    from .fuzzy_operations import TriangularFuzzyNumber


class SolutionValidationError(ValueError):
    """
    Raised when a scheduling or machine assignment string is invalid.
    """


@dataclass
class Solution:
    instance: Instance
    scheduling_string: Optional[List[int]] = None
    machine_assignment_string: Optional[List[int]] = None
    random_seed: Optional[int] = None
    skip_validation: bool = field(default=False, repr=False)
    makespan: Optional["TriangularFuzzyNumber"] = field(default=None, init=False)
    energy: Optional["TriangularFuzzyNumber"] = field(default=None, init=False)
    agreement: Optional["TriangularFuzzyNumber"] = field(default=None, init=False)
    evaluated: bool = field(default=False, init=False)
    schedule: Optional["Schedule"] = field(default=None, init=False)
    objective_key: Optional[tuple] = field(default=None, init=False, repr=False)
    # Cached scalar objectives used by dominance checks. Stored as:
    # (ms_c1, ms_c2, ms_c3, en_c1, en_c2, en_c3, ag_c1, ag_c2, ag_c3)
    objective_vector: Optional[tuple] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scheduling_string is None or self.machine_assignment_string is None:
            if self.skip_validation:
                raise SolutionValidationError("Cannot skip validation when strings are not provided.")
            self._initialise_with_heuristics()
        # 校验关闭：默认不再验证调度串/机器串合法性
        # if not self.skip_validation:
        #     self._validate_strings()

    def _initialise_with_heuristics(self) -> None:
        """
        Generate initial strings following Algorithm 1 heuristics and randomness.
        """
        rng = get_rng()
        alpha1 = rng.random()
        alpha2 = rng.random()
        from .heuristics import (
            heuristic_1_machine_assignment,
            heuristic_2_machine_assignment,
            heuristic_3_scheduling_string,
            heuristic_4_scheduling_string,
            random_machine_assignment,
            random_scheduling_string,
        )

        machine_assignment: List[int]
        if alpha1 < 0.4:
            machine_assignment = heuristic_1_machine_assignment(self.instance)
        elif alpha1 < 0.8:
            machine_assignment = heuristic_2_machine_assignment(self.instance)
        else:
            machine_assignment = random_machine_assignment(self.instance)

        if alpha2 < 0.4:
            scheduling = heuristic_3_scheduling_string(self.instance, machine_assignment)
        elif alpha2 < 0.8:
            scheduling = heuristic_4_scheduling_string(self.instance)
        else:
            scheduling = random_scheduling_string(self.instance)

        self.machine_assignment_string = machine_assignment
        self.scheduling_string = scheduling

    def _validate_strings(self) -> None:
        # 已禁用：保留函数占位，避免调用时报错
        return None

    def get_machine_for_operation(self, job_id: int, operation_idx: int) -> int:
        position = self.instance.get_operation_position(job_id, operation_idx)
        return self.machine_assignment_string[position]

    def calculate_objectives(self) -> None:
        if self.evaluated:
            return
        from .decoder import decode_solution
        from .metrics_calculator import calculate_all_metrics

        schedule = decode_solution(
            self.instance,
            self.scheduling_string,
            self.machine_assignment_string,
        )
        makespan, energy, agreement = calculate_all_metrics(schedule)
        self.schedule = schedule
        self.makespan = makespan
        self.energy = energy
        self.agreement = agreement
        self.evaluated = True
        self.objective_key = None
        self.objective_vector = (
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

    def evaluate(self) -> None:
        if self.evaluated:
            return
        calculate_objectives_with_count(self)

    def clone(self) -> "Solution":
        clone_solution = Solution(
            instance=self.instance,
            scheduling_string=list(self.scheduling_string),
            machine_assignment_string=list(self.machine_assignment_string),
            random_seed=self.random_seed,
            skip_validation=True,
        )
        if self.evaluated:
            clone_solution.makespan = self.makespan
            clone_solution.energy = self.energy
            clone_solution.agreement = self.agreement
            clone_solution.schedule = self.schedule
            clone_solution.evaluated = True
            clone_solution.objective_key = self.objective_key
            clone_solution.objective_vector = self.objective_vector
        return clone_solution

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return (
            f"Solution(jobs={self.instance.num_jobs}, evaluated={self.evaluated}, "
            f"makespan={self.makespan}, energy={self.energy}, agreement={self.agreement})"
        )
