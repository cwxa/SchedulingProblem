"""
COA-V1 implementation.

Design goals:
  - Keep all COA-specific logic together under src/COA/.
  - Reuse the existing instance / solution / evaluation / archive stack.
  - Remain batch-run compatible with the existing JSON input/output pipeline.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import fbea.evaluation_counter as eval_counter
from fbea.archive import Archive
from fbea.constants import EPS
from fbea.evaluation_counter import evaluate_solutions_with_count, get_evaluation_count
from fbea.heuristics import (
    heuristic_1_machine_assignment,
    heuristic_2_machine_assignment,
    heuristic_3_scheduling_string,
    heuristic_4_scheduling_string,
    heuristic_idle_first_machine_assignment,
    random_machine_assignment,
    random_scheduling_string,
)
from fbea.instance_loader import Instance
from fbea.random_manager import get_rng, set_global_seed
from fbea.solution import Solution

ObjectiveName = str

PACK_NAMES = (
    "P_ms",
    "P_en",
    "P_ag",
    "P_ms_en",
    "P_ms_ag",
    "P_en_ag",
    "P_all",
    "P_rand",
)

PACK_OBJECTIVES: Dict[str, Tuple[ObjectiveName, ...]] = {
    "P_ms": ("ms",),
    "P_en": ("en",),
    "P_ag": ("ag",),
    "P_ms_en": ("ms", "en"),
    "P_ms_ag": ("ms", "ag"),
    "P_en_ag": ("en", "ag"),
    "P_all": ("ms", "en", "ag"),
    "P_rand": tuple(),
}

INIT_PROFILE_WEIGHTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "P_ms": (
        ("ms_load", 0.40),
        ("idle_load", 0.30),
        ("ms_count", 0.20),
        ("random_mix", 0.10),
    ),
    "P_en": (
        ("en_count", 0.50),
        ("en_rand", 0.30),
        ("fast_count", 0.20),
    ),
    "P_ag": (
        ("ms_due", 0.50),
        ("idle_due", 0.30),
        ("rand_due", 0.20),
    ),
    "P_ms_en": (
        ("ms_load", 0.25),
        ("idle_load", 0.25),
        ("en_count", 0.25),
        ("en_rand", 0.25),
    ),
    "P_ms_ag": (
        ("ms_load", 0.30),
        ("idle_due", 0.25),
        ("ms_due", 0.25),
        ("random_mix", 0.20),
    ),
    "P_en_ag": (
        ("en_count", 0.30),
        ("en_rand", 0.30),
        ("idle_due", 0.20),
        ("rand_due", 0.20),
    ),
    "P_all": (
        ("ms_load", 0.20),
        ("en_count", 0.20),
        ("ms_due", 0.20),
        ("idle_due", 0.20),
        ("random_mix", 0.20),
    ),
    "P_rand": (
        ("full_random", 0.60),
        ("ms_rand", 0.20),
        ("rand_due", 0.20),
    ),
}


@dataclass(slots=True)
class CultureModel:
    machine_choices: List[Optional[int]] = field(default_factory=list)
    position_targets: Dict[Tuple[int, int], float] = field(default_factory=dict)
    transitions: Dict[int, List[Tuple[int, float, int]]] = field(default_factory=dict)


@dataclass(slots=True)
class PackEntry:
    solution: Solution
    bio_age: int = 0
    sol_stagnation: int = 0
    origin_pack: str = ""
    is_immigrant: bool = False
    front_rank: int = 0
    crowding: float = 0.0
    redundancy: float = 0.0


@dataclass(slots=True)
class PackState:
    name: str
    objectives: Tuple[ObjectiveName, ...]
    size: int
    members: List[PackEntry] = field(default_factory=list)
    leader_set: List[PackEntry] = field(default_factory=list)
    elite_window: Deque[List[Solution]] = field(default_factory=lambda: deque(maxlen=4))
    culture: CultureModel = field(default_factory=CultureModel)
    pack_stagnation: int = 0
    last_best_signature: Optional[Tuple[float, ...]] = None
    immigrant_limit: int = 2
    leader_count: int = 2
    pup_count_per_gen: int = 2
    elite_per_generation: int = 4
    culture_machine_threshold: float = 0.60
    culture_transition_threshold: float = 0.25
    similarity_threshold: float = 0.97
    stagnation_threshold: int = 5

    @property
    def is_random_pack(self) -> bool:
        return self.name == "P_rand"


def coa_v1_main_algorithm(
    instance: Instance,
    total_population_size: int = 100,
    crossover_probability: float = 0.7,
    mutation_probability: float = 0.2,
    random_seed: Optional[int] = None,
    progress_interval: int = 0,
):
    if random_seed is not None:
        set_global_seed(random_seed)

    pack_sizes = _split_population_size(total_population_size, len(PACK_NAMES))
    packs = [_build_pack(name, size) for name, size in zip(PACK_NAMES, pack_sizes)]
    archive = Archive()

    for pack in packs:
        pack.members = _initialise_pack_members(instance, pack)
        for entry in pack.members:
            archive.add_solution(entry.solution, None)
        _rank_pack_members(pack)
        _refresh_leaders(pack)
        pack.last_best_signature = _current_best_signature(pack)

    warmup_evaluations = max(int(eval_counter.MAX_EVALUATIONS * 0.15), total_population_size)
    generation = 0
    focused_packs = [pack for pack in packs if not pack.is_random_pack]
    random_pack = next(pack for pack in packs if pack.is_random_pack)

    while get_evaluation_count() < eval_counter.MAX_EVALUATIONS:
        generation += 1

        for pack in focused_packs:
            if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
                break
            _increment_member_ages(pack)
            _rank_pack_members(pack)
            _refresh_leaders(pack)

            culture_enabled = get_evaluation_count() >= warmup_evaluations
            if culture_enabled:
                _update_elite_window(pack)
                _update_culture(instance, pack)
            else:
                pack.culture = CultureModel(machine_choices=[None] * instance.total_operations)

            before_signature = _current_best_signature(pack)
            self_improved = _self_update_pack(
                instance,
                pack,
                archive,
                culture_enabled=culture_enabled,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            _rank_pack_members(pack)
            _refresh_leaders(pack)

            pups = _generate_pack_pups(
                instance,
                pack,
                culture_enabled=culture_enabled,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            if pups:
                evaluate_solutions_with_count([solution for solution, _ in pups])
                for solution, metadata in pups:
                    archive.add_solution(solution, None)
                    _admission_replace(pack, solution, metadata)

            _rank_pack_members(pack)
            _refresh_leaders(pack)
            after_signature = _current_best_signature(pack)
            if self_improved or _is_signature_better(after_signature, before_signature):
                pack.pack_stagnation = 0
            else:
                pack.pack_stagnation += 1
            pack.last_best_signature = after_signature

        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break

        _update_random_pack(instance, random_pack, archive)

        if generation % 3 == 0:
            for pack in focused_packs:
                immigrant = _select_immigrant_from_random_pack(random_pack, pack)
                if immigrant is not None:
                    archive.add_solution(immigrant, None)
                    _admission_replace(
                        pack,
                        immigrant,
                        {"origin_pack": random_pack.name, "is_immigrant": True},
                    )

        if progress_interval and generation % progress_interval == 0:
            print(
                f"[COA_V1] gen={generation} evals={get_evaluation_count()} "
                f"archive={len(archive.solutions)}"
            )

    return archive


def _build_pack(name: str, size: int) -> PackState:
    objectives = PACK_OBJECTIVES[name]
    leader_count = 2 if len(objectives) <= 1 else 3
    return PackState(
        name=name,
        objectives=objectives,
        size=size,
        leader_count=leader_count,
        pup_count_per_gen=max(1, min(2, size // 4 if size > 3 else 1)),
        immigrant_limit=max(1, min(2, size // 4 if size > 3 else 1)),
        elite_per_generation=min(4, size),
    )


def _split_population_size(total_population_size: int, num_packs: int) -> List[int]:
    base = max(1, total_population_size // num_packs)
    sizes = [base] * num_packs
    remainder = max(total_population_size - base * num_packs, 0)
    for idx in range(remainder):
        sizes[idx % num_packs] += 1
    return sizes


def _initialise_pack_members(instance: Instance, pack: PackState) -> List[PackEntry]:
    target_candidates = max(pack.size * 5, pack.size + 4)
    candidates: List[Solution] = []
    seen = set()
    while len(candidates) < target_candidates:
        solution = _generate_seed_solution(instance, pack.name)
        key = _solution_code_key(solution)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(solution)
    evaluate_solutions_with_count(candidates)
    ranked = _rank_solutions_for_pack(candidates, pack.objectives)
    selected: List[Solution] = []
    for solution in ranked:
        if len(selected) >= pack.size:
            break
        if all(_solution_similarity(solution, existing) < pack.similarity_threshold for existing in selected):
            selected.append(solution)
    for solution in ranked:
        if len(selected) >= pack.size:
            break
        if all(_solution_code_key(solution) != _solution_code_key(existing) for existing in selected):
            selected.append(solution)
    return [
        PackEntry(solution=solution, origin_pack=pack.name, is_immigrant=False)
        for solution in selected[: pack.size]
    ]


def _generate_seed_solution(instance: Instance, pack_name: str) -> Solution:
    rng = get_rng()
    profile = INIT_PROFILE_WEIGHTS[pack_name]
    recipe_names = [name for name, _ in profile]
    weights = [weight for _, weight in profile]
    recipe = rng.choices(recipe_names, weights=weights, k=1)[0]
    machine_assignment, scheduling = _build_strings_from_recipe(instance, recipe)
    return Solution(
        instance=instance,
        scheduling_string=scheduling,
        machine_assignment_string=machine_assignment,
        skip_validation=True,
    )


def _build_strings_from_recipe(instance: Instance, recipe: str) -> Tuple[List[int], List[int]]:
    if recipe == "ms_load":
        machines = heuristic_1_machine_assignment(instance)
        scheduling = heuristic_3_scheduling_string(instance, machines)
        return machines, scheduling
    if recipe == "idle_load":
        machines = heuristic_idle_first_machine_assignment(instance)
        scheduling = heuristic_3_scheduling_string(instance, machines)
        return machines, scheduling
    if recipe == "ms_count":
        machines = heuristic_1_machine_assignment(instance)
        scheduling = heuristic_4_scheduling_string(instance)
        return machines, scheduling
    if recipe == "en_count":
        machines = _heuristic_energy_machine_assignment(instance)
        scheduling = heuristic_4_scheduling_string(instance)
        return machines, scheduling
    if recipe == "en_rand":
        machines = _heuristic_energy_machine_assignment(instance)
        scheduling = random_scheduling_string(instance)
        return machines, scheduling
    if recipe == "fast_count":
        machines = heuristic_2_machine_assignment(instance)
        scheduling = heuristic_4_scheduling_string(instance)
        return machines, scheduling
    if recipe == "ms_due":
        machines = heuristic_1_machine_assignment(instance)
        scheduling = _heuristic_due_scheduling_string(instance, machines)
        return machines, scheduling
    if recipe == "idle_due":
        machines = heuristic_idle_first_machine_assignment(instance)
        scheduling = _heuristic_due_scheduling_string(instance, machines)
        return machines, scheduling
    if recipe == "rand_due":
        machines = random_machine_assignment(instance)
        scheduling = _heuristic_due_scheduling_string(instance, machines)
        return machines, scheduling
    if recipe == "ms_rand":
        machines = heuristic_1_machine_assignment(instance)
        scheduling = random_scheduling_string(instance)
        return machines, scheduling
    if recipe == "random_mix":
        machines = random_machine_assignment(instance)
        scheduling = heuristic_4_scheduling_string(instance)
        return machines, scheduling
    if recipe == "full_random":
        return random_machine_assignment(instance), random_scheduling_string(instance)
    return random_machine_assignment(instance), random_scheduling_string(instance)


def _heuristic_energy_machine_assignment(instance: Instance) -> List[int]:
    rng = get_rng()
    assignment: List[int] = [0] * instance.total_operations
    for job_id, operation_idx in instance.linearized_operations:
        operation = instance.jobs[job_id - 1].operations[operation_idx]
        best_score: Optional[float] = None
        best_machines: List[int] = []
        for option in operation.options:
            machine = instance.machines[option.machine_id]
            score = option.processing_time.m * machine.processing_power
            if best_score is None or score < best_score - EPS:
                best_score = score
                best_machines = [option.machine_id]
            elif abs(score - best_score) <= EPS:
                best_machines.append(option.machine_id)
        assignment[instance.get_operation_position(job_id, operation_idx)] = rng.choice(best_machines)
    return assignment


def _heuristic_due_scheduling_string(instance: Instance, machine_assignment: List[int]) -> List[int]:
    rng = get_rng()
    remaining_ops = {job_id: job.num_operations for job_id, job in enumerate(instance.jobs, start=1)}
    next_op = {job_id: 0 for job_id in range(1, instance.num_jobs + 1)}
    total_operations = instance.total_operations
    schedule: List[int] = []
    remaining_processing: Dict[int, List[float]] = {}

    for job_id, job in enumerate(instance.jobs, start=1):
        values = [0.0]
        tail_sum = 0.0
        per_op: List[float] = []
        for operation_idx in reversed(range(job.num_operations)):
            pos = instance.get_operation_position(job_id, operation_idx)
            machine_id = machine_assignment[pos]
            processing_map = instance.operation_processing_map[pos]
            tail_sum += processing_map[machine_id].m
            per_op.append(tail_sum)
        values.extend(reversed(per_op))
        remaining_processing[job_id] = values

    for _ in range(total_operations):
        candidates: List[int] = []
        best_slack: Optional[float] = None
        best_remaining: Optional[float] = None
        for job_id in range(1, instance.num_jobs + 1):
            if remaining_ops[job_id] <= 0:
                continue
            op_idx = next_op[job_id]
            due_2 = instance.jobs[job_id - 1].due_date[1]
            remain = remaining_processing[job_id][op_idx + 1]
            slack = due_2 - remain
            if best_slack is None or slack < best_slack - EPS:
                candidates = [job_id]
                best_slack = slack
                best_remaining = remain
            elif abs(slack - best_slack) <= EPS:
                if best_remaining is None or remain > best_remaining + EPS:
                    candidates = [job_id]
                    best_remaining = remain
                elif abs(remain - best_remaining) <= EPS:
                    candidates.append(job_id)
        chosen = rng.choice(candidates)
        schedule.append(chosen)
        remaining_ops[chosen] -= 1
        next_op[chosen] += 1
    return schedule


def _increment_member_ages(pack: PackState) -> None:
    for entry in pack.members:
        entry.bio_age += 1


def _self_update_pack(
    instance: Instance,
    pack: PackState,
    archive: Archive,
    *,
    culture_enabled: bool,
    crossover_probability: float,
    mutation_probability: float,
) -> bool:
    improved_any = False
    pending: List[Tuple[PackEntry, Solution]] = []
    current_members = list(pack.members)
    for entry in current_members:
        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break
        leader = _select_leader_for_base(pack, entry)
        child = _generate_child_solution(
            instance,
            base=entry.solution,
            leader=leader.solution if leader is not None else None,
            pack=pack,
            culture_enabled=culture_enabled,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )
        pending.append((entry, child))

    if not pending:
        return False

    remaining_budget = max(eval_counter.MAX_EVALUATIONS - get_evaluation_count(), 0)
    if remaining_budget <= 0:
        return False
    if len(pending) > remaining_budget:
        pending = pending[:remaining_budget]
    evaluate_solutions_with_count([child for _, child in pending])
    for entry, child in pending:
        archive.add_solution(child, None)
        if _pairwise_pack_compare(child, entry.solution, pack.objectives) < 0:
            entry.solution = child
            entry.bio_age = 0
            entry.sol_stagnation = 0
            entry.origin_pack = pack.name
            entry.is_immigrant = False
            improved_any = True
        else:
            entry.sol_stagnation += 1
    return improved_any


def _generate_pack_pups(
    instance: Instance,
    pack: PackState,
    *,
    culture_enabled: bool,
    crossover_probability: float,
    mutation_probability: float,
) -> List[Tuple[Solution, dict]]:
    pups: List[Tuple[Solution, dict]] = []
    if not pack.members:
        return pups
    remaining_budget = max(eval_counter.MAX_EVALUATIONS - get_evaluation_count(), 0)
    if remaining_budget <= 0:
        return pups
    for _ in range(min(pack.pup_count_per_gen, remaining_budget)):
        if get_evaluation_count() >= eval_counter.MAX_EVALUATIONS:
            break
        base = _tournament_select(pack, allow_leader=False)
        leader = _select_random_leader(pack)
        child = _generate_child_solution(
            instance,
            base=base.solution,
            leader=leader.solution if leader is not None else None,
            pack=pack,
            culture_enabled=culture_enabled,
            crossover_probability=min(1.0, crossover_probability + 0.1),
            mutation_probability=mutation_probability,
        )
        pups.append((child, {"origin_pack": pack.name, "is_immigrant": False}))
    return pups


def _generate_child_solution(
    instance: Instance,
    *,
    base: Solution,
    leader: Optional[Solution],
    pack: PackState,
    culture_enabled: bool,
    crossover_probability: float,
    mutation_probability: float,
) -> Solution:
    rng = get_rng()
    machine_assignment = list(base.machine_assignment_string)
    scheduling = list(base.scheduling_string)

    if leader is not None and rng.random() < crossover_probability:
        machine_positions = list(range(instance.total_operations))
        rng.shuffle(machine_positions)
        num_machine_updates = max(1, int(instance.total_operations * 0.30))
        for pos in machine_positions[:num_machine_updates]:
            source_roll = rng.random()
            chosen_machine: Optional[int] = None
            if source_roll < 0.40:
                chosen_machine = leader.machine_assignment_string[pos]
            elif source_roll < 0.80 and culture_enabled:
                chosen_machine = pack.culture.machine_choices[pos]
            if chosen_machine is None:
                valid = instance.operation_machine_options[pos]
                chosen_machine = rng.choice(valid)
            if chosen_machine in instance.operation_machine_options[pos]:
                machine_assignment[pos] = chosen_machine

        _apply_leader_schedule_move(scheduling, leader.scheduling_string)

    if culture_enabled:
        _apply_position_target_moves(scheduling, pack.culture.position_targets, move_count=2)
        _apply_transition_repairs(scheduling, pack.culture.transitions, repair_count=2)

    random_moves = 1
    if rng.random() < mutation_probability:
        random_moves += 1
    for _ in range(random_moves):
        _apply_random_schedule_edit(scheduling)
        if rng.random() < mutation_probability:
            _apply_random_machine_edit(instance, machine_assignment)

    return Solution(
        instance=instance,
        scheduling_string=scheduling,
        machine_assignment_string=machine_assignment,
        skip_validation=True,
    )


def _apply_leader_schedule_move(schedule: List[int], leader_schedule: Sequence[int]) -> None:
    if not schedule or not leader_schedule:
        return
    rng = get_rng()
    prefix_len = max(1, len(schedule) // 5)
    leader_prefix = list(leader_schedule[:prefix_len])
    leader_counts = Counter(leader_prefix)
    base_counts = Counter(schedule[:prefix_len])
    candidate_jobs = [job_id for job_id, count in leader_counts.items() if base_counts.get(job_id, 0) < count]
    if not candidate_jobs:
        candidate_jobs = list(dict.fromkeys(leader_prefix))
    if not candidate_jobs:
        return
    job_id = rng.choice(candidate_jobs)
    positions = [idx for idx, value in enumerate(schedule) if value == job_id]
    if not positions:
        return
    current_pos = positions[0]
    target_pos = rng.randint(0, max(prefix_len - 1, 0))
    _move_item(schedule, current_pos, target_pos)


def _apply_position_target_moves(
    schedule: List[int],
    position_targets: Dict[Tuple[int, int], float],
    *,
    move_count: int,
) -> None:
    if not schedule or not position_targets:
        return
    for _ in range(move_count):
        occurrences = _schedule_occurrence_positions(schedule)
        worst_key: Optional[Tuple[int, int]] = None
        worst_gap = 0.0
        for key, target in position_targets.items():
            positions = occurrences.get(key[0])
            if positions is None or len(positions) < key[1]:
                continue
            current = positions[key[1] - 1]
            gap = abs(current - target)
            if gap > worst_gap + EPS:
                worst_gap = gap
                worst_key = key
        if worst_key is None:
            return
        positions = occurrences[worst_key[0]]
        current_pos = positions[worst_key[1] - 1]
        target_pos = int(round(position_targets[worst_key]))
        max_step = 3
        if target_pos < current_pos:
            target_pos = max(target_pos, current_pos - max_step)
        else:
            target_pos = min(target_pos, current_pos + max_step)
        _move_item(schedule, current_pos, target_pos)


def _apply_transition_repairs(
    schedule: List[int],
    transitions: Dict[int, List[Tuple[int, float, int]]],
    *,
    repair_count: int,
) -> None:
    if not schedule or not transitions:
        return
    rng = get_rng()
    source_jobs = list(transitions.keys())
    if not source_jobs:
        return
    for _ in range(repair_count):
        a = rng.choice(source_jobs)
        options = transitions.get(a, [])
        strong = [item for item in options if item[1] >= 0.25 and item[2] >= 2]
        if not strong:
            continue
        b, _, _ = strong[0]
        for idx in range(len(schedule) - 1):
            if schedule[idx] != a:
                continue
            if schedule[idx + 1] == b:
                break
            try:
                next_b = schedule.index(b, idx + 1)
            except ValueError:
                break
            _move_item(schedule, next_b, idx + 1)
            break


def _apply_random_schedule_edit(schedule: List[int]) -> None:
    if len(schedule) < 2:
        return
    rng = get_rng()
    if rng.random() < 0.5:
        i, j = sorted(rng.sample(range(len(schedule)), 2))
        _move_item(schedule, j, i)
    else:
        i, j = rng.sample(range(len(schedule)), 2)
        schedule[i], schedule[j] = schedule[j], schedule[i]


def _apply_random_machine_edit(instance: Instance, machine_assignment: List[int]) -> None:
    if not machine_assignment:
        return
    rng = get_rng()
    pos = rng.randrange(len(machine_assignment))
    valid = instance.operation_machine_options[pos]
    machine_assignment[pos] = rng.choice(valid)


def _move_item(sequence: List[int], current_index: int, target_index: int) -> None:
    if current_index == target_index:
        return
    target_index = max(0, min(target_index, len(sequence) - 1))
    value = sequence.pop(current_index)
    sequence.insert(target_index, value)


def _rank_pack_members(pack: PackState) -> None:
    for entry in pack.members:
        entry.solution.evaluate()
    if pack.is_random_pack:
        for entry in pack.members:
            entry.front_rank = 0
            entry.crowding = 0.0
        return
    ranked_entries = _rank_entries_for_pack(pack.members, pack.objectives)
    pack.members[:] = ranked_entries


def _rank_entries_for_pack(entries: Sequence[PackEntry], objectives: Tuple[ObjectiveName, ...]) -> List[PackEntry]:
    entries = list(entries)
    if len(objectives) <= 1:
        objective = objectives[0]
        entries.sort(key=lambda item: (_objective_sort_value(item.solution, objective), item.bio_age))
        for index, entry in enumerate(entries):
            entry.front_rank = index
            entry.crowding = 0.0
        return entries

    fronts = _subspace_non_dominated_sort(entries, objectives)
    ranked: List[PackEntry] = []
    for front_rank, front in enumerate(fronts):
        crowding = _subspace_crowding_distance(front, objectives)
        front.sort(
            key=lambda item: (
                -crowding.get(id(item), 0.0),
                _objective_signature(item.solution, objectives),
            )
        )
        for entry in front:
            entry.front_rank = front_rank
            entry.crowding = crowding.get(id(entry), 0.0)
        ranked.extend(front)
    return ranked


def _rank_solutions_for_pack(solutions: Sequence[Solution], objectives: Tuple[ObjectiveName, ...]) -> List[Solution]:
    if not objectives:
        return list(solutions)
    entries = [PackEntry(solution=solution) for solution in solutions]
    ranked_entries = _rank_entries_for_pack(entries, objectives)
    return [entry.solution for entry in ranked_entries]


def _refresh_redundancy(entries: Sequence[PackEntry]) -> None:
    for entry in entries:
        entry.redundancy = 0.0
    size = len(entries)
    if size <= 1:
        return
    for i in range(size):
        sol_i = entries[i].solution
        for j in range(i + 1, size):
            similarity = _solution_similarity(sol_i, entries[j].solution)
            if similarity > entries[i].redundancy:
                entries[i].redundancy = similarity
            if similarity > entries[j].redundancy:
                entries[j].redundancy = similarity


def _refresh_leaders(pack: PackState) -> None:
    if not pack.members or pack.is_random_pack:
        pack.leader_set = []
        return
    pack.leader_set = pack.members[: min(pack.leader_count, len(pack.members))]


def _update_elite_window(pack: PackState) -> None:
    eligible = [
        entry.solution.clone()
        for entry in pack.members
        if entry.bio_age >= 2 and entry.sol_stagnation <= 6
    ]
    if not eligible:
        eligible = [entry.solution.clone() for entry in pack.members[: pack.elite_per_generation]]
    pack.elite_window.append(eligible[: pack.elite_per_generation])


def _update_culture(instance: Instance, pack: PackState) -> None:
    machine_choices: List[Optional[int]] = [None] * instance.total_operations
    position_buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    transition_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    unique_solutions: List[Solution] = []
    seen = set()
    for generation_elite in pack.elite_window:
        for solution in generation_elite:
            key = _solution_code_key(solution)
            if key in seen:
                continue
            seen.add(key)
            unique_solutions.append(solution)

    if not unique_solutions:
        pack.culture = CultureModel(machine_choices=machine_choices)
        return

    for pos in range(instance.total_operations):
        counts = Counter(solution.machine_assignment_string[pos] for solution in unique_solutions)
        machine, freq = counts.most_common(1)[0]
        if freq / len(unique_solutions) >= pack.culture_machine_threshold:
            machine_choices[pos] = machine

    for solution in unique_solutions:
        occurrences = _schedule_occurrence_positions(solution.scheduling_string)
        for job_id, positions in occurrences.items():
            for occ_idx, pos in enumerate(positions, start=1):
                position_buckets[(job_id, occ_idx)].append(pos)
        for idx in range(len(solution.scheduling_string) - 1):
            a = solution.scheduling_string[idx]
            b = solution.scheduling_string[idx + 1]
            transition_counts[a][b] += 1

    position_targets = {key: float(median(values)) for key, values in position_buckets.items() if values}
    transitions: Dict[int, List[Tuple[int, float, int]]] = {}
    for a, counter in transition_counts.items():
        total = sum(counter.values())
        ranked = []
        for b, count in counter.most_common():
            ratio = count / total if total else 0.0
            if ratio + EPS >= pack.culture_transition_threshold:
                ranked.append((b, ratio, count))
        if ranked:
            transitions[a] = ranked
    pack.culture = CultureModel(
        machine_choices=machine_choices,
        position_targets=position_targets,
        transitions=transitions,
    )


def _subspace_non_dominated_sort(
    entries: Sequence[PackEntry],
    objectives: Tuple[ObjectiveName, ...],
) -> List[List[PackEntry]]:
    size = len(entries)
    domination_counts = [0] * size
    dominated: List[List[int]] = [[] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            if _dominates(entries[i].solution, entries[j].solution, objectives):
                dominated[i].append(j)
                domination_counts[j] += 1
            elif _dominates(entries[j].solution, entries[i].solution, objectives):
                dominated[j].append(i)
                domination_counts[i] += 1
    current = [idx for idx, count in enumerate(domination_counts) if count == 0]
    fronts: List[List[PackEntry]] = []
    while current:
        front = [entries[idx] for idx in current]
        fronts.append(front)
        next_front: List[int] = []
        for idx in current:
            for dominated_idx in dominated[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    next_front.append(dominated_idx)
        current = next_front
    return fronts


def _subspace_crowding_distance(
    entries: Sequence[PackEntry],
    objectives: Tuple[ObjectiveName, ...],
) -> Dict[int, float]:
    if not entries:
        return {}
    size = len(entries)
    distances = [0.0] * size
    entry_ids = [id(entry) for entry in entries]

    for objective in objectives:
        reverse = objective == "ag"
        values = [_objective_raw_value(entry.solution, objective) for entry in entries]
        order = sorted(range(size), key=values.__getitem__, reverse=reverse)
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        min_value = values[order[-1]] if reverse else values[order[0]]
        max_value = values[order[0]] if reverse else values[order[-1]]
        if abs(max_value - min_value) <= EPS:
            continue
        denom = max_value - min_value
        for idx in range(1, size - 1):
            current = order[idx]
            prev_value = values[order[idx - 1]]
            next_value = values[order[idx + 1]]
            distances[current] += abs(next_value - prev_value) / denom
    return {entry_ids[idx]: distances[idx] for idx in range(size)}


def _dominates(sol_a: Solution, sol_b: Solution, objectives: Tuple[ObjectiveName, ...]) -> bool:
    better_or_equal = True
    strictly_better = False
    for objective in objectives:
        a = _objective_raw_value(sol_a, objective)
        b = _objective_raw_value(sol_b, objective)
        if objective == "ag":
            if a < b - EPS:
                better_or_equal = False
                break
            if a > b + EPS:
                strictly_better = True
        else:
            if a > b + EPS:
                better_or_equal = False
                break
            if a < b - EPS:
                strictly_better = True
    return better_or_equal and strictly_better


def _pairwise_pack_compare(sol_a: Solution, sol_b: Solution, objectives: Tuple[ObjectiveName, ...]) -> int:
    if not objectives:
        return 0
    if len(objectives) > 1:
        if _dominates(sol_a, sol_b, objectives):
            return -1
        if _dominates(sol_b, sol_a, objectives):
            return 1
    sig_a = _objective_signature(sol_a, objectives)
    sig_b = _objective_signature(sol_b, objectives)
    if sig_a < sig_b:
        return -1
    if sig_a > sig_b:
        return 1
    return 0


def _objective_signature(sol: Solution, objectives: Tuple[ObjectiveName, ...]) -> Tuple[float, ...]:
    return tuple(_objective_sort_value(sol, objective) for objective in objectives)


def _objective_sort_value(sol: Solution, objective: ObjectiveName) -> float:
    value = _objective_raw_value(sol, objective)
    return -value if objective == "ag" else value


def _objective_raw_value(sol: Solution, objective: ObjectiveName) -> float:
    sol.evaluate()
    if objective == "ms":
        return sol.makespan._c1()
    if objective == "en":
        return sol.energy._c1()
    if objective == "ag":
        return sol.agreement._c1()
    raise ValueError(f"Unknown objective: {objective}")


def _solution_code_key(solution: Solution) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return tuple(solution.scheduling_string), tuple(solution.machine_assignment_string)


def _solution_similarity(sol_a: Solution, sol_b: Solution) -> float:
    schedule_a = sol_a.scheduling_string
    schedule_b = sol_b.scheduling_string
    machine_a = sol_a.machine_assignment_string
    machine_b = sol_b.machine_assignment_string
    if not schedule_a or not schedule_b or not machine_a or not machine_b:
        return 0.0
    schedule_len = len(schedule_a)
    machine_len = len(machine_a)
    schedule_hits = 0
    for idx in range(schedule_len):
        if schedule_a[idx] == schedule_b[idx]:
            schedule_hits += 1
    machine_hits = 0
    for idx in range(machine_len):
        if machine_a[idx] == machine_b[idx]:
            machine_hits += 1
    schedule_match = schedule_hits / max(schedule_len, 1)
    machine_match = machine_hits / max(machine_len, 1)
    return 0.5 * (schedule_match + machine_match)


def _schedule_occurrence_positions(schedule: Sequence[int]) -> Dict[int, List[int]]:
    positions: Dict[int, List[int]] = defaultdict(list)
    for idx, job_id in enumerate(schedule):
        positions[job_id].append(idx)
    return positions


def _current_best_signature(pack: PackState) -> Optional[Tuple[float, ...]]:
    if not pack.members or pack.is_random_pack:
        return None
    return _objective_signature(pack.members[0].solution, pack.objectives)


def _is_signature_better(
    current: Optional[Tuple[float, ...]],
    previous: Optional[Tuple[float, ...]],
) -> bool:
    if current is None:
        return False
    if previous is None:
        return True
    return current < previous


def _select_leader_for_base(pack: PackState, base_entry: PackEntry) -> Optional[PackEntry]:
    if not pack.leader_set:
        return None
    non_self = [entry for entry in pack.leader_set if entry is not base_entry]
    if not non_self:
        return pack.leader_set[0]
    return get_rng().choice(non_self)


def _select_random_leader(pack: PackState) -> Optional[PackEntry]:
    if not pack.leader_set:
        return None
    return get_rng().choice(pack.leader_set)


def _tournament_select(pack: PackState, *, allow_leader: bool) -> PackEntry:
    rng = get_rng()
    pool = pack.members if allow_leader else [entry for entry in pack.members if entry not in pack.leader_set]
    if not pool:
        pool = pack.members
    competitors = rng.sample(pool, k=min(2, len(pool)))
    competitors.sort(key=lambda item: (item.front_rank, -item.crowding, item.sol_stagnation))
    return competitors[0]


def _admission_replace(pack: PackState, solution: Solution, metadata: dict) -> bool:
    if any(_solution_code_key(solution) == _solution_code_key(entry.solution) for entry in pack.members):
        return False
    if any(_solution_similarity(solution, entry.solution) >= pack.similarity_threshold for entry in pack.members):
        return False

    newcomer = PackEntry(
        solution=solution,
        bio_age=0,
        sol_stagnation=0,
        origin_pack=metadata.get("origin_pack", pack.name),
        is_immigrant=bool(metadata.get("is_immigrant", False)),
    )

    if len(pack.members) < pack.size:
        pack.members.append(newcomer)
        return True

    better_than = [
        entry
        for entry in pack.members
        if entry not in pack.leader_set and _pairwise_pack_compare(solution, entry.solution, pack.objectives) < 0
    ]
    if not better_than:
        return False

    if newcomer.is_immigrant:
        immigrant_count = sum(1 for entry in pack.members if entry.is_immigrant)
        if immigrant_count >= pack.immigrant_limit:
            better_than = [entry for entry in better_than if entry.is_immigrant]
            if not better_than:
                return False

    _refresh_redundancy(pack.members)
    candidate_pool = list(better_than)
    if len(candidate_pool) < max(1, pack.size // 4):
        extras = [
            entry
            for entry in pack.members
            if entry not in pack.leader_set and entry not in candidate_pool
        ]
        extras.sort(
            key=lambda item: (item.front_rank, item.redundancy, item.sol_stagnation, item.bio_age),
            reverse=True,
        )
        needed = max(1, pack.size // 4) - len(candidate_pool)
        candidate_pool.extend(extras[:needed])

    if not candidate_pool:
        return False

    victim = max(
        candidate_pool,
        key=lambda item: (
            item.redundancy * 0.5 + item.sol_stagnation * 0.3 + item.bio_age * 0.2,
            item.front_rank,
        ),
    )
    victim_idx = pack.members.index(victim)
    pack.members[victim_idx] = newcomer
    return True


def _update_random_pack(instance: Instance, pack: PackState, archive: Archive) -> None:
    _increment_member_ages(pack)
    _rank_pack_members(pack)
    _refresh_redundancy(pack.members)
    victim_count = max(1, len(pack.members) // 4)
    removable = sorted(
        pack.members,
        key=lambda entry: (entry.bio_age + entry.sol_stagnation, entry.redundancy),
        reverse=True,
    )
    survivors = [entry for entry in pack.members if entry not in removable[:victim_count]]
    pack.members = survivors
    newcomers: List[Solution] = []
    seen = {_solution_code_key(entry.solution) for entry in pack.members}
    remaining_budget = max(eval_counter.MAX_EVALUATIONS - get_evaluation_count(), 0)
    while (
        len(pack.members) + len(newcomers) < pack.size
        and len(newcomers) < remaining_budget
        and get_evaluation_count() < eval_counter.MAX_EVALUATIONS
    ):
        solution = _generate_seed_solution(instance, pack.name)
        key = _solution_code_key(solution)
        if key in seen:
            continue
        seen.add(key)
        newcomers.append(solution)
    if newcomers:
        evaluate_solutions_with_count(newcomers)
        for solution in newcomers:
            archive.add_solution(solution, None)
            pack.members.append(
                PackEntry(solution=solution, bio_age=0, sol_stagnation=0, origin_pack=pack.name)
            )
    _rank_pack_members(pack)


def _select_immigrant_from_random_pack(random_pack: PackState, target_pack: PackState) -> Optional[Solution]:
    if not random_pack.members:
        return None
    ranked = sorted(
        random_pack.members,
        key=lambda entry: _objective_signature(entry.solution, target_pack.objectives),
    )
    for entry in ranked:
        if any(_solution_code_key(entry.solution) == _solution_code_key(existing.solution) for existing in target_pack.members):
            continue
        if any(_solution_similarity(entry.solution, existing.solution) >= target_pack.similarity_threshold for existing in target_pack.members):
            continue
        return entry.solution.clone()
    return None
