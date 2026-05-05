"""Utilities for loading raw run outputs and constructing reference sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EPS = 1e-9
ROUND_DECIMALS = 4


def _point_key(point: dict) -> tuple:
    agreement_key = point.get("dissatisfaction_tuple", point.get("agreement_tuple", point["agreement"]))
    if isinstance(agreement_key, (list, tuple)):
        agreement_key = tuple(round(v, ROUND_DECIMALS) for v in agreement_key)
    else:
        agreement_key = round(agreement_key, ROUND_DECIMALS)
    return (
        tuple(round(v, ROUND_DECIMALS) for v in point["makespan"]),
        tuple(round(v, ROUND_DECIMALS) for v in point["energy"]),
        agreement_key,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _solution_from_dict(data: dict) -> dict:
    makespan = tuple(data["makespan"])
    energy = tuple(data["energy"])
    raw_dissatisfaction = data["dissatisfaction"] if "dissatisfaction" in data else data["agreement"]
    if isinstance(raw_dissatisfaction, (list, tuple)):
        dissatisfaction_tuple = tuple(raw_dissatisfaction)
        dissatisfaction = data.get("dissatisfaction_c1", data.get("agreement_c1"))
        if dissatisfaction is None:
            dissatisfaction = tfn_c1(dissatisfaction_tuple)
    else:
        dissatisfaction_tuple = None
        dissatisfaction = raw_dissatisfaction
    makespan_c1 = data.get("makespan_c1")
    if makespan_c1 is None:
        makespan_c1 = tfn_c1(makespan)
    energy_c1 = data.get("energy_c1")
    if energy_c1 is None:
        energy_c1 = tfn_c1(energy)
    return {
        "makespan": makespan,
        "energy": energy,
        "agreement": dissatisfaction,
        "agreement_tuple": dissatisfaction_tuple,
        "dissatisfaction": dissatisfaction,
        "dissatisfaction_tuple": dissatisfaction_tuple,
        "makespan_c1": makespan_c1,
        "energy_c1": energy_c1,
        "agreement_c1": dissatisfaction,
        "agreement_c2": data.get("dissatisfaction_c2", data.get("agreement_c2")),
        "agreement_c3": data.get("dissatisfaction_c3", data.get("agreement_c3")),
        "dissatisfaction_c1": dissatisfaction,
        "dissatisfaction_c2": data.get("dissatisfaction_c2", data.get("agreement_c2")),
        "dissatisfaction_c3": data.get("dissatisfaction_c3", data.get("agreement_c3")),
    }


def load_run_outputs(raw_root: str | Path) -> Dict[Tuple[str, str, str, int], List[dict]]:
    root = Path(raw_root)
    results: Dict[Tuple[str, str, str, int], List[dict]] = {}
    if not root.exists():
        return results
    for algo_dir in root.iterdir():
        if not algo_dir.is_dir():
            continue
        algorithm = algo_dir.name
        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_id = dataset_dir.name
            for instance_dir in dataset_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                instance_id = instance_dir.name
                for file in instance_dir.glob("seed_*.json"):
                    data = _load_json(file)
                    seed = int(data.get("algorithm_seed", file.stem.split("_")[-1]))
                    raw_solutions = data.get("solutions", [])
                    results[(algorithm, dataset_id, instance_id, seed)] = [
                        _solution_from_dict(sol) for sol in raw_solutions
                    ]
    return results


def load_metadata(raw_root: str | Path) -> Dict[Tuple[str, str, str, int], dict]:
    """
    Load metadata (without solution payload) for quick reference such as random factor.
    """
    root = Path(raw_root)
    meta: Dict[Tuple[str, str, str, int], dict] = {}
    if not root.exists():
        return meta
    for algo_dir in root.iterdir():
        if not algo_dir.is_dir():
            continue
        algorithm = algo_dir.name
        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_id = dataset_dir.name
            for instance_dir in dataset_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                instance_id = instance_dir.name
                for file in instance_dir.glob("seed_*.json"):
                    data = _load_json(file)
                    seed = int(data.get("algorithm_seed", file.stem.split("_")[-1]))
                    key = (algorithm, dataset_id, instance_id, seed)
                    random_factor_meta = data.get("random_factor_a")
                    job_factor_meta = data.get("job_random_factors_a")
                    if job_factor_meta is None and isinstance(random_factor_meta, list):
                        job_factor_meta = random_factor_meta
                    meta[key] = {
                        "random_factor_a": random_factor_meta,
                        "job_random_factors_a": job_factor_meta,
                    }
    return meta


def tfn_c1(values: Iterable[float]) -> float:
    l, m, r = values
    return (l + 2 * m + r) / 4.0


def dominates(a: dict, b: dict) -> bool:
    return (
        a["makespan_c1"] <= b["makespan_c1"] + EPS
        and a["energy_c1"] <= b["energy_c1"] + EPS
        and a["agreement"] <= b["agreement"] + EPS
        and (
            a["makespan_c1"] < b["makespan_c1"] - EPS
            or a["energy_c1"] < b["energy_c1"] - EPS
            or a["agreement"] < b["agreement"] - EPS
        )
    )


def solutions_equal(a: dict, b: dict) -> bool:
    return (
        abs(a["makespan_c1"] - b["makespan_c1"]) <= EPS
        and abs(a["energy_c1"] - b["energy_c1"]) <= EPS
        and abs(a["agreement"] - b["agreement"]) <= EPS
    )


def pareto_filter(points: List[dict]) -> List[dict]:
    non_dominated: List[dict] = []
    for i, point in enumerate(points):
        dominated_flag = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if dominates(other, point):
                dominated_flag = True
                break
        if not dominated_flag:
            non_dominated.append(point)
    return non_dominated


def build_reference_fronts(
    fronts: Dict[Tuple[str, str, str, int], List[dict]]
) -> Dict[Tuple[str, str], List[dict]]:
    combined: Dict[Tuple[str, str], List[dict]] = {}
    for (_, dataset_id, instance_id, _), solutions in fronts.items():
        key = (dataset_id, instance_id)
        combined.setdefault(key, []).extend(solutions)
    return {key: pareto_filter(deduplicate(points)) for key, points in combined.items()}


def aggregate_by_instance(
    fronts: Dict[Tuple[str, str, str, int], List[dict]]
) -> Dict[Tuple[str, str], Dict[str, List[dict]]]:
    aggregated: Dict[Tuple[str, str], Dict[str, List[dict]]] = {}
    for (algorithm, dataset_id, instance_id, _), solutions in fronts.items():
        key = (dataset_id, instance_id)
        aggregated.setdefault(key, {}).setdefault(algorithm, []).extend(solutions)
    for instance_dict in aggregated.values():
        for algorithm, points in instance_dict.items():
            instance_dict[algorithm] = deduplicate(points)
    return aggregated


def aggregate_by_instance_and_seed(
    fronts: Dict[Tuple[str, str, str, int], List[dict]]
) -> Dict[Tuple[str, str, int], Dict[str, List[dict]]]:
    aggregated: Dict[Tuple[str, str, int], Dict[str, List[dict]]] = {}
    for (algorithm, dataset_id, instance_id, seed), solutions in fronts.items():
        key = (dataset_id, instance_id, seed)
        aggregated.setdefault(key, {}).setdefault(algorithm, []).extend(solutions)
    for instance_dict in aggregated.values():
        for algorithm, points in instance_dict.items():
            instance_dict[algorithm] = deduplicate(points)
    return aggregated


def deduplicate(points: List[dict]) -> List[dict]:
    unique: List[dict] = []
    seen = set()
    for point in points:
        key = _point_key(point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return unique
