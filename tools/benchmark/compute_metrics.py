"""Compute IGD, rho, and coverage metrics from raw run outputs."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (PROJECT_ROOT, SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.benchmark.result_utils import (
    ROUND_DECIMALS,
    aggregate_by_instance,
    aggregate_by_instance_and_seed,
    build_reference_fronts,
    deduplicate,
    dominates,
    load_run_outputs,
    pareto_filter,
    solutions_equal,
)

EPS = 1e-12


def _normalize(value: float, low: float, high: float) -> float:
    if high - low <= EPS:
        return 0.0
    return (value - low) / (high - low)


def _objective_vector(point: dict, bounds: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """双目标：makespan + energy"""
    mk = _normalize(point["makespan_c1"], *bounds["makespan_c1"])
    en = _normalize(point["energy_c1"], *bounds["energy_c1"])
    return mk, en


def compute_bounds(points: List[dict]) -> Dict[str, Tuple[float, float]]:
    if not points:
        return {
            "makespan_c1": (0.0, 1.0),
            "energy_c1": (0.0, 1.0),
        }
    return {
        "makespan_c1": (
            min(p["makespan_c1"] for p in points),
            max(p["makespan_c1"] for p in points),
        ),
        "energy_c1": (
            min(p["energy_c1"] for p in points),
            max(p["energy_c1"] for p in points),
        ),
    }


def igd(front: List[dict], reference: List[dict], bounds: Dict[str, Tuple[float, float]]) -> float:
    if not reference or not front:
        return math.inf
    ref_vectors = [_objective_vector(point, bounds) for point in reference]
    front_vectors = [_objective_vector(point, bounds) for point in front]
    total = 0.0
    for ref_vec in ref_vectors:
        best = min(math.dist(ref_vec, front_vec) for front_vec in front_vectors)
        total += best
    return total / len(ref_vectors)


def rho(front: List[dict], reference: List[dict]) -> float:
    if not reference:
        return 0.0
    count = 0
    for ref_point in reference:
        if any(solutions_equal(ref_point, point) for point in front):
            count += 1
    return count / len(reference)


def coverage(front_a: List[dict], front_b: List[dict]) -> float:
    if not front_b:
        return 0.0
    dominated = 0
    for point_b in front_b:
        if any(dominates(point_a, point_b) for point_a in front_a):
            dominated += 1
    return dominated / len(front_b)


def write_csv(path: Path, headers: List[str], rows: List[List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark metrics")
    parser.add_argument("--raw", default="experiments/results", help="Directory containing raw run outputs")
    parser.add_argument(
        "--summary",
        default="experiments/results/summary_per_run.csv",
        help="CSV file for per-run IGD and rho metrics",
    )
    parser.add_argument(
        "--coverage",
        default="experiments/results/coverage_per_instance.csv",
        help="CSV file for per-instance coverage metrics",
    )
    parser.add_argument(
        "--per-seed-reference",
        action="store_true",
        help="Reference fronts are built per seed (combine algorithms within the same seed) instead of per instance.",
    )
    args = parser.parse_args()

    raw_outputs = load_run_outputs(args.raw)
    if not raw_outputs:
        raise SystemExit(f"No raw outputs found in {args.raw}")
    if args.per_seed_reference:
        # Build per-seed reference fronts (combine algorithms within same seed)
        per_key_all_points: Dict[Tuple[str, str, int], List[dict]] = {}
        per_key_reference: Dict[Tuple[str, str, int], List[dict]] = {}
        for (algorithm, dataset_id, instance_id, seed), points in raw_outputs.items():
            key = (dataset_id, instance_id, seed)
            per_key_all_points.setdefault(key, []).extend(points)
            per_key_reference.setdefault(key, []).extend(points)
        reference_fronts = {k: pareto_filter(deduplicate(v)) for k, v in per_key_reference.items()}

        summary_rows: List[List] = []
        for (algorithm, dataset_id, instance_id, seed), front in raw_outputs.items():
            front = pareto_filter(deduplicate(front))
            ref_key = (dataset_id, instance_id, seed)
            reference = reference_fronts.get(ref_key, [])
            bounds = compute_bounds(reference)
            if reference:
                agg_bounds = compute_bounds(per_key_all_points.get(ref_key, []))
                mixed_bounds: Dict[str, Tuple[float, float]] = {}
                for key in ("makespan_c1", "energy_c1"):
                    lo, hi = bounds[key]
                    if hi - lo <= EPS:
                        lo2, hi2 = agg_bounds[key]
                        if hi2 - lo2 > EPS:
                            lo, hi = lo2, hi2
                    mixed_bounds[key] = (lo, hi)
                bounds = mixed_bounds
            summary_rows.append(
                [
                    algorithm,
                    dataset_id,
                    instance_id,
                    seed,
                    igd(front, reference, bounds),
                    rho(front, reference),
                ]
            )
        summary_rows = sorted(summary_rows, key=lambda row: (row[1], row[2], row[3], row[0]))
        write_csv(Path(args.summary), ["algorithm", "dataset", "instance_id", "seed", "igd", "rho"], summary_rows)

        aggregated = {
            key: {algo: pareto_filter(points) for algo, points in algos.items()}
            for key, algos in aggregate_by_instance_and_seed(raw_outputs).items()
        }
        coverage_rows: List[List] = []
        for (dataset_id, instance_id, seed), alg_fronts in aggregated.items():
            algo_names = list(alg_fronts.keys())
            for algo_a in algo_names:
                for algo_b in algo_names:
                    if algo_a == algo_b:
                        continue
                    coverage_rows.append(
                        [
                            dataset_id,
                            instance_id,
                            seed,
                            algo_a,
                            algo_b,
                            coverage(alg_fronts[algo_a], alg_fronts[algo_b]),
                        ]
                    )
        write_csv(Path(args.coverage), ["dataset", "instance_id", "seed", "algorithm_a", "algorithm_b", "coverage"], coverage_rows)
    else:
        reference_fronts = {k: pareto_filter(v) for k, v in build_reference_fronts(raw_outputs).items()}

        # Build per-instance aggregates across all algorithms to provide robust bounds.
        # This avoids degenerate IGD=0 when the reference front collapses (max==min) on some objectives.
        per_instance_all_points: Dict[Tuple[str, str], List[dict]] = {}
        for (algorithm, dataset_id, instance_id, seed), points in raw_outputs.items():
            key = (dataset_id, instance_id)
            per_instance_all_points.setdefault(key, []).extend(points)

        summary_rows: List[List] = []
        for (algorithm, dataset_id, instance_id, seed), front in raw_outputs.items():
            front = pareto_filter(deduplicate(front))
            reference = reference_fronts.get((dataset_id, instance_id), [])
            bounds = compute_bounds(reference)
            # If any dimension in the reference bounds is degenerate, fall back to
            # per-instance aggregate bounds for that dimension to preserve distance information.
            if reference:
                agg_bounds = compute_bounds(per_instance_all_points.get((dataset_id, instance_id), []))
                mixed_bounds: Dict[str, Tuple[float, float]] = {}
                for key in ("makespan_c1", "energy_c1"):
                    lo, hi = bounds[key]
                    if hi - lo <= EPS:
                        lo2, hi2 = agg_bounds[key]
                        # Only replace when aggregate provides a non-degenerate span
                        if hi2 - lo2 > EPS:
                            lo, hi = lo2, hi2
                    mixed_bounds[key] = (lo, hi)
                bounds = mixed_bounds
            summary_rows.append(
                [
                    algorithm,
                    dataset_id,
                    instance_id,
                    seed,
                    igd(front, reference, bounds),
                    rho(front, reference),
                ]
            )
        summary_rows = sorted(summary_rows, key=lambda row: (row[1], row[2], row[3], row[0]))
        write_csv(Path(args.summary), ["algorithm", "dataset", "instance_id", "seed", "igd", "rho"], summary_rows)

        aggregated = {
            key: {algo: pareto_filter(points) for algo, points in algos.items()}
            for key, algos in aggregate_by_instance(raw_outputs).items()
        }
        coverage_rows: List[List] = []
        for (dataset_id, instance_id), alg_fronts in aggregated.items():
            algo_names = list(alg_fronts.keys())
            for algo_a in algo_names:
                for algo_b in algo_names:
                    if algo_a == algo_b:
                        continue
                    coverage_rows.append(
                        [
                            dataset_id,
                            instance_id,
                            algo_a,
                            algo_b,
                            coverage(alg_fronts[algo_a], alg_fronts[algo_b]),
                        ]
                    )
        write_csv(Path(args.coverage), ["dataset", "instance_id", "algorithm_a", "algorithm_b", "coverage"], coverage_rows)


if __name__ == "__main__":
    main()
