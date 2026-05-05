from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MK_FILTER = [f"mk{i:02d}" for i in range(1, 11)]
ALGO_KEY = "fbea_composable"
ALGO_LABEL = "FBEA_IDLE25_INIT_DEDUP_GAP50_Q_LS_2STEP_MS_EGO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run idle25 + init_dedup + gap50 + q_ls_2step + ms_ego on prepared mk01-mk10 JSONs."
    )
    parser.add_argument(
        "--prepared-instances-root",
        default="experiments/prepared_instances",
        help="Root directory containing prepared mkXX JSON instances.",
    )
    parser.add_argument(
        "--results-root",
        default="experiments/results",
        help="Root directory for algorithm results.",
    )
    parser.add_argument(
        "--mk-filter",
        nargs="+",
        default=DEFAULT_MK_FILTER,
        help="Datasets to run, default: mk01 ... mk10.",
    )
    parser.add_argument(
        "--instance-stems",
        nargs="+",
        default=None,
        help="Optional instance stem filter, e.g. seed_101 seed_102.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[10000, 50000, 150000],
        help="Evaluation budgets to run.",
    )
    parser.add_argument("--runs", type=int, default=20, help="Runs per instance-budget pair.")
    parser.add_argument("--total-pop", type=int, default=100, help="Total population size.")
    parser.add_argument(
        "--ms-ego-probability",
        type=float,
        default=0.2,
        help="Per-offspring probability of MS-EGO-lite.",
    )
    parser.add_argument(
        "--ms-ego-random-fill-probability",
        type=float,
        default=0.2,
        help="Probability of random donor fill at replaced MS loci.",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Also compute per-run metrics after each launcher call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only.",
    )
    return parser.parse_args()


def iter_instance_files(prepared_root: Path, mk_filter: set[str], instance_stems: set[str] | None):
    for mk_dir in sorted(path for path in prepared_root.iterdir() if path.is_dir() and path.name in mk_filter):
        instances = sorted(mk_dir.rglob("*.json"))
        if instance_stems is not None:
            instances = [path for path in instances if path.stem in instance_stems]
        yield mk_dir.name, instances


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[cmd]", " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    prepared_root = (PROJECT_ROOT / args.prepared_instances_root).resolve()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    mk_filter = {item.lower() for item in args.mk_filter}
    instance_stems = None if args.instance_stems is None else set(args.instance_stems)

    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared instances root not found: {prepared_root}")

    planned_jobs = 0
    cached_instances: list[tuple[str, list[Path]]] = []
    for mk_name, instances in iter_instance_files(prepared_root, mk_filter, instance_stems):
        cached_instances.append((mk_name, instances))
        planned_jobs += len(instances) * len(args.budgets)

    if planned_jobs == 0:
        raise SystemExit("No matching prepared JSON instances found.")

    print(f"Repository root: {PROJECT_ROOT}")
    print(f"Algorithm label: {ALGO_LABEL}")
    print(f"Planned launcher invocations: {planned_jobs}")
    print(f"Runs per invocation: {args.runs}")

    for mk_name, instances in cached_instances:
        if not instances:
            print(f"[skip] {mk_name}: no matching instances")
            continue

        for instance_path in instances:
            for budget in args.budgets:
                output_dir = results_root / mk_name / ALGO_LABEL / f"maxeval_{budget}"
                launcher_cmd = [
                    sys.executable,
                    "run_batch_launcher.py",
                    "--instance",
                    str(instance_path),
                    "--algos",
                    ALGO_KEY,
                    "--algorithm-label",
                    ALGO_LABEL,
                    "--output-dir",
                    str(output_dir),
                    "--runs",
                    str(args.runs),
                    "--total-pop",
                    str(args.total_pop),
                    "--crossover-prob",
                    "0.7",
                    "--mutation-prob",
                    "0.2",
                    "--max-evaluations",
                    str(budget),
                    "--initialization-strategy",
                    "idle_mix_25",
                    "--initial-dedup",
                    "--local-search-policy",
                    "q",
                    "--gap-strategy-enabled",
                    "--gap-trigger-interval",
                    "50",
                    "--gap-archive-sample-ratio",
                    "0.2",
                    "--feedback-mode",
                    "feedback",
                    "--ms-ego-enabled",
                    "--ms-ego-probability",
                    str(args.ms_ego_probability),
                    "--ms-ego-random-fill-probability",
                    str(args.ms_ego_random_fill_probability),
                ]
                print(
                    f"[run] mk={mk_name} instance={instance_path.stem} budget={budget} runs={args.runs}"
                )
                run_command(launcher_cmd, args.dry_run)

                if args.compute_metrics:
                    runs_dir = output_dir / instance_path.stem
                    metrics_cmd = [
                        sys.executable,
                        "run_batch_metrics.py",
                        "--runs-dir",
                        str(runs_dir),
                    ]
                    run_command(metrics_cmd, args.dry_run)

    print("All requested launcher invocations completed.")


if __name__ == "__main__":
    main()
