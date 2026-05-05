import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TOOLS_ROOT = PROJECT_ROOT / "tools"
for path in (PROJECT_ROOT, SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import shutil
import time
import json

import fbea.evaluation_counter as eval_counter
from fbea.instance_loader import load_instance, load_instance_from_json
from fbea.nsga2_algorithm import run_nsga2
from fbea.random_manager import set_global_seed
from tools.benchmark.result_serialization import write_run_output, serialise_solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II on a prepared instance.")
    parser.add_argument(
        "--instance-json",
        help="Path to a serialized instance JSON (if omitted, raw 'data' file is used).",
    )
    parser.add_argument(
        "--raw-source",
        default="data",
        help="Path to raw instance description used when no JSON is provided.",
    )
    parser.add_argument(
        "--instance-seed",
        type=int,
        help="Seed used when loading raw instance (ignored for JSON).",
    )
    parser.add_argument(
        "--algorithm-seed",
        type=int,
        default=42,
        help="Seed controlling algorithm randomness.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="Override maximum evaluation budget.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/nsga2",
        help="Directory to store raw result JSON files.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        help="Number of worker processes for objective evaluations (default: cpu_count).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        help="Identifier of the dataset (e.g., mk01). When omitted it is inferred from the JSON path.",
    )
    return parser.parse_args()


def load_prepared_instance(args: argparse.Namespace):
    if args.instance_json:
        return load_instance_from_json(args.instance_json)
    return load_instance(args.raw_source, random_seed=args.instance_seed)


def main() -> None:
    args = parse_args()
    instance = load_prepared_instance(args)
    set_global_seed(args.algorithm_seed)
    eval_counter.reset_global_counter()
    if args.eval_workers is not None:
        eval_counter.set_parallel_workers(args.eval_workers)
    if args.max_evaluations:
        eval_counter.set_max_evaluations(args.max_evaluations)
    budget = eval_counter.MAX_EVALUATIONS

    dataset_id = args.dataset_id or "default"
    instance_id = "raw"
    instance_seed = args.instance_seed
    instance_path_str = args.instance_json or args.raw_source
    if args.instance_json:
        instance_path = pathlib.Path(args.instance_json)
        instance_id = instance_path.stem
        if dataset_id == "default":
            dataset_candidate = instance_path.parent.name
            if not dataset_candidate.startswith("seed_"):
                dataset_id = dataset_candidate
        if instance_seed is None and instance_id.startswith("seed_"):
            try:
                instance_seed = int(instance_id.split("_")[-1])
            except ValueError:
                instance_seed = args.instance_seed
    elif args.instance_seed is not None:
        instance_id = f"raw_seed_{args.instance_seed}"

    def progress(gen: int, eval_count: int) -> None:
        if gen == 1 or gen % 100 == 0:
            print(f"Generation {gen} | evaluations {eval_count}/{budget}")

    output_dir = pathlib.Path(args.output_dir) / dataset_id / instance_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_generations = {10, 20, 50, 100, 200, 500}

    def snapshot_callback(gen: int, population) -> None:
        if gen not in snapshot_generations:
            return
        payload = {
            "algorithm": "NSGA2",
            "dataset_id": dataset_id,
            "instance_path": instance_path_str,
            "instance_id": instance_id,
            "algorithm_seed": args.algorithm_seed,
            "instance_seed": instance_seed,
            "generation": gen,
            "evaluations": eval_counter.get_evaluation_count(),
            "solutions": serialise_solutions(population),
        }
        snapshot_path = snapshot_dir / f"generation_{gen:04d}.json"
        snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    start_time = time.perf_counter()
    final_front = run_nsga2(
        instance,
        population_size=50,
        crossover_probability=0.7,
        mutation_probability=0.2,
        random_seed=args.algorithm_seed,
        generation_callback=progress,
        max_evaluations=budget,
        snapshot_callback=snapshot_callback,
    )
    runtime_seconds = time.perf_counter() - start_time

    job_random_factors = list(getattr(instance, "job_random_factors", []))
    print("\nFinal non-dominated solutions:", len(final_front))
    if job_random_factors:
        formatted_job_factors = ", ".join(f"{value:.6f}" for value in job_random_factors)
        print(f"Instance job random factors a: [{formatted_job_factors}]")
    print(f"Instance random factor a (mean) = {instance.random_factor_a:.6f}")
    for idx, solution in enumerate(final_front, start=1):
        solution.evaluate()
        print(
            f"[{idx}] makespan={solution.makespan} "
            f"energy={solution.energy} "
            f"agreement={solution.agreement._c1():.6f}"
        )
    print(f"\nTotal evaluations performed: {eval_counter.get_evaluation_count()}")
    output_path = output_dir / f"seed_{args.algorithm_seed}.json"
    write_run_output(
        output_path,
        algorithm="NSGA2",
        dataset_id=dataset_id,
        instance_path=instance_path_str,
        instance_id=instance_id,
        algorithm_seed=args.algorithm_seed,
        instance_seed=instance_seed,
        solutions=final_front,
        runtime_seconds=runtime_seconds,
        extra_metadata={
            "random_factor_a": instance.random_factor_a,
            "job_random_factors_a": job_random_factors,
        },
    )


if __name__ == "__main__":
    main()
