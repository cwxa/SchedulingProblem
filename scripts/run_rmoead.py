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
from rmoead.rmoead_enhanced import run_rmoead
from fbea.random_manager import set_global_seed
from tools.benchmark.result_serialization import write_run_output, serialise_solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RMOEA/D (Enhanced - Paper Compliant) on a prepared instance."
    )
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
        default="experiments/results/rmoead_enhanced",
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
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Population size (number of weight vectors). Paper recommends 100.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.8,
        help="Mutation rate. Paper recommends 0.8.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Q-learning learning rate. Paper recommends 0.4.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Q-learning discount factor. Paper recommends 0.6.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.8,
        help="E-greedy factor. Paper recommends 0.8.",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=40,
        help="VNS memory size LP. Paper recommends 40.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=200,
        help="Maximum generations. Paper recommends 200.",
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
        if gen == 1 or gen % 50 == 0:
            print(f"Generation {gen} | evaluations {eval_count}/{budget}")

    output_dir = pathlib.Path(args.output_dir) / dataset_id / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_generations = {50, 100, 150, 200}

    def snapshot_callback(gen: int, population) -> None:
        if gen not in snapshot_generations:
            return
        payload = {
            "algorithm": "RMOEAD_ENHANCED",
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
    final_front = run_rmoead(
        instance,
        population_size=args.population_size,
        mutation_probability=args.mutation_rate,
        random_seed=args.algorithm_seed,
        generation_callback=progress,
        max_evaluations=budget,
        snapshot_callback=snapshot_callback,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        memory_size=args.memory_size,
    )
    runtime_seconds = time.perf_counter() - start_time

    job_random_factors = list(getattr(instance, "job_random_factors", []))
    print("\n" + "=" * 80)
    print("RMOEA/D Enhanced (Paper Compliant) - Final Results")
    print("=" * 80)
    print(f"Final non-dominated solutions: {len(final_front)}")
    if job_random_factors:
        formatted_job_factors = ", ".join(f"{value:.6f}" for value in job_random_factors)
        print(f"Instance job random factors a: [{formatted_job_factors}]")
    print(f"Instance random factor a (mean) = {instance.random_factor_a:.6f}")
    print("\nPareto Front Solutions:")
    print("-" * 80)
    for idx, solution in enumerate(final_front, start=1):
        solution.evaluate()
        makespan_scalar = (solution.makespan._c1() + 2 * solution.makespan._c2() + solution.makespan._c3()) / 4
        energy_scalar = (solution.energy._c1() + 2 * solution.energy._c2() + solution.energy._c3()) / 4
        print(
            f"[{idx:2d}] Makespan={solution.makespan} (scalar={makespan_scalar:.2f}) "
            f"Energy={solution.energy} (scalar={energy_scalar:.2f}) "
            f"Agreement={solution.agreement._c1():.6f}"
        )
    print("=" * 80)
    print(f"\nTotal evaluations performed: {eval_counter.get_evaluation_count()}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    
    output_path = output_dir / f"seed_{args.algorithm_seed}.json"
    write_run_output(
        output_path,
        algorithm="RMOEAD_ENHANCED",
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
            "population_size": args.population_size,
            "mutation_rate": args.mutation_rate,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "memory_size": args.memory_size,
            "max_generations": args.max_generations,
            "paper_compliant": True,
        },
    )
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
