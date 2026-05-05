"""
Batch launcher: run one or multiple algorithms on a given prepared JSON instance
for specified seeds, and store outputs under a target directory.

Usage example (PowerShell):
  python run_batch_launcher.py `
    --instance experiments/prepared_instances/mk04/seed_7.json `
    --algos fbea fbea_unity_v2 `
    --seeds 1 2 3 `
    --output-dir experiments/results/mk04/fbeavsv2

Key behaviors:
  - Automatically inserts project src/tools/root into sys.path (no PYTHONPATH needed).
  - Supports multiple algorithms; add new ones by extending ALGO_MAP.
  - Writes one JSON per (algo, seed) to <output-dir>/<instance_stem>/<algo>_run<seed>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Sequence

# Ensure project modules are importable without relying on external PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "tools", ROOT):
    extra_str = extra.resolve().as_posix()
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run algorithms on a prepared JSON instance.")
    parser.add_argument(
        "--instance",
        required=True,
        help="Path to prepared instance JSON.",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["fbea", "fbea_unity_v2"],
        help=(
            "Algorithms to run (choices: fbea, fbea_composable, "
            "fbea_complement_migration, fbea_elitist_replacement, "
            "fbea_unity_init, fbea_v2_parent_selection_truncation, "
            "fbea_init_dedup, fbea_q_ls_2step, fbea_init_dedup_q_ls_2step, "
            "fbea_idle25_init_dedup_gap50_q_ls_2step, fbea_struct_local_search, "
            "fbea_archive_refine, fbea_unity_v2, coa_v1)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit seeds. If omitted, uses 1..runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per algorithm when --seeds not provided (defaults to 1..runs).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/batch_runs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--total-pop",
        type=int,
        default=100,
        help="Total population size.",
    )
    parser.add_argument(
        "--crossover-prob",
        type=float,
        default=0.7,
        help="Crossover probability.",
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.2,
        help="Mutation probability.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Progress print interval for algorithms that support it (0 to disable).",
    )
    parser.add_argument(
        "--initialization-strategy",
        choices=["heuristic", "random", "idle_mix_25", "unity_v2"],
        default="heuristic",
        help="Initialisation strategy used by fbea_composable.",
    )
    parser.add_argument(
        "--initial-dedup",
        action="store_true",
        help="Enable initial population de-duplication for fbea_composable.",
    )
    parser.add_argument(
        "--local-search-policy",
        choices=["random", "q"],
        default="random",
        help="Local-search policy used by fbea_composable; q defaults to two-step state design.",
    )
    parser.add_argument(
        "--gap-strategy-enabled",
        action="store_true",
        help="Enable periodic gap handling for fbea_composable.",
    )
    parser.add_argument(
        "--gap-trigger-interval",
        type=int,
        default=50,
        help="Gap-trigger interval for fbea_composable.",
    )
    parser.add_argument(
        "--gap-archive-sample-ratio",
        type=float,
        default=0.2,
        help="Archive sample ratio for fbea_composable gap handling.",
    )
    parser.add_argument(
        "--population-management-strategy",
        choices=["original", "v2_parent_selection_truncation"],
        default="original",
        help="Population management strategy for fbea_composable.",
    )
    parser.add_argument(
        "--replacement-strategy",
        choices=["original", "elitist_replacement"],
        default="original",
        help="Replacement strategy for fbea_composable.",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=["feedback", "fixed"],
        default="feedback",
        help="Feedback mode for fbea_composable.",
    )
    parser.add_argument(
        "--ms-ego-enabled",
        action="store_true",
        help="Enable the machine-selection EGO-lite post-recombination component.",
    )
    parser.add_argument(
        "--ms-ego-probability",
        type=float,
        default=0.2,
        help="Per-offspring probability of applying MS-EGO-lite when enabled.",
    )
    parser.add_argument(
        "--ms-ego-random-fill-probability",
        type=float,
        default=0.2,
        help="Probability that a replaced MS locus is filled from a random donor instead of the elite donor.",
    )
    parser.add_argument(
        "--struct-local-search-enabled",
        action="store_true",
        help="Enable the independent structure-aware local-search component for fbea_composable.",
    )
    parser.add_argument(
        "--struct-local-search-action-mode",
        default="addon19",
        help="Action-set mode for structure-aware local search.",
    )
    parser.add_argument(
        "--archive-refine-enabled",
        action="store_true",
        help="Enable the independent archive refine component for fbea_composable.",
    )
    parser.add_argument(
        "--archive-refine-interval",
        type=int,
        default=50,
        help="Trigger interval for archive refine in fbea_composable.",
    )
    parser.add_argument(
        "--archive-refine-sample-ratio",
        type=float,
        default=0.2,
        help="Archive sample ratio for archive refine in fbea_composable.",
    )
    parser.add_argument(
        "--archive-refine-candidate-topk",
        type=int,
        default=3,
        help="Maximum per-solution candidate count for archive refine.",
    )
    parser.add_argument(
        "--algorithm-label",
        help="Optional explicit output label for fbea_composable.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        help="Evaluation budget override.",
    )
    parser.add_argument("--migration-interval", type=int, default=50, help="Complement-migration interval.")
    parser.add_argument("--migration-k", type=int, default=2, help="Migrants per direction.")
    parser.add_argument("--migration-front-cap", type=int, default=2, help="Max sender front rank for migrants.")
    parser.add_argument(
        "--migration-cover-threshold",
        type=float,
        default=0.04,
        help="Receiver coverage threshold in normalized objective space.",
    )
    parser.add_argument(
        "--migration-pair-threshold",
        type=float,
        default=0.02,
        help="Minimum distance between selected migrants in one direction.",
    )
    return parser.parse_args(argv)


def default_args_for_quick_run() -> argparse.Namespace:
    """
    Quick-run defaults so you can right-click/run without typing CLI args.
    Adjust these values as needed.
    """
    ns = argparse.Namespace()
    # User default: run FBEA + V2 for 20 runs on mk01/a=0.9-1.5/seed_101 and write under experiments/results/mk01/fbeavsv2/0.9-1.5/<instance_stem>/.
    ns.instance = str(
        ROOT
        / "experiments"
        / "prepared_instances"
        / "mk01"
        / "a=0.9-1.5"
        / "mk01"
        / "seed_101.json"
    )
    ns.algos = ["fbea", "fbea_unity_v2"]
    ns.seeds = None  # will use runs range below
    ns.runs = 20
    ns.output_dir = str(ROOT / "experiments" / "results" / "mk01" / "fbeavsv2" / "0.9-1.5")
    ns.total_pop = 100
    ns.crossover_prob = 0.7
    ns.mutation_prob = 0.2
    ns.progress_interval = 0
    ns.initialization_strategy = "heuristic"
    ns.initial_dedup = False
    ns.local_search_policy = "random"
    ns.gap_strategy_enabled = False
    ns.gap_trigger_interval = 50
    ns.gap_archive_sample_ratio = 0.2
    ns.population_management_strategy = "original"
    ns.replacement_strategy = "original"
    ns.feedback_mode = "feedback"
    ns.ms_ego_enabled = False
    ns.ms_ego_probability = 0.2
    ns.ms_ego_random_fill_probability = 0.2
    ns.struct_local_search_enabled = False
    ns.struct_local_search_action_mode = "addon19"
    ns.archive_refine_enabled = False
    ns.archive_refine_interval = 50
    ns.archive_refine_sample_ratio = 0.2
    ns.archive_refine_candidate_topk = 3
    ns.algorithm_label = None
    ns.max_evaluations = None
    ns.migration_interval = 50
    ns.migration_k = 2
    ns.migration_front_cap = 2
    ns.migration_cover_threshold = 0.04
    ns.migration_pair_threshold = 0.02
    return ns


def _normalize_algorithm_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _default_fbea_composable_label(args: argparse.Namespace) -> str:
    tokens = ["fbea"]

    init_mode = args.initialization_strategy.lower()
    if init_mode == "random":
        tokens.append("random_init")
    elif init_mode == "idle_mix_25":
        tokens.append("idle25")
    elif init_mode == "unity_v2":
        tokens.append("unity_init")

    if args.initial_dedup:
        tokens.append("init_dedup")
    if args.gap_strategy_enabled:
        if args.gap_trigger_interval > 0:
            tokens.append(f"gap{args.gap_trigger_interval}")
        else:
            tokens.append("gap")
    if args.local_search_policy.lower() == "q":
        tokens.append("q_ls_2step")
    if args.population_management_strategy.lower() != "original":
        tokens.append(args.population_management_strategy.lower())
    if args.replacement_strategy.lower() != "original":
        tokens.append(args.replacement_strategy.lower())
    if args.feedback_mode.lower() != "feedback":
        tokens.append("fixed_feedback")
    if args.ms_ego_enabled:
        tokens.append("ms_ego")
    if args.struct_local_search_enabled:
        tokens.append("struct_ls")
    if args.archive_refine_enabled:
        tokens.append("archive_refine")

    return "_".join(tokens)


def _resolve_fbea_composable_label(args: argparse.Namespace) -> str:
    if args.algorithm_label:
        normalized = _normalize_algorithm_label(args.algorithm_label)
        if normalized:
            return normalized
    return _default_fbea_composable_label(args)


def _build_fbea_composable_kwargs(args: argparse.Namespace, seed: int) -> dict[str, object]:
    q_config = None
    if args.local_search_policy.lower() == "q":
        q_config = {"state_design": "two_step_eff"}
    return {
        "total_population_size": args.total_pop,
        "crossover_probability": args.crossover_prob,
        "mutation_probability": args.mutation_prob,
        "random_seed": seed,
        "initialization_strategy": args.initialization_strategy,
        "initial_dedup": args.initial_dedup,
        "local_search_policy": args.local_search_policy,
        "local_search_q_config": q_config,
        "gap_strategy_enabled": args.gap_strategy_enabled,
        "gap_trigger_interval": args.gap_trigger_interval,
        "gap_archive_sample_ratio": args.gap_archive_sample_ratio,
        "population_management_strategy": args.population_management_strategy,
        "replacement_strategy": args.replacement_strategy,
        "feedback_mode": args.feedback_mode,
        "ms_ego_enabled": args.ms_ego_enabled,
        "ms_ego_probability": args.ms_ego_probability,
        "ms_ego_random_fill_probability": args.ms_ego_random_fill_probability,
        "struct_local_search_enabled": args.struct_local_search_enabled,
        "struct_local_search_action_mode": args.struct_local_search_action_mode,
        "archive_refine_enabled": args.archive_refine_enabled,
        "archive_refine_interval": args.archive_refine_interval,
        "archive_refine_sample_ratio": args.archive_refine_sample_ratio,
        "archive_refine_candidate_topk": args.archive_refine_candidate_topk,
    }


def _resolve_run_label(algo_key: str, args: argparse.Namespace) -> str:
    if algo_key == "fbea_composable":
        return _resolve_fbea_composable_label(args)
    return algo_key


def _validate_unique_run_labels(algorithms: Sequence[str], args: argparse.Namespace) -> None:
    seen_labels: dict[str, str] = {}
    for algo in algorithms:
        algo_key = algo.lower()
        run_label = _resolve_run_label(algo_key, args)
        previous = seen_labels.get(run_label)
        if previous is not None:
            raise ValueError(
                "Duplicate output algorithm label detected: "
                f"'{run_label}' is produced by both '{previous}' and '{algo_key}'. "
                "Use --algorithm-label or run them separately."
            )
        seen_labels[run_label] = algo_key


def main() -> None:
    # If no CLI args are supplied, fall back to the quick-run defaults for convenience.
    args = parse_args() if len(sys.argv) > 1 else default_args_for_quick_run()

    # Lazy imports after sys.path setup to avoid ModuleNotFoundError
    try:
        import fbea.evaluation_counter as eval_counter  # type: ignore
        from COA.coa_v1 import coa_v1_main_algorithm  # type: ignore
        from fbea.instance_loader import load_instance_from_json  # type: ignore
        from fbea.random_manager import set_global_seed  # type: ignore
        from fbea.algorithm4_main_loop import (  # type: ignore
            fbea_main_algorithm,
            fbea_main_algorithm_archive_refine,
            fbea_main_algorithm_elitist_replacement,
            fbea_main_algorithm_idle25_init_dedup_gap50_q_local_search,
            fbea_main_algorithm_init_dedup,
            fbea_main_algorithm_init_dedup_q_local_search,
            fbea_main_algorithm_struct_local_search,
            fbea_main_algorithm_unity_init,
            fbea_main_algorithm_v2_parent_selection_truncation,
            fbea_main_algorithm_q_local_search,
        )
        from fbea.complement_migration import (  # type: ignore
            ComplementMigrationConfig,
            fbea_main_algorithm_complement_migration,
        )
        from tools.benchmark.result_serialization import write_run_output  # type: ignore
    except ModuleNotFoundError as exc:
        print("[ERROR] Failed to import project modules. sys.path is:")
        for p in sys.path[:10]:
            print("  ", p)
        raise exc

    selected_algorithms = {algo.lower() for algo in args.algos}
    fbea_unity_main_algorithm = None
    if "fbea_unity_v2" in selected_algorithms:
        from fbea_unity_v2 import fbea_unity_main_algorithm  # type: ignore

    instance_path = Path(args.instance).resolve()
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance JSON not found: {instance_path}")

    # Seed list
    seeds: Sequence[int] = args.seeds if args.seeds else list(range(1, args.runs + 1))
    if args.max_evaluations is not None:
        eval_counter.set_max_evaluations(args.max_evaluations)

    # Map algo name to runner factory
    ALGO_MAP: Dict[str, Callable[[object, int], object]] = {
        "fbea": lambda inst, seed: fbea_main_algorithm(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_composable": lambda inst, seed: fbea_main_algorithm(
            inst,
            **_build_fbea_composable_kwargs(args, seed),
        ),
        "fbea_complement_migration": lambda inst, seed: fbea_main_algorithm_complement_migration(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
            ms_ego_enabled=args.ms_ego_enabled,
            ms_ego_probability=args.ms_ego_probability,
            ms_ego_random_fill_probability=args.ms_ego_random_fill_probability,
            migration_config=ComplementMigrationConfig(
                interval=args.migration_interval,
                migrants_per_direction=args.migration_k,
                quality_front_cap=args.migration_front_cap,
                cover_threshold=args.migration_cover_threshold,
                migrant_pair_threshold=args.migration_pair_threshold,
            ),
        ),
        "fbea_elitist_replacement": lambda inst, seed: fbea_main_algorithm_elitist_replacement(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_unity_init": lambda inst, seed: fbea_main_algorithm_unity_init(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_v2_parent_selection_truncation": lambda inst, seed: fbea_main_algorithm_v2_parent_selection_truncation(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_init_dedup": lambda inst, seed: fbea_main_algorithm_init_dedup(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_q_ls_2step": lambda inst, seed: fbea_main_algorithm_q_local_search(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
            local_search_q_config={
                "state_design": "two_step_eff",
            },
        ),
        "fbea_init_dedup_q_ls_2step": lambda inst, seed: fbea_main_algorithm_init_dedup_q_local_search(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
            local_search_q_config={
                "state_design": "two_step_eff",
            },
        ),
        "fbea_idle25_init_dedup_gap50_q_ls_2step": lambda inst, seed: fbea_main_algorithm_idle25_init_dedup_gap50_q_local_search(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_struct_local_search": lambda inst, seed: fbea_main_algorithm_struct_local_search(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "fbea_archive_refine": lambda inst, seed: fbea_main_algorithm_archive_refine(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
        ),
        "coa_v1": lambda inst, seed: coa_v1_main_algorithm(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
            progress_interval=args.progress_interval,
        ),
    }
    if fbea_unity_main_algorithm is not None:
        ALGO_MAP["fbea_unity_v2"] = lambda inst, seed: fbea_unity_main_algorithm(
            inst,
            total_population_size=args.total_pop,
            crossover_probability=args.crossover_prob,
            mutation_probability=args.mutation_prob,
            random_seed=seed,
            progress_interval=args.progress_interval,
        )
    _validate_unique_run_labels(args.algos, args)

    # Output directories
    out_root = Path(args.output_dir).resolve() / instance_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Load instance once per run (fresh load to avoid mutation carry-over)
    for algo in args.algos:
        algo_key = algo.lower()
        if algo_key not in ALGO_MAP:
            print(f"[warn] Unknown algo '{algo}', skip.")
            continue
        runner = ALGO_MAP[algo_key]
        run_label = _resolve_run_label(algo_key, args)
        for seed in seeds:
            set_global_seed(seed)
            inst = load_instance_from_json(str(instance_path))
            eval_counter.reset_global_counter()
            t0 = time.perf_counter()
            archive = runner(inst, seed)
            runtime = time.perf_counter() - t0
            out_file = out_root / f"{run_label}_run{seed}.json"
            extra_metadata = {"max_evaluations": eval_counter.MAX_EVALUATIONS}
            migration_stats = getattr(archive, "complement_migration_stats", None)
            if migration_stats is not None:
                extra_metadata["migration_stats"] = migration_stats
            write_run_output(
                out_file,
                algorithm=run_label.upper(),
                dataset_id=instance_path.parent.name,
                instance_path=str(instance_path),
                instance_id=instance_path.stem,
                algorithm_seed=seed,
                instance_seed=seed,
                solutions=archive.solutions,
                runtime_seconds=runtime,
                extra_metadata=extra_metadata,
            )
            print(f"{algo} seed={seed} saved {out_file} | runtime={runtime:.2f}s")

    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
