"""CLI entrypoints for running EvalHarness experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_variance_decomposition, write_decomposition_outputs
from .harnesses import H1Minimal, H2ImprovedV2, H3FullV2
from .models import load_models_from_config
from .runtime import load_tasks
from .swebench import (
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_EVAL_DATASET,
    evaluate_swebench_predictions,
    load_swebench_tasks,
    run_swebench_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="evalharness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full experiment grid.")
    run_parser.add_argument("--models-config", required=True, help="Path to model config TOML/JSON.")
    run_parser.add_argument("--tasks", required=True, help="Path to task JSONL/JSON.")
    run_parser.add_argument("--harness-config-dir", default=str(_repo_root() / "configs" / "harnesses"))
    run_parser.add_argument("--results-dir", default="results/e1_raw")
    run_parser.add_argument("--scratch-dir", default="tmp/e1_scratch")
    run_parser.add_argument("--runs-per-cell", type=int, default=2)
    run_parser.add_argument("--limit", type=int, default=None, help="Optional task limit.")

    swebench_infer = subparsers.add_parser(
        "swebench-infer",
        help="Run EvalHarness inference on SWE-bench Verified and write all_preds.jsonl files.",
    )
    swebench_infer.add_argument("--models-config", required=True, help="Path to model config TOML/JSON.")
    swebench_infer.add_argument("--dataset-name", default=DEFAULT_SWEBENCH_DATASET)
    swebench_infer.add_argument("--split", default="test")
    swebench_infer.add_argument("--repo-cache-dir", default="cache/swebench_repos")
    swebench_infer.add_argument("--results-dir", default="results/swebench_lite")
    swebench_infer.add_argument("--scratch-dir", default="tmp/swebench_scratch")
    swebench_infer.add_argument("--harness-config-dir", default=str(_repo_root() / "configs" / "harnesses"))
    swebench_infer.add_argument("--models", default="", help="Model label to use (single model per run).")
    swebench_infer.add_argument("--harnesses", default="", help="Comma-separated harness labels to run.")
    swebench_infer.add_argument("--limit", type=int, default=None)
    swebench_infer.add_argument("--instance-ids", default="", help="Comma-separated instance IDs to run.")
    swebench_infer.add_argument("--subset", default="", help="Path to EvalHarness-format subset JSON.")
    swebench_infer.add_argument("--tag", default="test", help="Results subfolder tag (e.g. test, full).")
    swebench_infer.add_argument("--workers", type=int, default=1, help="Parallel tasks per harness (default 1, recommend 4-8).")
    swebench_infer.add_argument("--refresh-repo-cache", action="store_true")

    swebench_eval = subparsers.add_parser(
        "swebench-eval",
        help="Run the official SWE-bench evaluation harness on a predictions file.",
    )
    swebench_eval.add_argument("--predictions-path", required=True)
    swebench_eval.add_argument("--dataset-name", default=DEFAULT_SWEBENCH_EVAL_DATASET)
    swebench_eval.add_argument("--split", default="test")
    swebench_eval.add_argument("--run-id", default="")
    swebench_eval.add_argument("--max-workers", type=int, default=4)
    swebench_eval.add_argument("--cache-level", default="")
    swebench_eval.add_argument("--clean", action="store_true")
    swebench_eval.add_argument("--modal", action="store_true")
    swebench_eval.add_argument("--modal-parallelism", type=int, default=None)

    args = parser.parse_args()
    if args.command == "run":
        _run(args)
        return
    if args.command == "swebench-infer":
        _run_swebench_infer(args)
        return
    if args.command == "swebench-eval":
        _run_swebench_eval(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def _run(args: argparse.Namespace) -> None:
    models = load_models_from_config(args.models_config)
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    harnesses = _load_default_harnesses(Path(args.harness_config_dir))
    report = run_variance_decomposition(
        models=models,
        harnesses=harnesses,
        tasks=tasks,
        runs_per_cell=args.runs_per_cell,
        results_dir=args.results_dir,
        scratch_dir=args.scratch_dir,
    )
    write_decomposition_outputs(report, out_dir=args.results_dir)

    print(f"[EvalHarness] wrote outputs to {args.results_dir}")
    print(f"[EvalHarness] HPV_mean = {report.hpv_mean:.6f}")
    print(f"[EvalHarness] MPV_mean = {report.mpv_mean:.6f}")
    print(f"[EvalHarness] ratio = {report.ratio:.3f}x")
    print(f"[EvalHarness] ranking reversals = {len(report.ranking_reversals)}")


def _run_swebench_infer(args: argparse.Namespace) -> None:
    from .swebench import load_subset_ids

    all_models = load_models_from_config(args.models_config)

    # Single-model design: --models selects exactly one model
    if args.models:
        label = args.models.strip()
        if label not in all_models:
            raise ValueError(f"Unknown model: {label}. Available: {list(all_models.keys())}")
        model_label, model = label, all_models[label]
    elif len(all_models) == 1:
        model_label, model = next(iter(all_models.items()))
    else:
        raise ValueError(
            f"Multiple models in config ({list(all_models.keys())}). "
            f"Use --models to select one. Run separate processes for multiple models."
        )

    harnesses = _load_default_harnesses(Path(args.harness_config_dir))
    selected_harnesses = _select_keys(harnesses, args.harnesses)

    # Load instance IDs from subset file or CLI arg
    instance_ids: list[str] = []
    if args.subset:
        instance_ids = load_subset_ids(args.subset)
    elif args.instance_ids:
        instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]

    tasks = load_swebench_tasks(
        dataset_name=args.dataset_name,
        split=args.split,
        repo_cache_dir=args.repo_cache_dir,
        limit=args.limit,
        instance_ids=instance_ids or None,
        refresh_repo_cache=args.refresh_repo_cache,
    )
    run_dir = run_swebench_inference(
        model=model,
        model_label=model_label,
        harnesses=selected_harnesses,
        tasks=tasks,
        results_dir=args.results_dir,
        scratch_dir=args.scratch_dir,
        tag=args.tag,
        workers=args.workers,
    )
    print(f"[EvalHarness] Results saved to: {run_dir}")


def _run_swebench_eval(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions_path)
    run_id = args.run_id or predictions_path.parent.name
    evaluate_swebench_predictions(
        predictions_path=predictions_path,
        dataset_name=args.dataset_name,
        split=args.split,
        run_id=run_id,
        max_workers=args.max_workers,
        cache_level=args.cache_level or None,
        clean=args.clean,
        modal=args.modal,
        modal_parallelism=args.modal_parallelism,
    )


def _load_default_harnesses(config_dir: Path) -> dict[str, H1Minimal | H2ImprovedV2 | H3FullV2]:
    h2 = H2ImprovedV2(config_dir / "h2_improved.toml")
    h3 = H3FullV2(config_dir / "h3_full.toml")
    return {
        "H1": H1Minimal(config_dir / "h1_minimal.toml"),
        "H2": h2,
        "H3": h3,
    }


def _select_keys(mapping: dict[str, object], raw_labels: str) -> dict[str, object]:
    labels = [item.strip() for item in raw_labels.split(",") if item.strip()]
    if not labels:
        return mapping
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise ValueError(f"Unknown labels requested: {', '.join(missing)}")
    return {label: mapping[label] for label in labels}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
