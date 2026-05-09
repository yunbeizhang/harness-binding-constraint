"""SWE-bench Verified integration helpers for EvalHarness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .core import GradeResult, Harness, Model, TaskRuntime, TrajectoryResult
from .docker_runtime import DockerSweBenchRuntime
from .runtime import PreparedWorkspaceRuntime
from .utils import safe_jsonable, slugify


def load_subset_ids(subset_path: str | Path) -> list[str]:
    """Load instance IDs from an EvalHarness-format subset JSON file."""
    data = json.loads(Path(subset_path).read_text())
    return data["instance_ids"]


DEFAULT_SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SWEBENCH_EVAL_DATASET = "princeton-nlp/SWE-bench_Verified"


@dataclass
class SweBenchTask:
    """A SWE-bench instance backed by a cached GitHub repository checkout."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str | None = None
    fail_to_pass: Any = None
    pass_to_pass: Any = None
    metadata: dict[str, Any] | None = None
    repo_cache_dir: Path | None = None
    refresh_repo_cache: bool = False
    allowed_tools: list[str] | None = None

    @property
    def task_id(self) -> str:
        return self.instance_id

    @property
    def goal(self) -> str:
        sections = [
            f"You are fixing SWE-bench instance `{self.instance_id}` in repository `{self.repo}`.",
            f"The workspace is already checked out at base commit `{self.base_commit}`.",
            "Investigate the issue, edit the repository, and stop only when the repo changes are ready to be evaluated.",
            "Do not print the final patch manually. EvalHarness will collect `git diff --binary` automatically.",
            "",
            "Problem statement:",
            self.problem_statement.strip(),
        ]
        if self.hints_text:
            sections.extend(["", "Hints:", str(self.hints_text).strip()])
        if self.fail_to_pass:
            sections.extend(["", "Fail-to-pass tests:", str(self.fail_to_pass).strip()])
        if self.pass_to_pass:
            sections.extend(["", "Pass-to-pass tests:", str(self.pass_to_pass).strip()])
        return "\n".join(section for section in sections if section is not None)

    def make_runtime(self, scratch_root: str | Path, run_label: str) -> TaskRuntime:
        return DockerSweBenchRuntime(
            instance_id=self.instance_id,
            goal=self.goal,
            metadata={"repo": self.repo, "base_commit": self.base_commit},
        )


def load_swebench_tasks(
    *,
    dataset_name: str = DEFAULT_SWEBENCH_DATASET,
    split: str = "test",
    repo_cache_dir: str | Path = "cache/swebench_repos",
    limit: int | None = None,
    instance_ids: Iterable[str] | None = None,
    refresh_repo_cache: bool = False,
) -> list[SweBenchTask]:
    """Load SWE-bench tasks from Hugging Face via `datasets`."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Loading SWE-bench tasks requires the `datasets` package. "
            "Install it on your Linux workstation with `pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    wanted_ids = set(instance_ids) if instance_ids else None
    repo_cache_path = Path(repo_cache_dir).resolve()

    tasks: list[SweBenchTask] = []
    for row in dataset:
        instance_id = row["instance_id"]
        if wanted_ids and instance_id not in wanted_ids:
            continue
        task = SweBenchTask(
            instance_id=instance_id,
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            hints_text=row.get("hints_text"),
            fail_to_pass=row.get("FAIL_TO_PASS"),
            pass_to_pass=row.get("PASS_TO_PASS"),
            metadata={k: v for k, v in row.items() if k not in {"patch", "test_patch"}},
            repo_cache_dir=repo_cache_path,
            refresh_repo_cache=refresh_repo_cache,
            allowed_tools=[
                "list_files",
                "read_file",
                "write_file",
                "replace_in_file",
                "search_text",
                "run_command",
                "git_status",
                "git_diff",
            ],
        )
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


class SweBenchRuntime(PreparedWorkspaceRuntime):
    """A Git-backed runtime that materializes one SWE-bench repo workspace."""

    def __init__(self, task: SweBenchTask, scratch_root: str | Path, run_label: str):
        self.task = task
        self.task_id = task.instance_id
        self.goal = task.goal
        self.metadata = dict(task.metadata or {})
        self.metadata["repo"] = task.repo
        self.metadata["base_commit"] = task.base_commit

        scratch_root = Path(scratch_root).resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = scratch_root / slugify(task.instance_id) / slugify(run_label)
        if self.workspace_root.exists():
            shutil.rmtree(self.workspace_root)

        self._materialize_workspace(task, self.workspace_root)
        self.workspace_root = self.workspace_root.resolve()
        self.working_directory = self.workspace_root

    def list_tool_specs(self, schema_style: str, exposure: str):
        tool_specs = self._all_tool_specs()
        if exposure == "task_relevant" and self.task.allowed_tools:
            allowed = set(self.task.allowed_tools)
            tool_specs = [tool for tool in tool_specs if tool.name in allowed]
        return tool_specs

    def grade(self, final_output: Any, state) -> GradeResult:
        patch = self.extract_git_patch()
        success = bool(patch.strip())
        changed_files = self.list_changed_files()
        return GradeResult(
            success=success,
            score=1.0 if success else 0.0,
            stdout="",
            stderr="",
            details={
                "instance_id": self.task.instance_id,
                "repo": self.task.repo,
                "base_commit": self.task.base_commit,
                "model_patch": patch,
                "changed_files": changed_files,
                "final_output": "" if final_output is None else str(final_output),
            },
        )

    def extract_git_patch(self) -> str:
        proc = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "git diff failed")
        return proc.stdout

    def list_changed_files(self) -> list[str]:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return []
        files: list[str] = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            files.append(line[3:].strip())
        return files

    def _materialize_workspace(self, task: SweBenchTask, workspace_root: Path) -> None:
        mirror_dir = self._ensure_repo_cache(task)
        subprocess.run(
            ["git", "clone", str(mirror_dir), str(workspace_root)],
            check=True,
            capture_output=True,
            text=True,
            env=_git_env(),
        )
        subprocess.run(
            ["git", "-C", str(workspace_root), "checkout", task.base_commit],
            check=True,
            capture_output=True,
            text=True,
            env=_git_env(),
        )
        subprocess.run(
            ["git", "-C", str(workspace_root), "clean", "-fdx"],
            check=True,
            capture_output=True,
            text=True,
            env=_git_env(),
        )

    def _ensure_repo_cache(self, task: SweBenchTask) -> Path:
        if task.repo_cache_dir is None:
            raise RuntimeError("repo_cache_dir is not configured for SWE-bench tasks.")
        repo_slug = task.repo.replace("/", "__")
        mirror_dir = task.repo_cache_dir / repo_slug / "mirror.git"
        mirror_dir.parent.mkdir(parents=True, exist_ok=True)
        repo_url = f"https://github.com/{task.repo}.git"

        if not mirror_dir.exists():
            subprocess.run(
                ["git", "clone", "--mirror", repo_url, str(mirror_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=_git_env(),
            )
        elif task.refresh_repo_cache:
            subprocess.run(
                ["git", "-C", str(mirror_dir), "remote", "update", "--prune"],
                check=True,
                capture_output=True,
                text=True,
                env=_git_env(),
            )
        return mirror_dir


def run_swebench_inference(
    *,
    model: Model,
    model_label: str,
    harnesses: dict[str, Harness],
    tasks: list[SweBenchTask],
    results_dir: str | Path,
    scratch_dir: str | Path,
    tag: str = "test",
    workers: int = 1,
) -> Path:
    """Run a single model across harnesses and write results with cost tracking.

    Tasks within a harness run in parallel via ThreadPoolExecutor (docker
    containers are per-task and independent; LLM API calls are thread-safe).

    Returns the timestamped results directory path.
    """
    import concurrent.futures
    import copy
    import logging
    import threading
    import time
    from datetime import datetime

    logger = logging.getLogger("evalharness.swebench")
    results_root = Path(results_dir)
    scratch_root = Path(scratch_dir)
    scratch_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / tag / f"{timestamp}_{slugify(model_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results dir: %s (workers=%d)", run_dir, workers)

    experiment_start = time.time()
    all_harness_summaries = []

    def _process_one(task, harness_label, harness, harness_dir):
        inst_dir = harness_dir / task.instance_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        logger.info("[%s] %s — starting", harness_label, task.instance_id)
        try:
            run_label = f"{model_label}__{harness_label}"
            runtime = task.make_runtime(scratch_root=scratch_root, run_label=run_label)
            # Each task needs its own harness instance (state is per-task, but
            # Harness has instance attributes like _last_verify_usage that
            # would be shared across threads). Deep-copy is safer.
            h_copy = copy.copy(harness)
            h_copy._last_verify_usage = None
            trajectory = h_copy.run(model=model, runtime=runtime)
            patch = str(trajectory.grade_result.details.get("model_patch", ""))

            _persist_swebench_trajectory(raw_dir=inst_dir, trajectory=trajectory)

            elapsed = time.time() - t_start
            cost_info = {
                "instance_id": task.instance_id,
                "instance_cost": trajectory.total_cost,
                "api_calls": trajectory.api_calls,
                "input_tokens": trajectory.total_input_tokens,
                "output_tokens": trajectory.total_output_tokens,
                "cached_tokens": trajectory.total_cached_tokens,
                "total_tokens": trajectory.total_input_tokens + trajectory.total_output_tokens,
                "elapsed_seconds": round(elapsed, 1),
                "exit_status": trajectory.termination_reason,
                "step_count": trajectory.step_count,
                "has_patch": bool(patch.strip()),
            }
            (inst_dir / "cost.json").write_text(json.dumps(cost_info, indent=2))
            prediction = {
                "instance_id": task.instance_id,
                "model_name_or_path": f"{model_label}__{harness_label}",
                "model_patch": patch,
            }
            logger.info(
                "[%s] %s — done: %s, cost=$%.4f, tokens=%d, steps=%d, %.0fs",
                harness_label, task.instance_id, trajectory.termination_reason,
                trajectory.total_cost, trajectory.total_input_tokens + trajectory.total_output_tokens,
                trajectory.step_count, elapsed,
            )
            return prediction, cost_info
        except Exception as exc:
            logger.error("[%s] %s — error: %s", harness_label, task.instance_id, exc, exc_info=True)
            return None, None

    for harness_label, harness in harnesses.items():
        harness_dir = run_dir / harness_label.upper()
        harness_dir.mkdir(parents=True, exist_ok=True)

        predictions: list[dict[str, str]] = []
        instance_costs: list[dict[str, Any]] = []
        h_start = time.time()

        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_one, t, harness_label, harness, harness_dir): t for t in tasks}
                for future in concurrent.futures.as_completed(futures):
                    pred, cost_info = future.result()
                    if pred is not None:
                        predictions.append(pred)
                        instance_costs.append(cost_info)
        else:
            for task in tasks:
                pred, cost_info = _process_one(task, harness_label, harness, harness_dir)
                if pred is not None:
                    predictions.append(pred)
                    instance_costs.append(cost_info)

        # Write predictions JSONL for swebench eval
        preds_path = harness_dir / "all_preds.jsonl"
        preds_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=True) for r in predictions) + "\n"
        )

        # Per-harness summary
        h_elapsed = time.time() - h_start
        h_summary = {
            "harness": harness_label,
            "instances": instance_costs,
            "total_cost": round(sum(c["instance_cost"] for c in instance_costs), 6),
            "total_tokens": sum(c["total_tokens"] for c in instance_costs),
            "total_api_calls": sum(c["api_calls"] for c in instance_costs),
            "submitted": sum(1 for c in instance_costs if c["has_patch"]),
            "total_instances": len(instance_costs),
            "elapsed_seconds": round(h_elapsed, 1),
        }
        (harness_dir / "summary.json").write_text(json.dumps(h_summary, indent=2))
        all_harness_summaries.append(h_summary)

        print(f"\n--- {harness_label.upper()} done: "
              f"submitted={h_summary['submitted']}/{h_summary['total_instances']}, "
              f"cost=${h_summary['total_cost']:.4f}, "
              f"tokens={h_summary['total_tokens']:,} ---")

    # Grand summary
    experiment_elapsed = time.time() - experiment_start
    grand_summary = {
        "experiment": {
            "timestamp": timestamp,
            "model_name": model_label,
            "tag": tag,
            "num_instances": len(tasks),
            "instance_ids": [t.instance_id for t in tasks],
            "harnesses": list(harnesses.keys()),
            "total_cells": len(tasks) * len(harnesses),
            "elapsed_seconds": round(experiment_elapsed, 1),
        },
        "costs": {
            "total_cost": round(sum(s["total_cost"] for s in all_harness_summaries), 6),
            "total_input_tokens": sum(sum(c["input_tokens"] for c in s["instances"]) for s in all_harness_summaries),
            "total_output_tokens": sum(sum(c["output_tokens"] for c in s["instances"]) for s in all_harness_summaries),
            "total_cached_tokens": sum(sum(c["cached_tokens"] for c in s["instances"]) for s in all_harness_summaries),
            "total_tokens": sum(s["total_tokens"] for s in all_harness_summaries),
            "total_api_calls": sum(s["total_api_calls"] for s in all_harness_summaries),
        },
        "per_harness": all_harness_summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(grand_summary, indent=2))

    # Print report
    tc = grand_summary["costs"]["total_cost"]
    tt = grand_summary["costs"]["total_tokens"]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {run_dir}")
    print(f"{'='*70}")
    print(f"Model:      {model_label}")
    print(f"Instances:  {len(tasks)}")
    print(f"Cells:      {len(tasks) * len(harnesses)}")
    print(f"Time:       {experiment_elapsed:.0f}s ({experiment_elapsed/60:.1f}min)")
    print(f"")
    print(f"{'Harness':<12} {'Submitted':>10} {'Cost':>10} {'Tokens':>12} {'Calls':>8}")
    print(f"{'-'*52}")
    for s in all_harness_summaries:
        print(f"{s['harness'].upper():<12} {s['submitted']}/{s['total_instances']:>8} "
              f"${s['total_cost']:>8.4f} {s['total_tokens']:>11,} {s['total_api_calls']:>7}")
    print(f"{'-'*52}")
    print(f"{'TOTAL':<12} {'':>10} ${tc:>8.4f} {tt:>11,} {grand_summary['costs']['total_api_calls']:>7}")
    print(f"\nResults: {run_dir / 'summary.json'}")

    return run_dir


def evaluate_swebench_predictions(
    *,
    predictions_path: str | Path,
    dataset_name: str = DEFAULT_SWEBENCH_EVAL_DATASET,
    split: str = "test",
    run_id: str,
    max_workers: int = 4,
    cache_level: str | None = None,
    clean: bool | None = None,
    modal: bool = False,
    modal_parallelism: int | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Shell out to the official SWE-bench evaluation harness."""
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if split:
        command.extend(["--split", split])
    if cache_level:
        command.extend(["--cache_level", cache_level])
    if clean is not None:
        command.extend(["--clean", "True" if clean else "False"])
    if modal:
        command.extend(["--modal", "true"])
    if modal_parallelism is not None:
        command.extend(["--parallelism", str(modal_parallelism)])
    if extra_args:
        command.extend(extra_args)

    subprocess.run(command, check=True, env=dict(os.environ))


def _persist_swebench_trajectory(*, raw_dir: Path, trajectory: TrajectoryResult) -> None:
    output_path = raw_dir / f"{slugify(trajectory.task_id)}.json"
    output_path.write_text(
        json.dumps(safe_jsonable(asdict(trajectory)), indent=2),
        encoding="utf-8",
    )


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env
