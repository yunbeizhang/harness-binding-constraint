"""Experiment driver for model x harness variance decomposition."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import Harness, Model, TaskDefinition, TrajectoryResult
from .utils import safe_jsonable, slugify


@dataclass
class CellResult:
    """Aggregated results for one `(model, harness)` cell."""

    model_label: str
    harness_label: str
    task_scores: dict[str, list[float]]
    task_metrics: dict[str, list[dict[str, Any]]]
    mean_score: float
    std_score: float
    n_runs: int


@dataclass
class DecompositionReport:
    """Variance decomposition output used by the paper tables."""

    cells: dict[tuple[str, str], CellResult]
    mpv_per_harness: dict[str, float]
    hpv_per_model: dict[str, float]
    hpv_mean: float
    mpv_mean: float
    ratio: float
    ranking_reversals: list[dict[str, Any]]


def run_variance_decomposition(
    *,
    models: dict[str, Model],
    harnesses: dict[str, Harness],
    tasks: list[TaskDefinition],
    runs_per_cell: int = 2,
    results_dir: str | Path = "results/raw",
    scratch_dir: str | Path = "tmp/scratch",
) -> DecompositionReport:
    """Run the full factorial experiment grid and persist raw trajectories."""
    results_dir = Path(results_dir)
    scratch_dir = Path(scratch_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple[str, str], CellResult] = {}
    for model_label, model in models.items():
        for harness_label, harness in harnesses.items():
            task_scores: dict[str, list[float]] = {}
            task_metrics: dict[str, list[dict[str, Any]]] = {}
            for task in tasks:
                scores_this_task: list[float] = []
                metrics_this_task: list[dict[str, Any]] = []
                for run_idx in range(runs_per_cell):
                    run_label = f"{model_label}__{harness_label}__run{run_idx}"
                    runtime = task.make_runtime(scratch_root=scratch_dir, run_label=run_label)
                    trajectory = harness.run(model=model, runtime=runtime)
                    scores_this_task.append(trajectory.score)
                    metrics_this_task.append(trajectory.harness_metrics)
                    _persist_trajectory(
                        results_dir=results_dir,
                        model_label=model_label,
                        harness_label=harness_label,
                        run_idx=run_idx,
                        trajectory=trajectory,
                    )
                task_scores[task.task_id] = scores_this_task
                task_metrics[task.task_id] = metrics_this_task

            per_task_means = [sum(scores) / len(scores) for scores in task_scores.values()]
            mean_score = statistics.mean(per_task_means) if per_task_means else 0.0
            std_score = statistics.stdev(per_task_means) if len(per_task_means) > 1 else 0.0

            cells[(model_label, harness_label)] = CellResult(
                model_label=model_label,
                harness_label=harness_label,
                task_scores=task_scores,
                task_metrics=task_metrics,
                mean_score=mean_score,
                std_score=std_score,
                n_runs=runs_per_cell,
            )

    return _compute_decomposition(
        cells=cells,
        model_labels=list(models.keys()),
        harness_labels=list(harnesses.keys()),
    )


def write_decomposition_outputs(
    report: DecompositionReport,
    *,
    out_dir: str | Path,
) -> None:
    """Write JSON and LaTeX outputs for the decomposition report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    latex_path = out_dir / "table_factorial.tex"

    summary_payload = {
        "cells": {
            f"{model_label}::{harness_label}": safe_jsonable(asdict(cell))
            for (model_label, harness_label), cell in report.cells.items()
        },
        "mpv_per_harness": report.mpv_per_harness,
        "hpv_per_model": report.hpv_per_model,
        "hpv_mean": report.hpv_mean,
        "mpv_mean": report.mpv_mean,
        "ratio": report.ratio,
        "ranking_reversals": report.ranking_reversals,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    latex_path.write_text(_build_latex_table(report), encoding="utf-8")


def _compute_decomposition(
    *,
    cells: dict[tuple[str, str], CellResult],
    model_labels: list[str],
    harness_labels: list[str],
) -> DecompositionReport:
    mpv_per_harness: dict[str, float] = {}
    for harness_label in harness_labels:
        scores = [cells[(model_label, harness_label)].mean_score for model_label in model_labels]
        mpv_per_harness[harness_label] = statistics.variance(scores) if len(scores) > 1 else 0.0

    hpv_per_model: dict[str, float] = {}
    for model_label in model_labels:
        scores = [cells[(model_label, harness_label)].mean_score for harness_label in harness_labels]
        hpv_per_model[model_label] = statistics.variance(scores) if len(scores) > 1 else 0.0

    hpv_mean = statistics.mean(hpv_per_model.values()) if hpv_per_model else 0.0
    mpv_mean = statistics.mean(mpv_per_harness.values()) if mpv_per_harness else 0.0
    ratio = hpv_mean / mpv_mean if mpv_mean > 0 else float("inf")
    reversals = _find_ranking_reversals(cells, model_labels, harness_labels)

    return DecompositionReport(
        cells=cells,
        mpv_per_harness=mpv_per_harness,
        hpv_per_model=hpv_per_model,
        hpv_mean=hpv_mean,
        mpv_mean=mpv_mean,
        ratio=ratio,
        ranking_reversals=reversals,
    )


def _find_ranking_reversals(
    cells: dict[tuple[str, str], CellResult],
    model_labels: list[str],
    harness_labels: list[str],
) -> list[dict[str, Any]]:
    reversals: list[dict[str, Any]] = []
    for i, model_i in enumerate(model_labels):
        for j, model_j in enumerate(model_labels):
            if i >= j:
                continue
            for k, harness_k in enumerate(harness_labels):
                for l, harness_l in enumerate(harness_labels):
                    if k >= l:
                        continue
                    s_ik = cells[(model_i, harness_k)].mean_score
                    s_jk = cells[(model_j, harness_k)].mean_score
                    s_il = cells[(model_i, harness_l)].mean_score
                    s_jl = cells[(model_j, harness_l)].mean_score
                    if (s_ik > s_jk and s_il < s_jl) or (s_ik < s_jk and s_il > s_jl):
                        reversals.append(
                            {
                                "models": [model_i, model_j],
                                "harnesses": [harness_k, harness_l],
                                "scores": {
                                    f"B({model_i},{harness_k})": s_ik,
                                    f"B({model_j},{harness_k})": s_jk,
                                    f"B({model_i},{harness_l})": s_il,
                                    f"B({model_j},{harness_l})": s_jl,
                                },
                                "swing_pp": abs((s_ik - s_jk) - (s_il - s_jl)) * 100.0,
                            }
                        )
    return reversals


def _persist_trajectory(
    *,
    results_dir: Path,
    model_label: str,
    harness_label: str,
    run_idx: int,
    trajectory: TrajectoryResult,
) -> None:
    cell_dir = results_dir / slugify(model_label) / slugify(harness_label)
    cell_dir.mkdir(parents=True, exist_ok=True)
    output_path = cell_dir / f"{slugify(trajectory.task_id)}__run{run_idx}.json"
    output_path.write_text(
        json.dumps(safe_jsonable(asdict(trajectory)), indent=2),
        encoding="utf-8",
    )


def _build_latex_table(report: DecompositionReport) -> str:
    model_labels = sorted({cell.model_label for cell in report.cells.values()})
    harness_labels = sorted({cell.harness_label for cell in report.cells.values()})

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l" + "c" * (len(harness_labels) + 1) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(harness_labels) + r" & $\mathrm{HPV}(M)$ \\")
    lines.append(r"\midrule")
    for model_label in model_labels:
        row = [model_label]
        for harness_label in harness_labels:
            cell = report.cells[(model_label, harness_label)]
            row.append(f"{cell.mean_score * 100:.1f} ({cell.std_score * 100:.1f})")
        row.append(f"{report.hpv_per_model[model_label] * 10000:.2f}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    mpv_row = [r"$\mathrm{MPV}(H)$"]
    for harness_label in harness_labels:
        mpv_row.append(f"{report.mpv_per_harness[harness_label] * 10000:.2f}")
    mpv_row.append("")
    lines.append(" & ".join(mpv_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(f"% HPV_mean = {report.hpv_mean:.6f}")
    lines.append(f"% MPV_mean = {report.mpv_mean:.6f}")
    lines.append(f"% ratio = {report.ratio:.3f}x")
    lines.append(f"% reversals = {len(report.ranking_reversals)}")
    return "\n".join(lines)
