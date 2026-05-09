#!/usr/bin/env python3
"""Build paper-ready result tables for EvalHarness.

The script reads the finalized subset100 CSVs in ``results/`` and writes a
single tidy CSV with all values needed for the methodology/result tables:

- per-run pass rates
- mean/std over the two final runs
- HPV and MPV variance components
- HPV/MPV ratio and ranking-reversal count
- token, cost, and sequential-time totals

Output:
    assets/paper_results_tables.csv
    results/paper_results_tables.csv
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

MODELS = ["GLM-5.1", "GPT-5.4", "Kimi K2.6"]
HARNESSES = ["H1", "H2", "H3"]
RUN_LABELS = {
    "first_run_final": "first",
    "second_run_final": "second",
}
MODEL_LABELS = {
    "glm_5_1": "GLM-5.1",
    "gpt_5_4": "GPT-5.4",
    "kimi_k2_6": "Kimi K2.6",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def population_var(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def fmt_pp(x: float) -> str:
    return f"{x:.1f}"


def fmt_money(x: float) -> str:
    return f"${x:.2f}"


def fmt_tokens(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:.0f}"


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h"


def fmt_value(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def add_row(
    rows: list[dict[str, str]],
    *,
    section: str,
    metric: str,
    value: float | int | str,
    display_value: str | None = None,
    run: str = "",
    model: str = "",
    harness: str = "",
    comparison: str = "",
    unit: str = "",
    n: str | int = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "metric": metric,
            "run": run,
            "model": model,
            "harness": harness,
            "comparison": comparison,
            "value": fmt_value(value),
            "display_value": display_value if display_value is not None else str(value),
            "unit": unit,
            "n": str(n),
            "note": note,
        }
    )


def load_pass_rates() -> dict[tuple[str, str, str], float]:
    """Return pass rates in percentage points keyed by (run, model_label, harness)."""
    pass_rates: dict[tuple[str, str, str], float] = {}
    for row in read_csv(RESULTS_DIR / "final_summary_3x3.csv"):
        run = RUN_LABELS[row["run"]]
        model = MODEL_LABELS[row["model"]]
        harness = row["harness"]
        pass_rates[(run, model, harness)] = 100.0 * float(row["pass_rate"])
    return pass_rates


def load_resources() -> dict[tuple[str, str], dict[str, float]]:
    resources: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_csv(RESULTS_DIR / "resource_summary_total.csv"):
        key = (row["model"], row["harness"])
        resources[key] = {
            "cost_usd": float(row["cost_usd"]),
            "total_tokens": float(row["total_tokens"]),
            "elapsed_seconds_sequential": float(row["elapsed_seconds_sequential"]),
            "instances_with_cost": float(row["instances_with_cost"]),
        }
    return resources


def ranking_reversals(mean_scores: dict[tuple[str, str], float]) -> list[tuple[str, str, str, str]]:
    reversals: list[tuple[str, str, str, str]] = []
    for i, model_a in enumerate(MODELS):
        for model_b in MODELS[i + 1 :]:
            for hi, harness_a in enumerate(HARNESSES):
                for harness_b in HARNESSES[hi + 1 :]:
                    diff_a = mean_scores[(model_a, harness_a)] - mean_scores[(model_b, harness_a)]
                    diff_b = mean_scores[(model_a, harness_b)] - mean_scores[(model_b, harness_b)]
                    if diff_a == 0 or diff_b == 0:
                        continue
                    if diff_a * diff_b < 0:
                        reversals.append((model_a, model_b, harness_a, harness_b))
    return reversals


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pass_rates = load_pass_rates()

    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (run, model, harness), value in pass_rates.items():
        by_cell[(model, harness)].append(value)
        add_row(
            rows,
            section="pass_rate_by_run",
            metric="pass_rate",
            run=run,
            model=model,
            harness=harness,
            value=value,
            display_value=f"{fmt_pp(value)}%",
            unit="percentage_points",
            n=100,
        )

    mean_scores: dict[tuple[str, str], float] = {}
    std_scores: dict[tuple[str, str], float] = {}
    for model in MODELS:
        for harness in HARNESSES:
            values = by_cell[(model, harness)]
            m = mean(values)
            s = sample_std(values)
            mean_scores[(model, harness)] = m
            std_scores[(model, harness)] = s
            add_row(
                rows,
                section="factorial_scores",
                metric="mean_pass_rate",
                model=model,
                harness=harness,
                value=m,
                display_value=f"{fmt_pp(m)}%",
                unit="percentage_points",
                n=2,
            )
            add_row(
                rows,
                section="factorial_scores",
                metric="std_pass_rate",
                model=model,
                harness=harness,
                value=s,
                display_value=f"{fmt_pp(s)}pp",
                unit="percentage_points",
                n=2,
            )
            add_row(
                rows,
                section="factorial_scores",
                metric="paper_cell",
                model=model,
                harness=harness,
                value=f"{m:.3f},{s:.3f}",
                display_value=f"{fmt_pp(m)} ({fmt_pp(s)})",
                unit="mean_percent_std_pp",
                n=2,
                note="Use in Table factorial cells: mean pass@1 over two final runs, std in parentheses.",
            )

    hpv_values: list[float] = []
    for model in MODELS:
        scores = [mean_scores[(model, h)] for h in HARNESSES]
        hpv = population_var(scores)
        hpv_values.append(hpv)
        add_row(
            rows,
            section="variance_decomposition",
            metric="HPV",
            model=model,
            value=hpv,
            display_value=f"{hpv:.2f}",
            unit="pp_squared",
            n=3,
        )
        add_row(
            rows,
            section="variance_decomposition",
            metric="harness_range",
            model=model,
            value=max(scores) - min(scores),
            display_value=f"{max(scores) - min(scores):.1f}pp",
            unit="percentage_points",
            n=3,
        )

    mpv_values: list[float] = []
    for harness in HARNESSES:
        scores = [mean_scores[(m, harness)] for m in MODELS]
        mpv = population_var(scores)
        mpv_values.append(mpv)
        add_row(
            rows,
            section="variance_decomposition",
            metric="MPV",
            harness=harness,
            value=mpv,
            display_value=f"{mpv:.2f}",
            unit="pp_squared",
            n=3,
        )
        add_row(
            rows,
            section="variance_decomposition",
            metric="model_range",
            harness=harness,
            value=max(scores) - min(scores),
            display_value=f"{max(scores) - min(scores):.1f}pp",
            unit="percentage_points",
            n=3,
        )

    hpv_mean = mean(hpv_values)
    mpv_mean = mean(mpv_values)
    ratio = hpv_mean / mpv_mean if mpv_mean else math.inf
    reversals = ranking_reversals(mean_scores)
    possible_reversals = math.comb(len(MODELS), 2) * math.comb(len(HARNESSES), 2)
    add_row(
        rows,
        section="variance_decomposition",
        metric="mean_HPV",
        value=hpv_mean,
        display_value=f"{hpv_mean:.2f}",
        unit="pp_squared",
        n=len(hpv_values),
    )
    add_row(
        rows,
        section="variance_decomposition",
        metric="mean_MPV",
        value=mpv_mean,
        display_value=f"{mpv_mean:.2f}",
        unit="pp_squared",
        n=len(mpv_values),
    )
    add_row(
        rows,
        section="variance_decomposition",
        metric="HPV_over_MPV",
        value=ratio,
        display_value=f"{ratio:.2f}x",
        unit="ratio",
    )
    add_row(
        rows,
        section="ranking_reversals",
        metric="count",
        value=len(reversals),
        display_value=f"{len(reversals)} / {possible_reversals}",
        unit="pairs",
        n=possible_reversals,
    )
    for model_a, model_b, harness_a, harness_b in reversals:
        add_row(
            rows,
            section="ranking_reversals",
            metric="reversal",
            value=1,
            display_value=f"{model_a} vs {model_b}: {harness_a} -> {harness_b}",
            comparison=f"{model_a} vs {model_b}",
            harness=f"{harness_a}/{harness_b}",
        )

    resources = load_resources()
    totals = {"cost_usd": 0.0, "total_tokens": 0.0, "elapsed_seconds_sequential": 0.0}
    for model in MODELS:
        for harness in HARNESSES:
            res = resources[(model, harness)]
            totals["cost_usd"] += res["cost_usd"]
            totals["total_tokens"] += res["total_tokens"]
            totals["elapsed_seconds_sequential"] += res["elapsed_seconds_sequential"]
            add_row(
                rows,
                section="resources",
                metric="cost_usd",
                model=model,
                harness=harness,
                value=res["cost_usd"],
                display_value=fmt_money(res["cost_usd"]),
                unit="USD",
                n=int(res["instances_with_cost"]),
            )
            add_row(
                rows,
                section="resources",
                metric="total_tokens",
                model=model,
                harness=harness,
                value=int(res["total_tokens"]),
                display_value=fmt_tokens(res["total_tokens"]),
                unit="tokens",
                n=int(res["instances_with_cost"]),
            )
            add_row(
                rows,
                section="resources",
                metric="sequential_time_hours",
                model=model,
                harness=harness,
                value=res["elapsed_seconds_sequential"] / 3600,
                display_value=fmt_hours(res["elapsed_seconds_sequential"]),
                unit="hours",
                n=int(res["instances_with_cost"]),
            )

    add_row(
        rows,
        section="resources",
        metric="grand_total_cost_usd",
        value=totals["cost_usd"],
        display_value=fmt_money(totals["cost_usd"]),
        unit="USD",
    )
    add_row(
        rows,
        section="resources",
        metric="grand_total_tokens",
        value=int(totals["total_tokens"]),
        display_value=fmt_tokens(totals["total_tokens"]),
        unit="tokens",
    )
    add_row(
        rows,
        section="resources",
        metric="grand_total_sequential_time_hours",
        value=totals["elapsed_seconds_sequential"] / 3600,
        display_value=fmt_hours(totals["elapsed_seconds_sequential"]),
        unit="hours",
    )
    return rows


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "metric",
        "run",
        "model",
        "harness",
        "comparison",
        "value",
        "display_value",
        "unit",
        "n",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    output_name = "paper_results_tables.csv"
    write_rows(rows, RESULTS_DIR / output_name)
    write_rows(rows, ASSETS_DIR / output_name)
    print(f"Wrote {len(rows)} rows to {RESULTS_DIR / output_name}")
    print(f"Wrote {len(rows)} rows to {ASSETS_DIR / output_name}")


if __name__ == "__main__":
    main()
