#!/usr/bin/env python3
"""Plot step-by-step accuracy of ETD-k2 vs OLMo 2 1B baseline on 4 benchmarks.

Reads primary scores directly from the metrics.json files written by the eval
scripts, so the plots stay current as training/evaluation continues.

Layout expected (one metrics.json per step per benchmark):
  eval_results/baseline_olmo2_1B/step<N>/<benchmark>/metrics.json
  eval_results/running/replication/ETD_k2/step<N>/<benchmark>/metrics.json

Usage:
  python scripts/plot_etd_vs_baseline.py
  python scripts/plot_etd_vs_baseline.py --out-dir plots
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Benchmark dir name -> display title for the plot.
BENCHMARKS = {
    "arc_challenge": "ARC-Challenge",
    "agi_eval_english:1shot": "AGIEval-English",
    "socialiqa": "Social IQa",
    "openbookqa": "OpenBookQA",
}

# ETD-k2 final accuracies reported in the ETD paper (Table 5), keyed by
# benchmark dir name. Used as a fallback endpoint only when our own ETD-k2 run
# has not reached the final mid-training step (resource constraints).
PAPER_TABLE5_ETD = {
    "arc_challenge": 58.36,
    "agi_eval_english:1shot": 40.16,
    "socialiqa": 62.9,
    "openbookqa": 57.0,
}

# Full mid-training length for OLMo-2 1B (final checkpoint = step 23852).
# Falls back to this if the baseline's final step can't be inferred from disk.
TOTAL_STEPS = 23852

DEFAULT_BASELINE_DIR = "eval_results/baseline_olmo2_1B"
DEFAULT_ETD_DIR = "eval_results/running/replication/ETD_k2"

_STEP_RE = re.compile(r"step(\d+)$")


def read_primary_score(metrics_path: Path) -> float | None:
    """Return the aggregate ::olmes primary score (in %) or None if unreadable."""
    try:
        data = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    scores = data.get("all_primary_scores")
    if not scores:
        return None
    # Format: "<alias>: <value>"; alias itself contains colons, so split from right.
    try:
        value = float(scores[0].rsplit(": ", 1)[1])
    except (IndexError, ValueError):
        return None
    return value * 100.0


def collect(run_dir: Path, benchmark: str) -> list[tuple[int, float]]:
    """Collect (step, accuracy%) points for one run and benchmark, sorted by step."""
    points = []
    if not run_dir.is_dir():
        return points
    for step_dir in run_dir.iterdir():
        m = _STEP_RE.match(step_dir.name)
        if not m:
            continue
        metrics_path = step_dir / benchmark / "metrics.json"
        if not metrics_path.is_file():
            continue
        score = read_primary_score(metrics_path)
        if score is not None:
            points.append((int(m.group(1)), score))
    points.sort()
    return points


def plot_one(ax, base_pts, etd_pts, title, paper_value, final_step,
             legend=True) -> bool:
    """Draw one benchmark panel. Returns True if the paper fallback was used.

    If legend=True, the legend is drawn just outside the axes (right side).
    Pass legend=False to suppress it (e.g. when using a shared figure legend).
    """
    if base_pts:
        bx, by = zip(*base_pts)
        ax.plot(bx, by, marker="o", ms=3, lw=1.5,
                color="tab:gray", label="OLMo 2 1B baseline")
    used_paper = False
    if etd_pts:
        ex, ey = zip(*etd_pts)
        ax.plot(ex, ey, marker="o", ms=3, lw=1.5,
                color="tab:red", label="ETD-k2")
        # If our run never reached the final step, append the paper's reported
        # endpoint so the curve can still be compared against the baseline's end.
        etd_reached_end = ex[-1] >= final_step
        if not etd_reached_end and paper_value is not None:
            used_paper = True
            # Dashed connector from our last real checkpoint to the paper point.
            ax.plot([ex[-1], final_step], [ey[-1], paper_value],
                    ls="--", lw=1.2, color="tab:red", alpha=0.6)
            ax.plot([final_step], [paper_value], marker="*", ms=14,
                    color="tab:red", mec="black", mew=0.5,
                    label="ETD-k2 (paper, Table 5)", ls="none")

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  borderaxespad=0.0)
    return used_paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--etd-dir", default=DEFAULT_ETD_DIR)
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined 2x2 grid, skip the 4 individual PNGs.",
    )
    parser.add_argument(
        "--no-paper-fallback",
        action="store_true",
        help="Do not substitute the ETD paper's Table 5 value when the ETD-k2 "
             "run hasn't reached the final step.",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    etd_dir = Path(args.etd_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Final mid-training step: use the baseline's last evaluated step if present,
    # otherwise the known total length. This is the x position for the paper point.
    base_steps_all = [
        s for b in BENCHMARKS
        for s, _ in collect(baseline_dir, b)
    ]
    final_step = max(base_steps_all) if base_steps_all else TOTAL_STEPS

    # Combined 2x2 grid.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (bench_dir, title) in zip(axes, BENCHMARKS.items()):
        base_pts = collect(baseline_dir, bench_dir)
        etd_pts = collect(etd_dir, bench_dir)
        paper_value = None if args.no_paper_fallback else PAPER_TABLE5_ETD.get(bench_dir)

        # Suppress per-axis legends here; the grid uses one shared figure legend.
        used_paper = plot_one(ax, base_pts, etd_pts, title, paper_value,
                              final_step, legend=False)

        note = "  (+paper endpoint)" if used_paper else ""
        print(f"{title:18s} baseline:{len(base_pts):3d} pts  "
              f"ETD-k2:{len(etd_pts):3d} pts{note}")

        # Also emit individual figure per benchmark (legend outside, right).
        if not args.no_individual:
            fig_i, ax_i = plt.subplots(figsize=(7, 5))
            plot_one(ax_i, base_pts, etd_pts, title, paper_value, final_step)
            safe = bench_dir.replace(":", "_")
            fig_i.savefig(out_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_i)

    # One shared legend outside the grid (top), de-duplicated across panels.
    handles, labels = [], []
    for ax in axes:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=len(labels), frameon=True)

    fig.suptitle("ETD-k2 vs OLMo 2 1B baseline — step-by-step accuracy", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "etd_vs_baseline_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFinal step (paper-point x position): {final_step}")
    print(f"Wrote plots to {out_dir}/")


if __name__ == "__main__":
    main()
