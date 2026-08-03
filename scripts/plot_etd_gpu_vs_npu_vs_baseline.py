#!/usr/bin/env python3
"""Plot step-by-step accuracy of OLMo 2 1B baseline vs ETD-k2 trained on the
H100 GPU server vs ETD-k2 trained on the NPU server, on 4 benchmarks.

Reads primary scores directly from the metrics.json files written by the eval
scripts, so the plots stay current as training/evaluation continues on either
server.

Layout expected (one metrics.json per step per benchmark):
  eval_results/baseline_olmo2_1B/step<N>/<benchmark>/metrics.json
  eval_results/running/replication/ETD_k2/step<N>/<benchmark>/metrics.json
  eval_results/running/ETD_k2_npu/step<N>/<benchmark>/metrics.json

The NPU run's eval_results are produced on the NPU server and must be synced
into this repo before plotting, e.g. via a dedicated git branch used only to
carry the metrics.json files:

  # one-time, on the NPU server
  git checkout --orphan npu-eval-results
  git rm -rf --cached .
  git add -f eval_results/running/ETD_k2_npu
  git commit -m "npu eval results"
  git push origin npu-eval-results

  # after every new batch of NPU eval results
  git add -f eval_results/running/ETD_k2_npu
  git commit -m "update npu eval results up to step<N>"
  git push origin npu-eval-results

  # on this (GPU) server, to pull the latest NPU results without touching
  # any code on main
  git fetch origin npu-eval-results
  git checkout origin/npu-eval-results -- eval_results/running/ETD_k2_npu

Usage:
  python scripts/plot_etd_gpu_vs_npu_vs_baseline.py
  python scripts/plot_etd_gpu_vs_npu_vs_baseline.py --out-dir plots
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

DEFAULT_BASELINE_DIR = "eval_results/baseline_olmo2_1B"
DEFAULT_GPU_DIR = "eval_results/running/replication/ETD_k2"
DEFAULT_NPU_DIR = "eval_results/running/ETD_k2_npu"

# (label, color) for each series, keyed by the same names used for --*-dir args.
SERIES_STYLE = {
    "baseline": ("OLMo 2 1B baseline", "tab:gray"),
    "gpu": ("ETD-k2 (GPU)", "tab:red"),
    "npu": ("ETD-k2 (NPU)", "tab:blue"),
}

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


def plot_one(ax, series_points, title, legend=True) -> None:
    """Draw one benchmark panel.

    series_points: list of (series_key, points) pairs, drawn in order.
    If legend=True, the legend is drawn just outside the axes (right side).
    """
    for key, points in series_points:
        if not points:
            continue
        label, color = SERIES_STYLE[key]
        x, y = zip(*points)
        ax.plot(x, y, marker="o", ms=3, lw=1.5, color=color, label=label)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--gpu-dir", default=DEFAULT_GPU_DIR)
    parser.add_argument("--npu-dir", default=DEFAULT_NPU_DIR)
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined 2x2 grid, skip the 4 individual PNGs.",
    )
    args = parser.parse_args()

    run_dirs = {
        "baseline": Path(args.baseline_dir),
        "gpu": Path(args.gpu_dir),
        "npu": Path(args.npu_dir),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (bench_dir, title) in zip(axes, BENCHMARKS.items()):
        series_points = [
            (key, collect(run_dir, bench_dir)) for key, run_dir in run_dirs.items()
        ]
        plot_one(ax, series_points, title, legend=False)

        counts = "  ".join(
            f"{key}:{len(points):3d} pts" for key, points in series_points
        )
        print(f"{title:18s} {counts}")

        if not args.no_individual:
            fig_i, ax_i = plt.subplots(figsize=(7, 5))
            plot_one(ax_i, series_points, title)
            safe = bench_dir.replace(":", "_")
            fig_i.savefig(out_dir / f"{safe}_gpu_vs_npu.png", dpi=150, bbox_inches="tight")
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

    fig.suptitle("ETD-k2 (GPU) vs ETD-k2 (NPU) vs OLMo 2 1B baseline",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "etd_gpu_vs_npu_vs_baseline_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote plots to {out_dir}/")


if __name__ == "__main__":
    main()
