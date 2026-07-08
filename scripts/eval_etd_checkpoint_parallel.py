#!/usr/bin/env python3
"""
Evaluate a single ETD-k checkpoint on multiple benchmarks in parallel across
multiple GPUs, calling eval_etd_checkpoint.sh once per benchmark.

Requires the checkpoint to already be converted to HF format via
convert_etd_checkpoint.sh before running this script.

Usage:
    python scripts/eval_etd_checkpoint_parallel.py <step> <k> <run_dir> [--force]

Arguments:
    step      Training step number (integer)
    k         Number of ETD thinking iterations (integer)
    run_dir   Path to the training run directory (e.g. running/replication/ETD_k2)

Example:
    python scripts/eval_etd_checkpoint_parallel.py 23852 2 running/replication/ETD_k2

Edit BENCHMARK_GPU_MAP below to choose which benchmarks to run and which GPU
each one runs on. Benchmarks that map to the same GPU index run sequentially
on that GPU, while different GPU indices run concurrently. A benchmark is
skipped automatically if eval_results/<run_dir>/step<step>/<task_name>/metrics.json
already exists; pass --force to re-run it anyway.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

# ── Edit this mapping to choose which benchmarks to run and on which GPU ───
# Keys are olmes task strings (the full set from the ETD paper), values are
# GPU indices (as passed to CUDA_VISIBLE_DEVICES). Multiple benchmarks can
# share a GPU index; they run sequentially on that GPU while other GPUs run
# in parallel. Set a value to None to skip that benchmark entirely.
#
# Below, arc_challenge, agi_eval_english:1shot, socialiqa, and openbookqa are
# set to None because they were already evaluated for step 23852 of ETD_k2;
# the remaining 13 are spread round-robin across the 7 free GPUs (0,1,3,4,5,6,7).
BENCHMARK_GPU_MAP = {
    # Factual Knowledge
    "triviaqa::olmes": 0,
    "naturalqs::olmes": 0, # slow

    # Reading Comprehension
    "boolq::olmes": 1,
    "openbookqa::olmes": None,
    "drop::olmes": 1, # slow

    # Commonsense Reasoning
    "csqa::olmes": 3,          # CommonsenseQA
    "hellaswag::olmes": 4,
    "socialiqa::olmes": None,  # SocialQA
    "winogrande::olmes": 5,

    # Multi-Disciplinary Reasoning
    "arc_easy::olmes": 6,
    "arc_challenge::olmes": None,
    "mmlu::olmes": 3, # medium
    "mmlu_pro:mc::none": 4,    # MMLU-Pro # medium
    "agi_eval_english:1shot::olmes": None,  # AGIEval-English

    # BIG-Bench Hard
    "bbh:cot-v1::olmes": 5, # slow

    # Mathematical Reasoning
    "gsm8k::olmes": 6, # slow
    "minerva_math::olmes": 7,  # MATH # slow
}
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_etd_checkpoint.sh")


def results_exist(run_dir, step, task_name):
    metrics_path = os.path.join(
        REPO_ROOT, "eval_results", run_dir, f"step{step}", task_name, "metrics.json"
    )
    return os.path.isfile(metrics_path)


def run_task_group(gpu, tasks, step, k, run_dir, log_dir, force):
    """Run this GPU's tasks one after another; return list of (task_name, status)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    results = []
    for task in tasks:
        task_name = task.split("::")[0]
        if not force and results_exist(run_dir, step, task_name):
            print(f"[GPU {gpu}] {task_name}: already evaluated, skipping.", flush=True)
            results.append((task_name, "skipped"))
            continue

        log_path = os.path.join(log_dir, f"{task_name}.log")
        print(f"[GPU {gpu}] {task_name}: starting (log: {log_path})", flush=True)
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                ["bash", EVAL_SCRIPT, str(step), str(k), run_dir, task],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        dt = time.time() - t0
        if proc.returncode == 0:
            print(f"[GPU {gpu}] {task_name}: done in {dt / 60:.1f}m", flush=True)
            results.append((task_name, "ok"))
        else:
            print(f"[GPU {gpu}] {task_name}: FAILED (exit {proc.returncode}) - see {log_path}", flush=True)
            results.append((task_name, "failed"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", type=int, nargs="?", default=23852, help="Training step number (default: 23852)")
    parser.add_argument("k", type=int, nargs="?", default=2, help="Number of ETD thinking iterations (default: 2)")
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default="running/replication/ETD_k2",
        help="Path to the training run directory (default: running/replication/ETD_k2)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run benchmarks even if results already exist")
    args = parser.parse_args()

    gpu_to_tasks = {}
    excluded = []
    for task, gpu in BENCHMARK_GPU_MAP.items():
        if gpu is None:
            excluded.append(task.split("::")[0])
            continue
        gpu_to_tasks.setdefault(gpu, []).append(task)

    log_dir = os.path.join(REPO_ROOT, "eval_results", args.run_dir, f"step{args.step}", "_logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Evaluating ETD-k{args.k} step {args.step} from {args.run_dir}")
    n_to_run = sum(len(tasks) for tasks in gpu_to_tasks.values())
    print(f"Benchmarks: {n_to_run} to run across {len(gpu_to_tasks)} GPU(s): {sorted(gpu_to_tasks.keys())}")
    if excluded:
        print(f"Excluded (GPU=None): {', '.join(excluded)}")
    print()

    all_results = {}

    def worker(gpu, tasks):
        all_results[gpu] = run_task_group(gpu, tasks, args.step, args.k, args.run_dir, log_dir, args.force)

    threads = [threading.Thread(target=worker, args=(gpu, tasks)) for gpu, tasks in gpu_to_tasks.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok, skipped, failed = [], [], []
    for results in all_results.values():
        for task_name, status in results:
            {"ok": ok, "skipped": skipped, "failed": failed}[status].append(task_name)

    print()
    print("=" * 60)
    print(f"Done: {len(ok)} succeeded, {len(skipped)} skipped, {len(failed)} failed.")
    if failed:
        print("Failed benchmarks:")
        for t in failed:
            print(f"  - {t}")
        sys.exit(1)


if __name__ == "__main__":
    main()
