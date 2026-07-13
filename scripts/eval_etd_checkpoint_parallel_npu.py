#!/usr/bin/env python3
"""
NPU counterpart of eval_etd_checkpoint_parallel.py.

Evaluate a single ETD-k checkpoint on multiple benchmarks in parallel across
multiple Ascend NPU devices, calling eval_etd_checkpoint_npu.sh once per
benchmark.

Requires the checkpoint to already be converted to HF format via
convert_etd_checkpoint.sh before running this script.

Usage:
    python scripts/eval_etd_checkpoint_parallel_npu.py <step> <k> <run_dir> [--force]

Arguments:
    step      Training step number (integer)
    k         Number of ETD thinking iterations (integer)
    run_dir   Path to the training run directory (e.g. running/ETD_k2_npu)

Example:
    python scripts/eval_etd_checkpoint_parallel_npu.py 23852 2 running/ETD_k2_npu

Edit BENCHMARK_NPU_MAP below to choose which benchmarks to run and which NPU
device each one runs on. Benchmarks that map to the same device index run
sequentially on that device, while different device indices run concurrently.
A benchmark is skipped automatically if
eval_results/<run_dir>/step<step>/<task_name>/metrics.json already exists;
pass --force to re-run it anyway.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

# ── Edit this mapping to choose which benchmarks to run and on which device ─
# Keys are olmes task strings (the full set from the ETD paper), values are
# NPU device indices (passed to eval_etd_checkpoint_npu.sh via
# NPU_DEVICE_INDEX). Multiple benchmarks can share a device index; they run
# sequentially on that device while other devices run in parallel. Set a
# value to None to skip that benchmark entirely.
#
# The slow generative tasks (gsm8k, minerva_math, bbh) get dedicated devices;
# the remaining tasks are packed so each device's queue is roughly balanced.
BENCHMARK_NPU_MAP = {
    # Mathematical Reasoning
    "gsm8k::olmes": 0,         # slow
    "minerva_math::olmes": 1,  # MATH # slow

    # BIG-Bench Hard
    "bbh:cot-v1::olmes": 2,    # slow

    # Factual Knowledge
    "naturalqs::olmes": 3,     # slow
    "triviaqa::olmes": 3,

    # Reading Comprehension
    "drop::olmes": 4,          # slow
    "boolq::olmes": 4,
    "openbookqa::olmes": 7,

    # Commonsense Reasoning
    "csqa::olmes": 5,          # CommonsenseQA
    "socialiqa::olmes": 5,     # SocialQA
    "hellaswag::olmes": 6,
    "winogrande::olmes": 6,

    # Multi-Disciplinary Reasoning
    "mmlu::olmes": 5,          # medium
    "mmlu_pro:mc::none": 6,    # MMLU-Pro # medium
    "arc_easy::olmes": 7,
    "arc_challenge::olmes": 7,
    "agi_eval_english:1shot::olmes": 7,  # AGIEval-English
}
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_etd_checkpoint_npu.sh")


def results_exist(run_dir, step, task_name):
    metrics_path = os.path.join(
        REPO_ROOT, "eval_results", run_dir, f"step{step}", task_name, "metrics.json"
    )
    return os.path.isfile(metrics_path)


def run_task_group(device, tasks, step, k, run_dir, log_dir, force):
    """Run this device's tasks one after another; return list of (task_name, status)."""
    env = os.environ.copy()
    env["NPU_DEVICE_INDEX"] = str(device)
    # Isolated per-device module cache to avoid a shared-cache race between
    # parallel olmes processes (same pattern as inference_etd_checkpoints_parallel_npu.sh).
    env["HF_MODULES_CACHE"] = f"/tmp/hf_modules_etd_eval_npu{device}"
    results = []
    for task in tasks:
        task_name = task.split("::")[0]
        if not force and results_exist(run_dir, step, task_name):
            print(f"[NPU {device}] {task_name}: already evaluated, skipping.", flush=True)
            results.append((task_name, "skipped"))
            continue

        log_path = os.path.join(log_dir, f"{task_name}.log")
        print(f"[NPU {device}] {task_name}: starting (log: {log_path})", flush=True)
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
            print(f"[NPU {device}] {task_name}: done in {dt / 60:.1f}m", flush=True)
            results.append((task_name, "ok"))
        else:
            print(f"[NPU {device}] {task_name}: FAILED (exit {proc.returncode}) - see {log_path}", flush=True)
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
        default="running/ETD_k2_npu",
        help="Path to the training run directory (default: running/ETD_k2_npu)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run benchmarks even if results already exist")
    args = parser.parse_args()

    device_to_tasks = {}
    excluded = []
    for task, device in BENCHMARK_NPU_MAP.items():
        if device is None:
            excluded.append(task.split("::")[0])
            continue
        device_to_tasks.setdefault(device, []).append(task)

    log_dir = os.path.join(REPO_ROOT, "eval_results", args.run_dir, f"step{args.step}", "_logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Evaluating ETD-k{args.k} step {args.step} from {args.run_dir}")
    n_to_run = sum(len(tasks) for tasks in device_to_tasks.values())
    print(f"Benchmarks: {n_to_run} to run across {len(device_to_tasks)} NPU device(s): {sorted(device_to_tasks.keys())}")
    if excluded:
        print(f"Excluded (device=None): {', '.join(excluded)}")
    print()

    all_results = {}

    def worker(device, tasks):
        all_results[device] = run_task_group(device, tasks, args.step, args.k, args.run_dir, log_dir, args.force)

    threads = [threading.Thread(target=worker, args=(device, tasks)) for device, tasks in device_to_tasks.items()]
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
