#!/usr/bin/env python3
"""
Evaluate an ETD-k checkpoint on multiple benchmarks by calling
eval_etd_checkpoint.sh once per benchmark.

Requires the checkpoint to already be converted to HF format via
convert_etd_checkpoint.sh before running this script.

Usage:
    python scripts/eval_etd_checkpoint_multi.py <step> <k> <run_dir> [--gpu N]

Arguments:
    step      Training step number (integer)
    k         Number of ETD thinking iterations (integer)
    run_dir   Path to the training run directory (e.g. running/train/ETD_k2)

Examples:
    python scripts/eval_etd_checkpoint_multi.py 100 2 running/sanity/ETD_k2_per_step
    python scripts/eval_etd_checkpoint_multi.py 5000 2 running/train/ETD_k2 --gpu 6

Results are saved to:
    eval_results/<run_dir>/step<step>/<task_name>/

Edit BENCHMARKS below to choose which tasks to run.
"""

import argparse
import os
import subprocess
import sys

# ── Edit this list to choose which benchmarks to evaluate ──────────────────
ALL_BENCHMARKS = [
    # Factual Knowledge
    "triviaqa::olmes",
    "naturalqs::olmes",

    # Reading Comprehension
    "boolq::olmes",
    "openbookqa::olmes",
    "drop::olmes",

    # Commonsense Reasoning
    "csqa::olmes",           # CommonsenseQA
    "hellaswag::olmes",
    "socialiqa::olmes",      # SocialQA
    "winogrande::olmes",

    # Multi-Disciplinary Reasoning
    "arc_easy::olmes",
    "arc_challenge::olmes",
    "mmlu::olmes",
    "mmlu_pro:mc::none",     # MMLU-Pro
    "agi_eval_english:1shot::olmes",  # AGIEval-English

    # BIG-Bench Hard
    "bbh:cot-v1::olmes",

    # Mathematical Reasoning
    "gsm8k::olmes",
    "minerva_math::olmes",   # MATH
]

BENCHMARKS = [
    # Multi-Disciplinary Reasoning
    "arc_challenge::olmes",
    "agi_eval_english:1shot::olmes",  # AGIEval-English

    # Commonsense Reasoning
    "socialiqa::olmes",      # SocialQA

    # Reading Comprehension
    "openbookqa::olmes",
]
# ───────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_etd_checkpoint.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", type=int, help="Training step number")
    parser.add_argument("k", type=int, help="Number of ETD thinking iterations")
    parser.add_argument("run_dir", type=str, help="Path to the training run directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: inherit CUDA_VISIBLE_DEVICES or 6)")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Evaluating ETD-k{args.k} step {args.step} from {args.run_dir}")
    print(f"Benchmarks ({len(BENCHMARKS)}): {', '.join(t.split('::')[0] for t in BENCHMARKS)}")
    print()

    failed = []
    for i, task in enumerate(BENCHMARKS, 1):
        task_name = task.split("::")[0]
        print(f"[{i}/{len(BENCHMARKS)}] {task_name} ...", flush=True)
        result = subprocess.run(
            ["bash", EVAL_SCRIPT, str(args.step), str(args.k), args.run_dir, task],
            env=env,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            failed.append(task)
        else:
            print(f"  Done.")

    print()
    if failed:
        print(f"Completed with {len(failed)} failure(s):")
        for t in failed:
            print(f"  - {t}")
        sys.exit(1)
    else:
        print(f"All {len(BENCHMARKS)} benchmarks completed successfully.")


if __name__ == "__main__":
    main()
