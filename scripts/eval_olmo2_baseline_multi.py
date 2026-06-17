#!/usr/bin/env python3
"""
Evaluate an OLMo 2 1B baseline checkpoint on multiple benchmarks by calling
eval_olmo2_baseline.sh once per benchmark.

Usage:
    python scripts/eval_olmo2_baseline_multi.py <step> <revision> [--gpu N]

Example:
    python scripts/eval_olmo2_baseline_multi.py 23852 stage2-ingredient3-step23852-tokens97B
    python scripts/eval_olmo2_baseline_multi.py 1000 stage2-ingredient3-step1000-tokens3B --gpu 7

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
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_olmo2_baseline.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", type=int, help="Training step number")
    parser.add_argument("revision", type=str, help="HuggingFace revision string")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: inherit CUDA_VISIBLE_DEVICES or 7)")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Evaluating OLMo 2 1B baseline — step {args.step}, revision {args.revision}")
    print(f"Benchmarks ({len(BENCHMARKS)}): {', '.join(t.split('::')[0] for t in BENCHMARKS)}")
    print()

    failed = []
    for i, task in enumerate(BENCHMARKS, 1):
        task_name = task.split("::")[0]
        print(f"[{i}/{len(BENCHMARKS)}] {task_name} ...", flush=True)
        result = subprocess.run(
            ["bash", EVAL_SCRIPT, str(args.step), args.revision, task],
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
