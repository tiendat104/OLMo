#!/usr/bin/env python3
"""
Evaluate multiple intermediate OLMo 2 1B baseline checkpoints on multiple
benchmarks by calling eval_olmo2_baseline.sh once per (checkpoint, benchmark).

Usage:
    python scripts/eval_olmo2_baseline_multi_checkpoints.py [--gpu N]

Configure which checkpoints and benchmarks to run by editing CHECKPOINTS and
BENCHMARKS below. CHECKPOINTS is a list of (step, revision) pairs.

Example:
    python scripts/eval_olmo2_baseline_multi_checkpoints.py
    python scripts/eval_olmo2_baseline_multi_checkpoints.py --gpu 7
"""

import argparse
import os
import subprocess
import sys

# ── Map each step to its HuggingFace revision string ───────────────────────
# Fill this in from the OLMo 2 1B revisions list on HuggingFace
# (https://huggingface.co/allenai/OLMo-2-0425-1B/tree/main -> "main" dropdown).
# The revision encodes both step and tokens, e.g. stage2-ingredient3-step1000-tokens3B
STEP_TO_REVISION = {
    1000:  "stage2-ingredient3-step1000-tokens3B",
    2000:  "stage2-ingredient3-step2000-tokens5B",
    3000:  "stage2-ingredient3-step3000-tokens7B",
    4000:  "stage2-ingredient3-step4000-tokens9B",
    5000:  "stage2-ingredient3-step5000-tokens11B",
    6000:  "stage2-ingredient3-step6000-tokens13B",
    7000:  "stage2-ingredient3-step7000-tokens15B",
    8000:  "stage2-ingredient3-step8000-tokens17B",
    9000:  "stage2-ingredient3-step9000-tokens19B",
    10000: "stage2-ingredient3-step10000-tokens21B",
    11000: "stage2-ingredient3-step11000-tokens24B",
    12000: "stage2-ingredient3-step12000-tokens26B",
    13000: "stage2-ingredient3-step13000-tokens28B",
    14000: "stage2-ingredient3-step14000-tokens30B",
    15000: "stage2-ingredient3-step15000-tokens32B",
    16000: "stage2-ingredient3-step16000-tokens34B",
    17000: "stage2-ingredient3-step17000-tokens36B",
    18000: "stage2-ingredient3-step18000-tokens38B",
    19000: "stage2-ingredient3-step19000-tokens40B",
    20000: "stage2-ingredient3-step20000-tokens42B",
    21000: "stage2-ingredient3-step21000-tokens45B",
    22000: "stage2-ingredient3-step22000-tokens47B",
    23000: "stage2-ingredient3-step23000-tokens49B",
    23852: "stage2-ingredient3-step23852-tokens51B",
}

# ── Edit this list to choose which checkpoints to evaluate ─────────────────
# Only list the steps here; revisions are looked up in STEP_TO_REVISION above.
STEPS = [
    2000,
    3000,
    4000,
    6000,
    7000,
    8000,
    9000,
    11000,
    12000,
    13000,
    14000,
    15000,
    16000,
    17000,
    18000,
    19000,
    20000,
    21000,
    22000,
    23000,
]

# ── Edit this list to choose which benchmarks to evaluate ──────────────────
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
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: inherit CUDA_VISIBLE_DEVICES or 7)")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Resolve each step to its revision; fail early if any step is unmapped.
    missing = [step for step in STEPS if step not in STEP_TO_REVISION]
    if missing:
        print(f"ERROR: no revision in STEP_TO_REVISION for step(s): "
              f"{', '.join(str(s) for s in missing)}")
        sys.exit(1)
    checkpoints = [(step, STEP_TO_REVISION[step]) for step in STEPS]

    print(f"Evaluating OLMo 2 1B baseline over {len(checkpoints)} checkpoint(s) "
          f"and {len(BENCHMARKS)} benchmark(s)")
    print(f"Checkpoints: {', '.join(str(step) for step, _ in checkpoints)}")
    print(f"Benchmarks:  {', '.join(t.split('::')[0] for t in BENCHMARKS)}")
    print()

    total = len(checkpoints) * len(BENCHMARKS)
    failed = []
    n = 0
    for step, revision in checkpoints:
        print(f"=== step {step} ({revision}) ===")
        for task in BENCHMARKS:
            n += 1
            task_name = task.split("::")[0]
            print(f"[{n}/{total}] step {step} — {task_name} ...", flush=True)
            result = subprocess.run(
                ["bash", EVAL_SCRIPT, str(step), revision, task],
                env=env,
            )
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                failed.append((step, task))
            else:
                print(f"  Done.")
        print()

    if failed:
        print(f"Completed with {len(failed)} failure(s):")
        for step, task in failed:
            print(f"  - step {step}: {task}")
        sys.exit(1)
    else:
        print(f"All {total} evaluation(s) completed successfully.")


if __name__ == "__main__":
    main()
