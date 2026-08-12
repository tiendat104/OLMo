"""Compare cache-off and cache-on evaluation runs example by example.

Aggregate accuracy is a weak signal: two runs can score identically while producing
completely different text, or differ by a point purely because one borderline example
flipped.  What actually establishes the KV-cache fix is how many *individual
generations* are byte-identical, and where the rest first diverge.

Expected outcome if the fix is correct: the large majority of generations identical,
the remainder diverging some way into the text rather than at the first token.  bf16
caching perturbs logits by ~1e-2, which flips argmax whenever the top-2 gap is smaller
than that; with trained weights that is uncommon but not rare, and once a single token
differs everything after it is a different rollout.

A near-zero identical rate, or divergence at token 0, means the cache is feeding the
model wrong state -- not floating-point noise.

Usage:
    python scripts/etd_kv_cache_eval_diff.py
    python scripts/etd_kv_cache_eval_diff.py --task gsm8k --show 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_ROOT = REPO_ROOT / "kvcache_check" / "eval_results"


def find_predictions(run_dir: Path) -> Optional[Path]:
    """Locate the predictions file olmes wrote, whatever it chose to call it."""
    for pattern in ("**/*predictions*.jsonl", "**/*-predictions.jsonl", "**/predictions.jsonl"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load(path: Path) -> Dict[str, dict]:
    """Index records by whatever stable per-example id is present."""
    records: Dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("doc_id", rec.get("native_id", rec.get("idx", len(records))))
            records[str(key)] = rec
    return records


def continuation(rec: dict) -> str:
    """Pull the generated text out, tolerating differing olmes schema versions."""
    for key in ("continuation", "prediction", "model_output", "output", "generated_text"):
        val = rec.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0]
        if isinstance(val, list) and val and isinstance(val[0], dict):
            for inner in ("continuation", "text", "prediction"):
                if inner in val[0]:
                    return str(val[0][inner])
    preds = rec.get("predictions")
    if isinstance(preds, list) and preds and isinstance(preds[0], dict):
        for inner in ("continuation", "text", "prediction", "model_output"):
            if inner in preds[0]:
                return str(preds[0][inner])
    return ""


def first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def metrics_of(run_dir: Path) -> Optional[dict]:
    hits = sorted(run_dir.glob("**/metrics.json"))
    if not hits:
        return None
    with hits[0].open() as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) and data else data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, help="task name, e.g. gsm8k (default: auto-detect)")
    ap.add_argument("--show", type=int, default=3, help="how many differing examples to print")
    args = ap.parse_args()

    pairs: List[tuple] = []
    for off_dir in sorted(CHECK_ROOT.glob("**/*-cache-off")):
        on_dir = off_dir.with_name(off_dir.name.replace("-cache-off", "-cache-on"))
        task = off_dir.name.replace("-cache-off", "")
        if args.task and task != args.task:
            continue
        if on_dir.exists():
            pairs.append((task, off_dir, on_dir))

    if not pairs:
        print(f"No off/on pairs found under {CHECK_ROOT}")
        print("Run both modes first:")
        print("  bash scripts/etd_kv_cache_eval_check.sh off")
        print("  bash scripts/etd_kv_cache_eval_check.sh on")
        return 1

    overall_ok = True
    for task, off_dir, on_dir in pairs:
        print("=" * 78)
        print(f"TASK: {task}")
        print("=" * 78)

        for label, d in (("cache off", off_dir), ("cache on", on_dir)):
            m = metrics_of(d)
            scores = {k: v for k, v in (m or {}).items() if isinstance(v, (int, float))}
            print(f"  {label:10s} metrics: {json.dumps(scores)[:180] if scores else '(none found)'}")

        off_p, on_p = find_predictions(off_dir), find_predictions(on_dir)
        if not off_p or not on_p:
            print(f"\n  Could not locate prediction files (off={bool(off_p)} on={bool(on_p)}).")
            print(f"  Looked under {off_dir} and {on_dir}.")
            overall_ok = False
            continue

        off_recs, on_recs = load(off_p), load(on_p)
        shared = sorted(set(off_recs) & set(on_recs), key=lambda s: (len(s), s))
        if not shared:
            print("\n  No overlapping example ids between the two runs.")
            overall_ok = False
            continue

        identical, differing = 0, []
        for key in shared:
            a, b = continuation(off_recs[key]), continuation(on_recs[key])
            if a == b:
                identical += 1
            else:
                differing.append((key, first_divergence(a, b), a, b))

        pct = 100.0 * identical / len(shared)
        print(f"\n  compared        : {len(shared)} examples")
        print(f"  byte-identical  : {identical} ({pct:.1f}%)")
        print(f"  differing       : {len(differing)}")

        if differing:
            positions = [d[1] for d in differing]
            at_start = sum(1 for p in positions if p == 0)
            print(f"  first divergence: min {min(positions)}, median {sorted(positions)[len(positions)//2]}, max {max(positions)} chars in")
            print(f"  diverging at char 0: {at_start}")
            for key, pos, a, b in differing[: args.show]:
                print(f"\n  --- example {key}, diverges at char {pos} ---")
                lo = max(0, pos - 60)
                print(f"    off: ...{a[lo:pos + 80]!r}")
                print(f"    on : ...{b[lo:pos + 80]!r}")

        print()
        if pct >= 90.0:
            print("  VERDICT: consistent with correct caching (bf16 argmax flips on close calls).")
        elif pct >= 50.0:
            print("  VERDICT: UNCLEAR -- more divergence than float noise alone would explain.")
            print("           Inspect the examples above; check whether divergence clusters at char 0.")
            overall_ok = False
        else:
            print("  VERDICT: FAIL -- the cache is almost certainly feeding wrong state.")
            overall_ok = False
        print()

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
