"""Compare a cached evaluation run against the stored replication run, example by example.

A subset run cannot be compared to a published full-set score directly. But the stored
replication run recorded per-example predictions, so its accuracy can be recomputed on
exactly the subset that was just evaluated -- which makes even a 200-example run a
rigorous comparison rather than an approximation.

Reports three things, in increasing order of strength:

  1. accuracy on the shared subset, cached run vs stored run
  2. per-example metric agreement (how many examples scored the same either way)
  3. byte-identical generations (the strongest signal, and the one that distinguishes
     bf16 argmax flips from a cache feeding wrong state)

Usage:
    python scripts/etd_kv_cache_eval_vs_reference.py
    python scripts/etd_kv_cache_eval_vs_reference.py --task gsm8k --show 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLICATION = Path(
    "/home/n84449292/tiendat/projects/Loop_Transformer_project/Work/replication/rep_ETD/OLMo"
)


def find_predictions(run_dir: Path) -> Optional[Path]:
    for pattern in ("**/*predictions*.jsonl", "**/predictions.jsonl"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_requests(run_dir: Path) -> Optional[Path]:
    hits = sorted(run_dir.glob("**/*requests*.jsonl"))
    return hits[0] if hits else None


def question_by_doc_id(run_dir: Path) -> Dict[str, str]:
    """Map this run's local doc_id to the question text.

    Necessary because olmes --limit SUBSAMPLES rather than taking the first N, and
    renumbers doc_id within the subset: doc_id 0 of a limited run is not doc_id 0 of the
    full run. Matching on ids across runs therefore compares unrelated questions.  The
    question text is the only stable key.
    """
    path = find_requests(run_dir)
    mapping: Dict[str, str] = {}
    if not path:
        return mapping
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc = rec.get("doc") or {}
            q = doc.get("question") or doc.get("query") or doc.get("context")
            key = rec.get("doc_id", rec.get("idx"))
            if q is not None and key is not None:
                mapping[str(key)] = " ".join(str(q).split())
    return mapping


def load(path: Path, questions: Dict[str, str]) -> Dict[str, dict]:
    """Index predictions by question text where possible, falling back to doc_id."""
    out: Dict[str, dict] = {}
    for i, line in enumerate(path.open()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        doc_id = str(rec.get("doc_id", rec.get("idx", i)))
        doc = rec.get("doc") or {}
        q = doc.get("question") or doc.get("query") or questions.get(doc_id)
        key = " ".join(str(q).split()) if q else f"__docid__{doc_id}"
        out[key] = rec
    return out


def score_of(rec: dict) -> Optional[float]:
    """Pull the per-example primary metric out, tolerating schema differences."""
    for container in (rec.get("metrics"), rec):
        if isinstance(container, dict):
            for key in ("exact_match", "acc", "accuracy", "em", "f1", "primary_score"):
                val = container.get(key)
                if isinstance(val, (int, float, bool)):
                    return float(val)
    return None


def continuation(rec: dict) -> str:
    for key in ("continuation", "prediction", "model_output", "output", "generated_text"):
        val = rec.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val:
            if isinstance(val[0], str):
                return val[0]
            if isinstance(val[0], dict):
                for inner in ("continuation", "text", "prediction", "model_output"):
                    if inner in val[0]:
                        return str(val[0][inner])
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--mode", default="on", choices=("on", "off"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--step", default="23852")
    ap.add_argument("--run-name", default="running/ETD_k2_npu")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else (
        REPO_ROOT / "kvcache_check" / "eval_results" / args.run_name / f"step{args.step}"
        / f"{args.task}-cache-{args.mode}"
    )
    ref_dir = Path(args.reference) if args.reference else (
        DEFAULT_REPLICATION / "eval_results" / args.run_name / f"step{args.step}" / args.task
    )

    print("=" * 78)
    print(f"Cached run vs stored replication run   task={args.task}  cache={args.mode}")
    print(f"  run       : {run_dir}")
    print(f"  reference : {ref_dir}")
    print("=" * 78)

    run_p, ref_p = find_predictions(run_dir), find_predictions(ref_dir)
    if not run_p:
        print(f"\nNo predictions found under {run_dir}")
        return 1
    if not ref_p:
        print(f"\nNo stored predictions found under {ref_dir}")
        print("The reference run may have kept only metrics.json; compare aggregates by hand.")
        return 1
    print(f"\n  run predictions       : {run_p.name}")
    print(f"  reference predictions : {ref_p.name}")

    run_recs = load(run_p, question_by_doc_id(run_dir))
    ref_recs = load(ref_p, question_by_doc_id(ref_dir))

    by_docid = sum(1 for k in run_recs if k.startswith("__docid__"))
    if by_docid:
        print(f"\n  WARNING: {by_docid} run records had no question text and fell back to doc_id.")
        print("           olmes --limit subsamples and renumbers doc_id, so id-based matching")
        print("           can compare unrelated questions. Treat those rows with suspicion.")

    shared = sorted(set(run_recs) & set(ref_recs))
    print(f"  examples: run {len(run_recs)}, reference {len(ref_recs)}, matched by question {len(shared)}")
    if not shared:
        print("\nNo questions in common -- cannot compare.")
        print("If the run has only a couple of examples, it was probably skipped by olmes")
        print("because its output directory already held results. Delete it and re-run.")
        return 1
    if len(shared) < len(run_recs):
        print(f"  note: {len(run_recs) - len(shared)} run examples had no match in the reference")

    scored = [(k, score_of(run_recs[k]), score_of(ref_recs[k])) for k in shared]
    usable = [(k, a, b) for k, a, b in scored if a is not None and b is not None]

    print()
    if usable:
        run_acc = 100.0 * sum(a for _, a, _ in usable) / len(usable)
        ref_acc = 100.0 * sum(b for _, _, b in usable) / len(usable)
        agree = sum(1 for _, a, b in usable if a == b)
        print(f"  accuracy on the shared {len(usable)} examples")
        print(f"    cached run : {run_acc:6.2f}%")
        print(f"    stored run : {ref_acc:6.2f}%   (recomputed on this same subset)")
        print(f"    difference : {run_acc - ref_acc:+.2f} points")
        print(f"  per-example metric agreement: {agree}/{len(usable)} ({100.0 * agree / len(usable):.1f}%)")
    else:
        print("  Could not locate a per-example metric field in one of the files.")
        print(f"  Keys present in a run record: {sorted(run_recs[shared[0]])[:12]}")

    identical, differing = 0, []
    for k in shared:
        a, b = continuation(run_recs[k]), continuation(ref_recs[k])
        if a and a == b:
            identical += 1
        elif a or b:
            differing.append((k, a, b))
    if identical or differing:
        total = identical + len(differing)
        print(f"\n  byte-identical generations: {identical}/{total} ({100.0 * identical / total:.1f}%)")
        for k, a, b in differing[: args.show]:
            pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"\n    --- example {k}, diverges at char {pos} ---")
            print(f"      run    : ...{a[max(0, pos - 50):pos + 70]!r}")
            print(f"      ref    : ...{b[max(0, pos - 50):pos + 70]!r}")

    print("\n" + "=" * 78)
    if usable:
        delta = abs(run_acc - ref_acc)
        if delta <= 1.0:
            print("VERDICT: accuracy matches within 1 point -- consistent with correct caching.")
            print("         Residual differences are bf16 argmax flips on close calls.")
        elif delta <= 2.5:
            print("VERDICT: BORDERLINE. Inspect the diverging examples above before accepting.")
        else:
            print("VERDICT: FAIL -- too large to attribute to floating-point nondeterminism.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
