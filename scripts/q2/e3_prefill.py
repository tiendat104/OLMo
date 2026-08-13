"""E3 -- prefill: latency and peak memory as a function of the loop count k.

Prefill is the single forward pass that consumes the prompt and populates the KV
cache. One variable is swept, k; everything else is held fixed at the same operating
point used for decode (B=8, S=4096), so the two sections of the report describe the
same request shape and can be read side by side.

Differences from the decode measurement, and why:

  - Prefill is one forward pass, not a sequence of steps. Latency is therefore
    measured by repeating independent passes and taking the median, rather than by
    timing a run of sequential steps and dividing. Each pass repeats the identical
    computation; the repetition is for measurement stability only.
  - Peak memory is reset before EVERY measured pass and the maximum taken across
    them. In decode the cache grows monotonically, so one reset and a final read is
    correct; here each pass is independent, so per-pass peaks are the right thing.
  - `last_logits_only=True`. With it off, prefill materialises a B x S x vocab logits
    tensor -- 6.6 GB at B=8, S=4096 -- which is constant in k and would dominate peak
    memory, diluting the k-dependence being measured. Computing logits only for the
    final position is also what a serving system does, since generation needs no more.

Protocol as established in E0: CPU pinning (unpinned, timings drifted up to 22%
within a single process from core migration), first timing block discarded, all k in
one process in randomised order, canary at start and end.

Usage:
    PYTHONPATH=$PWD ../.venv/bin/python scripts/q2/e3_prefill.py
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD taskset -c 0-47 \
        python scripts/q2/e3_prefill.py --profile full
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.q2.bench_common import (  # noqa: E402
    PROFILES,
    DeviceMemory,
    IncrementalWriter,
    analytical_kv_bytes,
    build_model,
    canary,
    effective_depth,
    gb,
    is_oom,
    kv_cache_bytes,
    param_bytes,
    resolve_device,
    run_metadata,
    warn_if_unpinned,
    write_csv,
)

DEFAULTS = dict(
    tiny=dict(ks=(1, 2, 3, 4, 5, 8), batch=2, seq=512, warmup=3, measured=10),
    full=dict(ks=(1, 2, 3, 4, 5, 8), batch=8, seq=4096, warmup=5, measured=20),
)


def prefill_point(
    model, profile: str, device: str, mem: DeviceMemory, batch: int, seq: int, warmup: int, measured: int
) -> Dict[str, Any]:
    """Warmup passes discarded, then `measured` independent timed passes."""
    vocab = PROFILES[profile]["vocab_size"]
    ids = torch.randint(0, vocab, (batch, seq), device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model(ids, use_cache=True, last_logits_only=True)
    mem.sync()

    samples: List[float] = []
    peaks: List[int] = []
    kv_bytes = 0
    for _ in range(measured):
        mem.reset_peak()          # per pass: each prefill is independent
        mem.sync()
        start = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True, last_logits_only=True)
        mem.sync()
        samples.append(time.perf_counter() - start)
        peaks.append(mem.peak_allocated())
        kv_bytes = kv_cache_bytes(out.attn_key_values)
        del out

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    del ids
    mem.empty_cache()

    return dict(
        latency_ms=1000.0 * median,
        latency_ms_mean=1000.0 * sum(samples) / n,
        latency_ms_min=1000.0 * samples[0],
        latency_ms_max=1000.0 * samples[-1],
        spread_pct=100.0 * (samples[-1] - samples[0]) / median if median else 0.0,
        peak_bytes=max(peaks),
        peak_bytes_min=min(peaks),
        kv_bytes=kv_bytes,
        throughput_tok_s=batch * seq / median,
        samples=n,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ks", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--measured", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = DEFAULTS[args.profile]
    device = resolve_device(args.device)
    mem = DeviceMemory(device)
    ks = [int(x) for x in args.ks.split(",")] if args.ks else list(d["ks"])
    batch = args.batch or d["batch"]
    seq = args.seq or d["seq"]
    warmup = args.warmup if args.warmup is not None else d["warmup"]
    measured = args.measured or d["measured"]
    meta = run_metadata(args.profile, device)

    print("=" * 92)
    print(f"E3 PREFILL   profile={args.profile}  device={device}  dtype={meta['dtype']}")
    print(f"B={batch}  S={seq}  warmup={warmup}  measured passes={measured}  k={ks}")
    print(f"affinity: {meta['host']['affinity']}")
    print("=" * 92)
    warn_if_unpinned()

    run_name = f"e3_prefill_{args.profile}_B{batch}_S{seq}"
    writer = IncrementalWriter(run_name)
    rows: List[Dict[str, Any]] = []
    c_start = canary(device, mem)
    print(f"  [canary start] {c_start * 1000:.3f} ms")

    # Randomised order so any drift over the run does not bias one particular k.
    order = list(ks)
    random.Random(args.seed).shuffle(order)
    print(f"  measurement order: {order}\n")
    print(f"  {'k':>2} {'D':>3} {'latency ms':>11} {'spread':>7} {'peak GB':>8} {'KV GB':>7} {'KV pred':>8}")

    for k in order:
        depth = effective_depth(args.profile, "etd", k)
        try:
            model = build_model(args.profile, device, "etd", k)
        except Exception as exc:  # noqa: BLE001
            print(f"  {k:>2} {depth:>3}   model build failed: {type(exc).__name__}: {exc}")
            mem.empty_cache()
            continue
        weights = param_bytes(model)
        try:
            r = prefill_point(model, args.profile, device, mem, batch, seq, warmup, measured)
        except Exception as exc:  # noqa: BLE001
            del model
            mem.empty_cache()
            status = "OOM" if is_oom(exc) else f"FAILED: {type(exc).__name__}"
            print(f"  {k:>2} {depth:>3}   {status}")
            row = dict(k=k, depth=depth, batch=batch, seq=seq, weight_bytes=weights, oom=is_oom(exc))
            writer.add(row)
            rows.append(row)
            continue

        kv_pred = analytical_kv_bytes(args.profile, depth, batch, seq)
        row = dict(
            k=k, depth=depth, batch=batch, seq=seq, weight_bytes=weights, oom=False,
            kv_predicted_bytes=kv_pred,
            kv_measured_over_predicted=round(r["kv_bytes"] / kv_pred, 4) if kv_pred else 0,
            **r,
        )
        writer.add(row)
        rows.append(row)
        print(
            f"  {k:>2} {depth:>3} {r['latency_ms']:>11.2f} {r['spread_pct']:>6.1f}% "
            f"{gb(r['peak_bytes']):>8.2f} {gb(r['kv_bytes']):>7.2f} "
            f"{r['kv_bytes'] / kv_pred if kv_pred else 0:>8.3f}"
        )
        del model
        mem.empty_cache()

    c_end = canary(device, mem)
    drift = 100.0 * abs(c_end - c_start) / c_start if c_start else 0.0
    print(f"\n  [canary end] {c_end * 1000:.3f} ms   drift {drift:.1f}%")
    if drift > 5:
        print("  WARNING: conditions changed while the sweep ran; re-examine before trusting ratios.")

    # Report in k order with ratios against the baseline, as the report table needs.
    done = sorted([r for r in rows if not r.get("oom") and "latency_ms" in r], key=lambda r: r["k"])
    if done:
        base = done[0]
        print("\n" + "=" * 92)
        print(f"  {'k':>2} {'D':>3} {'latency ms':>11} {'vs base':>8} {'peak GB':>8} {'vs base':>8} "
              f"{'KV GB':>7} {'vs base':>8} {'FLOP':>6}")
        def ratio(x, y):
            # Memory counters read 0 on CPU (no device memory API), so guard the divide.
            return f"{x / y:>8.2f}" if y else f"{'-':>8}"

        for r in done:
            print(
                f"  {r['k']:>2} {r['depth']:>3} {r['latency_ms']:>11.2f} "
                f"{ratio(r['latency_ms'], base['latency_ms'])} {gb(r['peak_bytes']):>8.2f} "
                f"{ratio(r['peak_bytes'], base['peak_bytes'])} {gb(r['kv_bytes']):>7.2f} "
                f"{ratio(r['kv_bytes'], base['kv_bytes'])} {r['depth'] / base['depth']:>6.2f}"
            )

    csv_path = write_csv(run_name, rows)
    print(f"\n  raw: {writer.path}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
