"""E4 -- decode: latency, throughput and peak memory across k, batch size and context.

Decode is where looping should hurt most, and where the previous experiment design was
weakest: it fixed a single operating point (B=4, S=4k). LT2 section 3.2 sweeps decode
across sequence length *and* batch size, and their central finding -- the decode cliff,
where a looped model loses half its throughput between 4k and 32k because its KV cache
grows per loop iteration -- only appears across that grid. A single point cannot show it.

Two arms at every depth, because "looping costs more than not looping" is not a useful
conclusion. ETD(k) and Dense(D) execute the same number of layers, so they have
identical FLOPs and identical KV; they differ only in stored weights and in how serial
the execution is. Hierarchical vs Flat section 5.6 reports a dense stack running 1.9x
faster than a recurrent one at equal depth, attributing it to independent layers
admitting parallelism that shared weights cannot. This sweep tests that directly.

Metric definitions follow `Metrics definition.md`:

    decode latency per token = t_decode / G     (divide by G, NOT by B)
    decode throughput        = B * G / t_decode

The G steps are sequential, so time divides by G; the B sequences run in parallel
within each step, so it does not divide by B.

Measurement protocol, from what E0 established:

  - CPU pinning is mandatory. Unpinned, latency stepped 22% within a single process
    from core migration; pinned it holds to 0.5%. The harness warns if unpinned.
  - The first timing block is discarded, not merely the first iterations: settling
    takes longer than a per-block warmup covers.
  - All k are measured inside one process, so any residual drift is common-mode and
    cancels out of the ratios that the report actually uses.
  - A canary runs at the start, middle and end; if it drifts, conditions changed.
  - OOM is a result, not an error: it defines the frontier, and larger batches at that
    (arm, k, S) are skipped rather than retried.

Usage:
    PYTHONPATH=$PWD ../.venv/bin/python scripts/q2/e4_decode.py                 # toy
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD taskset -c 0-47 \
        python scripts/q2/e4_decode.py --profile full
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    tiny=dict(ks=(1, 2, 3, 5), batches=(1, 2, 4), seqs=(128, 512), gen=16, warmup=8),
    full=dict(ks=(1, 2, 3, 4, 5, 8), batches=(1, 2, 4, 8, 16, 32), seqs=(1024, 4096, 16384), gen=128, warmup=16),
)


def decode_point(
    model,
    profile: str,
    device: str,
    mem: DeviceMemory,
    batch: int,
    seq: int,
    gen: int,
    warmup: int,
) -> Dict[str, Any]:
    """One (B, S) point: prefill, discard a warmup block, then time G decode steps."""
    vocab = PROFILES[profile]["vocab_size"]
    ids = torch.randint(0, vocab, (batch, seq), device=device)

    # Prefill populates the cache. Excluded from the decode timing entirely.
    with torch.no_grad():
        out = model(ids, use_cache=True, last_logits_only=True)
    cache = out.attn_key_values
    kv_after_prefill = kv_cache_bytes(cache)
    mem.sync()

    step_tok = torch.randint(0, vocab, (batch, 1), device=device)

    # Warmup block, discarded: E0 showed settling on a longer timescale than per-call
    # warmup covers. These steps still extend the cache, which is realistic.
    with torch.no_grad():
        for _ in range(warmup):
            out = model(step_tok, past_key_values=cache, use_cache=True, last_logits_only=True)
            cache = out.attn_key_values
    mem.sync()

    # Timed block. Peak counter reset here: decode memory grows monotonically as the
    # cache fills, so the high-water mark is at the final step.
    mem.reset_peak()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen):
            out = model(step_tok, past_key_values=cache, use_cache=True, last_logits_only=True)
            cache = out.attn_key_values
    mem.sync()
    t_decode = time.perf_counter() - start

    peak = mem.peak_allocated()
    kv_final = kv_cache_bytes(cache)
    del ids, step_tok, out, cache
    mem.empty_cache()

    return dict(
        t_decode_s=t_decode,
        latency_ms_per_token=1000.0 * t_decode / gen,   # divide by G, not by B
        throughput_tok_s=batch * gen / t_decode,        # B * G / t_decode
        peak_bytes=peak,
        kv_after_prefill_bytes=kv_after_prefill,
        kv_final_bytes=kv_final,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--arms", default="etd,dense", help="comma-separated: etd, dense")
    ap.add_argument("--ks", default=None)
    ap.add_argument("--batches", default=None)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for the randomised k order")
    args = ap.parse_args()

    d = DEFAULTS[args.profile]
    device = resolve_device(args.device)
    mem = DeviceMemory(device)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ks = [int(x) for x in args.ks.split(",")] if args.ks else list(d["ks"])
    batches = [int(x) for x in args.batches.split(",")] if args.batches else list(d["batches"])
    seqs = [int(x) for x in args.seqs.split(",")] if args.seqs else list(d["seqs"])
    gen = args.gen or d["gen"]
    warmup = args.warmup if args.warmup is not None else d["warmup"]
    meta = run_metadata(args.profile, device)

    # A restricted run measures a different thing from the full sweep and must not
    # land on top of it: the sweep backs the capacity table in the report, and
    # overwriting 190 rows with a handful would destroy that evidence silently.
    # A single operating point is named for it; a genuine sweep keeps the plain name.
    run_name = f"e4_decode_{args.profile}"
    if len(batches) == 1 and len(seqs) == 1:
        run_name += f"_B{batches[0]}_S{seqs[0]}"

    print("=" * 96)
    print(f"E4 DECODE SWEEP   profile={args.profile}  device={device}  dtype={meta['dtype']}")
    print(f"arms={arms}  k={ks}  B={batches}  S={seqs}  G={gen}  warmup={warmup}")
    print(f"affinity: {meta['host']['affinity']}")
    print("=" * 96)
    warn_if_unpinned()

    writer = IncrementalWriter(run_name)
    rows: List[Dict[str, Any]] = []
    canaries: List[Dict[str, Any]] = []

    def take_canary(label: str) -> None:
        value = canary(device, mem)
        canaries.append(dict(label=label, seconds=value))
        print(f"  [canary {label}] {value * 1000:.3f} ms")

    take_canary("start")

    # k order randomised so that any residual drift becomes noise rather than a
    # systematic bias on one k -- the k-ratios are what the report depends on.
    jobs = [(arm, k) for arm in arms for k in ks]
    random.Random(args.seed).shuffle(jobs)
    total_points = len(jobs) * len(seqs) * len(batches)
    done = 0
    t0 = time.time()

    for job_idx, (arm, k) in enumerate(jobs):
        depth = effective_depth(args.profile, "etd", k)
        try:
            model = build_model(args.profile, device, arm, k)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{arm} k={k} (D={depth}): model build failed: {type(exc).__name__}: {exc}")
            mem.empty_cache()
            continue
        weights = param_bytes(model)
        print(f"\n{arm:6s} k={k}  D={depth:>3d}  weights {gb(weights):.2f} GB")
        print(f"  {'S':>6s} {'B':>4s} {'lat ms/tok':>11s} {'tok/s':>10s} {'peak GB':>9s} {'KV GB':>8s} {'KV pred':>9s}")

        for seq in seqs:
            oom_batch: Optional[int] = None
            for batch in batches:
                if oom_batch is not None:
                    break  # larger batches at this S cannot fit either
                try:
                    r = decode_point(model, args.profile, device, mem, batch, seq, gen, warmup)
                except Exception as exc:  # noqa: BLE001
                    mem.empty_cache()
                    if is_oom(exc):
                        oom_batch = batch
                        row = dict(arm=arm, k=k, depth=depth, batch=batch, seq=seq, gen=gen,
                                   weight_bytes=weights, oom=True)
                        writer.add(row)
                        rows.append(row)
                        print(f"  {seq:>6d} {batch:>4d}   OOM -- frontier for this (arm, k, S)")
                        continue
                    print(f"  {seq:>6d} {batch:>4d}   FAILED: {type(exc).__name__}: {exc}")
                    continue

                kv_pred = analytical_kv_bytes(args.profile, depth, batch, seq + warmup + gen)
                row = dict(
                    arm=arm, k=k, depth=depth, batch=batch, seq=seq, gen=gen,
                    weight_bytes=weights, oom=False,
                    kv_predicted_bytes=kv_pred,
                    kv_measured_over_predicted=round(r["kv_final_bytes"] / kv_pred, 4) if kv_pred else 0,
                    **r,
                )
                writer.add(row)
                rows.append(row)
                done += 1
                print(
                    f"  {seq:>6d} {batch:>4d} {r['latency_ms_per_token']:>11.2f} "
                    f"{r['throughput_tok_s']:>10.1f} {gb(r['peak_bytes']):>9.2f} "
                    f"{gb(r['kv_final_bytes']):>8.2f} {r['kv_final_bytes'] / kv_pred:>9.3f}"
                )

        del model
        mem.empty_cache()
        if job_idx == len(jobs) // 2:
            take_canary("middle")

    take_canary("end")

    # Canary drift is the honest check on whether conditions held for the whole sweep.
    values = [c["seconds"] for c in canaries]
    drift = 100.0 * (max(values) - min(values)) / min(values) if values and min(values) > 0 else 0.0
    print("\n" + "=" * 96)
    print(f"canary drift across the sweep: {drift:.1f}%")
    if drift > 5:
        print("  WARNING: conditions changed while the sweep ran. Re-examine before trusting ratios.")
    print(f"points measured: {done}/{total_points}   elapsed: {(time.time() - t0) / 60:.1f} min")

    csv_path = write_csv(run_name, rows, meta=meta)
    print(f"  raw: {writer.path}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
