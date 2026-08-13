"""E1 -- static characterisation of every arm: what looping costs, and what it saves.

The cost side of Q2 (FLOPs, KV, latency, memory) is easy to over-report and the
benefit side is easy to leave out. Looping's entire selling point is that stored
parameters stay **constant** in k while effective depth grows, so a report that
tabulates only costs describes half the trade. Hierarchical vs Flat section 5.6 puts
params, latency and peak memory in a single row for exactly this reason.

This experiment establishes, per arm and per depth:

  - measured parameter counts, split into embeddings and layer stack
  - stored weight bytes
  - KV bytes per token (the term that scales with executed layers)
  - FLOP ratio versus the 16-layer baseline

Parameters are **measured, not derived**. Deriving them as D/16 would be wrong here:
`weight_tying: false` means ~411 M embedding parameters that do not scale with depth,
so a D/16 derivation overstates the dense arm by ~16% at k=5 -- flattering looping in
precisely the comparison the report turns on.

No timing, so this is insensitive to CPU pinning and to the shared machine. Weights
are allocated but not randomly initialised: byte counts and shapes are identical and
it saves a lot of time at 3.4 B parameters.

Usage:
    PYTHONPATH=$PWD ../.venv/bin/python scripts/q2/e1_static.py
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/q2/e1_static.py --profile full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.q2.bench_common import (  # noqa: E402
    PROFILES,
    DeviceMemory,
    analytical_kv_bytes,
    build_model,
    effective_depth,
    mb,
    param_bytes,
    param_count,
    resolve_device,
    run_metadata,
    write_csv,
    write_result,
)

DEFAULT_KS = (1, 2, 3, 4, 5, 8)


def embedding_params(model) -> int:
    """Parameters that do not scale with depth: token embedding + output projection."""
    total = 0
    for name, p in model.named_parameters():
        if "wte" in name or "ff_out" in name or "wpe" in name:
            total += p.numel()
    return total


def characterise(profile: str, device: str, arm: str, k: int, mem: DeviceMemory) -> Dict[str, Any]:
    depth = effective_depth(profile, "etd", k) if arm != "baseline" else PROFILES[profile]["n_layers"]
    model = build_model(profile, device, arm, k, init_params=False)

    total = param_count(model)
    emb = embedding_params(model)
    layers = total - emb
    stored_layers = model.config.n_layers

    row = dict(
        arm=arm,
        k=k if arm == "etd" else "-",
        stored_layers=stored_layers,
        executed_layers=depth,
        params_total=total,
        params_embedding=emb,
        params_layers=layers,
        params_per_layer=layers // stored_layers if stored_layers else 0,
        weight_bytes=param_bytes(model),
        kv_bytes_per_token=analytical_kv_bytes(profile, depth, 1, 1),
        flops_ratio=round(depth / PROFILES[profile]["n_layers"], 4),
    )
    del model
    mem.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    args = ap.parse_args()

    device = resolve_device(args.device)
    mem = DeviceMemory(device)
    ks = [int(x) for x in args.ks.split(",")]
    meta = run_metadata(args.profile, device)

    print("=" * 100)
    print(f"E1 STATIC CHARACTERISATION   profile={args.profile}  device={device}  dtype={meta['dtype']}")
    print("=" * 100)

    rows: List[Dict[str, Any]] = [characterise(args.profile, device, "baseline", 1, mem)]
    for k in ks:
        rows.append(characterise(args.profile, device, "etd", k, mem))
        rows.append(characterise(args.profile, device, "dense", k, mem))

    baseline = rows[0]
    base_params = baseline["params_total"]

    header = (
        f"{'arm':10s} {'k':>3s} {'stored':>7s} {'exec':>5s} "
        f"{'params':>10s} {'weights':>10s} {'KV/token':>10s} {'FLOP':>6s} {'param':>6s}"
    )
    print("\n" + header)
    print(f"{'':10s} {'':>3s} {'layers':>7s} {'D':>5s} {'':>10s} {'':>10s} {'':>10s} {'ratio':>6s} {'ratio':>6s}")
    print("-" * len(header))
    for r in rows:
        r["params_ratio"] = round(r["params_total"] / base_params, 4)
        print(
            f"{r['arm']:10s} {str(r['k']):>3s} {r['stored_layers']:>7d} {r['executed_layers']:>5d} "
            f"{r['params_total'] / 1e6:>9.1f}M {mb(r['weight_bytes']):>9.0f}M "
            f"{r['kv_bytes_per_token'] / 1024:>9.1f}K {r['flops_ratio']:>6.2f} {r['params_ratio']:>6.2f}"
        )

    # The benefit column: what looping saves against the same depth built densely.
    print("\n" + "=" * 100)
    print("What looping saves: stored weights, at equal executed depth")
    print("=" * 100)
    print(f"{'D':>4s} {'k':>3s} {'ETD weights':>14s} {'Dense weights':>16s} {'saved':>13s} {'saved %':>9s}")
    print("-" * 60)
    savings = []
    for k in ks:
        etd = next(r for r in rows if r["arm"] == "etd" and r["k"] == k)
        dense = next(r for r in rows if r["arm"] == "dense" and r["executed_layers"] == etd["executed_layers"])
        saved = dense["weight_bytes"] - etd["weight_bytes"]
        pct = 100.0 * saved / dense["weight_bytes"] if dense["weight_bytes"] else 0.0
        savings.append(dict(k=k, depth=etd["executed_layers"], saved_bytes=saved, saved_pct=round(pct, 2)))
        print(
            f"{etd['executed_layers']:>4d} {k:>3d} {mb(etd['weight_bytes']):>13.0f}M "
            f"{mb(dense['weight_bytes']):>15.0f}M {mb(saved):>12.0f}M {pct:>8.1f}%"
        )

    print("\n  Read alongside the cost tables: at equal executed depth the two arms have identical")
    print("  FLOPs and identical KV, and differ only in stored weights and in how serial they are.")
    print("  Weights amortise across a batch; KV does not. So this saving is largest at batch 1 and")
    print("  shrinks as batch and context grow -- which E4 and E5 measure.")

    payload = dict(metadata=meta, rows=rows, savings=savings)
    raw = write_result(f"e1_static_{args.profile}", payload)
    csv_path = write_csv(f"e1_static_{args.profile}", rows, meta=meta)
    print(f"\n  raw: {raw}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
