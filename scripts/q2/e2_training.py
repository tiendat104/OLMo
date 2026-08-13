"""E2 -- training: peak memory and step time as a function of the loop count k.

One variable is swept, k; batch size and sequence length are fixed across the whole
sweep, chosen so the most demanding configuration (k=8) fits. Larger is better up to
that limit: activations are the only term that scales with k, so a bigger batch makes
the term under study a larger share of the total and the effect easier to resolve. If
k=8 runs out of memory, halve the batch and re-run the whole sweep -- rows measured at
different batch sizes are not comparable.

Why the peak counter runs across the WHOLE step. `Metrics definition.md` expects the
peak in early backward, when nearly all activations are still live and the first
gradients have been allocated -- but it also notes the peak can fall at the loss
instead when full logits are materialised, which training cannot avoid. The E0
preflight showed the peak sitting 2.6 GB above every sampled phase boundary, so
sampling cannot locate it. The counter is therefore reset once at the start of the
step and read at the end, capturing the high-water mark wherever it occurs. Phase
samples are recorded alongside as supporting detail, and because the peak's *location*
may shift with k: at low k the logits and their gradient are a large share of the
footprint, while at high k retained activations dominate.

What is expected to scale. Of the five terms in

    M_train = M_weights + M_opt + M_grad + A_stored(D) + W_bwd

only A_stored carries k. Weights are the 16 stored layers regardless of k; gradients
and optimizer state are allocated one per *parameter*, not per use, so the k passes
through a looped layer accumulate into the same buffer; backward workspace is
transient. Every loop iteration produces its own activation stack and all of them must
be retained until backward consumes them.

Optimizer. Plain AdamW over bf16 parameters, so the moments are bf16 -- about 2x the
parameter bytes. A production mixed-precision recipe (fp32 master weights and fp32
moments) would add a fixed ~11 GB to every row without changing the comparison, since
optimizer state is constant in k. The simple setup is used deliberately: adding a
large constant to every row would make looping's effect look smaller as a proportion
while leaving it unchanged, and would consume batch-size headroom that is better spent
on the term actually under study.

Gradient checkpointing is off. With recompute enabled, activations are discarded and
rebuilt during backward, so the one term that scales with k would stop doing so and
the effect under measurement would disappear.

Usage:
    PYTHONPATH=$PWD ../.venv/bin/python scripts/q2/e2_training.py
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD taskset -c 0-47 \
        python scripts/q2/e2_training.py --profile full
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
    build_model,
    canary,
    effective_depth,
    gb,
    grad_bytes,
    is_oom,
    logits_bytes,
    optimizer_state_bytes,
    param_bytes,
    resolve_device,
    run_metadata,
    warn_if_unpinned,
    write_csv,
)

DEFAULTS = dict(
    tiny=dict(ks=(1, 2, 3, 4, 5, 8), batch=2, seq=256, warmup=3, measured=10),
    full=dict(ks=(1, 2, 3, 4, 5, 8), batch=4, seq=2048, warmup=5, measured=20),
)


def training_point(
    model, profile: str, device: str, mem: DeviceMemory, batch: int, seq: int, warmup: int, measured: int
) -> Dict[str, Any]:
    """Warmup steps discarded, then `measured` timed forward/backward/update steps."""
    vocab = PROFILES[profile]["vocab_size"]
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def step():
        logits = model(ids).logits
        loss = logits.float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    mem.sync()

    # Component sizes, from a step left in flight so the gradients still exist.
    logits = model(ids).logits
    logits.float().mean().backward()
    mem.sync()
    components = dict(
        weight_bytes=param_bytes(model),
        grad_bytes=grad_bytes(model),
        logits_bytes=logits_bytes(profile, batch, seq),
    )
    opt.step()
    components["optimizer_bytes"] = optimizer_state_bytes(opt)
    opt.zero_grad(set_to_none=True)
    del logits
    mem.sync()

    # Phase samples: supporting detail only. The peak is not assumed to coincide with
    # any of them -- E0 measured it 2.6 GB above every sampled boundary.
    phases: Dict[str, int] = {}
    mem.reset_peak()
    phases["step_start"] = mem.allocated()
    out = model(ids).logits
    mem.sync()
    phases["after_forward"] = mem.allocated()
    loss = out.float().mean()
    mem.sync()
    phases["after_loss"] = mem.allocated()
    loss.backward()
    mem.sync()
    phases["after_backward"] = mem.allocated()
    opt.step()
    opt.zero_grad(set_to_none=True)
    mem.sync()
    phases["after_update"] = mem.allocated()
    phase_peak = mem.peak_allocated()
    del out, loss

    # Phase timing: forward, backward and optimizer separately, with a device sync
    # between phases. The syncs prevent any overlap between phases, so the three
    # parts need not sum exactly to the whole-step time below -- they are for
    # attribution, not a replacement for it. Forward includes the loss.
    #
    # Why this exists: the whole-step fit has a fixed cost (intercept) that cannot
    # be attributed from the fit alone. The optimizer phase is measured directly
    # here and should be constant in k; whatever fixed cost remains inside
    # forward+backward belongs to the depth-independent ends of the network
    # (embedding, final norm, vocabulary projection, loss).
    phase_samples = {"forward_ms": [], "backward_ms": [], "optimizer_ms": []}
    for _ in range(max(5, measured // 2)):
        mem.sync()
        t0 = time.perf_counter()
        logits = model(ids).logits
        loss = logits.float().mean()
        mem.sync()
        t1 = time.perf_counter()
        loss.backward()
        mem.sync()
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        mem.sync()
        t3 = time.perf_counter()
        phase_samples["forward_ms"].append(1000.0 * (t1 - t0))
        phase_samples["backward_ms"].append(1000.0 * (t2 - t1))
        phase_samples["optimizer_ms"].append(1000.0 * (t3 - t2))
        del logits, loss
    phase_ms = {name: sorted(v)[len(v) // 2] for name, v in phase_samples.items()}

    # Timed block, with the peak counter running across whole steps.
    mem.reset_peak()
    mem.sync()
    start = time.perf_counter()
    for _ in range(measured):
        step()
    mem.sync()
    elapsed = time.perf_counter() - start
    peak = mem.peak_allocated()

    del ids, opt
    mem.empty_cache()

    return dict(
        step_time_ms=1000.0 * elapsed / measured,
        peak_bytes=max(peak, phase_peak),
        phases=phases,
        steps=measured,
        **phase_ms,
        **components,
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

    print("=" * 96)
    print(f"E2 TRAINING   profile={args.profile}  device={device}  dtype={meta['dtype']}")
    print(f"B={batch}  S={seq}  warmup={warmup}  measured steps={measured}  k={ks}")
    print("gradient checkpointing OFF; AdamW over bf16 parameters")
    print(f"affinity: {meta['host']['affinity']}")
    print("=" * 96)
    warn_if_unpinned()

    writer = IncrementalWriter(f"e2_training_{args.profile}")
    rows: List[Dict[str, Any]] = []
    c_start = canary(device, mem)
    print(f"  [canary start] {c_start * 1000:.3f} ms")

    order = list(ks)
    random.Random(args.seed).shuffle(order)
    print(f"  measurement order: {order}\n")
    print(f"  {'k':>2} {'D':>3} {'step ms':>10} {'fwd ms':>8} {'bwd ms':>8} {'opt ms':>8} "
          f"{'peak GB':>8} {'residual':>9}")

    for k in order:
        depth = effective_depth(args.profile, "etd", k)
        try:
            model = build_model(args.profile, device, "etd", k)
            r = training_point(model, args.profile, device, mem, batch, seq, warmup, measured)
        except Exception as exc:  # noqa: BLE001
            mem.empty_cache()
            status = "OOM" if is_oom(exc) else f"FAILED: {type(exc).__name__}: {exc}"
            print(f"  {k:>2} {depth:>3}   {status}")
            row = dict(k=k, depth=depth, batch=batch, seq=seq, oom=is_oom(exc))
            writer.add(row)
            rows.append(row)
            continue

        # Everything the peak is not accounted for by the constant terms. This is
        # A_stored + W_bwd, which cannot be separated further.
        residual = r["peak_bytes"] - (
            r["weight_bytes"] + r["grad_bytes"] + r["optimizer_bytes"] + r["logits_bytes"]
        )
        row = dict(k=k, depth=depth, batch=batch, seq=seq, oom=False, residual_bytes=residual, **r)
        writer.add(row)
        rows.append(row)
        print(
            f"  {k:>2} {depth:>3} {r['step_time_ms']:>10.1f} {r['forward_ms']:>8.1f} "
            f"{r['backward_ms']:>8.1f} {r['optimizer_ms']:>8.1f} "
            f"{gb(r['peak_bytes']):>8.2f} {gb(residual):>9.2f}"
        )
        del model
        mem.empty_cache()

    c_end = canary(device, mem)
    drift = 100.0 * abs(c_end - c_start) / c_start if c_start else 0.0
    print(f"\n  [canary end] {c_end * 1000:.3f} ms   drift {drift:.1f}%")

    done = sorted([r for r in rows if not r.get("oom") and "step_time_ms" in r], key=lambda r: r["k"])
    if done:
        base = done[0]

        def ratio(x, y):
            return f"{x / y:>8.2f}" if y else f"{'-':>8}"

        print("\n" + "=" * 96)
        print(f"  {'k':>2} {'D':>3} {'peak GB':>8} {'vs base':>8} {'step ms':>10} {'vs base':>8} {'FLOP':>6}")
        for r in done:
            print(
                f"  {r['k']:>2} {r['depth']:>3} {gb(r['peak_bytes']):>8.2f} "
                f"{ratio(r['peak_bytes'], base['peak_bytes'])} {r['step_time_ms']:>10.1f} "
                f"{ratio(r['step_time_ms'], base['step_time_ms'])} {r['depth'] / base['depth']:>6.2f}"
            )

        print("\n  Where the peak falls (allocated GB at each phase boundary; the peak counter runs")
        print("  across the whole step and is not assumed to coincide with any of them):")
        print(f"  {'k':>2} " + " ".join(f"{p:>15}" for p in
              ("step_start", "after_forward", "after_loss", "after_backward", "after_update")) + f" {'PEAK':>9}")
        for r in done:
            ph = r["phases"]
            print(f"  {r['k']:>2} " + " ".join(f"{gb(ph.get(p, 0)):>15.2f}" for p in
                  ("step_start", "after_forward", "after_loss", "after_backward", "after_update"))
                  + f" {gb(r['peak_bytes']):>9.2f}")

    csv_path = write_csv(f"e2_training_{args.profile}", [{k: v for k, v in r.items() if k != "phases"} for r in rows])
    print(f"\n  raw: {writer.path}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
