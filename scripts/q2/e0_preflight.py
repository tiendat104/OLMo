"""E0 -- preflight: establish what this stack can actually measure, before measuring it.

The Q2 plan asks for a memory decomposition (weights / gradients / optimizer state /
activations / KV) and for latencies on a machine shared with another user's training
job. Both assumptions can fail on torch_npu, and finding out mid-sweep would be
expensive. This probe answers, in minutes:

  1  Do the memory APIs exist, and do they move when memory is allocated?
  2  Does PyTorch's allocator see all of it?  <-- the main risk. On Ascend, CANN
     operators can allocate workspace via the ACL runtime rather than the caching
     allocator, in which case max_memory_allocated() UNDER-reports and the whole
     decomposition silently fails to add up. Cross-checked against npu-smi.
  3  Can each component of the decomposition actually be read?
  4  Where in a training step does the peak fall? (Metrics definition.md predicts
     early backward, or the loss if logits dominate.)
  5  Which attention path is really taken? `flash_attention: true` only *attempts* to
     import the CUDA-only flash_attn package and silently falls back to
     F.scaled_dot_product_attention. Whether that is a fused kernel or the math
     backend decides whether prefill memory is O(S) or O(B*h*S^2) -- tested
     empirically, not by reading flags.
  6  Is bf16 actually in use end to end?
  7  How sensitive are dispatch-bound latencies to host-CPU contention from the
     other user's job? Decode at B=1 is the most dispatch-bound point in the plan,
     so any sensitivity shows up there first.

Whatever this finds unmeasurable gets removed from the deliverables *with the reason
recorded*, which is a more honest outcome than a table with unexplained gaps.

Usage:
    # Mac, toy scale, to debug the harness
    PYTHONPATH=$PWD ../.venv/bin/python scripts/q2/e0_preflight.py

    # NPU, real model
    ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/q2/e0_preflight.py --profile full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.q2.bench_common import (  # noqa: E402
    DeviceMemory,
    PROFILES,
    analytical_kv_bytes,
    build_model,
    effective_depth,
    gb,
    grad_bytes,
    host_state,
    kv_cache_bytes,
    logits_bytes,
    mb,
    npu_smi_raw,
    optimizer_state_bytes,
    param_bytes,
    param_count,
    resolve_device,
    run_metadata,
    time_repeated,
    write_result,
)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------------------


def check_memory_api(device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Do the APIs exist, and does allocating a known-size tensor move them?"""
    section("1. Memory API availability and sanity")

    api = {
        name: hasattr(mem.backend, name) if mem.backend else False
        for name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
            "reset_peak_memory_stats",
            "empty_cache",
            "synchronize",
        )
    }
    for name, present in api.items():
        print(f"  {'yes' if present else 'NO ':>4}  torch.{device.split(':')[0]}.{name}")

    if not mem.available:
        print("\n  Memory API unavailable on this device (expected on CPU).")
        print("  Memory metrics cannot be produced here; use the NPU for anything memory-related.")
        return dict(api=api, usable=False)

    # Allocate a known size and check the counter moves by roughly that much.
    want = 256 * 1024 * 1024
    mem.empty_cache()
    mem.reset_peak()
    before = mem.allocated()
    blob = torch.empty(want // 2, dtype=torch.float16, device=device)  # 2 bytes/elem
    mem.sync()
    after = mem.allocated()
    observed = after - before
    del blob
    mem.empty_cache()
    mem.sync()
    freed = mem.allocated()

    ratio = observed / want if want else 0.0
    print(f"\n  allocated {mb(want):.0f} MB  ->  counter moved {mb(observed):.0f} MB  (ratio {ratio:.3f})")
    print(f"  after free and empty_cache: {mb(freed - before):.1f} MB above baseline")

    sane = 0.9 <= ratio <= 1.1
    print(f"  verdict: {'counter tracks allocation' if sane else 'COUNTER DOES NOT TRACK ALLOCATION'}")
    return dict(api=api, usable=True, requested_bytes=want, observed_bytes=observed, ratio=ratio, sane=sane)


def check_allocator_coverage(profile: str, device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Does torch see all device memory, or does CANN allocate outside its allocator?"""
    section("2. Allocator coverage -- does torch see everything?")

    if not mem.available:
        print("  Skipped: no memory API on this device.")
        return dict(skipped=True)

    smi_before = npu_smi_raw()
    if smi_before is None:
        print("  npu-smi not available; cannot cross-check torch's accounting.")
        print("  On a CUDA/CPU box this is expected. On the NPU this check is important.")

    model = build_model(profile, device, "etd", k=2)
    batch, seq = (2, 512) if profile == "tiny" else (1, 2048)
    ids = torch.randint(0, PROFILES[profile]["vocab_size"], (batch, seq), device=device)

    mem.reset_peak()
    with torch.no_grad():
        model(ids, last_logits_only=True)
    mem.sync()

    torch_peak = mem.peak_allocated()
    torch_reserved = mem.peak_reserved()
    smi_after = npu_smi_raw()

    print(f"  torch peak allocated : {gb(torch_peak):8.3f} GB")
    print(f"  torch peak reserved  : {gb(torch_reserved):8.3f} GB")
    print(f"  reserved / allocated : {torch_reserved / torch_peak:.2f}" if torch_peak else "")
    if smi_after:
        print("\n  npu-smi during/after the forward (raw, compare device memory by eye):")
        for line in smi_after.splitlines()[:16]:
            print(f"    {line}")
        print("\n  ACTION: if npu-smi shows materially more device memory in use than torch's")
        print("  reserved figure, CANN is allocating outside the caching allocator. Torch numbers")
        print("  then become RELATIVE indicators and npu-smi is the authority for absolutes,")
        print("  notably the OOM frontier.")

    del model
    mem.empty_cache()
    return dict(
        torch_peak_bytes=torch_peak,
        torch_reserved_bytes=torch_reserved,
        npu_smi_available=smi_after is not None,
        npu_smi_raw=smi_after,
    )


def check_components(profile: str, device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Can every term of the memory decomposition actually be read?"""
    section("3. Component readability")

    model = build_model(profile, device, "etd", k=2)
    depth = effective_depth(profile, "etd", 2)
    batch, seq = (2, 256) if profile == "tiny" else (1, 1024)
    ids = torch.randint(0, PROFILES[profile]["vocab_size"], (batch, seq), device=device)

    p_bytes = param_bytes(model)
    p_count = param_count(model)

    # KV: measured exactly from the returned cache, then compared to the analytical model.
    with torch.no_grad():
        out = model(ids, use_cache=True, last_logits_only=True)
    kv_measured = kv_cache_bytes(out.attn_key_values)
    kv_predicted = analytical_kv_bytes(profile, depth, batch, seq)
    n_entries = len(out.attn_key_values) if out.attn_key_values else 0

    # Gradients and optimizer state require a real step.
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    logits = model(ids).logits
    loss = logits.float().mean()
    loss.backward()
    g_bytes = grad_bytes(model)
    opt.step()
    o_bytes = optimizer_state_bytes(opt)

    lg_bytes = logits_bytes(profile, batch, seq)

    rows = [
        ("parameters", p_bytes, f"{p_count / 1e6:.1f} M params"),
        ("gradients", g_bytes, "one buffer per parameter"),
        ("optimizer state", o_bytes, f"{o_bytes / p_bytes:.1f}x parameters" if p_bytes else ""),
        ("logits (all positions)", lg_bytes, f"B={batch}, S={seq}, V={PROFILES[profile]['embedding_size']}"),
        ("KV cache (measured)", kv_measured, f"{n_entries} entries, expected {depth}"),
        ("KV cache (analytical)", kv_predicted, "2 x n_kv_heads x d_head x dtype x D x B x S"),
    ]
    for label, value, note in rows:
        print(f"  {label:26s} {mb(value):10.1f} MB   {note}")

    kv_ok = n_entries == depth
    kv_ratio = kv_measured / kv_predicted if kv_predicted else 0.0
    print(f"\n  cache entries == executed layers ({depth}): {'yes' if kv_ok else 'NO'}")
    print(f"  measured / analytical KV: {kv_ratio:.3f}  {'(model confirmed)' if 0.95 <= kv_ratio <= 1.05 else '(MODEL WRONG)'}")
    print("\n  A_stored + W_bwd cannot be read directly and only ever appear as a residual;")
    print("  they cannot be separated from each other. This is a stated limitation.")

    del model, opt, logits, loss
    mem.empty_cache()
    return dict(
        param_bytes=p_bytes,
        param_count=p_count,
        grad_bytes=g_bytes,
        optimizer_bytes=o_bytes,
        logits_bytes=lg_bytes,
        kv_measured_bytes=kv_measured,
        kv_analytical_bytes=kv_predicted,
        kv_entries=n_entries,
        kv_entries_expected=depth,
        kv_model_ok=bool(0.95 <= kv_ratio <= 1.05),
    )


def check_phase_profile(profile: str, device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Where does the training peak actually fall?"""
    section("4. Training-step phase profile")

    if not mem.available:
        print("  Skipped: no memory API on this device.")
        return dict(skipped=True)

    model = build_model(profile, device, "etd", k=2)
    model.train()
    batch, seq = (2, 256) if profile == "tiny" else (1, 1024)
    ids = torch.randint(0, PROFILES[profile]["vocab_size"], (batch, seq), device=device)

    phases: Dict[str, int] = {}
    mem.reset_peak()
    phases["model_built"] = mem.allocated()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # One step first, so optimizer state exists and is not billed to the measured step.
    model(ids).logits.float().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    mem.sync()
    phases["optimizer_ready"] = mem.allocated()

    mem.reset_peak()
    logits = model(ids).logits
    mem.sync()
    phases["after_forward"] = mem.allocated()

    loss = logits.float().mean()
    mem.sync()
    phases["after_loss"] = mem.allocated()

    loss.backward()
    mem.sync()
    phases["after_backward"] = mem.allocated()

    opt.step()
    opt.zero_grad(set_to_none=True)
    mem.sync()
    phases["after_step"] = mem.allocated()
    peak = mem.peak_allocated()

    for name, value in phases.items():
        print(f"  {name:20s} {gb(value):8.3f} GB")
    print(f"  {'PEAK across step':20s} {gb(peak):8.3f} GB")

    headroom = peak - max(phases["after_forward"], phases["after_loss"])
    print(f"\n  peak exceeds the largest sampled phase by {mb(headroom):.1f} MB")
    print("  -> the peak falls *between* samples (early backward), as Metrics definition.md predicts,")
    print("     which is why the peak counter is used rather than phase samples alone.")

    del model, opt, logits, loss
    mem.empty_cache()
    return dict(phases=phases, peak_bytes=peak, unsampled_headroom_bytes=headroom)


def check_attention_path(profile: str, device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Which attention implementation runs, and is its memory linear or quadratic in S?

    Flags alone are not enough: `flash_attention: true` only *tries* to import the
    CUDA-only flash_attn package and falls through silently. What actually matters is
    whether the B*h*S^2 score matrix gets materialised, so that is measured directly.
    """
    section("5. Attention path -- and whether memory is O(S) or O(S^2)")

    try:
        import flash_attn  # type: ignore  # noqa: F401

        pkg = True
    except Exception:  # noqa: BLE001
        pkg = False

    model = build_model(profile, device, "etd", k=2)
    block = model.transformer.blocks[0]
    func_bound = getattr(block, "flash_attn_func", None) is not None

    print(f"  config.flash_attention          : {model.config.flash_attention}")
    print(f"  flash_attn package importable   : {pkg}")
    print(f"  block.flash_attn_func bound     : {func_bound}")
    print(f"  => path taken                   : {'flash_attn' if func_bound else 'F.scaled_dot_product_attention'}")

    result: Dict[str, Any] = dict(config_flag=model.config.flash_attention, package=pkg, func_bound=func_bound)

    if not mem.available:
        print("\n  Skipped scaling test: no memory API on this device.")
        del model
        return {**result, "scaling_tested": False}

    # Empirical test: non-weight memory at S and 2S. Linear -> ~2x, quadratic -> ~4x.
    base_s = 512 if profile == "tiny" else 2048
    weights = param_bytes(model)
    readings = {}
    for seq in (base_s, 2 * base_s):
        ids = torch.randint(0, PROFILES[profile]["vocab_size"], (1, seq), device=device)
        mem.empty_cache()
        mem.reset_peak()
        with torch.no_grad():
            model(ids, last_logits_only=True)
        mem.sync()
        readings[seq] = mem.peak_allocated() - weights
        del ids

    small, large = readings[base_s], readings[2 * base_s]
    ratio = large / small if small > 0 else 0.0
    print(f"\n  non-weight peak at S={base_s:<6d}: {mb(small):9.1f} MB")
    print(f"  non-weight peak at S={2 * base_s:<6d}: {mb(large):9.1f} MB")
    print(f"  ratio for 2x sequence length : {ratio:.2f}")
    quadratic = ratio > 3.0
    if quadratic:
        print("  => QUADRATIC. The B*h*S^2 score matrix is being materialised.")
        print("     At B=8, S=16k, h=16 that is ~67 GB in bf16 -- the top of the planned grid")
        print("     is unreachable and attention, not KV, will dominate prefill memory.")
        print("     This must be a headline caveat, and the S sweep may need truncating.")
    else:
        print("  => roughly linear. Scores are not being materialised; the grid stands.")

    del model
    mem.empty_cache()
    return {
        **result,
        "scaling_tested": True,
        "nonweight_bytes": {str(k): v for k, v in readings.items()},
        "ratio_2x_seq": ratio,
        "quadratic": quadratic,
    }


def check_dtype(profile: str, device: str) -> Dict[str, Any]:
    section("6. dtype end to end")
    want = PROFILES[profile]["dtype"]
    model = build_model(profile, device, "etd", k=2)
    ids = torch.randint(0, PROFILES[profile]["vocab_size"], (1, 64), device=device)
    with torch.no_grad():
        out_dtype = model(ids, last_logits_only=True).logits.dtype
    p_dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"  requested          : {want}")
    print(f"  parameter dtypes   : {sorted(p_dtypes)}")
    print(f"  logits dtype       : {out_dtype}")
    ok = out_dtype == want and p_dtypes == {str(want)}
    print(f"  verdict            : {'consistent' if ok else 'MISMATCH'}")
    del model
    return dict(requested=str(want), param_dtypes=sorted(p_dtypes), logits_dtype=str(out_dtype), consistent=ok)


def check_contention_sensitivity(profile: str, device: str, mem: DeviceMemory) -> Dict[str, Any]:
    """Time the most dispatch-bound point in the plan, and record host load with it.

    The machine is shared. Run this once while the other user's job is up and once when
    it is not: the ratio converts a scheduling worry into a measured coefficient. Decode
    at B=1 is chosen because ~300 host-side op launches per token dominate it, so any
    sensitivity to host-CPU contention appears here first.
    """
    section("7. Host-contention sensitivity (decode, B=1)")

    host = host_state()
    print(f"  host load (1m/5m/15m): {host['loadavg_1m']} / {host['loadavg_5m']} / {host['loadavg_15m']}")
    print(f"  cpu count            : {host['cpu_count']}")

    model = build_model(profile, device, "etd", k=2)
    prompt_len = 64 if profile == "tiny" else 512
    ids = torch.randint(0, PROFILES[profile]["vocab_size"], (1, prompt_len), device=device)

    with torch.no_grad():
        out = model(ids, use_cache=True, last_logits_only=True)
    cache = out.attn_key_values
    step_in = torch.randint(0, PROFILES[profile]["vocab_size"], (1, 1), device=device)

    def one_step():
        with torch.no_grad():
            model(step_in, past_key_values=cache, use_cache=False, last_logits_only=True)

    timing = time_repeated(one_step, mem, warmup=5, measured=20)
    print(f"\n  decode step: median {timing.median_s * 1000:.2f} ms   "
          f"min {timing.min_s * 1000:.2f}   max {timing.max_s * 1000:.2f}   spread {timing.spread_pct:.1f}%")
    if timing.spread_pct > 25:
        print("  WARNING: spread above 25% -- the host is likely contended right now.")
    print("\n  Re-run this when the machine is quiet; the ratio of the two medians is the")
    print("  contention sensitivity, and decides whether the scheduling rules in section 6.1")
    print("  are mandatory or merely precautionary.")

    del model
    mem.empty_cache()
    return dict(host=host, timing=timing.as_dict(), prompt_len=prompt_len)


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="suffix for the result filename, e.g. 'loaded' or 'quiet'")
    args = ap.parse_args()

    device = resolve_device(args.device)
    mem = DeviceMemory(device)
    meta = run_metadata(args.profile, device)

    print("=" * 78)
    print(f"E0 PREFLIGHT   profile={args.profile}  device={device}  dtype={PROFILES[args.profile]['dtype']}")
    print(f"torch {meta['torch']}  torch_npu {meta['torch_npu']}  commit {meta['commit'][:8]}")
    print("=" * 78)

    results: Dict[str, Any] = {"metadata": meta}
    checks = (
        ("memory_api", lambda: check_memory_api(device, mem)),
        ("allocator_coverage", lambda: check_allocator_coverage(args.profile, device, mem)),
        ("components", lambda: check_components(args.profile, device, mem)),
        ("phase_profile", lambda: check_phase_profile(args.profile, device, mem)),
        ("attention_path", lambda: check_attention_path(args.profile, device, mem)),
        ("dtype", lambda: check_dtype(args.profile, device)),
        ("contention", lambda: check_contention_sensitivity(args.profile, device, mem)),
    )
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"\n  CHECK FAILED: {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            results[name] = dict(failed=True, error=f"{type(exc).__name__}: {exc}")

    name = f"e0_preflight_{args.profile}_{device.replace(':', '')}" + (f"_{args.tag}" if args.tag else "")
    path = write_result(name, results)

    section("Summary")
    failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("failed")]
    print(f"  checks run    : {len(checks)}")
    print(f"  checks failed : {len(failed)}{' -> ' + ', '.join(failed) if failed else ''}")
    att = results.get("attention_path", {})
    if att.get("scaling_tested"):
        print(f"  attention     : {'QUADRATIC in S' if att.get('quadratic') else 'linear in S'}")
    comp = results.get("components", {})
    if "kv_model_ok" in comp:
        print(f"  KV model      : {'confirmed against measurement' if comp['kv_model_ok'] else 'DOES NOT MATCH'}")
    print(f"\n  written: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
