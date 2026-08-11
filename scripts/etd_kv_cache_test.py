"""Test suite for KV caching under looped (k>1) ETD inference.

Written BEFORE the implementation, as the contract the fix must satisfy.
See ``KV_cache_test_plan.md`` for the design rationale.

Background
----------
KV caching is upstream OLMo machinery and works fine; it simply has no notion of
loop iterations.  Each block's cache entry is looked up by *block index*, which
stops identifying a position once a block runs more than once per forward pass.
ETD-off and k=1 execute every block exactly once, so block index == position and
caching already works for them -- that makes them a genuine control group.  For
k>1 the mapping breaks, and rather than silently produce wrong tokens the code
currently raises.

The fix indexes the cache by *position in the unrolled schedule*, giving one entry
per executed layer (``n_enc + n_think*k + n_dec``) instead of one per block.

Tolerances
----------
Cache-vs-no-cache is NOT bit-exact: a length-1 query attends with different kernel
shapes and reduction order than a length-S query.  A tolerance is unavoidable, and
one chosen after watching an implementation fail would be worthless.  So the values
here are inherited verbatim from upstream ``tests/model_test.py::test_forward``,
which performs exactly this comparison:  ``atol=1e-2, rtol=1e3`` under half
precision, ``assert_close`` defaults otherwise.

Passing the absolute threshold is necessary but not sufficient, so every
configuration also reports its observed maximum deviation.  The k>1 numbers must
land in the same order of magnitude as the control group -- if ETD-off disagrees by
1e-7 and k=5 by 1e-3, both pass the ceiling yet something is clearly wrong.

Usage
-----
    # Mac: toy model, CPU, fp32
    PYTHONPATH=$PWD ../.venv/bin/python scripts/etd_kv_cache_test.py

    # NPU: real 1B model, bf16
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/etd_kv_cache_test.py --profile full

    # after the fix: treat "expected pre-fix failure" as a hard failure
    PYTHONPATH=$PWD ../.venv/bin/python scripts/etd_kv_cache_test.py --strict

Exit code 0 if nothing failed unexpectedly; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from olmo.model import OLMo
from scripts.etd_snapshot import (  # noqa: E402  -- one source of truth for configs
    PROFILES,
    SEED,
    build_models,
    check as snapshot_check,
    enable_etd_kv_cache,
    expected_cache_len,
    make_config,
    resolve_device,
    snapshot_path,
    supports_etd_kv_cache,
)

# The pre-fix failure mode.  A k>1 test that fails for any *other* reason means the
# test itself is wrong, not the model -- which is why the message is matched.
PREFIX_ERROR_FRAGMENT = "incompatible with use_cache"

BATCH = 2
SEQ_LEN = 16
DECODE_PROMPT_LEN = 8
DECODE_STEPS = 16

PASS, FAIL, RED, SKIP = "PASS", "FAIL", "RED-EXPECTED", "SKIP"


class Results:
    """Collects outcomes so the whole suite runs even when parts fail."""

    def __init__(self, strict: bool) -> None:
        self.rows: List[Tuple[str, str, str, str]] = []
        self.strict = strict

    def add(self, test: str, case: str, status: str, detail: str = "") -> None:
        self.rows.append((test, case, status, detail))
        mark = {PASS: "  ok ", FAIL: " FAIL", RED: "  red", SKIP: " skip"}[status]
        print(f"  [{mark}] {case}{('  ' + detail) if detail else ''}")

    def failed(self) -> bool:
        bad = {FAIL} if not self.strict else {FAIL, RED}
        return any(r[2] in bad for r in self.rows)

    def summary(self) -> None:
        counts = {s: sum(1 for r in self.rows if r[2] == s) for s in (PASS, RED, SKIP, FAIL)}
        print("\n" + "=" * 78)
        print(
            f"SUMMARY   pass={counts[PASS]}  expected-red={counts[RED]}  "
            f"skip={counts[SKIP]}  FAIL={counts[FAIL]}"
        )
        if counts[FAIL]:
            print("\nUnexpected failures:")
            for test, case, status, detail in self.rows:
                if status == FAIL:
                    print(f"  {test} / {case}: {detail}")
        if counts[RED] and not self.strict:
            print(
                "\nExpected-red entries are the tests that target the fix. They must fail\n"
                "before the implementation exists ('red before green'). Re-run with --strict\n"
                "once the fix is in -- they must all turn green."
            )
        print("=" * 78)


# --------------------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------------------


def tolerances(dtype: torch.dtype) -> dict:
    """Inherited verbatim from upstream tests/model_test.py::test_forward."""
    if dtype in (torch.float16, torch.bfloat16):
        return dict(rtol=1e3, atol=1e-2)
    return {}


def compare(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, float]:
    """Return (within tolerance, observed max absolute deviation)."""
    deviation = (a.detach().double() - b.detach().double()).abs().max().item()
    try:
        torch.testing.assert_close(a, b, **tolerances(dtype))
        return True, deviation
    except AssertionError:
        return False, deviation


def is_prefix_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) and PREFIX_ERROR_FRAGMENT in str(exc)


def fixed_input(profile: str, device: str, batch: int, seq_len: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, PROFILES[profile]["vocab_size"], (batch, seq_len), device=device)


def configured_model(profile: str, device: str, etd: bool, k: int, want_cache: bool) -> Tuple[OLMo, bool]:
    """Build one model, optionally requesting the opt-in KV-cache switch.

    Returns (model, switch_was_available).  Pre-fix the switch does not exist, so
    callers must probe rather than pass it as a constructor argument.
    """
    dtype = PROFILES[profile]["dtype"]
    cfg = make_config(profile, device, etd=etd, k=k)
    available = enable_etd_kv_cache(cfg) if want_cache else supports_etd_kv_cache(cfg)

    torch.manual_seed(SEED)
    base_cfg = make_config(profile, device, etd=False)
    base = OLMo(base_cfg, init_params=True).to(dtype).eval()

    model = OLMo(cfg, init_params=False).to(dtype).eval()
    model.load_state_dict(base.state_dict())
    return model, available


CASES = [("etd_off", False, 1), ("etd_k1", True, 1), ("etd_k2", True, 2), ("etd_k3", True, 3), ("etd_k5", True, 5)]
CONTROL = {"etd_off", "etd_k1"}


# --------------------------------------------------------------------------------------
# Test 1 -- the existing ETD invariant must survive
# --------------------------------------------------------------------------------------


def test_1_k1_equals_etd_off(profile: str, device: str, r: Results) -> None:
    """ETD k=1 must remain BIT-FOR-BIT identical to ETD-off.

    At k=1 the unrolled schedule is 0..n-1, exactly the standard loop, so equality
    is exact rather than approximate.  This is the invariant used to validate the
    original ETD implementation, and the fix must not disturb it.
    """
    print("\nTest 1: ETD k=1 == ETD off (uncached, bit-exact)")
    input_ids = fixed_input(profile, device, BATCH, SEQ_LEN)
    models = build_models(profile, device)
    with torch.no_grad():
        off = models["etd_off"](input_ids).logits
        k1 = models["etd_k1"](input_ids).logits
    if torch.equal(off, k1):
        r.add("1", "k=1 vs ETD-off", PASS, "bit-for-bit identical")
    else:
        dev = (k1.double() - off.double()).abs().max().item()
        r.add("1", "k=1 vs ETD-off", FAIL, f"differ, max abs diff {dev:.3e}")

    # Guard against the opposite silent failure: if the ETD branch were never
    # entered, k>1 would also match ETD-off.  With random weights, agreement is
    # impossible unless the loop simply did not run.
    with torch.no_grad():
        k2 = models["etd_k2"](input_ids).logits
    if torch.equal(off, k2):
        r.add("1", "k=2 vs ETD-off (must differ)", FAIL, "identical - ETD loop did not run")
    else:
        r.add("1", "k=2 vs ETD-off (must differ)", PASS, "differs, loop is active")


# --------------------------------------------------------------------------------------
# Tests 2-4 -- cache vs no-cache, one decode step
# --------------------------------------------------------------------------------------


def test_2_3_4_cache_equivalence(profile: str, device: str, r: Results) -> None:
    """Cached inference must reproduce the uncached forward pass.

    The uncached full-sequence forward is the ground truth: it is what the
    replication used and what the paper describes.

    ETD-off and k=1 are the CONTROL GROUP -- they exercise upstream's own caching on
    code ETD does not alter, and must pass before the fix exists.  If they fail, the
    harness is wrong, not the model.
    """
    print("\nTests 2-4: cache == no-cache (single step)")
    dtype = PROFILES[profile]["dtype"]
    input_ids = fixed_input(profile, device, BATCH, SEQ_LEN)
    deviations = {}

    for label, etd, k in CASES:
        model, _ = configured_model(profile, device, etd, k, want_cache=True)
        test_id = "2/3" if label in CONTROL else "4"
        try:
            with torch.no_grad():
                full = model(input_ids).logits[:, -1]
                prefill = model(input_ids[:, :-1], use_cache=True)
                stepped = model(
                    input_ids[:, -1:], past_key_values=prefill.attn_key_values, use_cache=True
                ).logits[:, -1]
            ok, dev = compare(full, stepped, dtype)
            deviations[label] = dev
            if ok:
                r.add(test_id, label, PASS, f"max abs dev {dev:.3e}")
            else:
                r.add(test_id, label, FAIL, f"outside tolerance, max abs dev {dev:.3e}")
        except Exception as exc:  # noqa: BLE001
            if is_prefix_error(exc) and label not in CONTROL:
                r.add(test_id, label, RED, f"{type(exc).__name__}: {exc}")
            else:
                r.add(test_id, label, FAIL, f"{type(exc).__name__}: {exc}")

    _report_relative_calibration(deviations, r)


def _report_relative_calibration(deviations: dict, r: Results) -> None:
    """Absolute tolerance is not enough -- k>1 must also be the same order as control.

    A run where ETD-off disagrees by 1e-7 while k=5 disagrees by 1e-3 passes every
    absolute threshold yet is clearly wrong.  Control-group magnitudes are measured
    on upstream code before the fix exists, so this baseline cannot be gamed.
    """
    control = [d for label, d in deviations.items() if label in CONTROL]
    looped = {label: d for label, d in deviations.items() if label not in CONTROL}
    if not control or not looped:
        return
    ceiling = max(max(control) * 100.0, 1e-12)
    print(f"\n  relative calibration: control max dev {max(control):.3e}, ceiling {ceiling:.3e} (100x)")
    for label, dev in looped.items():
        if dev <= ceiling:
            r.add("4", f"{label} calibration", PASS, f"{dev:.3e} within 100x of control")
        else:
            r.add("4", f"{label} calibration", FAIL, f"{dev:.3e} exceeds 100x control ({ceiling:.3e})")


# --------------------------------------------------------------------------------------
# Test 5 -- the cache must have the right shape, not merely produce right answers
# --------------------------------------------------------------------------------------


def test_5_cache_length(profile: str, device: str, r: Results) -> None:
    """One cache entry per EXECUTED layer, each holding the full sequence.

    Logits could agree by coincidence while the cache is silently the wrong length
    or holding stale entries, so this checks the mechanism rather than the output.
    Expected length is n_enc + n_think*k + n_dec, i.e. 12 + 4k at 1B scale.
    """
    print("\nTest 5: cache length == one entry per executed layer")
    input_ids = fixed_input(profile, device, BATCH, SEQ_LEN)

    for label, etd, k in CASES:
        model, _ = configured_model(profile, device, etd, k, want_cache=True)
        want = expected_cache_len(profile, etd, k)
        try:
            with torch.no_grad():
                out = model(input_ids, use_cache=True)
            got = len(out.attn_key_values)
            if got != want:
                r.add("5", f"{label} length", FAIL, f"{got} entries, expected {want}")
                continue
            bad = [
                i
                for i, (key, value) in enumerate(out.attn_key_values)
                if key.shape[-2] != SEQ_LEN or value.shape[-2] != SEQ_LEN
            ]
            if bad:
                r.add("5", f"{label} entry shapes", FAIL, f"entries {bad} have wrong sequence length")
            else:
                r.add("5", f"{label}", PASS, f"{got} entries, each seq_len={SEQ_LEN}")
        except Exception as exc:  # noqa: BLE001
            if is_prefix_error(exc) and label not in CONTROL:
                r.add("5", label, RED, f"{type(exc).__name__}: {exc}")
            else:
                r.add("5", label, FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Test 6 -- many decode steps, not just one
# --------------------------------------------------------------------------------------


def test_6_multi_step_decode(profile: str, device: str, r: Results) -> None:
    """Correctness must hold over many steps, not just the first.

    Upstream's test checks a SINGLE cached step.  A loop-indexing bug can be right
    for the first generated token and drift afterwards -- and the Q2 decode
    benchmarks run hundreds of steps, so one-step correctness would be a false
    green.  Both paths are driven by the same token sequence so that any divergence
    is attributable to caching rather than to different inputs.
    """
    print(f"\nTest 6: multi-step decode ({DECODE_STEPS} steps)")
    dtype = PROFILES[profile]["dtype"]
    prompt = fixed_input(profile, device, BATCH, DECODE_PROMPT_LEN)

    for label, etd, k in CASES:
        model, _ = configured_model(profile, device, etd, k, want_cache=True)
        try:
            # Ground truth: repeated full-sequence forwards, greedy.
            tokens = prompt.clone()
            reference = []
            with torch.no_grad():
                for _ in range(DECODE_STEPS):
                    logits = model(tokens).logits[:, -1]
                    reference.append(logits)
                    tokens = torch.cat([tokens, logits.argmax(-1, keepdim=True)], dim=1)
            generated = tokens[:, DECODE_PROMPT_LEN:]

            # Cached: prefill once, then one token at a time, driven by the same tokens.
            with torch.no_grad():
                out = model(prompt, use_cache=True)
                cache = out.attn_key_values
                cached = [out.logits[:, -1]]
                for step in range(DECODE_STEPS - 1):
                    out = model(
                        generated[:, step : step + 1], past_key_values=cache, use_cache=True
                    )
                    cache = out.attn_key_values
                    cached.append(out.logits[:, -1])

            worst = 0.0
            worst_step = -1
            all_ok = True
            for step, (ref, cur) in enumerate(zip(reference, cached)):
                ok, dev = compare(ref, cur, dtype)
                all_ok &= ok
                if dev > worst:
                    worst, worst_step = dev, step
            if all_ok:
                r.add("6", label, PASS, f"{DECODE_STEPS} steps, worst dev {worst:.3e} @ step {worst_step}")
            else:
                r.add("6", label, FAIL, f"drift, worst dev {worst:.3e} @ step {worst_step}")
        except Exception as exc:  # noqa: BLE001
            if is_prefix_error(exc) and label not in CONTROL:
                r.add("6", label, RED, f"{type(exc).__name__}: {exc}")
            else:
                r.add("6", label, FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Test 7 -- the change must be inert unless explicitly requested
# --------------------------------------------------------------------------------------


def test_7_default_is_off(profile: str, device: str, r: Results) -> None:
    """With the switch unset, k>1 must refuse to cache exactly as it does today.

    This is what guarantees the replication's reported accuracy numbers stay
    reproducible: the default code path is unchanged, so evaluation behaves
    identically unless caching is deliberately turned on.

    Passes both before and after the fix -- before, because the switch does not yet
    exist; after, because it defaults to off.
    """
    print("\nTest 7: switch defaults to off (k>1 still refuses to cache)")
    input_ids = fixed_input(profile, device, BATCH, SEQ_LEN)

    for label, etd, k in CASES:
        if label in CONTROL:
            continue
        model, switch_exists = configured_model(profile, device, etd, k, want_cache=False)
        try:
            with torch.no_grad():
                model(input_ids, use_cache=True)
            r.add("7", label, FAIL, "cached with the switch off - default is not inert")
        except Exception as exc:  # noqa: BLE001
            if is_prefix_error(exc):
                note = "switch exists, defaults off" if switch_exists else "switch not yet added"
                r.add("7", label, PASS, f"refused as expected ({note})")
            else:
                r.add("7", label, FAIL, f"wrong error: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Test 8 -- the path evaluation actually uses
# --------------------------------------------------------------------------------------


def test_8_hf_generation(profile: str, device: str, r: Results) -> None:
    """The HuggingFace generation path must actually use the cache.

    Fixing the core forward pass alone is not enough: the HF wrapper currently
    disables caching for k>1 outright, and prepare_inputs_for_generation only trims
    the input to the last token when k <= 1.  Without those, generation keeps
    re-running the full sequence and the fix is invisible to the benchmarks it
    exists to enable.

    Two assertions, because tokens alone would be a false green -- if caching is
    silently disabled, both runs take the identical uncached path and trivially
    agree.  So we also instrument the inner model to confirm that cached decoding
    really does feed one token per step.
    """
    print("\nTest 8: HuggingFace generation path")
    try:
        from hf_olmo import OLMoConfig, OLMoForCausalLM
    except Exception as exc:  # noqa: BLE001
        r.add("8", "import hf_olmo", SKIP, f"{type(exc).__name__}: {exc}")
        return

    dtype = PROFILES[profile]["dtype"]
    prompt = fixed_input(profile, device, BATCH, DECODE_PROMPT_LEN)

    for label, etd, k in CASES:
        try:
            model, _ = configured_model(profile, device, etd, k, want_cache=True)
            cfg_kwargs = model.config.asdict()
            hf_config = OLMoConfig(**cfg_kwargs, use_cache=True)
            hf = OLMoForCausalLM(hf_config, model=model).to(device).eval()

            seen: List[int] = []
            inner_forward = hf.model.forward

            def recording_forward(*args, _orig=inner_forward, _seen=seen, **kwargs):
                ids = kwargs.get("input_ids", args[0] if args else None)
                if ids is not None:
                    _seen.append(int(ids.shape[1]))
                return _orig(*args, **kwargs)

            hf.model.forward = recording_forward

            with torch.no_grad():
                cached_out = hf.generate(
                    prompt, max_new_tokens=DECODE_STEPS, do_sample=False, use_cache=True
                )
            cached_calls = list(seen)
            seen.clear()
            with torch.no_grad():
                uncached_out = hf.generate(
                    prompt, max_new_tokens=DECODE_STEPS, do_sample=False, use_cache=False
                )

            tokens_match = torch.equal(cached_out, uncached_out)
            # After the prefill call, a genuinely cached run feeds one token at a time.
            really_cached = len(cached_calls) > 1 and all(n == 1 for n in cached_calls[1:])

            if tokens_match and really_cached:
                r.add("8", label, PASS, f"tokens match; per-step input lengths {cached_calls[:4]}...")
            elif not really_cached:
                status = RED if label not in CONTROL else FAIL
                r.add("8", label, status, f"cache not used; per-step input lengths {cached_calls[:4]}...")
            else:
                r.add("8", label, FAIL, "cached and uncached generation produced different tokens")
        except Exception as exc:  # noqa: BLE001
            if is_prefix_error(exc) and label not in CONTROL:
                r.add("8", label, RED, f"{type(exc).__name__}: {exc}")
            else:
                r.add("8", label, FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


# --------------------------------------------------------------------------------------
# Test 9 -- nothing else changed
# --------------------------------------------------------------------------------------


def test_9_snapshot(profile: str, device: str, r: Results) -> None:
    """Existing behaviour must be bit-for-bit unchanged.

    The catch-all: it detects accidental damage anywhere in the file, including
    edits with nothing to do with caching.  The reference is captured from pre-fix
    code (on the NPU, from the frozen replication checkout, which has provably never
    been edited).
    """
    print("\nTest 9: before/after snapshot (bit-exact)")
    path = snapshot_path(profile, device)
    if not path.exists():
        r.add("9", "reference snapshot", SKIP, f"not found: {path.name} - run etd_snapshot.py --write")
        return
    code = snapshot_check(profile, device, verbose=False)
    if code == 0:
        r.add("9", "snapshot", PASS, f"identical to {path.name}")
    else:
        r.add("9", "snapshot", FAIL, f"differs from {path.name} - see output above")


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None, help="override auto-detected device")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="require the fix: expected-pre-fix failures are treated as hard failures",
    )
    args = ap.parse_args()

    device = args.device or resolve_device()
    dtype = PROFILES[args.profile]["dtype"]
    print("=" * 78)
    print(f"ETD KV-cache test suite   profile={args.profile}  device={device}  dtype={dtype}")
    print(f"torch {torch.__version__}   tolerances={tolerances(dtype) or 'assert_close defaults'}")
    print(f"mode: {'STRICT (fix required)' if args.strict else 'pre-fix (red-before-green allowed)'}")
    print("=" * 78)

    r = Results(strict=args.strict)
    for fn in (
        test_1_k1_equals_etd_off,
        test_2_3_4_cache_equivalence,
        test_5_cache_length,
        test_6_multi_step_decode,
        test_7_default_is_off,
        test_8_hf_generation,
        test_9_snapshot,
    ):
        try:
            fn(args.profile, device, r)
        except Exception as exc:  # noqa: BLE001
            r.add(fn.__name__, "harness", FAIL, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    r.summary()
    return 1 if r.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
