"""Diagnose why cached generation does or does not engage for an exported checkpoint.

The unit suite constructs OLMoForCausalLM directly and calls generate() with an
explicit use_cache=True.  olmes instead loads the exported folder through
AutoModelForCausalLM(trust_remote_code=True) and calls generate() with no explicit
use_cache, relying on the model's generation_config.  Those are different paths, and
the cache can be lost in any of several places between them.

This probe walks that chain and prints what each link actually holds, then runs
generation both ways while recording the input length passed to the inner model at
every step.  A genuinely cached run shows [prompt_len, 1, 1, 1, ...]; an uncached one
shows [prompt_len, prompt_len+1, prompt_len+2, ...].

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/etd_kv_cache_generate_probe.py \
        --model-path kvcache_check/running/ETD_k2_npu/step23852-hf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sync(device: str) -> None:
    """Wait for the device to finish. Kernel launches are async, so timing without
    this measures launch time rather than execution time."""
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default=None, help="e.g. npu:0 (default: auto)")
    ap.add_argument("--new-tokens", type=int, default=12)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="also time cached vs uncached generation at realistic gsm8k-like sizes, "
        "with no evaluation harness in the loop",
    )
    ap.add_argument("--bench-prompt-len", type=int, default=750)
    ap.add_argument("--bench-new-tokens", type=int, default=150)
    args = ap.parse_args()

    from olmo.npu_util import is_npu_available

    if args.device:
        device = args.device
    elif is_npu_available():
        device = "npu:0"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model_path = str(Path(args.model_path).resolve())
    print("=" * 78)
    print("KV-cache generation probe")
    print(f"  model  : {model_path}")
    print(f"  device : {device}")
    print("=" * 78)

    # ---- what is on disk -------------------------------------------------------
    cfg_path = Path(model_path) / "config.json"
    with cfg_path.open() as f:
        raw = json.load(f)
    print("\n[1] config.json as written")
    for key in ("use_cache", "etd_kv_cache", "etd_num_iterations", "etd_encoder_layers", "etd_thinking_layers"):
        print(f"      {key:22s} {raw.get(key, '(absent)')}")

    gen_path = Path(model_path) / "generation_config.json"
    if gen_path.exists():
        with gen_path.open() as f:
            print(f"      generation_config.json  {json.load(f)}")
    else:
        print("      generation_config.json  (absent)")

    # Does the exported copy of the modeling code contain the fix?
    src = (Path(model_path) / "modeling_olmo.py").read_text()
    print(f"      exported modeling_olmo.py has the fix: {'kv_cache_is_populated' in src}")

    # ---- load exactly as olmes does --------------------------------------------
    from transformers import AutoModelForCausalLM

    print("\n[2] loading via AutoModelForCausalLM(trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    print(f"      hf config.use_cache          {getattr(model.config, 'use_cache', '(absent)')}")
    print(f"      hf config.etd_kv_cache       {getattr(model.config, 'etd_kv_cache', '(absent)')}")
    print(f"      hf config.etd_num_iterations {getattr(model.config, 'etd_num_iterations', '(absent)')}")
    gc = getattr(model, "generation_config", None)
    print(f"      generation_config.use_cache  {getattr(gc, 'use_cache', '(no generation_config)')}")

    inner = model.model  # the olmo.model.OLMo instance
    print(f"      inner ModelConfig.etd_kv_cache       {getattr(inner.config, 'etd_kv_cache', '(absent)')}")
    print(f"      inner ModelConfig.etd_num_iterations {getattr(inner.config, 'etd_num_iterations', '(absent)')}")
    if hasattr(inner, "expected_kv_cache_len"):
        print(f"      inner expected_kv_cache_len()        {inner.expected_kv_cache_len()}")
    else:
        print("      inner expected_kv_cache_len()        (absent - stale module cache?)")

    # ---- instrument and generate ------------------------------------------------
    vocab = getattr(model.config, "vocab_size", 100278)
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (1, args.prompt_len), device=device)

    def run(label, **gen_kwargs):
        seen = []
        original = inner.forward

        def recording(*a, _o=original, **kw):
            ids = kw.get("input_ids", a[0] if a else None)
            if ids is not None:
                seen.append(int(ids.shape[1]))
            return _o(*a, **kw)

        inner.forward = recording
        try:
            started = time.time()
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=args.new_tokens, do_sample=False, **gen_kwargs)
            elapsed = time.time() - started
        finally:
            inner.forward = original

        cached = len(seen) > 1 and all(n == 1 for n in seen[1:])
        print(f"\n      {label}")
        print(f"        per-step input lengths : {seen}")
        print(f"        cache in use           : {cached}")
        print(f"        wall-clock             : {elapsed:.2f}s")
        return cached

    print("\n[3] generation, as olmes calls it (no explicit use_cache)")
    implicit = run("generate(...)")

    print("\n[4] generation with use_cache=True passed explicitly")
    explicit = run("generate(..., use_cache=True)")

    # ---- direct cost measurement, no evaluation harness involved -----------------
    if args.benchmark:
        print("\n[5] cached vs uncached generation cost, harness excluded")
        print(f"      prompt {args.bench_prompt_len} tokens, generating {args.bench_new_tokens}")
        torch.manual_seed(1)
        bench_prompt = torch.randint(0, vocab, (1, args.bench_prompt_len), device=device)

        def timed(use_cache: bool) -> float:
            # One short warm-up so kernel compilation is not billed to the measurement.
            with torch.no_grad():
                model.generate(bench_prompt, max_new_tokens=4, do_sample=False, use_cache=use_cache)
            _sync(device)
            started = time.time()
            with torch.no_grad():
                model.generate(
                    bench_prompt,
                    max_new_tokens=args.bench_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
            _sync(device)
            return time.time() - started

        t_cached = timed(True)
        t_uncached = timed(False)
        per_tok_c = 1000 * t_cached / args.bench_new_tokens
        per_tok_u = 1000 * t_uncached / args.bench_new_tokens
        print(f"        cached   : {t_cached:6.2f}s   {per_tok_c:6.1f} ms/token")
        print(f"        uncached : {t_uncached:6.2f}s   {per_tok_u:6.1f} ms/token")
        print(f"        speedup  : {t_uncached / t_cached:.2f}x")
        print()
        if t_uncached / t_cached >= 1.5:
            print("      The model DOES get materially faster with the cache. If an evaluation")
            print("      harness shows no speedup, the harness dominates its own wall-clock")
            print("      (per-step detokenisation for stop-sequence checks is the usual cause),")
            print("      and harness timing cannot be used to judge decode cost.")
        else:
            print("      The model itself gains little from caching at this size. Worth")
            print("      confirming against the batching sweep before drawing conclusions:")
            print("      at batch 1 a 1B model may be dominated by fixed per-layer overhead.")

    # ---- verdict -----------------------------------------------------------------
    print("\n" + "=" * 78)
    if implicit and explicit:
        print("Cache engages on both paths. If olmes is still slow, the cause is elsewhere.")
    elif explicit and not implicit:
        print("Cache engages ONLY when use_cache=True is passed explicitly.")
        print("=> generation_config is not carrying use_cache through. Fix by writing")
        print("   generation_config.json with use_cache=true into the exported folder.")
    elif not explicit:
        print("Cache does NOT engage even when requested explicitly.")
        print("=> the break is inside the model/wrapper, not in configuration. The")
        print("   per-step input lengths above show whether prepare_inputs_for_generation")
        print("   is trimming; if it is not, past_key_values is not surviving the round")
        print("   trip through generate().")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
