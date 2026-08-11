"""Before/after behavioural snapshot for the ETD KV-cache work.

Records the exact logits produced by the *current* code on fixed inputs with fixed
seeds, so that after the KV-cache fix lands we can prove nothing on the existing
code paths changed.  The comparison is ``torch.equal`` -- bit-for-bit identical,
not merely close.

This file is intentionally SELF-CONTAINED (it imports nothing from the other new
scripts).  On the NPU server the reference snapshot must be generated inside the
*frozen replication checkout*, which sits at the ``replication-validated`` tag and
therefore does not contain any of the new test files.  Dropping this single
untracked file into that checkout adds no tracked change, so the reference
provably comes from code that has never been edited.

``scripts/etd_kv_cache_test.py`` imports the profile/config helpers from here so
that both tools build models identically -- one source of truth, no drift.

Usage
-----
    # in the frozen replication checkout (pre-fix), write the reference
    PYTHONPATH=$PWD python scripts/etd_snapshot.py --write

    # in the work checkout (post-fix), compare against it
    PYTHONPATH=$PWD python scripts/etd_snapshot.py --check

    # real 1B model on the NPU
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/etd_snapshot.py --write --profile full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from olmo.config import ActivationType, BlockType, InitFnType, LayerNormType, ModelConfig
from olmo.model import OLMo
from olmo.npu_util import is_npu_available

# --------------------------------------------------------------------------------------
# Fixed experimental constants.  These must not change once a reference snapshot
# has been written, or the comparison is meaningless.
# --------------------------------------------------------------------------------------

SEED = 42
SNAPSHOT_BATCH = 2
SNAPSHOT_SEQ_LEN = 16

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_fixtures" / "etd"


# --------------------------------------------------------------------------------------
# Profiles: the same suite runs at toy scale on a laptop CPU and at real scale on
# the NPU.  The KV-cache bug is *bookkeeping* -- which cache slot each loop
# iteration reads and writes -- and that logic is identical at 1B parameters and
# at a few thousand, so the toy profile is a legitimate correctness test.
# --------------------------------------------------------------------------------------

PROFILES = {
    # Toy model, runs on a CPU in seconds.  ETD split 3-2*k-3 over 8 layers.
    "tiny": dict(
        d_model=64,
        n_heads=4,
        n_layers=8,
        mlp_ratio=4,
        max_sequence_length=128,
        vocab_size=256,
        embedding_size=256,
        eos_token_id=0,
        pad_token_id=1,
        rope_theta=500_000,
        flash_attention=False,  # unavailable on CPU
        etd_encoder_layers=3,
        etd_thinking_layers=2,
        dtype=torch.float32,
    ),
    # The real OLMo 2 1B mid-training architecture.  ETD split 7-4*k-5 over 16 layers.
    "full": dict(
        d_model=2048,
        n_heads=16,
        n_layers=16,
        mlp_ratio=8,
        max_sequence_length=4096,
        vocab_size=100278,
        embedding_size=100352,
        eos_token_id=100257,
        pad_token_id=100277,
        rope_theta=500_000,
        flash_attention=True,
        etd_encoder_layers=7,
        etd_thinking_layers=4,
        dtype=torch.bfloat16,
    ),
}


def resolve_device() -> str:
    """NPU -> CUDA -> CPU, whichever is available first."""
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_config(profile: str, device: str, etd: bool, k: int = 1) -> ModelConfig:
    """Build a ModelConfig for the given profile, with ETD on or off.

    ``etd=False`` produces the plain OLMo configuration (no ETD fields set), which
    is the code path the ETD branch must remain bit-for-bit identical to at k=1.
    """
    p = PROFILES[profile]
    return ModelConfig(
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=p["n_layers"],
        mlp_ratio=p["mlp_ratio"],
        weight_tying=False,
        alibi=False,
        rope=True,
        rope_theta=p["rope_theta"],
        flash_attention=p["flash_attention"],
        attention_dropout=0.0,
        include_bias=False,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        layer_norm_eps=1e-6,
        bias_for_layer_norm=False,
        attention_layer_norm=True,
        attention_layer_norm_with_affine=True,
        norm_after=True,
        activation_type=ActivationType.swiglu,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        max_sequence_length=p["max_sequence_length"],
        vocab_size=p["vocab_size"],
        embedding_size=p["embedding_size"],
        eos_token_id=p["eos_token_id"],
        pad_token_id=p["pad_token_id"],
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_cutoff_factor=3,
        init_device=device,
        etd_encoder_layers=p["etd_encoder_layers"] if etd else None,
        etd_thinking_layers=p["etd_thinking_layers"] if etd else None,
        etd_num_iterations=k if etd else 1,
    )


def expected_cache_len(profile: str, etd: bool, k: int) -> int:
    """How many cache entries a forward pass should return.

    One entry per *executed* layer.  Without ETD that is ``n_layers``; with ETD the
    thinking block runs k times, giving ``n_encoder + n_thinking*k + n_decoder``.
    """
    p = PROFILES[profile]
    if not etd:
        return p["n_layers"]
    enc, think = p["etd_encoder_layers"], p["etd_thinking_layers"]
    dec = p["n_layers"] - enc - think
    return enc + think * k + dec


def supports_etd_kv_cache(cfg: ModelConfig) -> bool:
    """True once the opt-in KV-cache switch exists in ModelConfig.

    Pre-fix the field does not exist at all, so tests must probe for it rather than
    passing it as a constructor argument (which would raise TypeError).
    """
    return hasattr(cfg, "etd_kv_cache")


def enable_etd_kv_cache(cfg: ModelConfig) -> bool:
    """Turn the opt-in switch on if this build has it.  Returns whether it did."""
    if supports_etd_kv_cache(cfg):
        setattr(cfg, "etd_kv_cache", True)
        return True
    return False


def build_models(profile: str, device: str) -> Dict[str, OLMo]:
    """Build every configuration under test, all sharing one set of weights.

    Weights are generated once from a fixed seed on the ETD-off model and copied
    into the others, so any output difference is attributable to the forward pass
    rather than to initialisation.
    """
    dtype = PROFILES[profile]["dtype"]

    torch.manual_seed(SEED)
    base = OLMo(make_config(profile, device, etd=False), init_params=True).to(dtype).eval()
    state_dict = base.state_dict()

    models: Dict[str, OLMo] = {"etd_off": base}
    for k in (1, 2, 5):
        m = OLMo(make_config(profile, device, etd=True, k=k), init_params=False).to(dtype).eval()
        m.load_state_dict(state_dict)
        models[f"etd_k{k}"] = m
    return models


def weights_fingerprint(model: OLMo) -> torch.Tensor:
    """A cheap scalar summary of the weights.

    Not cryptographic -- its purpose is diagnostic.  If a snapshot mismatch occurs,
    this distinguishes "the weights changed" (initialisation drifted) from "the
    forward pass changed", which are very different problems.
    """
    total = 0.0
    for name in sorted(dict(model.named_parameters()).keys()):
        p = dict(model.named_parameters())[name]
        total += float(p.detach().float().sum().item())
    return torch.tensor(total, dtype=torch.float64)


def snapshot_path(profile: str, device: str) -> Path:
    dtype_name = str(PROFILES[profile]["dtype"]).replace("torch.", "")
    return FIXTURE_DIR / f"snapshot_{profile}_{device}_{dtype_name}.pt"


def capture(profile: str, device: str) -> Dict[str, torch.Tensor]:
    """Run every configuration on fixed input and collect the outputs."""
    torch.manual_seed(SEED)
    input_ids = torch.randint(
        0, PROFILES[profile]["vocab_size"], (SNAPSHOT_BATCH, SNAPSHOT_SEQ_LEN), device=device
    )

    models = build_models(profile, device)
    out: Dict[str, torch.Tensor] = {"input_ids": input_ids.cpu()}

    for label, model in models.items():
        with torch.no_grad():
            # Uncached forward -- this is the path training and current evaluation use.
            out[f"{label}/logits"] = model(input_ids).logits.cpu()
        out[f"{label}/weights_fingerprint"] = weights_fingerprint(model)

        # Cached forward, but only where it is currently legal.  ETD-off and k=1
        # both execute each block exactly once, so upstream's block-index lookup is
        # correct for them and caching already works.  Capturing these guards the
        # upstream cached path against regressions too.
        etd_off_or_k1 = label in ("etd_off", "etd_k1")
        if etd_off_or_k1:
            with torch.no_grad():
                cached = model(input_ids, use_cache=True)
            out[f"{label}/cached_logits"] = cached.logits.cpu()
            out[f"{label}/cache_len"] = torch.tensor(len(cached.attn_key_values))

    return out


def write(profile: str, device: str) -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(profile, device)
    data = capture(profile, device)
    torch.save(data, path)
    print(f"Wrote snapshot: {path}")
    print(f"  torch {torch.__version__}, device={device}, profile={profile}")
    for key in sorted(data):
        v = data[key]
        print(f"  {key:40s} {tuple(v.shape) if v.ndim else float(v)}")
    return 0


def check(profile: str, device: str, verbose: bool = True) -> int:
    """Compare current behaviour against the stored reference.  0 = identical."""
    path = snapshot_path(profile, device)
    if not path.exists():
        print(f"FAIL: no reference snapshot at {path}")
        print("      Run with --write on the pre-fix code first.")
        return 1

    reference: Dict[str, torch.Tensor] = torch.load(path, weights_only=True)
    current = capture(profile, device)

    failures = []

    missing = sorted(set(reference) - set(current))
    extra = sorted(set(current) - set(reference))
    if missing:
        failures.append(f"keys missing from current run: {missing}")
    if extra:
        # Extra keys are not a failure in themselves (the fix legitimately adds
        # cached outputs for k>1), but they are reported for visibility.
        if verbose:
            print(f"  note: current run produced new keys (expected after the fix): {extra}")

    for key in sorted(set(reference) & set(current)):
        ref, cur = reference[key], current[key]
        if ref.shape != cur.shape:
            failures.append(f"{key}: shape {tuple(cur.shape)} != reference {tuple(ref.shape)}")
        elif not torch.equal(ref, cur):
            diff = (cur.double() - ref.double()).abs().max().item()
            failures.append(f"{key}: not bit-for-bit identical (max abs diff {diff:.3e})")
        elif verbose:
            print(f"  OK  {key}")

    if failures:
        print(f"\nFAIL: {len(failures)} snapshot mismatch(es) against {path.name}")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nPASS: current behaviour is bit-for-bit identical to {path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="write a new reference snapshot")
    g.add_argument("--check", action="store_true", help="compare against the stored reference")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="tiny")
    ap.add_argument("--device", default=None, help="override auto-detected device")
    args = ap.parse_args()

    device = args.device or resolve_device()
    print(f"profile={args.profile} device={device} dtype={PROFILES[args.profile]['dtype']}")

    return write(args.profile, device) if args.write else check(args.profile, device)


if __name__ == "__main__":
    raise SystemExit(main())
