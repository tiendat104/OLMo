"""
Sanity check for the ETD (Encode-Think-Decode) forward pass implementation.

Verifies five properties:
  1. ETD-k=1 produces bit-for-bit identical logits to ETD-disabled (standard
     forward pass) when given the same weights and input.
  2. ETD-k=2,3,4,5 run without error and produce the correct output shape.
  3. ETD-k=2 produces DIFFERENT logits from ETD-disabled — proving the ETD
     branch is actually entered and the thinking loop changes the computation.
  4. The first thinking block (block 7) is called exactly k times during a
     forward pass with etd_num_iterations=k — directly verifying the loop count.
  5. When etd_encoder_layers is not set in the config (defaults to None), the
     ETD-off (else) branch executes: block 7 is called exactly once and the
     output is bit-for-bit identical to the ETD-disabled baseline.

Architecture matches the OLMo 2 1B mid-training config exactly:
  16 layers, d_model=2048, n_heads=16, mlp_ratio=8, RoPE θ=500k,
  SwiGLU, RMSNorm, norm_after, QK-norm. ETD split: 7 encoder + 4 thinking + 5 decoder.

Requires Step 2 (ETD fields in olmo/config.py and ETD branch in olmo/model.py)
to be applied before this script can run.

Usage:
    CUDA_VISIBLE_DEVICES=<n>          python scripts/test_etd_sanity.py  # H100/CUDA
    ASCEND_RT_VISIBLE_DEVICES=<n>     python scripts/test_etd_sanity.py  # Huawei NPU
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.config import ActivationType, BlockType, InitFnType, LayerNormType, ModelConfig
from olmo.model import OLMo
from olmo.npu_util import is_npu_available


def _get_device() -> str:
    """Return 'npu' if a Huawei NPU is reachable, else 'cuda'."""
    return "npu" if is_npu_available() else "cuda"


def _empty_cache() -> None:
    """Free cached memory on whichever accelerator is active."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_npu_available():
        torch.npu.empty_cache()


def make_config(
    etd_encoder_layers=None,
    etd_thinking_layers=None,
    etd_num_iterations=1,
):
    """Return the exact OLMo 2 1B ModelConfig, optionally with ETD enabled."""
    return ModelConfig(
        d_model=2048,
        n_heads=16,
        n_layers=16,
        mlp_ratio=8,
        weight_tying=False,
        alibi=False,
        rope=True,
        rope_theta=500_000,
        flash_attention=True,
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
        max_sequence_length=4096,
        vocab_size=100278,
        embedding_size=100352,
        eos_token_id=100257,
        pad_token_id=100277,
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_cutoff_factor=3,
        init_device=_get_device(),
        etd_encoder_layers=etd_encoder_layers,
        etd_thinking_layers=etd_thinking_layers,
        etd_num_iterations=etd_num_iterations,
    )


def main():
    assert is_npu_available() or torch.cuda.is_available(), (
        "This script requires a CUDA GPU or a Huawei NPU. "
        "Run with CUDA_VISIBLE_DEVICES=<n> or ASCEND_RT_VISIBLE_DEVICES=<n>."
    )

    device = _get_device()
    torch.manual_seed(42)
    dtype = torch.bfloat16

    batch_size, seq_len = 2, 64
    # embedding_size=100352 (not vocab_size=100278) because weight_tying=False
    expected_shape = (batch_size, seq_len, 100352)

    input_ids = torch.randint(0, 100278, (batch_size, seq_len), device=device)

    # Build the baseline model (ETD disabled) with random weights.
    print("Building baseline model (ETD disabled, random weights) ...")
    model_std = OLMo(make_config(), init_params=True).to(dtype).eval()

    with torch.no_grad():
        logits_std = model_std(input_ids).logits

    # Save weights so all ETD models use the same parameters.
    state_dict = model_std.state_dict()

    # ------------------------------------------------------------------
    # Test 1: ETD-k=1 must be bit-for-bit identical to ETD-disabled.
    #
    # When etd_num_iterations=1 the block index sequence is [0..15],
    # which is identical to the original enumerate(blocks) loop, so
    # torch.equal() must hold exactly — not just approximately.
    # ------------------------------------------------------------------
    print("Test 1: ETD-k=1 vs ETD-disabled ...")
    cfg_k1 = make_config(etd_encoder_layers=7, etd_thinking_layers=4, etd_num_iterations=1)
    model_k1 = OLMo(cfg_k1, init_params=False).to(dtype).eval()
    model_k1.load_state_dict(state_dict)

    with torch.no_grad():
        logits_k1 = model_k1(input_ids).logits

    assert torch.equal(logits_std, logits_k1), (
        f"FAIL: ETD-k=1 logits differ from ETD-disabled.\n"
        f"  max abs diff: {(logits_k1 - logits_std).abs().max().item()}"
    )
    print("  PASS: ETD-k=1 is bit-for-bit identical to ETD-disabled.")

    del model_k1
    _empty_cache()

    # ------------------------------------------------------------------
    # Test 2: ETD-k=2,3,4,5 run without error and produce the right shape.
    # ------------------------------------------------------------------
    for k in [2, 3, 4, 5]:
        print(f"Test 2: ETD-k={k} forward pass ...")
        cfg_k = make_config(etd_encoder_layers=7, etd_thinking_layers=4, etd_num_iterations=k)
        model_k = OLMo(cfg_k, init_params=False).to(dtype).eval()
        model_k.load_state_dict(state_dict)

        with torch.no_grad():
            out = model_k(input_ids)

        assert out.logits.shape == expected_shape, (
            f"FAIL: ETD-k={k} output shape {out.logits.shape} != expected {expected_shape}"
        )
        print(f"  PASS: ETD-k={k} output shape {out.logits.shape}.")

        del model_k
        _empty_cache()

    del model_std
    _empty_cache()

    # ------------------------------------------------------------------
    # Test 3: ETD-k=2 must produce DIFFERENT logits from ETD-disabled.
    #
    # If the ETD branch is not entered (e.g., a bug in the if-condition
    # causes the standard loop to run instead), the outputs would be
    # identical to ETD-disabled. This catches that silent failure.
    # With random weights the probability of accidental equality is zero.
    # ------------------------------------------------------------------
    print("Test 3: ETD-k=2 logits must differ from ETD-disabled ...")
    model_std2 = OLMo(make_config(), init_params=True).to(dtype).eval()
    state_dict2 = model_std2.state_dict()
    input_ids2 = torch.randint(0, 100278, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits_std2 = model_std2(input_ids2).logits

    cfg_k2 = make_config(etd_encoder_layers=7, etd_thinking_layers=4, etd_num_iterations=2)
    model_k2 = OLMo(cfg_k2, init_params=False).to(dtype).eval()
    model_k2.load_state_dict(state_dict2)

    with torch.no_grad():
        logits_k2 = model_k2(input_ids2).logits

    assert not torch.equal(logits_std2, logits_k2), (
        "FAIL: ETD-k=2 produced identical logits to ETD-disabled — "
        "the ETD branch may not have been entered."
    )
    print("  PASS: ETD-k=2 logits differ from ETD-disabled (ETD branch is active).")

    # ------------------------------------------------------------------
    # Test 4: The first thinking block (block index 7) is called exactly
    # k times per forward pass.
    #
    # We patch the block's forward method at the instance level to count
    # calls without affecting other blocks or the class definition.
    # ------------------------------------------------------------------
    print("Test 4: thinking block call count equals etd_num_iterations ...")
    for k in [1, 2, 3]:
        cfg_k = make_config(etd_encoder_layers=7, etd_thinking_layers=4, etd_num_iterations=k)
        model_k = OLMo(cfg_k, init_params=False).to(dtype).eval()
        model_k.load_state_dict(state_dict2)

        think_block = model_k.transformer.blocks[7]
        call_count = [0]
        original_forward = think_block.forward

        def counting_forward(*args, _orig=original_forward, _count=call_count, **kwargs):
            _count[0] += 1
            return _orig(*args, **kwargs)

        think_block.forward = counting_forward

        with torch.no_grad():
            model_k(input_ids2)

        assert call_count[0] == k, (
            f"FAIL: ETD-k={k} — thinking block called {call_count[0]} times, expected {k}."
        )
        print(f"  PASS: ETD-k={k} — thinking block called exactly {k} time(s).")

        del model_k
        _empty_cache()

    # ------------------------------------------------------------------
    # Test 5: When etd_encoder_layers is not set (defaults to None),
    # the ETD-off (else) branch executes.
    #
    # Two checks together prove this:
    #   (a) Block 7 is called exactly once — standard sequential pass,
    #       no looping. If the ETD branch ran with k>1 it would be >1.
    #   (b) Output is identical to the ETD-disabled baseline — the else
    #       branch is the original code, so outputs must match exactly.
    # ------------------------------------------------------------------
    print("Test 5: ETD-off branch executes when etd_encoder_layers is not set ...")
    model_off = OLMo(make_config(), init_params=False).to(dtype).eval()
    model_off.load_state_dict(state_dict2)

    think_block_off = model_off.transformer.blocks[7]
    call_count_off = [0]
    original_forward_off = think_block_off.forward

    def counting_forward_off(*args, _orig=original_forward_off, _count=call_count_off, **kwargs):
        _count[0] += 1
        return _orig(*args, **kwargs)

    think_block_off.forward = counting_forward_off

    with torch.no_grad():
        logits_off = model_off(input_ids2).logits

    assert call_count_off[0] == 1, (
        f"FAIL: ETD-off — block 7 called {call_count_off[0]} times, expected 1."
    )
    assert torch.equal(logits_std2, logits_off), (
        "FAIL: ETD-off output differs from ETD-disabled baseline."
    )
    print("  PASS: ETD-off branch executed — block 7 called exactly once.")
    print("  PASS: ETD-off output is bit-for-bit identical to ETD-disabled baseline.")

    del model_off, model_std2, model_k2
    _empty_cache()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
