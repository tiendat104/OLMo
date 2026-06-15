"""
Sanity check for the ETD (Encode-Think-Decode) forward pass implementation.

Verifies four properties:
  1. ETD-k=1 produces bit-for-bit identical logits to ETD-disabled (standard
     forward pass) when given the same weights and input.
  2. ETD-k=2,3,4,5 run without error and produce the correct output shape.
  3. ETD-k=2 produces DIFFERENT logits from ETD-disabled — proving the ETD
     branch is actually entered and the thinking loop changes the computation.
  4. The first thinking block (block 7) is called exactly k times during a
     forward pass with etd_num_iterations=k — directly verifying the loop count.

Architecture matches the OLMo 2 1B mid-training config exactly:
  16 layers, d_model=2048, n_heads=16, mlp_ratio=8, RoPE θ=500k,
  SwiGLU, RMSNorm, norm_after, QK-norm. ETD split: 7 encoder + 4 thinking + 5 decoder.

Requires Step 2 (ETD fields in olmo/config.py and ETD branch in olmo/model.py)
to be applied before this script can run.

Usage:
    CUDA_VISIBLE_DEVICES=<n> python scripts/test_etd_sanity.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo.config import ActivationType, BlockType, InitFnType, LayerNormType, ModelConfig
from olmo.model import OLMo


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
        init_device="cuda",
        etd_encoder_layers=etd_encoder_layers,
        etd_thinking_layers=etd_thinking_layers,
        etd_num_iterations=etd_num_iterations,
    )


def main():
    assert torch.cuda.is_available(), (
        "This script requires a CUDA GPU. Run with CUDA_VISIBLE_DEVICES=<n>."
    )

    torch.manual_seed(42)
    dtype = torch.bfloat16

    batch_size, seq_len = 2, 64
    # embedding_size=100352 (not vocab_size=100278) because weight_tying=False
    expected_shape = (batch_size, seq_len, 100352)

    input_ids = torch.randint(0, 100278, (batch_size, seq_len), device="cuda")

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
    torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()

    del model_std
    torch.cuda.empty_cache()

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
    input_ids2 = torch.randint(0, 100278, (batch_size, seq_len), device="cuda")

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
        torch.cuda.empty_cache()

    del model_std2, model_k2
    torch.cuda.empty_cache()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
