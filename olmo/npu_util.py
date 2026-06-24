"""Minimal utilities for optional Huawei Ascend NPU support via torch_npu.

This module is intentionally import-light.  It only imports `torch` at module
level; `torch_npu` is imported lazily inside `is_npu_available()` so that
importing OLMo on a machine without torch_npu has zero cost and raises no
errors.
"""

import torch


def is_npu_available() -> bool:
    """Return True iff torch_npu is installed and at least one NPU device is reachable.

    Calling this on a machine without torch_npu always returns False.
    Existing CUDA / MPS / CPU code paths are unaffected when this returns False.
    """
    try:
        import torch_npu  # noqa: F401  – patches torch.npu namespace
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())
