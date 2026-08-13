"""Shared infrastructure for the Q2 benchmarks.

Everything here is used by every experiment (E0-E7), so it is the single place where
model construction, memory accounting and result recording are defined. See
`Documents/final_report/Q2/Experiment_plan.md` for the design and
`Documents/final_report/Q2/Metrics definition.md` for the authoritative metric
definitions -- this module implements those definitions and does not reinterpret them.

Two profiles, one source of truth:

  tiny  -- toy model, CPU, fp32. Debug harness logic locally in seconds.
  full  -- the real OLMo-2 1B, NPU, bf16. The measurements that go in the report.

Three arms per depth, so the report can answer "what does looping cost *compared to
the alternative way of buying the same depth*" rather than merely "looping costs more
than not looping":

  baseline  n_layers layers, executed once
  etd       n_layers layers, thinking block executed k times   -> D executed layers
  dense     D layers, executed once                            -> same D, more weights
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Long sweeps are normally run through `| tee`, and Python block-buffers stdout when
# it is piped -- so progress appears only in multi-KB bursts, and a healthy run looks
# hung. Line-buffer stdout on import so every experiment prints as it goes.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001  -- not available on every stream type
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "q2_results"
RAW_DIR = RESULTS_DIR / "raw"

ARMS = ("baseline", "etd", "dense")


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, Any]] = {
    # Toy model for harness development on a laptop CPU. ETD split 3-2*k-3.
    "tiny": dict(
        d_model=64,
        n_heads=4,
        n_layers=8,
        mlp_ratio=4,
        max_sequence_length=4096,
        vocab_size=256,
        embedding_size=256,
        eos_token_id=0,
        pad_token_id=1,
        rope_theta=500_000,
        flash_attention=False,
        etd_encoder_layers=3,
        etd_thinking_layers=2,
        dtype=torch.float32,
    ),
    # The real OLMo-2 1B mid-training architecture. ETD split 7-4*k-5.
    # max_sequence_length is raised from the trained 4096 so the sweep can reach 16k:
    # valid for a *cost* measurement, not a quality one (Experiment_plan.md section 8).
    "full": dict(
        d_model=2048,
        n_heads=16,
        n_layers=16,
        mlp_ratio=8,
        max_sequence_length=32768,
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


def effective_depth(profile: str, arm: str, k: int = 1) -> int:
    """Number of *executed* layers -- `L_eff` / `D` / `n_eff_kv_layers` in the docs.

    This is the quantity every cost in the report scales with, and it is deliberately
    not the number of *stored* layers, which is what looping holds fixed.
    """
    p = PROFILES[profile]
    if arm in ("baseline",):
        return p["n_layers"]
    depth = p["n_layers"] + p["etd_thinking_layers"] * (k - 1)
    return depth  # 'etd' executes it by looping; 'dense' by having that many layers


def resolve_device(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    from olmo.npu_util import is_npu_available

    if is_npu_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def make_config(profile: str, device: str, arm: str, k: int = 1, **overrides):
    """Build a ModelConfig for one arm.

    `dense` gets `n_layers = D` and no ETD fields, so it executes the same number of
    layers as `etd` at the same k but stores each of them separately.
    """
    from olmo.config import (
        ActivationType,
        BlockType,
        InitFnType,
        LayerNormType,
        ModelConfig,
    )

    assert arm in ARMS, f"unknown arm {arm!r}"
    p = PROFILES[profile]

    if arm == "dense":
        n_layers = effective_depth(profile, "etd", k)
        etd_enc = etd_think = None
        etd_k = 1
    elif arm == "etd":
        n_layers = p["n_layers"]
        etd_enc, etd_think = p["etd_encoder_layers"], p["etd_thinking_layers"]
        etd_k = k
    else:  # baseline
        n_layers = p["n_layers"]
        etd_enc = etd_think = None
        etd_k = 1

    cfg = ModelConfig(
        d_model=p["d_model"],
        n_heads=p["n_heads"],
        n_layers=n_layers,
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
        init_device=device,  # NOT 'meta' -- weights must be materialised
        etd_encoder_layers=etd_enc,
        etd_thinking_layers=etd_think,
        etd_num_iterations=etd_k,
    )
    # Opt-in KV caching for looped inference; harmless when the field is absent.
    if arm == "etd" and hasattr(cfg, "etd_kv_cache"):
        cfg.etd_kv_cache = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def build_model(profile: str, device: str, arm: str, k: int = 1, init_params: bool = True, **overrides):
    """Instantiate on the target device, in the profile's dtype, in eval mode.

    ``init_params=False`` allocates the parameters without running the random
    initialiser. Byte counts and shapes are identical, so it is the right choice when
    only the sizes matter (E1) and saves a lot of time at 3.4 B parameters.
    """
    from olmo.model import OLMo

    cfg = make_config(profile, device, arm, k, **overrides)
    model = OLMo(cfg, init_params=init_params).to(PROFILES[profile]["dtype"]).eval()
    # Guard against the `init_device: meta` trap: meta tensors report zero bytes and
    # every memory number downstream would be silently meaningless.
    assert param_bytes(model) > 0, "model has no materialised parameters"
    return model


# --------------------------------------------------------------------------------------
# Byte accounting -- measured, never assumed
# --------------------------------------------------------------------------------------


def param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def grad_bytes(model) -> int:
    return sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)


def optimizer_state_bytes(optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def kv_cache_bytes(attn_key_values) -> int:
    """Exact KV size, by summing the returned cache tensors.

    Preferred over the analytical `2 * n_kv_heads * d_head * dtype * D * B * S`, which
    then becomes a prediction this can be checked against rather than an assumption.
    """
    if not attn_key_values:
        return 0
    total = 0
    for entry in attn_key_values:
        for t in entry:
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
    return total


def analytical_kv_bytes(profile: str, depth: int, batch: int, seq_len: int) -> int:
    """2 (K,V) x n_kv_heads x d_head x dtype x D x B x S."""
    p = PROFILES[profile]
    head_dim = p["d_model"] // p["n_heads"]
    dtype_bytes = torch.finfo(p["dtype"]).bits // 8
    return 2 * p["n_heads"] * head_dim * dtype_bytes * depth * batch * seq_len


def logits_bytes(profile: str, batch: int, positions: int) -> int:
    p = PROFILES[profile]
    dtype_bytes = torch.finfo(p["dtype"]).bits // 8
    return batch * positions * p["embedding_size"] * dtype_bytes


# --------------------------------------------------------------------------------------
# Device memory and timing
# --------------------------------------------------------------------------------------


class DeviceMemory:
    """Thin wrapper over the accelerator memory API, tolerant of it being absent.

    Reports *allocated* memory: reserved reflects the caching allocator's pool and
    carries history across calls, while allocated tracks live tensors and is
    reproducible. Reserved is still exposed, because it is what actually causes OOM.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        if device.startswith("npu"):
            self.backend = getattr(torch, "npu", None)
        elif device.startswith("cuda"):
            self.backend = torch.cuda
        else:
            self.backend = None

    @property
    def available(self) -> bool:
        return self.backend is not None and hasattr(self.backend, "max_memory_allocated")

    def _call(self, name: str) -> int:
        fn = getattr(self.backend, name, None) if self.backend else None
        return int(fn()) if fn else 0

    def reset_peak(self) -> None:
        fn = getattr(self.backend, "reset_peak_memory_stats", None) if self.backend else None
        if fn:
            fn()

    def empty_cache(self) -> None:
        fn = getattr(self.backend, "empty_cache", None) if self.backend else None
        if fn:
            fn()

    def allocated(self) -> int:
        return self._call("memory_allocated")

    def peak_allocated(self) -> int:
        return self._call("max_memory_allocated")

    def reserved(self) -> int:
        return self._call("memory_reserved")

    def peak_reserved(self) -> int:
        return self._call("max_memory_reserved")

    def sync(self) -> None:
        """Kernel launches are async: without this we time the launch, not the work."""
        fn = getattr(self.backend, "synchronize", None) if self.backend else None
        if fn:
            fn()


@dataclass
class Timing:
    median_s: float
    mean_s: float
    min_s: float
    max_s: float
    spread_pct: float
    samples: int

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            median_s=self.median_s,
            mean_s=self.mean_s,
            min_s=self.min_s,
            max_s=self.max_s,
            spread_pct=self.spread_pct,
            samples=self.samples,
        )


def time_repeated(fn, mem: DeviceMemory, warmup: int = 5, measured: int = 20) -> Timing:
    """Median-of-N timing with warmup and explicit synchronisation.

    Median rather than mean because the machine is shared: a contended sample is an
    outlier, and the spread is reported so a contaminated point is visible rather than
    silently averaged in.
    """
    for _ in range(warmup):
        fn()
    mem.sync()

    samples = []
    for _ in range(measured):
        mem.sync()
        start = time.perf_counter()
        fn()
        mem.sync()
        samples.append(time.perf_counter() - start)

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    mean = sum(samples) / n
    spread = 100.0 * (samples[-1] - samples[0]) / median if median > 0 else 0.0
    return Timing(median, mean, samples[0], samples[-1], spread, n)


# --------------------------------------------------------------------------------------
# Provenance -- a number is never orphaned from the conditions that produced it
# --------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cpu_affinity() -> Dict[str, Any]:
    """Which CPUs this process may run on, and whether it is pinned.

    E0 established that CPU pinning is not optional here. Unpinned, decode latency
    stepped between roughly 50, 54 and 61 ms within a single process -- 22% drift, far
    larger than the effects the sweeps must resolve. Pinned to one NUMA node the same
    measurement holds to 0.5%. Every result therefore records its affinity, so an
    unpinned run cannot silently contaminate the results.
    """
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return dict(supported=False, pinned=None, n_cpus=None)
    total = os.cpu_count() or len(cpus)
    return dict(
        supported=True,
        pinned=len(cpus) < total,
        n_cpus=len(cpus),
        total_cpus=total,
        range=f"{cpus[0]}-{cpus[-1]}" if cpus else "",
    )


def warn_if_unpinned() -> None:
    """Print a loud warning if the process is free to migrate across cores."""
    aff = cpu_affinity()
    if aff.get("supported") and not aff.get("pinned"):
        print(
            "\n  *** WARNING: process is NOT pinned to a CPU set. E0 measured up to 22% latency\n"
            "  *** drift from core migration when unpinned, against 0.5% when pinned. Re-run as:\n"
            "  ***     taskset -c 0-47 python <script> ...\n"
            "  *** Memory results are unaffected; latency results will not be trustworthy.\n"
        )


def host_state() -> Dict[str, Any]:
    """Host load and CPU affinity, recorded with every result -- the machine is shared."""
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:  # noqa: BLE001
        load1 = load5 = load15 = -1.0
    return dict(
        hostname=socket.gethostname(),
        cpu_count=os.cpu_count(),
        loadavg_1m=round(load1, 2),
        loadavg_5m=round(load5, 2),
        loadavg_15m=round(load15, 2),
        affinity=cpu_affinity(),
    )


def npu_smi_raw() -> Optional[str]:
    """Raw `npu-smi info`, for cross-checking torch's allocator accounting.

    On Ascend, CANN operators can allocate workspace outside PyTorch's caching
    allocator, in which case torch under-reports. Captured raw rather than parsed,
    because the output format is not stable enough to depend on.
    """
    try:
        return subprocess.check_output(["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30).decode()
    except Exception:  # noqa: BLE001
        return None


def run_metadata(profile: str, device: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    try:
        import torch_npu  # type: ignore

        torch_npu_version = getattr(torch_npu, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        torch_npu_version = None
    return dict(
        timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        commit=_git("rev-parse", "HEAD"),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        # Scoped away from q2_results: a run writes its own outputs there, so an
        # unscoped check reports every run as dirty and the flag stops meaning
        # anything. What matters is whether the *code* differs from `commit`.
        dirty=bool(_git("status", "--porcelain", "--", ":(exclude)q2_results")),
        profile=profile,
        device=device,
        dtype=str(PROFILES[profile]["dtype"]),
        torch=torch.__version__,
        torch_npu=torch_npu_version,
        python=platform.python_version(),
        host=host_state(),
    )


def write_result(name: str, payload: Dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def gb(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def mb(num_bytes: float) -> float:
    return num_bytes / (1024**2)


def write_csv(name: str, rows, fieldnames=None, meta=None) -> Path:
    """Aggregated results, committed to the branch so they reach the Mac by git pull.

    A `<name>.meta.json` sidecar is written alongside. The CSV carries measurements
    only; without the sidecar a number in the report cannot be traced back to the
    code, device and machine state that produced it, which is the point of keeping
    the file at all.
    """
    import csv
    import json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    rows = list(rows)
    if not rows:
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if meta is not None:
        (RESULTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Sweep machinery -- shared by E3 (prefill), E4 (decode) and E5 (serving frontier)
# --------------------------------------------------------------------------------------


def is_oom(exc: BaseException) -> bool:
    """Is this exception an out-of-memory condition rather than a real failure?

    OOM is a *result* in these sweeps -- it defines the frontier -- so it must be
    distinguished from a genuine error. torch_npu and CUDA word it differently.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in ("out of memory", "oom", "insufficient memory", "alloc failed"))


class IncrementalWriter:
    """Append each grid point as it completes, so a crash costs one point, not a sweep.

    The machine is shared, sweeps run for tens of minutes, and another user's job can
    destabilise the node. Writing at the end would risk losing everything.
    """

    def __init__(self, name: str) -> None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RAW_DIR / f"{name}.jsonl"
        # Result files are evidence: a re-run must not silently destroy a previous
        # run's data. An existing file is moved aside with a numeric suffix.
        if self.path.exists() and self.path.stat().st_size > 0:
            n = 1
            while (backup := RAW_DIR / f"{name}.prev{n}.jsonl").exists():
                n += 1
            self.path.rename(backup)
            print(f"  [note] existing {self.path.name} moved to {backup.name}")
        self.rows: list = []
        self.path.write_text("")

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def canary(device: str, mem: DeviceMemory, size: int = 2048) -> float:
    """Time a fixed tiny workload, to detect drift over the course of a sweep.

    Run at the start, middle and end. If the canary moves, conditions changed while
    the sweep was running and the results need re-examining -- regardless of what
    anyone believed about the machine being quiet.
    """
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    for _ in range(3):
        a @ a
    mem.sync()
    start = time.perf_counter()
    for _ in range(10):
        a @ a
    mem.sync()
    elapsed = (time.perf_counter() - start) / 10
    del a
    mem.empty_cache()
    return elapsed
