#!/usr/bin/env python3
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="OLMo authoritative dataset mirror check")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--log-dir", default=None)
    return p.parse_args()


def load_training_params(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data = cfg.get("data", {})
    model = cfg.get("model", {})

    if data.get("datasets"):
        raise ValueError("data.datasets is set; this script only handles data.paths.")

    paths = data.get("paths")
    if not paths:
        raise ValueError("data.paths is empty or missing.")

    dtype_name = data.get("memmap_dtype", "uint16")
    chunk_size = model.get("max_sequence_length", 1024)
    vocab_size = model.get("vocab_size", 65536)

    dtype = getattr(np, dtype_name)
    item_size = int(np.dtype(dtype).itemsize)

    return {
        "paths": paths,
        "dtype": dtype,
        "dtype_name": dtype_name,
        "item_size": item_size,
        "chunk_size": chunk_size,
        "bytes_per_instance": item_size * chunk_size,
        "vocab_size": vocab_size,
    }


def load_manifest(manifest_path):
    records = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            records[row["local_path"]] = {
                "url": row["url"],
                "expected_bytes": int(row["expected_bytes"]),
                "got_bytes": int(row["got_bytes"]),
                "status": row["status"],
            }
    return records


def verify_file(path, dtype, chunk_size, vocab_size, manifest_expected_bytes):
    issues = []

    if not os.path.exists(path):
        return False, ["file-missing"], 0, 0, 0

    size_bytes = os.path.getsize(path)

    if size_bytes == 0:
        return False, ["file-empty"], 0, 0, 0

    item_size = int(np.dtype(dtype).itemsize)
    token_count = size_bytes // item_size
    instance_count = size_bytes // (item_size * chunk_size)

    if manifest_expected_bytes > 0 and size_bytes != manifest_expected_bytes:
        issues.append(
            f"size-mismatch: disk={size_bytes:,} manifest={manifest_expected_bytes:,}"
        )

    try:
        arr = np.memmap(path, dtype=dtype, mode="r")
        n = len(arr)

        if n == 0:
            issues.append("memmap-array-length-zero")
        else:
            first = int(arr[0])
            last = int(arr[n - 1])
            del arr

            if not (0 <= first < vocab_size):
                issues.append(f"first-token-out-of-range: {first} vocab_size={vocab_size}")
            if not (0 <= last < vocab_size):
                issues.append(f"last-token-out-of-range: {last} vocab_size={vocab_size}")

    except Exception as exc:
        issues.append(f"memmap-error: {exc}")
        token_count = 0
        instance_count = 0

    ok = len(issues) == 0
    return ok, issues, size_bytes, token_count, instance_count


def main():
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logfh = None
    log_path = None

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"dataset_check_{ts}.log")
        logfh = open(log_path, "w", buffering=1)

    def log(msg=""):
        print(msg, flush=True)
        if logfh:
            logfh.write(msg + "\n")

    log("=" * 68)
    log("  OLMo Authoritative Dataset Check")
    log("=" * 68)
    log(f"  Timestamp : {ts}")
    log(f"  Config    : {args.config}")
    log(f"  Manifest  : {args.manifest}")
    if log_path:
        log(f"  Log       : {log_path}")
    log()

    log("--- [1/5] Parse training config ---")
    try:
        params = load_training_params(args.config)
    except Exception as exc:
        log(f"  ERROR loading config: {exc}")
        sys.exit(1)

    paths = params["paths"]

    log(f"  data.paths entries       : {len(paths)}")
    log(f"  data.memmap_dtype        : {params['dtype_name']}")
    log(f"  model.max_sequence_length: {params['chunk_size']} tokens")
    log(f"  bytes per instance       : {params['bytes_per_instance']}")
    log(f"  model.vocab_size         : {params['vocab_size']}")
    log(f"  first expected path      : {paths[0]}")
    log(f"  last  expected path      : {paths[-1]}")
    log()

    remote = [p for p in paths if p.startswith("http://") or p.startswith("https://")]
    if remote:
        log(f"  ERROR: {len(remote)} paths are still remote URLs.")
        log(f"  Example: {remote[0]}")
        log("  FINAL RESULT: FAIL")
        sys.exit(1)

    log("--- [2/5] Load manifest ---")
    try:
        manifest = load_manifest(args.manifest)
    except Exception as exc:
        log(f"  ERROR loading manifest: {exc}")
        sys.exit(1)

    log(f"  Manifest entries         : {len(manifest)}")
    log()

    log("--- [3/5] Cross-reference: config paths vs manifest local_path ---")
    config_set = set(paths)
    manifest_set = set(manifest.keys())

    in_config_not_manifest = sorted(config_set - manifest_set)
    in_manifest_not_config = sorted(manifest_set - config_set)

    if not in_config_not_manifest:
        log(f"  OK : all {len(paths)} config paths present in manifest")
    else:
        log(f"  MISMATCH: {len(in_config_not_manifest)} config paths not in manifest")
        for p in in_config_not_manifest:
            log(f"    MISSING-FROM-MANIFEST {p}")

    if not in_manifest_not_config:
        log("  OK : no extra paths in manifest beyond config")
    else:
        log(f"  MISMATCH: {len(in_manifest_not_config)} manifest paths not in config")
        for p in in_manifest_not_config:
            log(f"    EXTRA-IN-MANIFEST {p}")

    log()

    log("--- [4/5] Per-file verification ---")
    log(
        f"  Checking {len(paths)} files "
        f"[dtype={params['dtype_name']}, chunk={params['chunk_size']}, vocab={params['vocab_size']}]"
    )
    log()

    bad_files = []
    total_size = 0
    total_tokens = 0
    total_instances = 0

    for i, path in enumerate(paths):
        exp_bytes = manifest.get(path, {}).get("expected_bytes", -1)

        ok, issues, size_bytes, tok_count, inst_count = verify_file(
            path,
            dtype=params["dtype"],
            chunk_size=params["chunk_size"],
            vocab_size=params["vocab_size"],
            manifest_expected_bytes=exp_bytes,
        )

        if ok:
            total_size += size_bytes
            total_tokens += tok_count
            total_instances += inst_count
        else:
            bad_files.append((path, issues))
            log(f"  FAIL [{i + 1}/{len(paths)}] {path}")
            for issue in issues:
                log(f"       -> {issue}")

        if (i + 1) % 100 == 0 and (i + 1) < len(paths):
            log(f"  ... {i + 1}/{len(paths)} checked, {len(bad_files)} issue(s) so far ...")

    log()

    log("--- [5/5] Summary ---")
    log(f"  Expected files from data.paths        : {len(paths)}")
    log(f"  Files verified OK                     : {len(paths) - len(bad_files)}")
    log(f"  Files with issues                     : {len(bad_files)}")
    log()
    log(f"  Total on-disk size of OK files        : {total_size / 1024**3:.4f} GiB")
    log(f"                                          {total_size:,} bytes")
    log(f"  Total token count of OK files         : {total_tokens:,}")
    log(f"                                          ~{total_tokens / 1e9:.4f} BT")
    log(f"  Total training instances              : {total_instances:,}")
    log(f"                                          ~{total_instances * params['chunk_size'] / 1e9:.4f} BT consumed")
    log()
    log("  Manifest cross-check:")
    log(f"    Config paths missing from manifest  : {len(in_config_not_manifest)}")
    log(f"    Manifest paths not in config        : {len(in_manifest_not_config)}")
    log()

    if bad_files:
        log("  BAD FILE DETAIL:")
        for path, issues in bad_files:
            log(f"    {path}")
            for issue in issues:
                log(f"      -> {issue}")
        log()

    overall_pass = (
        len(bad_files) == 0
        and len(in_config_not_manifest) == 0
        and len(in_manifest_not_config) == 0
    )

    log("=" * 68)

    if overall_pass:
        log("  FINAL RESULT : PASS")
        log()
        log("  Every file in data.paths is present on disk, non-empty,")
        log("  matches the manifest Content-Length, opens as a valid")
        log("  memmap array, and has first/last tokens within vocab range.")
        log("  Config and manifest are in full agreement.")
        log("  The local mirror is ready for training.")
    else:
        log("  FINAL RESULT : FAIL")
        if bad_files:
            log(f"    {len(bad_files)} file(s) failed verification")
        if in_config_not_manifest:
            log(f"    {len(in_config_not_manifest)} config path(s) missing from manifest")
        if in_manifest_not_config:
            log(f"    {len(in_manifest_not_config)} manifest path(s) not expected by config")

    log("=" * 68)

    if logfh:
        logfh.close()
        print(f"\nLog saved: {log_path}", flush=True)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
