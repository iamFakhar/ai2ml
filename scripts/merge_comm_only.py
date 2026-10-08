#!/usr/bin/env python3
"""Merge --profile_only comm/overlap sidecars into each run's summary.json,
preserving the existing (clean PASS-1) timing. No GPU work.

`python -m src.runner --profile_only` skips PASS 1 and writes a
<exp>/comm_only.json holding the freshly measured comm_metrics + (boundary-fixed,
multi-rank) overlap_trace. This script, for each sidecar, reloads the run's
existing summary.json + per_step_rank0.json and calls the SAME aggregate() the
runner uses — fed the fresh comm/overlap but the ORIGINAL timing records — then
writes back ONLY the comm/overlap/backward-split columns. Timing, throughput,
memory, straggler and MFU are left byte-for-byte untouched.

Pipeline:
    python -m src.runner --profile_only ...            # per cell (GPU)
    python -m scripts.merge_comm_only \
        --exp_root outputs/experiments --base configs/base.yaml
    python -m scripts.rebuild_master_csv --inplace     # summary.json -> master CSV
    python -m scripts.augment_recompute_calibration outputs/all_experiments_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from copy import deepcopy

import yaml

from src.instrumentation.metrics import MetricsAggregator, StepRecord

# Only these columns are overwritten; everything else in summary.json is kept.
MERGE_COLS = [
    # comm (parse_comm_metrics passthrough)
    "total_nccl_comm_time_ms", "nccl_allreduce_time_ms", "nccl_allgather_time_ms",
    "nccl_reducescatter_time_ms", "nccl_allreduce_count", "nccl_allgather_count",
    "nccl_reducescatter_count", "nccl_collective_count", "cuda_kernel_count",
    "mean_kernel_duration_us", "mean_collective_size_bytes",
    "measured_comm_volume_bytes",
    # volume + achieved bandwidth
    "measured_comm_volume_gb", "theoretical_comm_volume_gb",
    "achieved_bw_gbps", "bandwidth_efficiency",
    # overlap + QA (notes/07)
    "overlap_percentage_trace", "overlap_analyzed_steps", "overlap_step_spread_pp",
    "overlap_stable", "overlap_n_ranks",
    # backward split (bwd_exposed_comm changes -> bwd_compute changes)
    "recompute_ms", "bwd_exposed_comm_ms", "bwd_compute_ms",
]

_STEP_FIELDS = StepRecord.__dataclass_fields__.keys()


def _load_steps(path):
    data = json.load(open(path))
    steps = data if isinstance(data, list) else data.get("steps", [])
    return [StepRecord(**{k: s[k] for k in _STEP_FIELDS if k in s}) for s in steps]


def _load_bw_curve(cfg):
    path = cfg.get("hardware", {}).get("nccl_bw_curve_path")
    if path and os.path.exists(path):
        return json.load(open(path))
    return None


def _gc_flag(summary):
    gc = summary.get("grad_checkpoint", True)
    return gc if isinstance(gc, bool) else (str(gc) == "True")


def merge_one(exp_dir, base, bw_curve):
    sc = json.load(open(os.path.join(exp_dir, "comm_only.json")))
    comm = sc.get("comm_metrics") or None
    overlap = sc.get("overlap_trace") or None
    if not comm:
        return "skip: sidecar has no comm_metrics (Pass-2 OOM?)"
    sumf = os.path.join(exp_dir, "summary.json")
    psf = os.path.join(exp_dir, "per_step_rank0.json")
    if not (os.path.exists(sumf) and os.path.exists(psf)):
        return "skip: missing summary.json or per_step_rank0.json"

    summary = json.load(open(sumf))
    cfg = deepcopy(base)
    cfg["grad_accum_steps"] = int(summary.get("gradient_accumulation_steps") or 1)
    cfg.setdefault("workload", {})
    cfg["workload"]["gradient_checkpointing"] = _gc_flag(summary)

    strat = summary["strategy"]
    ws = int(summary["world_size"])
    P = int(summary.get("trainable_parameters") or 1)

    agg = MetricsAggregator(cfg, strat, P, ws)
    for rec in _load_steps(psf):
        agg.record(rec)
    # Same call the runner uses (timing from the loaded records, comm/overlap
    # from the sidecar). straggler/per-rank/mfu columns aren't in MERGE_COLS,
    # so their recomputation here is discarded.
    out = agg.aggregate(comm_metrics=comm, bw_curve=bw_curve, overlap_trace=overlap)

    patched = [c for c in MERGE_COLS if c in out]
    for c in patched:
        summary[c] = out[c]
    with open(sumf, "w") as f:
        json.dump(summary, f, indent=2)
    stab = out.get("overlap_stable", 1.0)
    return (f"ok: patched {len(patched)} cols, "
            f"overlap={out.get('overlap_percentage_trace', 0):.1f}% "
            f"{'STABLE' if stab else 'UNSTABLE'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="outputs/experiments",
                    help="dir containing <experiment_name>/comm_only.json")
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    bw_curve = _load_bw_curve(base)
    sidecars = sorted(glob.glob(os.path.join(args.exp_root, "*", "comm_only.json")))
    print(f"found {len(sidecars)} comm_only.json sidecar(s)")
    ok = 0
    for f in sidecars:
        d = os.path.dirname(f)
        try:
            status = merge_one(d, base, bw_curve)
        except Exception as e:  # noqa: BLE001 - report and continue
            status = f"FAIL {type(e).__name__}: {e}"
        if status.startswith("ok"):
            ok += 1
        print(f"  {os.path.basename(d)}: {status}")
    print(f"\nmerged {ok}/{len(sidecars)}. Next: "
          f"python -m scripts.rebuild_master_csv --inplace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
