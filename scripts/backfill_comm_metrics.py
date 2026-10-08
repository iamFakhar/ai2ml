#!/usr/bin/env python3
"""Backfill the newly-wired comm/overlap/attribution columns into an existing
experiment CSV, WITHOUT re-running any GPU work.

For each row it reconstructs the real MetricsAggregator from the saved
per_step_rank0.json + summary.json, computes timeline overlap from the Kineto
trace already on disk, and calls the SAME aggregate() the runner uses — then
patches only the previously-blank columns back into the row:

    measured_comm_volume_gb, mean_collective_size_bytes,
    achieved_bw_gbps, bandwidth_efficiency, overlap_percentage_trace,
    bwd_compute_ms, recompute_ms, bwd_exposed_comm_ms

Everything else in the row is left untouched. Output is written to a new file
(<csv>_backfilled.csv) so the original is preserved.

Usage:
  python -m scripts.backfill_comm_metrics \
      --exp_root outputs_ddp_decomp --base configs/base_ddp_decomp.yaml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from copy import deepcopy

import yaml

from src.instrumentation.metrics import MetricsAggregator, StepRecord
from src.instrumentation.profiler import compute_overlap_from_trace

# Columns this backfill (re)populates; all others are copied verbatim.
PATCH_COLS = [
    "measured_comm_volume_gb", "mean_collective_size_bytes",
    "achieved_bw_gbps", "bandwidth_efficiency", "overlap_percentage_trace",
    "bwd_compute_ms", "recompute_ms", "bwd_exposed_comm_ms",
]

_STEP_FIELDS = StepRecord.__dataclass_fields__.keys()


def _params_from_theoretical(theo_gb: float, strategy: str, ws: int) -> int:
    if theo_gb <= 0 or ws <= 1:
        return 1  # single-GPU: comm path is inert, P is unused
    factor = 3 * (ws - 1) / ws if strategy == "fsdp_zero3" else 2 * (ws - 1) / ws
    return int(round(theo_gb * 1e9 / (factor * 2)))


def _load_steps(path: str):
    data = json.load(open(path))
    steps = data if isinstance(data, list) else data.get("steps", [])
    recs = []
    for s in steps:
        recs.append(StepRecord(**{k: s[k] for k in _STEP_FIELDS if k in s}))
    return recs


def _comm_metrics_from_row(row: dict):
    t = float(row.get("total_nccl_comm_time_ms") or 0.0)
    if t <= 0:
        return None
    g = lambda k: float(row.get(k) or 0.0)
    return {
        "total_nccl_comm_time_ms": t,
        "nccl_allreduce_time_ms": g("nccl_allreduce_time_ms"),
        "nccl_allgather_time_ms": g("nccl_allgather_time_ms"),
        "nccl_reducescatter_time_ms": g("nccl_reducescatter_time_ms"),
        "nccl_allreduce_count": g("nccl_allreduce_count"),
        "nccl_allgather_count": g("nccl_allgather_count"),
        "nccl_reducescatter_count": g("nccl_reducescatter_count"),
        "cuda_kernel_count": g("cuda_kernel_count"),
        "mean_kernel_duration_us": g("mean_kernel_duration_us"),
    }


def backfill_row(row: dict, base: dict, bw_curve, exp_root: str) -> dict:
    exp = row["experiment_name"]
    d = os.path.join(exp_root, "experiments", exp)
    ps = os.path.join(d, "per_step_rank0.json")
    if not os.path.exists(ps):
        print(f"  SKIP {exp}: no per_step_rank0.json")
        return row

    ws = int(row["world_size"])
    strat = row["strategy"]
    theo = float(row.get("theoretical_comm_volume_gb") or 0.0)
    P = _params_from_theoretical(theo, strat, ws)

    cfg = deepcopy(base)
    cfg["grad_accum_steps"] = int(row.get("gradient_accumulation_steps") or 1)
    cfg.setdefault("workload", {})["gradient_checkpointing"] = (
        str(row.get("grad_checkpoint", "True")) == "True")

    comm_metrics = _comm_metrics_from_row(row)

    overlap_trace = None
    if comm_metrics:
        prof_cfg = base.get("instrumentation", {}).get("profiler", {})
        analyze_steps = int(prof_cfg.get("analyze_steps", prof_cfg.get("active", 5)))
        tol = float(prof_cfg.get("stability_tol_pp", 8.0))
        # Prefer the current per-rank export naming; fall back to legacy traces.
        traces = sorted(glob.glob(os.path.join(d, "traces", "trace_rank*.json")))
        if not traces:
            traces = sorted(glob.glob(os.path.join(d, "traces", "*.pt.trace.json")))
        per_rank = [compute_overlap_from_trace(t, analyze_steps, tol) for t in traces]
        per_rank = [o for o in per_rank if o and o.get("comm_busy_us", 0.0) > 0.0]
        if per_rank:
            # comm-weighted overlap; per-rank-mean busy/overlapped for per-step exposed.
            n = len(per_rank)
            sum_busy = sum(o["comm_busy_us"] for o in per_rank)
            sum_ovl = sum(o["comm_overlapped_us"] for o in per_rank)
            overlap_trace = {
                "overlap_percentage_trace": 100.0 * sum_ovl / sum_busy if sum_busy > 0 else 0.0,
                "comm_busy_us": sum_busy / n,
                "comm_overlapped_us": sum_ovl / n,
                "analyzed_steps": float(analyze_steps),
                "overlap_ranks": list(range(n)),
                "overlap_step_spread_pp": max(o.get("overlap_step_spread_pp", 0.0) for o in per_rank),
                "overlap_stable": all(o.get("overlap_stable", True) for o in per_rank),
            }

    agg_o = MetricsAggregator(cfg, strat, P, ws)
    for rec in _load_steps(ps):
        agg_o.record(rec)
    out = agg_o.aggregate(comm_metrics=comm_metrics, bw_curve=bw_curve,
                          overlap_trace=overlap_trace)

    patched = dict(row)
    for col in PATCH_COLS:
        if col in out:
            patched[col] = out[col]
    return patched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_root", default="outputs_ddp_decomp")
    p.add_argument("--base", default="configs/base_ddp_decomp.yaml")
    args = p.parse_args()

    base = yaml.safe_load(open(args.base))
    curve_path = base["hardware"].get("nccl_bw_curve_path")
    bw_curve = json.load(open(curve_path)) if curve_path and os.path.exists(curve_path) else None

    csv_path = os.path.join(args.exp_root, "all_experiments_summary.csv")
    rows = list(csv.DictReader(open(csv_path)))
    fieldnames = list(rows[0].keys()) if rows else []
    # The pre-fix CSV lacks the new granularity column; insert it after counts.
    if "mean_collective_size_bytes" not in fieldnames:
        anchor = "nccl_reducescatter_count"
        idx = fieldnames.index(anchor) + 1 if anchor in fieldnames else len(fieldnames)
        fieldnames.insert(idx, "mean_collective_size_bytes")

    out_rows = []
    for r in rows:
        print(f"backfilling {r['experiment_name']} ...")
        out_rows.append(backfill_row(r, base, bw_curve, args.exp_root))

    out_path = csv_path.replace(".csv", "_backfilled.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
