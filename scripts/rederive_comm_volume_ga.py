#!/usr/bin/env python3
"""Re-derive the ga-aware communication VOLUME + BANDWIDTH columns in place.

The bug: measured_comm_volume_gb was computed from the ga=1 per-step bus volume
(_theoretical_comm_gb x n_prof), but ZeRO-3 re-gathers parameters every
micro-batch, so the real per-optimizer-step volume scales ~ga while the measured
NCCL time already covers all micro-batches. Result: achieved_bw understated ~1/ga
at ga>1 for ZeRO-3 (ZeRO-2/DDP unaffected — their collectives occur once/step).

This recomputes, for each PROFILED ZeRO-3 ga>1 run, the four affected fields
  measured_comm_volume_gb, mean_collective_size_bytes,
  achieved_bw_gbps, bandwidth_efficiency
using the SAME code path as the runner (MetricsAggregator._bus_volume_gb /
_raw_data_gb + profiler.score_bandwidth), from values already in summary.json —
NO GPU work. It is idempotent: everything is derived from theoretical_comm_volume_gb
(the ga=1 reference, never rewritten) and n_prof, not by scaling current values.

Patches per-run summary.json (authoritative); run scripts.rebuild_master_csv
afterwards to refresh the master CSV.

Usage:
  python -m scripts.rederive_comm_volume_ga [--exp_root outputs/experiments]
                                            [--base configs/base.yaml]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import yaml

from src.instrumentation.metrics import MetricsAggregator
from src.instrumentation.profiler import score_bandwidth


def _num(d, k):
    try:
        return float(d.get(k) or "")
    except (TypeError, ValueError):
        return None


def _infer_n_prof(summaries) -> int:
    """n_prof (profiler active steps) is a constant; recover it from ga=1
    profiled rows where measured_comm_volume_gb == _theoretical_comm_gb x n_prof."""
    votes = {}
    for s in summaries:
        if int(s.get("gradient_accumulation_steps") or 1) != 1:
            continue
        theo, meas = _num(s, "theoretical_comm_volume_gb"), _num(s, "measured_comm_volume_gb")
        if theo and meas and theo > 0:
            k = round(meas / theo)
            votes[k] = votes.get(k, 0) + 1
    return max(votes, key=votes.get) if votes else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="outputs/experiments")
    ap.add_argument("--base", default="configs/base.yaml")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    curve_path = base.get("hardware", {}).get("nccl_bw_curve_path")
    bw_curve = (json.load(open(curve_path))
                if curve_path and os.path.exists(curve_path) else None)

    paths = glob.glob(os.path.join(args.exp_root, "*", "summary.json"))
    summaries = [(p, json.load(open(p))) for p in paths]
    n_prof = _infer_n_prof([s for _, s in summaries])
    print(f"n_prof (profiler active steps) = {n_prof}")

    patched = 0
    for path, s in summaries:
        strat = s.get("strategy")
        ga = int(s.get("gradient_accumulation_steps") or 1)
        ws = int(s.get("world_size") or 1)
        theo = _num(s, "theoretical_comm_volume_gb")
        t_ms = _num(s, "total_nccl_comm_time_ms")
        # only profiled cells whose volume actually changes: ZeRO-3, ga>1
        if strat != "fsdp_zero3" or ga <= 1 or not t_ms or t_ms <= 0 or not theo:
            continue

        # recover P from the ga=1 theoretical bus volume: theo = 3(n-1)/n * P*2/1e9
        f = (ws - 1) / ws
        P = int(round(theo * 1e9 / (3 * f * 2)))

        cfg = {"grad_accum_steps": ga,
               "protocol": {"warmup_steps_discard": 0},
               "instrumentation": {"profiler": {"active": n_prof}}}
        agg = MetricsAggregator(cfg, strat, P, ws)
        bus_gb = agg._bus_volume_gb() * n_prof
        raw_gb = agg._raw_data_gb() * n_prof

        counts = (_num(s, "nccl_allreduce_count") or 0) + \
                 (_num(s, "nccl_allgather_count") or 0) + \
                 (_num(s, "nccl_reducescatter_count") or 0)
        mean_size = (raw_gb * 1e9 / counts) if counts > 0 else 0.0
        times = {"allreduce": _num(s, "nccl_allreduce_time_ms") or 0.0,
                 "allgather": _num(s, "nccl_allgather_time_ms") or 0.0,
                 "reducescatter": _num(s, "nccl_reducescatter_time_ms") or 0.0}
        dominant = max(times, key=times.get)
        bw = score_bandwidth(measured_bytes=bus_gb * 1e9, comm_time_ms=t_ms,
                             bw_curve=bw_curve, world_size=ws,
                             collective=dominant, lookup_size_bytes=mean_size)

        old_bw = _num(s, "achieved_bw_gbps")
        s["measured_comm_volume_gb"] = bus_gb
        s["mean_collective_size_bytes"] = mean_size
        s["achieved_bw_gbps"] = bw["achieved_bw_gbps"]
        s["bandwidth_efficiency"] = bw["bandwidth_efficiency"]
        json.dump(s, open(path, "w"), indent=2)
        patched += 1
        print(f"  {s['experiment_name']:44s} ga={ga} "
              f"achieved_bw {old_bw:.1f} -> {bw['achieved_bw_gbps']:.1f} GB/s")

    print(f"\npatched {patched} ZeRO-3 ga>1 summary.json files. "
          f"Run: python -m scripts.rebuild_master_csv --inplace")


if __name__ == "__main__":
    main()
