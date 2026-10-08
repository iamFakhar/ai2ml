#!/usr/bin/env python3
"""Comm-time attribution: split measured NCCL device time into TRANSFER vs WAIT,
and report exposed vs overlapped vs idle — to settle whether the DDP/ZeRO-2
achieved_bw drop at ga>1 is a real fabric slowdown or a wait/overlap artifact.

Per profiler step window (one optimizer step) on each exported Kineto trace:
  comm_time_sum_us  = Sum of NCCL device-kernel self-times (== total_nccl basis)
  comm_busy_us      = |union of NCCL kernel intervals|      (wall-clock device busy)
  transfer_floor_us = bus_volume_step / busbw_peak(size)    (irreducible transfer)
  wait_us           = max(0, comm_time_sum - transfer_floor)  (spin / sync / overhead)
  overlapped_us     = |comm_busy ∩ compute_busy|            (hidden behind compute)
  exposed_us        = comm_busy - overlapped                (on the critical path)
  idle_us           = window - |comm ∪ compute|             (pure bubble)

Reads runs under outputs/comm_attrib/experiments/<name>/{summary.json,traces/}.
Averages analyzed windows and exported ranks. Emits a per-(strategy, ga) table
+ a stacked figure. Read-only: does NOT modify the master CSV or live metrics.

Usage:
  python -m analysis.comm_attribution [--root outputs/comm_attrib] [--analyze_steps 5]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

from src.instrumentation.profiler import (
    _merge_intervals, _gpu_step_windows, _NCCL_DEVKERNEL_RE, _lookup_peak_bw,
)

STRAT_ORDER = ["ddp", "fsdp_zero2", "fsdp_zero3"]
STRAT_LABEL = {"ddp": "DDP", "fsdp_zero2": "ZeRO-2", "fsdp_zero3": "ZeRO-3"}


def _union_len(intervals):
    return sum(b - a for a, b in _merge_intervals([list(x) for x in intervals]))


def _intersect_len(a_u, b_u):
    i = j = 0; tot = 0.0
    while i < len(a_u) and j < len(b_u):
        lo = max(a_u[i][0], b_u[j][0]); hi = min(a_u[i][1], b_u[j][1])
        if hi > lo: tot += hi - lo
        if a_u[i][1] < b_u[j][1]: i += 1
        else: j += 1
    return tot


def _load_trace(path):
    """Return (comm_intervals, comp_intervals, windows). Comm carries dur so we
    can also sum self-times; compute is union-only."""
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return None
    ev = data.get("traceEvents", [])
    comm, comp = [], []
    for e in ev:
        if e.get("ph") != "X" or str(e.get("cat", "")).lower() != "kernel":
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or not dur:
            continue
        (comm if _NCCL_DEVKERNEL_RE.search(e.get("name", "")) else comp).append((ts, ts + dur))
    if not comm:
        return None
    return comm, comp, _gpu_step_windows(ev)


def _clip(intervals, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def attribute_cell(exp_dir, bw_curve, analyze_steps):
    """Average per-window attribution across the exported ranks of one run."""
    summ = json.load(open(os.path.join(exp_dir, "summary.json")))
    ws = int(summ["world_size"]); ga = int(summ["gradient_accumulation_steps"])
    n_prof = int(summ.get("num_measured_steps", 0)) or 8
    # per-optimizer-step bus volume: measured volume is bus_vol_per_step * active.
    active = 8
    bus_vol_total_gb = float(summ.get("measured_comm_volume_gb") or 0.0)
    bus_vol_step_gb = bus_vol_total_gb / active if active else 0.0
    size_b = float(summ.get("mean_collective_size_bytes") or 0.0)
    # dominant collective by device time
    times = {"allreduce": summ.get("nccl_allreduce_time_ms", 0.0),
             "allgather": summ.get("nccl_allgather_time_ms", 0.0),
             "reducescatter": summ.get("nccl_reducescatter_time_ms", 0.0)}
    dominant = max(times, key=times.get)
    peak = _lookup_peak_bw(bw_curve, size_b, ws, dominant) or 0.0
    transfer_floor_us = (bus_vol_step_gb / peak * 1e6) if peak > 0 else 0.0

    per_rank = []
    for tpath in sorted(glob.glob(os.path.join(exp_dir, "traces", "trace_rank*.json"))):
        loaded = _load_trace(tpath)
        if not loaded:
            continue
        comm, comp, windows = loaded
        used = windows[:max(1, analyze_steps)] if windows else []
        if not used:
            continue
        acc = {k: [] for k in ("sum", "busy", "ov", "exp", "idle")}
        for lo, hi in used:
            c = _clip(comm, lo, hi); p = _clip(comp, lo, hi)
            sum_us = sum(e - s for s, e in c)          # self-time sum (metric basis)
            cu = _merge_intervals([list(x) for x in c])
            pu = _merge_intervals([list(x) for x in p])
            busy = sum(b - a for a, b in cu)
            ov = _intersect_len(cu, pu)
            union_all = _union_len(c + p)
            acc["sum"].append(sum_us)
            acc["busy"].append(busy)
            acc["ov"].append(ov)
            acc["exp"].append(busy - ov)
            acc["idle"].append(max(0.0, (hi - lo) - union_all))
        per_rank.append({k: float(np.mean(v)) for k, v in acc.items()})
    if not per_rank:
        return None
    agg = {k: float(np.mean([r[k] for r in per_rank])) for k in per_rank[0]}
    comm_sum = agg["sum"]
    wait = max(0.0, comm_sum - transfer_floor_us)
    return {
        "strategy": summ["strategy"], "ga": ga, "ws": ws, "n_ranks": len(per_rank),
        "dominant": dominant, "peak_busbw_gbps": peak,
        "bus_vol_step_gb": bus_vol_step_gb,
        "comm_time_sum_us": comm_sum,
        "comm_busy_us": agg["busy"],
        "transfer_floor_us": transfer_floor_us,
        "wait_us": wait,
        "wait_frac": (wait / comm_sum) if comm_sum > 0 else 0.0,
        "overlapped_us": agg["ov"],
        "exposed_us": agg["exp"],
        "idle_us": agg["idle"],
        # achieved busbw as the live metric computes it (sum-based) vs the
        # transfer-implied fabric busbw (== peak, by construction of the floor).
        "achieved_busbw_sumbased_gbps": (bus_vol_step_gb / (comm_sum / 1e6)) if comm_sum > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/comm_attrib")
    ap.add_argument("--analyze_steps", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/figures")
    args = ap.parse_args()

    bw_curve = None
    for p in ("outputs/nccl_tests/bw_curves.json",):
        if os.path.exists(p):
            bw_curve = json.load(open(p)); break

    rows = []
    for exp_dir in sorted(glob.glob(os.path.join(args.root, "experiments", "*"))):
        if not os.path.exists(os.path.join(exp_dir, "summary.json")):
            continue
        if not glob.glob(os.path.join(exp_dir, "traces", "trace_rank*.json")):
            continue
        r = attribute_cell(exp_dir, bw_curve, args.analyze_steps)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: (STRAT_ORDER.index(r["strategy"]) if r["strategy"] in STRAT_ORDER else 9, r["ga"]))

    # ---- table ----
    hdr = ("strat", "ga", "comm_sum_us", "transfer_us", "wait_us", "wait%",
           "exposed_us", "overlap%", "idle_us", "busbw_sum", "peak_bw")
    print("\n%-7s %3s %11s %11s %10s %6s %10s %8s %9s %9s %8s" % hdr)
    print("-" * 108)
    for r in rows:
        ov = 100 * r["overlapped_us"] / r["comm_busy_us"] if r["comm_busy_us"] > 0 else 0
        print("%-7s %3d %11.0f %11.0f %10.0f %5.1f%% %10.0f %7.1f%% %9.0f %9.1f %8.1f" % (
            STRAT_LABEL.get(r["strategy"], r["strategy"]), r["ga"],
            r["comm_time_sum_us"], r["transfer_floor_us"], r["wait_us"],
            100 * r["wait_frac"], r["exposed_us"], ov, r["idle_us"],
            r["achieved_busbw_sumbased_gbps"], r["peak_busbw_gbps"]))

    out_json = os.path.join(args.root, "comm_attribution.json")
    json.dump(rows, open(out_json, "w"), indent=2)
    print("\nwrote", out_json, "(%d cells)" % len(rows))
    _plot(rows, args.out_dir)


def _plot(rows, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(plot skipped:", e, ")"); return
    str016 = [s for s in STRAT_ORDER if any(r["strategy"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(str016), figsize=(6 * len(str016), 5), sharey=True)
    if len(str016) == 1:
        axes = [axes]
    for ax, strat in zip(axes, str016):
        sub = sorted([r for r in rows if r["strategy"] == strat], key=lambda r: r["ga"])
        gas = [r["ga"] for r in sub]
        transfer = [r["transfer_floor_us"] / 1e3 for r in sub]
        wait = [r["wait_us"] / 1e3 for r in sub]
        x = range(len(gas))
        ax.bar(x, transfer, label="transfer (floor)", color="#4c72b0")
        ax.bar(x, wait, bottom=transfer, label="wait / spin / overhead", color="#c44e52")
        ax.set_xticks(list(x)); ax.set_xticklabels(["ga%d" % g for g in gas])
        ax.set_title(STRAT_LABEL.get(strat, strat))
        ax.set_xlabel("grad accumulation")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Comm device time per opt-step (ms)  [sum of NCCL kernels]")
    axes[-1].legend(fontsize=9, loc="upper left")
    fig.suptitle("Comm-time attribution (4B, 8 GPUs): transfer floor vs wait, by strategy & ga")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comm_attribution_transfer_vs_wait.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
