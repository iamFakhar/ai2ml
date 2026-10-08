# # src/instrumentation/profiler.py
"""PyTorch profiler wrapper + NCCL/kernel trace parsing (design doc §9 group E/F).

Two responsibilities:
  1. Build the torch.profiler schedule (only for validated configs).
  2. Parse key_averages() into per-collective time/count, kernel stats, and a
     timeline-based overlap estimate.

CORRECTNESS NOTES (validated via quick_probe.py + quick_probe_fsdp.py):
  - The SAME collective appears at multiple abstraction layers, all reporting
    identical INCLUSIVE cuda_time_total:
        c10d::allreduce_            (CPU wrapper)
        record_param_comms         (bookkeeping wrapper)
        nccl:all_reduce            (CPU wrapper)
        ncclDevKernel_AllReduce_*  (THE actual device kernel)  <-- self time
    Summing cuda_time_total would 4x-overcount. We therefore:
      (a) read SELF device time, not inclusive, and
      (b) match ONLY ncclDevKernel_* device kernels.
  - Confirmed kernel name strings on this build (PyTorch 2.11 / bundled NCCL):
        ncclDevKernel_AllReduce_Sum_bf16_RING_LL(...)
        ncclDevKernel_AllGather_RING_LL(...)
        ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(...)
  - The scheduled profiler CLEARS events at the end of each cycle unless
    acc_events=True is set (otherwise key_averages() is empty/partial after
    the loop). build_profiler() sets acc_events=True.

The overlap estimate here is the TRACE method (field 41). The ABLATION method
(field 42) is computed in the runner by comparing iter_time with overlap
suppressed. Chapter 4 reconciles the two.
"""
from __future__ import annotations

import os
import re
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
from torch.autograd import DeviceType
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)


# --------------------------------------------------------------------------
# Kernel-name matching
# --------------------------------------------------------------------------
# Match ONLY actual NCCL device kernels (they carry the real SELF device time).
# Wrappers (c10d::*, record_param_comms, nccl:*) hold inclusive time and are
# deliberately excluded to prevent multi-layer overcounting.
_NCCL_DEVKERNEL_RE = re.compile(r"ncclDevKernel|ncclKernel", re.IGNORECASE)

# Collective-type classifiers. Names observed: AllReduce / AllGather /
# ReduceScatter (no separator, mixed case) -> plain substring, case-insensitive.
# Order matters: reducescatter is checked before allreduce/allgather so the
# "reduce" substring in "ReduceScatter" is not misclassified.
_REDUCESCATTER_RE = re.compile(r"reducescatter", re.IGNORECASE)
_ALLGATHER_RE = re.compile(r"allgather", re.IGNORECASE)
_ALLREDUCE_RE = re.compile(r"allreduce", re.IGNORECASE)


# --------------------------------------------------------------------------
# Profiler construction
# --------------------------------------------------------------------------

def build_profiler(cfg: dict, trace_dir: str, rank: int = 0,
                   export_ranks: Optional[List[int]] = None):
    """Return a torch.profiler context, or nullcontext if disabled.

    Boundary protection (see notes/07): the custom on_trace_ready calls
    torch.cuda.synchronize() BEFORE exporting, which guarantees the last active
    step's trailing NCCL kernel (FSDP reduce-scatter / DDP all-reduce) has a
    valid end-timestamp in the trace — this is precisely what prevents the
    truncation that would otherwise resurface as decomposition residual.

    Only ranks in `export_ranks` write a trace file (exposed comm is per-rank
    and straggler-dependent, so we analyze several ranks and average; other
    ranks still record so their overhead matches, but skip the ~200 MB write).
    """
    inst = cfg["instrumentation"]
    if not inst.get("enable_profiler_trace", False):
        return nullcontext()

    os.makedirs(trace_dir, exist_ok=True)
    p = inst["profiler"]
    if export_ranks is None:
        export_ranks = p.get("export_ranks", [0])

    def on_ready(prof):
        # Ensure the GPU finished the tail of the last active step before export.
        torch.cuda.synchronize()
        if (not export_ranks) or (rank in export_ranks):
            prof.export_chrome_trace(
                os.path.join(trace_dir, f"trace_rank{rank}.json"))

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            skip_first=p["skip_first"],
            wait=p["wait"],
            warmup=p["warmup"],
            active=p["active"],
            repeat=p["repeat"],
        ),
        on_trace_ready=on_ready,
        record_shapes=p.get("record_shapes", False),
        profile_memory=p.get("profile_memory", False),
        with_stack=p.get("with_stack", False),
        with_flops=p.get("with_flops", False),
        acc_events=True,   # retain events across schedule cycles (critical)
    )


def profiler_step(prof) -> None:
    """Advance the profiler schedule; no-op for nullcontext."""
    if prof is not None and hasattr(prof, "step"):
        prof.step()


# --------------------------------------------------------------------------
# Event time extraction
# --------------------------------------------------------------------------

def _self_cuda_us(e) -> float:
    """Return SELF device time (exclusive of children) in microseconds.

    Uses self_* attributes so multi-layer collective wrappers (which report
    inclusive time) do not cause overcounting. Falls back across PyTorch
    version attribute-name differences.
    """
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return float(v)
    return 0.0


def _classify_nccl_kernel(name: str) -> Optional[str]:
    """Classify ONLY actual NCCL device kernels.

    Returns 'allreduce' | 'allgather' | 'reducescatter', or None for:
      - non-NCCL kernels,
      - CPU wrappers (c10d::*, record_param_comms, nccl:* python-side ops),
      - local ops that merely contain 'reduce' (e.g. reducer::mul_out,
        aten reduce_kernel).
    """
    if not _NCCL_DEVKERNEL_RE.search(name):
        return None
    if _REDUCESCATTER_RE.search(name):
        return "reducescatter"
    if _ALLGATHER_RE.search(name):
        return "allgather"
    if _ALLREDUCE_RE.search(name):
        return "allreduce"
    return None


# --------------------------------------------------------------------------
# Trace parsing
# --------------------------------------------------------------------------

def parse_comm_metrics(prof) -> Dict[str, float]:
    """Extract per-collective time (ms) + counts and kernel stats.

    Returns a flat dict with stable defaults (0.0) so the CSV schema holds
    even when a collective type is absent (e.g. no reduce-scatter under DDP,
    no all-reduce under ZeRO-3).

    Byte volume is NOT estimated from shapes here (shapes are unreliable for
    NCCL kernels and returned 0). It is filled ANALYTICALLY in metrics.py from
    the theoretical comm volume + per-collective counts.
    """
    out: Dict[str, float] = {
        # per-collective device time
        "nccl_allreduce_time_ms": 0.0,
        "nccl_allgather_time_ms": 0.0,
        "nccl_reducescatter_time_ms": 0.0,
        "total_nccl_comm_time_ms": 0.0,
        # per-collective counts (for RQ2 granularity analysis)
        "nccl_allreduce_count": 0.0,
        "nccl_allgather_count": 0.0,
        "nccl_reducescatter_count": 0.0,
        "nccl_collective_count": 0.0,
        # aggregate kernel stats
        "cuda_kernel_count": 0.0,
        "mean_kernel_duration_us": 0.0,
        # volume (filled analytically downstream)
        "mean_collective_size_bytes": 0.0,
        "measured_comm_volume_bytes": 0.0,
    }
    if prof is None or not hasattr(prof, "key_averages"):
        return out

    events = prof.key_averages()

    total_comm_us = 0.0
    total_count = 0
    per_coll_count = {"allreduce": 0, "allgather": 0, "reducescatter": 0}

    total_kernel_us = 0.0
    total_kernel_count = 0

    for e in events:
        # Only genuine device kernels carry trustworthy self device time.
        # Under acc_events=True the profiler attributes a collective's device
        # self-time up onto its CPU wrappers (c10d::allreduce_,
        # record_param_comms) AND onto its user annotation (nccl:all_reduce,
        # which surfaces with device_type=CUDA). Counting those double/triple-
        # counts device time and inflates cuda_kernel_count / mean duration, so
        # we skip anything that is not a real, non-annotation CUDA kernel.
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            continue
        if getattr(e, "is_user_annotation", False):
            continue

        self_us = _self_cuda_us(e)
        if self_us <= 0:
            continue  # zero-device-time ops

        # Aggregate kernel-level stats: any real GPU kernel with self time.
        total_kernel_us += self_us
        total_kernel_count += e.count

        coll = _classify_nccl_kernel(e.key)
        if coll is None:
            continue

        out[f"nccl_{coll}_time_ms"] += self_us / 1e3
        per_coll_count[coll] += e.count
        total_comm_us += self_us
        total_count += e.count

    out["total_nccl_comm_time_ms"] = total_comm_us / 1e3
    out["nccl_allreduce_count"] = float(per_coll_count["allreduce"])
    out["nccl_allgather_count"] = float(per_coll_count["allgather"])
    out["nccl_reducescatter_count"] = float(per_coll_count["reducescatter"])
    out["nccl_collective_count"] = float(total_count)
    out["cuda_kernel_count"] = float(total_kernel_count)
    out["mean_kernel_duration_us"] = (
        total_kernel_us / total_kernel_count if total_kernel_count else 0.0
    )
    return out


# --------------------------------------------------------------------------
# Overlap + bandwidth scoring
# --------------------------------------------------------------------------

def compute_overlap_trace(total_nccl_comm_time_ms: float,
                          exposed_comm_time_ms: float) -> Dict[str, float]:
    """Timeline overlap estimate (field 41).

    overlap = 1 - exposed / total   (clamped to [0, 1])

    exposed_comm_time_ms is supplied by the aggregator from the attribution
    model (iter - attributed compute). Total comm is the real device time from
    parse_comm_metrics. Exposed cannot exceed total, so the ratio is bounded.
    """
    if total_nccl_comm_time_ms <= 0:
        return {"overlap_percentage_trace": 0.0}
    overlap = 1.0 - (exposed_comm_time_ms / total_nccl_comm_time_ms)
    overlap = max(0.0, min(1.0, overlap))
    return {"overlap_percentage_trace": overlap * 100.0}


def _merge_intervals(intervals):
    """Merge a list of (start, end) into sorted, disjoint intervals."""
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _intersect_len(a_u, b_u) -> float:
    """Total length of the intersection of two sorted, disjoint interval sets."""
    i = j = 0
    total = 0.0
    while i < len(a_u) and j < len(b_u):
        lo = max(a_u[i][0], b_u[j][0])
        hi = min(a_u[i][1], b_u[j][1])
        if hi > lo:
            total += hi - lo
        if a_u[i][1] < b_u[j][1]:
            i += 1
        else:
            j += 1
    return total


def _gpu_step_windows(events) -> List[tuple]:
    """GPU-side ProfilerStep windows [(ts, ts+dur), ...] sorted by start time.

    These delimit each profiled iteration on the device timeline (cat
    'gpu_user_annotation', name 'ProfilerStep#N'), letting us score overlap
    per-step and drop the truncated trailing step(s).
    """
    w = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "gpu_user_annotation":
            continue
        if "ProfilerStep" not in str(e.get("name", "")):
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or not dur:
            continue
        w.append((ts, ts + dur))
    w.sort()
    return w


def _overlap_in_window(comm, comp, lo, hi) -> tuple:
    """Restrict comm/comp intervals to [lo, hi]; return (busy_us, overlapped_us)."""
    c = [(max(s, lo), min(e, hi)) for s, e in comm if e > lo and s < hi]
    p = [(max(s, lo), min(e, hi)) for s, e in comp if e > lo and s < hi]
    cu = _merge_intervals(c)
    if not cu:
        return 0.0, 0.0
    pu = _merge_intervals(p)
    busy = sum(b - a for a, b in cu)
    return busy, _intersect_len(cu, pu)


def compute_overlap_from_trace(trace_path: str,
                               analyze_steps: Optional[int] = None,
                               stability_tol_pp: float = 8.0
                               ) -> Optional[Dict[str, float]]:
    """Timeline comm/compute overlap from a Kineto Chrome trace (this rank).

        overlap = |comm_busy ∩ compute_busy| / |comm_busy|

    comm_busy = union of NCCL device-kernel intervals; compute_busy = union of
    all other GPU-kernel intervals (cf. HTA get_comm_comp_overlap).

    Boundary protection (notes/07): the profiler's final active step is
    truncated by profiler-stop (its collectives are captured without the compute
    that hides them, and its comm_busy is inflated), which understates overlap —
    worst for ZeRO-3. We therefore score overlap PER profiler step and use only
    the representative leading steps:
      - analyze_steps set   -> use the first `analyze_steps` step windows
        (the remaining recorded steps are the trailing guard, dropped);
      - analyze_steps None  -> drop just the last step window (HTA convention);
      - no step markers      -> fall back to whole-trace (legacy traces).
    A per-step stability spread is reported so non-steady-state windows are
    flagged rather than silently averaged. Returns None if unreadable / no NCCL.
    """
    import json
    try:
        with open(trace_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None

    events = data.get("traceEvents", [])
    comm, comp = [], []
    for e in events:
        if e.get("ph") != "X" or str(e.get("cat", "")).lower() != "kernel":
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or not dur:
            continue
        if _NCCL_DEVKERNEL_RE.search(e.get("name", "")):
            comm.append((ts, ts + dur))
        else:
            comp.append((ts, ts + dur))
    if not comm:
        return None

    windows = _gpu_step_windows(events)
    if analyze_steps is not None and windows:
        used = windows[:max(1, int(analyze_steps))]
        dropped = len(windows) - len(used)
    elif len(windows) >= 2:
        used = windows[:-1]          # default: drop the truncated last step
        dropped = 1
    else:
        used = None                  # legacy fallback: whole trace
        dropped = 0

    per_step: List[float] = []
    if used:
        busy_tot = ov_tot = 0.0
        for lo, hi in used:
            b, o = _overlap_in_window(comm, comp, lo, hi)
            if b > 0:
                per_step.append(100.0 * o / b)
                busy_tot += b
                ov_tot += o
        n_used = len(per_step)
    else:
        cu = _merge_intervals(list(comm))
        pu = _merge_intervals(list(comp))
        busy_tot = sum(b - a for a, b in cu)
        ov_tot = _intersect_len(cu, pu)
        n_used = 0

    if busy_tot <= 0:
        return None
    pct = max(0.0, min(1.0, ov_tot / busy_tot)) * 100.0
    spread = (max(per_step) - min(per_step)) if len(per_step) >= 2 else 0.0
    return {
        "overlap_percentage_trace": pct,
        "comm_busy_us": busy_tot,
        "comm_overlapped_us": ov_tot,
        # steps the busy/overlapped totals are summed over — metrics.py divides
        # the exposed window by exactly this to get per-step exposed comm.
        "analyzed_steps": float(n_used),
        "overlap_per_step": per_step,
        "overlap_step_spread_pp": spread,
        "overlap_stable": bool(spread <= stability_tol_pp),
        "profiler_steps_dropped": float(dropped),
    }


def score_bandwidth(measured_bytes: float, comm_time_ms: float,
                    bw_curve: Optional[dict], world_size: int,
                    collective: str = "allreduce",
                    lookup_size_bytes: Optional[float] = None
                    ) -> Dict[str, float]:
    """achieved_bw_gbps and bandwidth_efficiency vs nccl-test curve (fields 48/49).

    measured_bytes is the analytical BUS volume (from metrics.py), since NCCL
    kernel shapes are not reliably captured. comm_time_ms is the REAL device
    time from parse_comm_metrics. achieved busbw = bus_volume / time.

    The efficiency denominator is looked up on the curve for the DOMINANT
    `collective` at the per-collective message size (`lookup_size_bytes`), NOT
    at the total window volume — otherwise the nearest-size lookup indexes the
    curve far past its plateau. Falls back to measured_bytes if no size given.
    """
    out = {"achieved_bw_gbps": 0.0, "bandwidth_efficiency": 0.0}
    if comm_time_ms <= 0 or measured_bytes <= 0:
        return out

    achieved_gbps = (measured_bytes / 1e9) / (comm_time_ms / 1e3)
    out["achieved_bw_gbps"] = achieved_gbps

    size_for_lookup = (lookup_size_bytes if lookup_size_bytes and lookup_size_bytes > 0
                       else measured_bytes)
    peak = _lookup_peak_bw(bw_curve, size_for_lookup, world_size, collective)
    if peak and peak > 0:
        out["bandwidth_efficiency"] = min(1.0, achieved_gbps / peak) * 100.0
    return out


def _lookup_peak_bw(bw_curve: Optional[dict], nbytes: float,
                    world_size: int, collective: str = "allreduce"
                    ) -> Optional[float]:
    """Look up peak busbw at the nearest message size for this GPU count.

    bw_curve format (from nccl_tests.sh -> bw_curves.json):
      {"8": {"allreduce": [[size_bytes, bw_gbps], ...],
             "allgather": [...], "reducescatter": [...]}, "4": {...}, ...}
    Uses the curve for `collective` (the strategy's dominant collective) and
    falls back to the allreduce curve if that collective is absent.
    """
    if not bw_curve:
        return None
    ws_key = str(world_size)
    if ws_key not in bw_curve:
        return None
    curve = bw_curve[ws_key].get(collective) or bw_curve[ws_key].get("allreduce")
    if not curve:
        return None
    best = min(curve, key=lambda pt: abs(pt[0] - nbytes))
    return best[1]