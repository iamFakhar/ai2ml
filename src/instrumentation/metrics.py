# src/instrumentation/metrics.py
"""Per-step recording + per-experiment aggregation (design doc §9, 58 fields).

The core reframe (v2): iteration time is ATTRIBUTED, not treated as a
residual. We record raw per-rank per-step data, then aggregate over the
measured window and reconstruct:

  iter = fwd + bwd + opt + exposed_comm + straggler_wait + residual

The residual (pct_residual) is a methodology quality metric; the gate is
checked against protocol.residual_quality_gate_pct on validated configs.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .profiler import score_bandwidth

# Attribution residual below this (ms) is float cancellation noise, not signal
# (e.g. single-GPU, where exposed_comm == idle_sync makes attributed == iter
# exactly up to ~1e-14 ms fp error). Clamped to 0 rather than reported.
_RESIDUAL_EPS_MS = 1e-6


# --------------------------------------------------------------------------
# Per-step record (one per step, per rank)
# --------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    iter_time_ms: float
    fwd_time_ms: float
    bwd_time_ms: float
    opt_time_ms: float
    loss: float
    local_tokens: int
    # Memory (peak within this step)
    peak_mem_allocated_gb: float
    peak_mem_reserved_gb: float
    headroom_to_80gb_gb: float
    fragmentation_gb: float
    num_alloc_retries: float
    num_ooms: float

    @property
    def compute_time_ms(self) -> float:
        return self.fwd_time_ms + self.bwd_time_ms + self.opt_time_ms

    @property
    def idle_sync_time_ms(self) -> float:
        return self.iter_time_ms - self.compute_time_ms

    def to_dict(self) -> Dict[str, float]:
        d = self.__dict__.copy()
        d["compute_time_ms"] = self.compute_time_ms
        d["idle_sync_time_ms"] = self.idle_sync_time_ms
        return d


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------

def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}
    mean = statistics.fmean(xs)
    std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "cv": (std / mean) if mean else 0.0,
    }


class MetricsAggregator:
    """Collects StepRecords for THIS rank; aggregation happens post-run.

    Cross-rank reductions (straggler ratio, per-rank arrays) are done by the
    runner AFTER training using distributed.gather_* helpers — never inside
    the timed region (design doc §8 timing hygiene).
    """

    def __init__(self, cfg: dict, strategy_name: str,
                 num_params: int, world_size: int) -> None:
        self.cfg = cfg
        self.strategy_name = strategy_name
        self.num_params = num_params
        self.world_size = world_size
        self.warmup = cfg["protocol"]["warmup_steps_discard"]

        self.all_records: List[StepRecord] = []      # every step (for JSON)
        self.measured: List[StepRecord] = []          # post-warmup only

    def record(self, rec: StepRecord) -> None:
        self.all_records.append(rec)
        if rec.step >= self.warmup:
            self.measured.append(rec)

    # ---- throughput ------------------------------------------------------

    def _global_tokens_per_step(self) -> int:
        w = self.cfg["workload"]
        ga = self.cfg.get("grad_accum_steps", 1)
        return w["per_device_batch_size"] * self.world_size * ga * w["seq_len"]

    def _throughput_series(self) -> List[float]:
        toks = self._global_tokens_per_step()
        return [toks / (r.iter_time_ms / 1e3) for r in self.measured]

    # ---- main aggregation ------------------------------------------------

    def aggregate(self,
                  comm_metrics: Optional[Dict[str, float]] = None,
                  overlap_ablation_pct: Optional[float] = None,
                  overlap_trace: Optional[Dict[str, float]] = None,
                  bw_curve: Optional[dict] = None,
                  straggler_ratio: Optional[float] = None,
                  per_rank_iter_ms: Optional[List[float]] = None,
                  per_rank_peak_mem_gb: Optional[List[float]] = None,
                  single_gpu_throughput: Optional[float] = None,
                  compute_flops: Optional[float] = None) -> Dict[str, float]:
        m = self.measured
        if not m:
            return {"status": "no_measured_steps"}

        iter_ms = [r.iter_time_ms for r in m]
        fwd_ms = [r.fwd_time_ms for r in m]
        bwd_ms = [r.bwd_time_ms for r in m]
        opt_ms = [r.opt_time_ms for r in m]
        compute_ms = [r.compute_time_ms for r in m]

        it = _stats(iter_ms)
        thr = _stats(self._throughput_series())

        agg: Dict[str, float] = {
            "num_measured_steps": len(m),
            # iter_time
            "iter_time_ms_mean": it["mean"],
            "iter_time_ms_std": it["std"],
            "iter_time_ms_median": it["median"],
            "iter_time_ms_min": it["min"],
            "iter_time_ms_max": it["max"],
            "iter_time_ms_cv": it["cv"],
            # phase means
            "fwd_time_ms_mean": statistics.fmean(fwd_ms),
            "bwd_time_ms_mean": statistics.fmean(bwd_ms),
            "opt_time_ms_mean": statistics.fmean(opt_ms),
            "compute_time_ms_mean": statistics.fmean(compute_ms),
            # throughput
            "throughput_tokens_per_sec_mean": thr["mean"],
            "throughput_tokens_per_sec_std": thr["std"],
            # loss (sanity)
            "loss_mean": statistics.fmean([r.loss for r in m]),
            # memory (peak over measured steps)
            "peak_memory_allocated_gb": max(r.peak_mem_allocated_gb for r in m),
            "peak_memory_reserved_gb": max(r.peak_mem_reserved_gb for r in m),
            "peak_memory_semantics": self._semantics(),
            "headroom_to_80gb_gb": min(r.headroom_to_80gb_gb for r in m),
            "fragmentation_gb": max(r.fragmentation_gb for r in m),
            "num_alloc_retries": max(r.num_alloc_retries for r in m),
            "num_ooms": max(r.num_ooms for r in m),
            # theoretical comm volume (GB)
            "theoretical_comm_volume_gb":
                self._theoretical_comm_gb(),
        }

        # --- exposed comm + attribution --------------------------------------
        exposed_comm = self._exposed_comm_ms(comm_metrics)
        agg["exposed_comm_time_ms"] = exposed_comm

        straggler = straggler_ratio or 1.0
        # straggler wait ≈ portion of idle attributable to barrier imbalance
        straggler_wait = max(0.0, (straggler - 1.0) * it["mean"])
        agg["straggler_ratio"] = straggler
        agg["straggler_wait_ms"] = straggler_wait

        attributed = (agg["fwd_time_ms_mean"] + agg["bwd_time_ms_mean"]
                      + agg["opt_time_ms_mean"] + exposed_comm + straggler_wait)
        residual = max(0.0, it["mean"] - attributed)
        if residual < _RESIDUAL_EPS_MS:
            residual = 0.0  # clamp sub-nanosecond fp cancellation noise
        agg["residual_ms"] = residual

        total = it["mean"] if it["mean"] > 0 else 1.0
        agg["pct_forward"] = 100.0 * agg["fwd_time_ms_mean"] / total
        agg["pct_backward"] = 100.0 * agg["bwd_time_ms_mean"] / total
        agg["pct_optimizer"] = 100.0 * agg["opt_time_ms_mean"] / total
        agg["pct_exposed_comm"] = 100.0 * exposed_comm / total
        agg["pct_residual"] = 100.0 * residual / total

        # --- outliers (counted, not removed) ---------------------------------
        agg["num_outlier_steps"] = self._count_outliers(iter_ms, it["mean"], it["std"])

        # --- backward internal split (RQ1 attribution) ----------------------
        # bwd_time = bwd_compute + recompute + bwd_exposed_comm.
        #   bwd_exposed_comm_ms : MEASURED from the trace = non-overlapped comm
        #       per step = (comm_busy - comm_overlapped) / n_profiled_steps.
        #       (For FSDP this is step-total exposed comm, ~all in backward.)
        #   recompute_ms        : ANALYTICAL ≈ forward compute, since full-layer
        #       activation checkpointing recomputes the whole forward once in
        #       backward. Validate magnitude against the group4a (GC on/off)
        #       recompute-tax ablation.
        #   bwd_compute_ms      : remainder (true gradient computation).
        gc_on = self.cfg.get("workload", {}).get("gradient_checkpointing", True)
        recompute_ms = agg["fwd_time_ms_mean"] if gc_on else 0.0
        bwd_exposed_ms = 0.0
        if overlap_trace and self.world_size > 1:
            # comm_busy/overlapped are summed over the ANALYZED step windows
            # (notes/07: leading steps only, trailing guard dropped). Normalize
            # by that same count — NOT `active` — so exposed comm is per-step.
            n_prof = int(overlap_trace.get("analyzed_steps") or 0) or self._profiled_steps()
            exposed_window_us = max(
                0.0, overlap_trace.get("comm_busy_us", 0.0)
                - overlap_trace.get("comm_overlapped_us", 0.0))
            bwd_exposed_ms = exposed_window_us / n_prof / 1e3
        agg["recompute_ms"] = recompute_ms
        agg["bwd_exposed_comm_ms"] = bwd_exposed_ms
        agg["bwd_compute_ms"] = max(
            0.0, agg["bwd_time_ms_mean"] - recompute_ms - bwd_exposed_ms)

        # --- comm metrics (from profiler) ------------------------------------
        if comm_metrics:
            # Merge the FULL comm field set into the row. Previously only the
            # per-collective counts were copied, so total_nccl_comm_time_ms,
            # the per-collective times and the kernel stats were left blank in
            # the CSV even when the profiler measured them — comm time read as
            # "0"/empty regardless of the profiler. Copy times + kernel stats too.
            agg.update({
                # per-collective device time (ms)
                "total_nccl_comm_time_ms": comm_metrics.get("total_nccl_comm_time_ms", 0.0),
                "nccl_allreduce_time_ms": comm_metrics.get("nccl_allreduce_time_ms", 0.0),
                "nccl_allgather_time_ms": comm_metrics.get("nccl_allgather_time_ms", 0.0),
                "nccl_reducescatter_time_ms": comm_metrics.get("nccl_reducescatter_time_ms", 0.0),
                # per-collective counts
                "nccl_allreduce_count": comm_metrics.get("nccl_allreduce_count", 0.0),
                "nccl_allgather_count": comm_metrics.get("nccl_allgather_count", 0.0),
                "nccl_reducescatter_count": comm_metrics.get("nccl_reducescatter_count", 0.0),
                # aggregate kernel stats
                "cuda_kernel_count": comm_metrics.get("cuda_kernel_count", 0.0),
                "mean_kernel_duration_us": comm_metrics.get("mean_kernel_duration_us", 0.0),
            })

            # --- analytical comm volume + granularity + bandwidth ------------
            # NCCL device kernels report shape 0, so volume is analytical (the
            # nccl-tests convention: known size x MEASURED time -> busbw). Both
            # volume and comm time are summed over the same profiled window.
            n_prof = self._profiled_steps()
            bus_gb = self._bus_volume_gb() * n_prof           # busbw volume (ga-aware)
            raw_gb = self._raw_data_gb() * n_prof             # raw data moved (ga-aware)
            agg["measured_comm_volume_gb"] = bus_gb

            total_count = (agg["nccl_allreduce_count"]
                           + agg["nccl_allgather_count"]
                           + agg["nccl_reducescatter_count"])
            mean_size = (raw_gb * 1e9 / total_count) if total_count > 0 else 0.0
            agg["mean_collective_size_bytes"] = mean_size

            # dominant collective (by device time) selects the busbw curve
            times = {"allreduce": agg["nccl_allreduce_time_ms"],
                     "allgather": agg["nccl_allgather_time_ms"],
                     "reducescatter": agg["nccl_reducescatter_time_ms"]}
            dominant = max(times, key=times.get)
            bw = score_bandwidth(
                measured_bytes=bus_gb * 1e9,
                comm_time_ms=agg["total_nccl_comm_time_ms"],
                bw_curve=bw_curve, world_size=self.world_size,
                collective=dominant, lookup_size_bytes=mean_size,
            )
            agg["achieved_bw_gbps"] = bw["achieved_bw_gbps"]
            agg["bandwidth_efficiency"] = bw["bandwidth_efficiency"]

        # overlap_percentage_trace is now the REAL timeline overlap measured
        # from the Kineto trace (comm_busy ∩ compute_busy / comm_busy), supplied
        # by the runner via overlap_trace. The old idle-sync proxy
        # (1 - exposed/total) was tautological (~100%) and is removed; if the
        # trace is unavailable the field is left blank rather than misleading.
        if overlap_trace is not None:
            agg["overlap_percentage_trace"] = overlap_trace.get(
                "overlap_percentage_trace", 0.0)
            # QA columns (notes/07): steady-state windowing + multi-rank support.
            agg["overlap_analyzed_steps"] = float(
                overlap_trace.get("analyzed_steps", 0.0))
            agg["overlap_step_spread_pp"] = float(
                overlap_trace.get("overlap_step_spread_pp", 0.0))
            agg["overlap_stable"] = 1.0 if overlap_trace.get(
                "overlap_stable", True) else 0.0
            agg["overlap_n_ranks"] = float(
                len(overlap_trace.get("overlap_ranks", [])) or 1)

        if overlap_ablation_pct is not None:
            agg["overlap_percentage_ablation"] = overlap_ablation_pct

        # --- MFU + scaling ---------------------------------------------------
        if compute_flops:
            agg["compute_flops"] = compute_flops
            agg["mfu"] = self._mfu(compute_flops, agg["compute_time_ms_mean"])

        if single_gpu_throughput and single_gpu_throughput > 0 and self.world_size > 1:
            # NOTE: values >100% are expected at high grad-accum and are NOT
            # superlinear speedup. The single-GPU baseline is measured at ga=1,
            # while these cells use larger ga; the higher effective batch
            # amortizes fixed per-step overhead better, so per-GPU throughput
            # rises vs the ga=1 baseline. This is a baseline-mismatch artifact
            # to document, not a scaling anomaly.
            agg["scaling_efficiency"] = (
                agg["throughput_tokens_per_sec_mean"]
                / (self.world_size * single_gpu_throughput) * 100.0
            )

        # --- per-rank arrays -------------------------------------------------
        if per_rank_iter_ms is not None:
            agg["per_rank_iter_time_ms_mean"] = per_rank_iter_ms
        if per_rank_peak_mem_gb is not None:
            agg["per_rank_peak_memory_gb"] = per_rank_peak_mem_gb

        agg["status"] = "ok"
        return agg

    # ---- helpers ---------------------------------------------------------

    def _semantics(self) -> str:
        return "full_replica" if self.strategy_name in ("ddp", "single") else "sharded"

    def _theoretical_comm_gb(self) -> float:
        """Per-step BUS volume (GB): nccl-tests busbw convention.

        busbw factor = 2(n-1)/n for DDP all-reduce and ZeRO-2 (all-gather +
        reduce-scatter), 3(n-1)/n for ZeRO-3 (2x all-gather + reduce-scatter).
        """
        if self.world_size <= 1:
            return 0.0
        n = self.world_size
        factor = 3 * (n - 1) / n if self.strategy_name == "fsdp_zero3" else 2 * (n - 1) / n
        return factor * self.num_params * 2 / 1e9

    def _raw_data_gb_per_step(self) -> float:
        """Per-step RAW data volume (GB) moved across collectives (no busbw
        (n-1)/n factor), at ga=1. Kept as a ga=1 reference; the ga-aware version
        used for volume/bandwidth is _raw_data_gb().
          DDP        -> 1x params (grad all-reduce)
          ZeRO-2     -> 2x params (fwd all-gather + grad reduce-scatter)
          ZeRO-3     -> 3x params (fwd + bwd all-gather + grad reduce-scatter)
        """
        if self.world_size <= 1:
            return 0.0
        raw_factor = {"ddp": 1, "fsdp_zero2": 2, "fsdp_zero3": 3}.get(
            self.strategy_name, 2)
        return raw_factor * self.num_params * 2 / 1e9

    def _accum_steps(self) -> int:
        return int(self.cfg.get("grad_accum_steps", 1) or 1)

    def _bus_volume_gb(self) -> float:
        """Per-optimizer-step BUS volume (GB), ACCOUNTING FOR accumulation depth.

        With gradient accumulation the collectives that repeat every micro-batch
        scale with ga, while those deferred by no_sync to the final micro-batch
        occur once per optimizer step. This matches the measured collective
        counts (ZeRO-3 all-gather count scales ~ga x; ZeRO-2/DDP counts are flat):
          ZeRO-3 : all-gather (fwd+bwd) per micro-batch -> x ga ; reduce-scatter once
          ZeRO-2 : all-gather (fwd) + reduce-scatter, both once  -> ga-independent
          DDP    : all-reduce once                                -> ga-independent
        At ga=1 this equals _theoretical_comm_gb(). Using the ga=1 value while the
        measured NCCL time covers all micro-batches under-states achieved_bw by
        ~ga at ga>1 for ZeRO-3 — the bug this method fixes.
        """
        if self.world_size <= 1:
            return 0.0
        f = (self.world_size - 1) / self.world_size
        p2 = self.num_params * 2 / 1e9
        ga = self._accum_steps()
        if self.strategy_name == "fsdp_zero3":
            return (2 * ga * f + f) * p2      # AG (fwd+bwd) x ga + RS x 1
        return 2 * f * p2                     # ZeRO-2 (AG+RS) / DDP (AR): once

    def _raw_data_gb(self) -> float:
        """Per-step RAW volume (GB), ga-aware (mirrors _bus_volume_gb's
        repetition structure). Drives the mean per-collective size, which must
        stay ~constant across ga since counts and raw volume scale together."""
        if self.world_size <= 1:
            return 0.0
        p2 = self.num_params * 2 / 1e9
        ga = self._accum_steps()
        if self.strategy_name == "fsdp_zero3":
            return (2 * ga + 1) * p2         # AG (fwd+bwd) x ga + RS x 1
        if self.strategy_name == "fsdp_zero2":
            return 2 * p2
        return 1 * p2                         # ddp

    def _profiled_steps(self) -> int:
        """Number of steps the profiler actually recorded (PASS 2 active window).
        total_nccl_comm_time and counts are summed over exactly these steps, so
        analytical volume must be scaled by the same count to align units.
        """
        try:
            return int(self.cfg["instrumentation"]["profiler"]["active"])
        except (KeyError, TypeError):
            return 1

    def _exposed_comm_ms(self, comm_metrics: Optional[Dict[str, float]]) -> float:
        """Exposed comm = idle_sync mean (best available residual proxy).

        When profiler comm data exists, we prefer the smaller of (idle_sync,
        total_comm) since exposed comm cannot exceed total comm.

        Because this is an idle-sync proxy (not a timeline measurement), it goes
        to ~0 on compute-bound cells and makes overlap_percentage_trace collapse
        to ~100% tautologically. The ablation method (overlap_percentage_ablation)
        is the authoritative overlap metric.
        """
        idle = statistics.fmean([r.idle_sync_time_ms for r in self.measured])
        idle = max(0.0, idle)
        if comm_metrics:
            total_comm = comm_metrics.get("total_nccl_comm_time_ms", 0.0)
            if total_comm > 0:
                return min(idle, total_comm)
        return idle

    def _count_outliers(self, xs: List[float], mean: float, std: float) -> int:
        if std <= 0:
            return 0
        return sum(1 for x in xs if abs(x - mean) > 2 * std)

    def _mfu(self, flops_per_step: float, compute_ms: float) -> float:
        peak = self.cfg["hardware"]["a100_peak_bf16_tflops"] * 1e12
        if compute_ms <= 0:
            return 0.0
        achieved = flops_per_step / (compute_ms / 1e3) / self.world_size
        return min(1.0, achieved / peak) * 100.0