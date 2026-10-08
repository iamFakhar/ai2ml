# src/runner.py
"""Unified training runner — the ONE entrypoint (design doc §8).

All timing, aggregation, warmup protocol, and metric definitions are SHARED.
Strategy-specific behavior (wrap / optimizer timing / no_sync / clip /
mixed-precision / memory semantics) is dispatched through the strategy layer.

Launch (torchrun):
  torchrun --standalone --nproc_per_node=<N> src/runner.py \
      --config configs/base.yaml \
      --model qwen3-4b --strategy fsdp_zero3 \
      --world_size 8 --grad_accum_steps 1 --rep 0

The orchestrator (orchestration/launch_matrix.py) constructs these commands.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, set_seed as hf_set_seed

# --- project modules --------------------------------------------------------
from .distributed import (
    enable_nccl_tracing, setup_distributed, cleanup_distributed,
    is_main_process, get_rank, get_world_size, barrier,
    all_reduce_mean, gather_scalar, gather_dict,
)
from .strategies import build_strategy
from .instrumentation.cuda_timers import CudaTimers
from .instrumentation.memory import (
    reset_peak_memory_stats, get_memory_stats,
)
from .instrumentation.profiler import (
    build_profiler, profiler_step, parse_comm_metrics, compute_overlap_from_trace,
)
from .instrumentation.metrics import MetricsAggregator, StepRecord
from .data.synthetic import build_dataloader, MicroBatchFeeder
from . import io_utils


# ==========================================================================
# Config loading + overlay
# ==========================================================================

def load_configs(base_path: str, models_path: str) -> tuple[dict, dict]:
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(models_path) as f:
        models = yaml.safe_load(f)["models"]
    return base, models


def build_run_config(base: dict, models: dict, args) -> dict:
    """Overlay CLI selections onto the locked base config."""
    cfg = dict(base)  # shallow copy is fine; we only add scalar keys
    model_spec = models[args.model]

    cfg["experiment"] = dict(cfg["experiment"])
    cfg["experiment"]["name"] = args.experiment_name
    cfg["experiment"]["seed"] = base["experiment"]["seed"]

    cfg["model_key"] = args.model
    cfg["model_spec"] = model_spec
    cfg["strategy"] = args.strategy
    cfg["world_size_requested"] = args.world_size
    cfg["grad_accum_steps"] = args.grad_accum_steps
    cfg["rep"] = args.rep

    # Ablation / override flags
    if args.suppress_overlap:
        cfg.setdefault("ablation", {})["suppress_overlap"] = True
    if args.no_grad_checkpoint:
        cfg["workload"] = dict(cfg["workload"])
        cfg["workload"]["gradient_checkpointing"] = False
    if args.enable_profiler_trace:
        cfg["instrumentation"] = dict(cfg["instrumentation"])
        cfg["instrumentation"]["enable_profiler_trace"] = True
    if args.keep_traces:
        cfg["instrumentation"] = dict(cfg["instrumentation"])
        cfg["instrumentation"]["keep_traces"] = True
    if getattr(args, "profile_only", False):
        # Skip PASS 1: reuse the clean timing already on disk; run PASS 2 only
        # and write a comm_only sidecar to be merged later (notes/07).
        cfg["instrumentation"] = dict(cfg["instrumentation"])
        cfg["instrumentation"]["enable_profiler_trace"] = True
        cfg["instrumentation"]["profile_only"] = True
    return cfg


# ==========================================================================
# Model loading
# ==========================================================================

def get_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(cfg: dict, device: torch.device):
    spec = cfg["model_spec"]
    w = cfg["workload"]
    dtype = get_dtype()

    _ = AutoConfig.from_pretrained(spec["hf_name"])  # validate name early
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_name"],
        torch_dtype=dtype,
        attn_implementation=w["attn_implementation"],
    )
    if w["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(device)
    model.train()
    return model


def resolve_decoder_layer_cls(model, spec: dict):
    """Locate the transformer decoder layer class for FSDP auto-wrap."""
    target = spec["decoder_layer_cls"]
    for module in model.modules():
        if type(module).__name__ == target:
            return type(module)
    raise RuntimeError(
        f"Could not find decoder layer class '{target}' in model. "
        f"Check transformers version / config."
    )


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================================================
# FLOPs estimate (for MFU) — standard 6ND causal-LM approximation
# ==========================================================================

def estimate_flops_per_step(cfg: dict, num_params: int) -> float:
    w = cfg["workload"]
    ga = cfg.get("grad_accum_steps", 1)
    tokens = w["per_device_batch_size"] * w["seq_len"] * ga  # per-GPU per step
    # 6 * N * tokens for fwd+bwd (activation checkpointing adds recompute,
    # but MFU is reported against non-recompute FLOPs by convention).
    return 6.0 * num_params * tokens


# ==========================================================================
# Training
# ==========================================================================

def train(cfg: dict) -> Dict:
    enable_nccl_tracing()
    rank, world_size, local_rank, device = setup_distributed(cfg["distributed"]["backend"])

    # Seed differs per rank so sampler shards differ, but data corpus is
    # deterministic (fixed dataset seed in synthetic.py).
    hf_set_seed(cfg["experiment"]["seed"] + rank)

    if cfg.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    exp_name = cfg["experiment"]["name"]
    out_root = cfg["experiment"]["output_root"]
    exp_dir = os.path.join(out_root, "experiments", exp_name)
    trace_dir = os.path.join(exp_dir, "traces")
    tb_dir = os.path.join(out_root, "tb_root", exp_name)
    master_csv = os.path.join(out_root, "all_experiments_summary.csv")

    if is_main_process():
        io_utils.ensure_dir(exp_dir)
        print(f"[rank {rank}] START {exp_name} "
              f"| strategy={cfg['strategy']} ws={world_size} "
              f"ga={cfg['grad_accum_steps']}", flush=True)

    # Provenance (rank 0 only).
    provenance = io_utils.capture_provenance(cfg) if is_main_process() else None

    # ---- model + strategy -------------------------------------------------
    strategy = build_strategy(cfg["strategy"], cfg)
    model = load_model(cfg, device)
    num_params = count_trainable_params(model)
    decoder_cls = resolve_decoder_layer_cls(model, cfg["model_spec"])
    vocab_size = model.config.vocab_size

    model = strategy.wrap_model(model, local_rank, decoder_cls)
    optimizer = strategy.build_optimizer(model)          # AFTER wrap (FSDP)
    autocast_ctx = strategy.autocast_context(get_dtype())

    # ---- data -------------------------------------------------------------
    loader = build_dataloader(cfg, world_size, rank, vocab_size)
    feeder = MicroBatchFeeder(loader, device, cfg["grad_accum_steps"])

    # ---- instrumentation --------------------------------------------------
    aggregator = MetricsAggregator(cfg, cfg["strategy"], num_params, world_size)
    tb = io_utils.TBLogger(
        enabled=cfg["instrumentation"]["enable_tensorboard"] and is_main_process(),
        log_dir=tb_dir,
    )
    suppress_overlap = cfg.get("ablation", {}).get("suppress_overlap", False)
    profiling_enabled = cfg["instrumentation"]["enable_profiler_trace"]
    profile_only = cfg["instrumentation"].get("profile_only", False)

    gpu_mem_gb = cfg["hardware"]["gpu_mem_gb"]
    max_grad_norm = cfg["optimizer"]["max_grad_norm"]
    log_every = cfg["instrumentation"]["log_every"]
    num_steps = cfg["protocol"]["num_steps"]
    tokens_per_step_global = (cfg["workload"]["per_device_batch_size"]
                              * world_size * cfg["grad_accum_steps"]
                              * cfg["workload"]["seq_len"])

    # _run_one_step positional args shared by both passes.
    step_args = (model, optimizer, strategy, feeder, autocast_ctx, device,
                 gpu_mem_gb, max_grad_norm, cfg["grad_accum_steps"],
                 suppress_overlap)

    status = "ok"
    comm_metrics = None
    try:
        # ---- PASS 1: clean timing — measured window with NO profiler -------
        # The torch profiler perturbs the very iterations it observes (kernel
        # hooks, shape/memory recording). Running it over the timed loop is
        # what inflated iter-time CV (0.15-0.45 vs ~0.01), per-rank spread and
        # the straggler ratio on exactly the validated (profiled) configs.
        # Timing and comm profiling are therefore two SEPARATE passes: the
        # measured window here has no profiler attached (design doc §8/§10).
        # In --profile_only mode PASS 1 is skipped entirely (the clean timing is
        # reused from a prior full run; only PASS 2 comm/overlap is refreshed).
        for step in ([] if profile_only else range(num_steps)):
            rec = _run_one_step(step, *step_args)
            aggregator.record(rec)

            throughput = tokens_per_step_global / (rec.iter_time_ms / 1e3)
            if is_main_process():
                tb.log_step(step, rec, throughput)
                if step % log_every == 0:
                    print(
                        f"[{exp_name}] step={step:03d} "
                        f"loss={rec.loss:.4f} iter={rec.iter_time_ms:.2f}ms "
                        f"fwd={rec.fwd_time_ms:.2f} bwd={rec.bwd_time_ms:.2f} "
                        f"opt={rec.opt_time_ms:.2f} "
                        f"tok/s={throughput:.0f} "
                        f"mem={rec.peak_mem_reserved_gb:.1f}GB",
                        flush=True,
                    )

        # ---- PASS 2: comm/kernel profiling — timings intentionally discarded
        # Only per-collective device time + counts are extracted here. This is
        # best-effort: a failure in the profiling pass must not invalidate the
        # clean timing already gathered in PASS 1.
        if profiling_enabled:
            try:
                p = cfg["instrumentation"]["profiler"]
                export_ranks = p.get("export_ranks", [0])
                # Run real steps past the active window (trailing guard) so the
                # last active step's tail is fully captured, then stop early to
                # avoid wasting compute (notes/07).
                last_active = p["skip_first"] + p["wait"] + p["warmup"] + p["active"] - 1
                stop_step = min(num_steps - 1,
                                last_active + int(p.get("trailing_guard_steps", 2)))
                with build_profiler(cfg, trace_dir, rank, export_ranks) as prof:
                    for step in range(num_steps):
                        _run_one_step(step, *step_args)   # StepRecord discarded
                        profiler_step(prof)
                        if step >= stop_step:
                            break
                comm_metrics = parse_comm_metrics(prof)
            except torch.cuda.OutOfMemoryError:
                if is_main_process():
                    print(f"[{exp_name}] profiling pass OOM; keeping PASS 1 "
                          f"timing, comm metrics skipped", flush=True)
                torch.cuda.empty_cache()
                comm_metrics = None

    except torch.cuda.OutOfMemoryError as e:
        status = "oom"
        if is_main_process():
            print(f"[{exp_name}] OOM caught: {e}", flush=True)
        torch.cuda.empty_cache()

    # ---- post-run aggregation (cross-rank; OUTSIDE timed region) ----------
    # Volume + bandwidth are computed inside aggregate() (it has num_params /
    # strategy / world_size); we only supply the nccl-test busbw curve here.
    bw_curve = _load_bw_curve(cfg) if (status == "ok" and comm_metrics) else None

    # Real timeline comm/compute overlap from the Kineto trace(s). Exposed comm
    # is per-rank & straggler-dependent, so EVERY exported rank computes overlap
    # on its OWN trace; results are gathered to rank 0 and averaged (notes/07).
    # The metric uses only the leading `analyze_steps` per-step windows, dropping
    # the truncated trailing guard. Heavy traces are discarded unless keep_traces.
    overlap_trace = None
    # Gate the gather on rank-CONSISTENT conditions only (profiling_enabled is
    # from config; status is already the guard the later _cross_rank_stats gather
    # shares). comm_metrics may be None on just the rank whose Pass-2 OOM'd, so
    # it must NOT gate this collective — otherwise ranks diverge and deadlock.
    if profiling_enabled and status == "ok":
        p = cfg["instrumentation"]["profiler"]
        analyze_steps = int(p.get("analyze_steps", p.get("active", 5)))
        tol = float(p.get("stability_tol_pp", 8.0))
        own_trace = _find_own_trace(trace_dir, rank) if comm_metrics else None
        local = (compute_overlap_from_trace(own_trace, analyze_steps, tol)
                 if own_trace else None)
        # Every rank must participate in the gather (collective op).
        payload = {
            "rank": float(rank),
            "ok": bool(local),
            "overlap": local.get("overlap_percentage_trace", 0.0) if local else 0.0,
            "busy": local.get("comm_busy_us", 0.0) if local else 0.0,
            "ovl": local.get("comm_overlapped_us", 0.0) if local else 0.0,
            "spread": local.get("overlap_step_spread_pp", 0.0) if local else 0.0,
            "stable": bool(local.get("overlap_stable", True)) if local else True,
            "nsteps": local.get("analyzed_steps", 0.0) if local else 0.0,
        }
        gathered = gather_dict(payload, device)
        if is_main_process():
            overlap_trace = _aggregate_overlap(gathered, analyze_steps)
            if overlap_trace:
                warn = "" if overlap_trace["overlap_stable"] else "  [UNSTABLE per-step spread]"
                print(f"[{exp_name}] timeline overlap = "
                      f"{overlap_trace['overlap_percentage_trace']:.1f}% "
                      f"(ranks={overlap_trace['overlap_ranks']}, "
                      f"spread={overlap_trace['overlap_step_spread_pp']:.1f}pp){warn}",
                      flush=True)
            else:
                print(f"[{exp_name}] overlap: no NCCL kernels in trace(s)", flush=True)
        if own_trace and not cfg["instrumentation"].get("keep_traces", False):
            try:
                os.remove(own_trace)
            except OSError:
                pass

    # --- profile_only: emit a comm/overlap sidecar and STOP. PASS 1 was skipped
    # so there is no fresh timing to aggregate; the sidecar is merged into the
    # existing summary.json later (scripts.merge_comm_only), preserving timing.
    if profile_only:
        if is_main_process():
            sidecar = {
                "experiment_name": exp_name,
                "world_size": world_size,
                "profiler_active_steps": int(
                    cfg["instrumentation"]["profiler"].get("active", 0)),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "comm_metrics": comm_metrics or {},
                "overlap_trace": overlap_trace or {},
            }
            io_utils.write_json(os.path.join(exp_dir, "comm_only.json"), sidecar)
            print(f"[{exp_name}] profile_only: wrote comm_only.json "
                  f"(status={status}, comm={'yes' if comm_metrics else 'no'})",
                  flush=True)
        barrier()
        cleanup_distributed()
        return {"experiment_name": exp_name, "status": status, "profile_only": True}

    straggler_ratio, per_rank_iter, per_rank_mem = _cross_rank_stats(
        aggregator, device, world_size,
    )

    single_gpu_thr = _load_single_gpu_throughput(cfg) if world_size > 1 else None
    flops = estimate_flops_per_step(cfg, num_params)

    if status == "oom":
        summary = _oom_summary(cfg, num_params, world_size, aggregator, gpu_mem_gb)
    else:
        agg = aggregator.aggregate(
            comm_metrics=comm_metrics,
            bw_curve=bw_curve,
            overlap_trace=overlap_trace,
            straggler_ratio=straggler_ratio,
            per_rank_iter_ms=per_rank_iter,
            per_rank_peak_mem_gb=per_rank_mem,
            single_gpu_throughput=single_gpu_thr,
            compute_flops=flops,
        )
        summary = _assemble_summary(cfg, num_params, world_size, agg)

    # ---- write outputs (rank 0) ------------------------------------------
    if is_main_process():
        summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        io_utils.append_master_row(master_csv, summary)
        io_utils.write_json(os.path.join(exp_dir, "summary.json"), summary)
        io_utils.write_json(os.path.join(exp_dir, "experiment_config.json"),
                            _serializable_cfg(cfg))
        if provenance is not None:
            io_utils.snapshot_gpu_end(provenance)
            io_utils.write_json(os.path.join(exp_dir, "metadata.json"), provenance)
        tb.hparams(
            {"model": cfg["model_key"], "strategy": cfg["strategy"],
             "world_size": world_size, "grad_accum": cfg["grad_accum_steps"]},
            {"hp/throughput": summary.get("throughput_tokens_per_sec_mean", 0.0),
             "hp/iter_ms": summary.get("iter_time_ms_mean", 0.0),
             "hp/peak_mem_gb": summary.get("peak_memory_reserved_gb", 0.0),
             "hp/mfu": summary.get("mfu", 0.0)},
        )
        tb.close()

    # Per-rank raw step dump (every rank writes its own file).
    io_utils.write_json(
        os.path.join(exp_dir, f"per_step_rank{rank}.json"),
        {"rank": rank, "steps": [r.to_dict() for r in aggregator.all_records]},
    )

    barrier()
    cleanup_distributed()

    if is_main_process():
        print(f"[{exp_name}] DONE status={status} "
              f"iter={summary.get('iter_time_ms_mean', 0):.2f}ms "
              f"resid%={summary.get('pct_residual', 0):.1f}", flush=True)
    return summary


# ==========================================================================
# Single step
# ==========================================================================

def _run_one_step(step, model, optimizer, strategy, feeder, autocast_ctx,
                  device, gpu_mem_gb, max_grad_norm, grad_accum,
                  suppress_overlap) -> StepRecord:
    model.train()
    reset_peak_memory_stats(device)
    micro_batches = feeder.next_step()
    optimizer.zero_grad(set_to_none=True)

    timers = CudaTimers()
    timers.start("iter")

    last_loss = 0.0
    for i, batch in enumerate(micro_batches):
        is_last = (i == grad_accum - 1)
        sync_ctx = strategy.maybe_no_sync(model, is_last)
        with sync_ctx:
            timers.start("fwd")
            with autocast_ctx:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss / grad_accum
            timers.stop("fwd")

            timers.start("bwd")
            loss.backward()
            timers.stop("bwd")
        last_loss = out.loss.detach().float().item()

    timers.start("opt")
    strategy.clip_grad_norm(model, max_grad_norm)
    optimizer.step()
    timers.stop("opt")

    # Overlap-suppression ablation: force a device sync between bwd and opt so
    # comm cannot hide behind compute (design doc §6 Group 4b).
    if suppress_overlap:
        torch.cuda.synchronize(device)

    timers.stop("iter")
    torch.cuda.synchronize(device)

    t = timers.read()
    mem = get_memory_stats(device, gpu_mem_gb)

    return StepRecord(
        step=step,
        iter_time_ms=t["iter_time_ms"],
        fwd_time_ms=t["fwd_time_ms"],
        bwd_time_ms=t["bwd_time_ms"],
        opt_time_ms=t["opt_time_ms"],
        loss=last_loss,
        local_tokens=sum(b["input_ids"].numel() for b in micro_batches),
        peak_mem_allocated_gb=mem["peak_mem_allocated_gb"],
        peak_mem_reserved_gb=mem["peak_mem_reserved_gb"],
        headroom_to_80gb_gb=mem["headroom_to_80gb_gb"],
        fragmentation_gb=mem["fragmentation_gb"],
        num_alloc_retries=mem["num_alloc_retries"],
        num_ooms=mem["num_ooms"],
    )


# ==========================================================================
# Cross-rank stats (post-run only — never in timed region)
# ==========================================================================

def _cross_rank_stats(aggregator: MetricsAggregator, device, world_size):
    if not aggregator.measured:
        return 1.0, None, None
    local_iter_mean = sum(r.iter_time_ms for r in aggregator.measured) / len(aggregator.measured)
    local_peak_mem = max(r.peak_mem_reserved_gb for r in aggregator.measured)

    per_rank_iter = gather_scalar(local_iter_mean, device)
    per_rank_mem = gather_scalar(local_peak_mem, device)

    if world_size > 1 and per_rank_iter:
        straggler = max(per_rank_iter) / max(min(per_rank_iter), 1e-9)
    else:
        straggler = 1.0
    return straggler, per_rank_iter, per_rank_mem


# ==========================================================================
# Summary assembly
# ==========================================================================

def _assemble_summary(cfg, num_params, world_size, agg) -> Dict:
    w = cfg["workload"]
    summary = {
        "experiment_name": cfg["experiment"]["name"],
        "model": cfg["model_key"],
        "strategy": cfg["strategy"],
        "world_size": world_size,
        "per_device_batch_size": w["per_device_batch_size"],
        "seq_length": w["seq_len"],
        "gradient_accumulation_steps": cfg["grad_accum_steps"],
        "precision": cfg["precision"]["mode"],
        "grad_checkpoint": w["gradient_checkpointing"],
        "trainable_parameters": num_params,
    }
    summary.update(agg)
    return summary


def _oom_summary(cfg, num_params, world_size, aggregator, gpu_mem_gb) -> Dict:
    w = cfg["workload"]
    # Populate memory up to failure point if any steps ran.
    peak_resv = (max((r.peak_mem_reserved_gb for r in aggregator.all_records),
                     default=gpu_mem_gb))
    return {
        "experiment_name": cfg["experiment"]["name"],
        "model": cfg["model_key"],
        "strategy": cfg["strategy"],
        "world_size": world_size,
        "per_device_batch_size": w["per_device_batch_size"],
        "seq_length": w["seq_len"],
        "gradient_accumulation_steps": cfg["grad_accum_steps"],
        "precision": cfg["precision"]["mode"],
        "grad_checkpoint": w["gradient_checkpointing"],
        "trainable_parameters": num_params,
        "peak_memory_reserved_gb": peak_resv,
        "peak_memory_semantics":
            "full_replica" if cfg["strategy"] in ("ddp", "single") else "sharded",
        "num_measured_steps": 0,
        "status": "oom",
    }


# ==========================================================================
# Helpers: BW curve + single-GPU baseline lookups
# ==========================================================================

def _load_bw_curve(cfg) -> Optional[dict]:
    import json
    path = cfg["hardware"].get("nccl_bw_curve_path")
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _find_own_trace(trace_dir: str, rank: int) -> Optional[str]:
    """This rank's Kineto trace file.

    The current profiler exports 'trace_rank{rank}.json' (see build_profiler).
    Falls back to the legacy tensorboard_trace_handler naming
    ('{host}_{pid}.{id}.pt.trace.json') for older runs."""
    cand = os.path.join(trace_dir, f"trace_rank{rank}.json")
    if os.path.exists(cand):
        return cand
    import glob
    matches = glob.glob(os.path.join(trace_dir, f"*_{os.getpid()}.*.pt.trace.json"))
    return matches[0] if matches else None


def _aggregate_overlap(gathered, analyze_steps):
    """Combine per-rank overlap payloads (from exported ranks) into one dict.

    overlap% is comm-weighted across ranks (== mean of per-rank overlap since
    the /len cancels); comm_busy/overlapped are per-rank means so metrics.py's
    per-step exposed-comm term reflects the multi-rank average. Returns None if
    no rank produced a usable trace."""
    valid = [g for g in gathered if g and g.get("ok") and g.get("busy", 0.0) > 0.0]
    if not valid:
        return None
    n = len(valid)
    sum_busy = sum(g["busy"] for g in valid)
    sum_ovl = sum(g["ovl"] for g in valid)
    pct = 100.0 * sum_ovl / sum_busy if sum_busy > 0 else 0.0
    return {
        "overlap_percentage_trace": pct,
        "comm_busy_us": sum_busy / n,
        "comm_overlapped_us": sum_ovl / n,
        "analyzed_steps": float(analyze_steps),
        "overlap_ranks": sorted(int(g["rank"]) for g in valid),
        "overlap_per_rank": [g["overlap"] for g in valid],
        "overlap_step_spread_pp": max(g.get("spread", 0.0) for g in valid),
        "overlap_stable": all(g.get("stable", True) for g in valid),
    }


def _load_single_gpu_throughput(cfg) -> Optional[float]:
    """Read the matching single-GPU baseline throughput from master CSV.

    Enables scaling_efficiency computation. Returns None if the baseline for
    this model hasn't been run yet (Group 2 must precede Group 1 analysis;
    scaling_efficiency can also be back-filled in analysis).
    """
    import csv
    master = os.path.join(cfg["experiment"]["output_root"],
                          "all_experiments_summary.csv")
    if not os.path.exists(master):
        return None
    model = cfg["model_key"]
    vals = []
    with open(master) as f:
        for row in csv.DictReader(f):
            if (row.get("model") == model
                    and row.get("strategy") == "single"
                    and row.get("status") == "ok"):
                try:
                    vals.append(float(row["throughput_tokens_per_sec_mean"]))
                except (KeyError, ValueError):
                    pass
    return sum(vals) / len(vals) if vals else None


def _serializable_cfg(cfg: dict) -> dict:
    """Strip non-serializable model_spec objects for JSON dump."""
    out = {}
    for k, v in cfg.items():
        if k == "model_spec":
            out[k] = dict(v)
        elif isinstance(v, dict):
            out[k] = v
        else:
            out[k] = v
    return out


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified distributed training runner")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--models_config", default="configs/models.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--strategy", required=True,
                   choices=["ddp", "single", "fsdp_zero2", "fsdp_zero3"])
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--experiment_name", required=True)
    # ablation / overrides
    p.add_argument("--suppress_overlap", action="store_true")
    p.add_argument("--no_grad_checkpoint", action="store_true")
    p.add_argument("--enable_profiler_trace", action="store_true")
    p.add_argument("--keep_traces", action="store_true",
                   help="Keep the ~200MB/rank Kineto trace after overlap is "
                        "computed (default: discard to bound disk at scale). "
                        "Use for Nsight/HTA validation cells.")
    p.add_argument("--profile_only", action="store_true",
                   help="Skip PASS 1 (clean timing) and run only PASS 2 "
                        "(profiling). Writes a comm_only.json sidecar into the "
                        "experiment dir instead of a master-CSV row; merge it "
                        "later with scripts.merge_comm_only (keeps existing "
                        "timing). Requires a prior full run for this cell.")
    return p.parse_args()


def main():
    args = parse_args()
    base, models = load_configs(args.config, args.models_config)
    if args.model not in models:
        print(f"Unknown model '{args.model}'. Options: {list(models)}", file=sys.stderr)
        sys.exit(1)
    cfg = build_run_config(base, models, args)
    train(cfg)


if __name__ == "__main__":
    main()