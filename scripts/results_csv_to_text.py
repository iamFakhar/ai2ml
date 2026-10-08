#!/usr/bin/env python3
"""Render the master results CSV as a human-readable .txt report.

One labelled block per run, grouped into sections (config, timing, compute
breakdown, time distribution, throughput, memory, scaling, kernels,
communication, status). Values are copied verbatim from the CSV (full
precision preserved); blank cells render as an em dash.

Fields are mapped by COLUMN NAME, not position, so the report is stable across
schema changes (added/reordered columns). Any mapped column absent from the CSV
simply renders as "—".

Usage:
  python scripts/results_csv_to_text.py
  python scripts/results_csv_to_text.py --csv outputs/all_experiments_summary.csv \
                                        --out outputs/all_experiments_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import os

EM = "—"                     # blank-cell placeholder
WIDTH = 78                   # border / rule width
LABEL_W = 29                 # label field width (longest label fits exactly)

# (Section title, [(label, csv_column), ...]) — order defines the report layout.
SECTIONS = [
    ("Experiment Configuration", [
        ("Experiment name", "experiment_name"),
        ("Model", "model"),
        ("Strategy", "strategy"),
        ("World size", "world_size"),
        ("Per-device batch size", "per_device_batch_size"),
        ("Sequence length", "seq_length"),
        ("Grad accumulation steps", "gradient_accumulation_steps"),
        ("Precision", "precision"),
        ("Gradient checkpointing", "grad_checkpoint"),
        ("Measured steps", "num_measured_steps"),
    ]),
    ("Iteration Timing (ms)", [
        ("Mean", "iter_time_ms_mean"),
        ("Std", "iter_time_ms_std"),
        ("Min", "iter_time_ms_min"),
        ("Max", "iter_time_ms_max"),
        ("Median", "iter_time_ms_median"),
        ("Coeff. of variation", "iter_time_ms_cv"),
    ]),
    ("Compute Breakdown (ms)", [
        ("Forward", "fwd_time_ms_mean"),
        ("Backward", "bwd_time_ms_mean"),
        ("Optimizer", "opt_time_ms_mean"),
        ("Total compute", "compute_time_ms_mean"),
        ("Backward compute", "bwd_compute_ms"),
        ("Backward compute (calibrated)", "bwd_compute_ms_calibrated"),
        ("Recompute", "recompute_ms"),
        ("Recompute (calibrated)", "recompute_ms_calibrated"),
        ("Backward exposed comm", "bwd_exposed_comm_ms"),
    ]),
    ("Time Distribution", [
        ("Forward %", "pct_forward"),
        ("Backward %", "pct_backward"),
        ("Optimizer %", "pct_optimizer"),
        ("Exposed comm %", "pct_exposed_comm"),
        ("Residual %", "pct_residual"),
        ("Exposed comm (ms)", "exposed_comm_time_ms"),
        ("Straggler wait (ms)", "straggler_wait_ms"),
        ("Residual (ms)", "residual_ms"),
    ]),
    ("Throughput & Loss", [
        ("Throughput mean (tok/s)", "throughput_tokens_per_sec_mean"),
        ("Throughput std (tok/s)", "throughput_tokens_per_sec_std"),
        ("Loss mean", "loss_mean"),
    ]),
    ("Memory", [
        ("Peak allocated (GB)", "peak_memory_allocated_gb"),
        ("Peak reserved (GB)", "peak_memory_reserved_gb"),
        ("Semantics", "peak_memory_semantics"),
        ("Headroom to 80GB (GB)", "headroom_to_80gb_gb"),
        ("Fragmentation (GB)", "fragmentation_gb"),
        ("Alloc retries", "num_alloc_retries"),
        ("OOMs", "num_ooms"),
        ("Transient peak (GB)", "transient_peak_gb"),
        ("Per-rank peak (GB)", "per_rank_peak_memory_gb"),
    ]),
    ("Scaling & Efficiency", [
        ("Per-rank iter time (ms)", "per_rank_iter_time_ms_mean"),
        ("Straggler ratio", "straggler_ratio"),
        ("Scaling efficiency", "scaling_efficiency"),
        ("MFU", "mfu"),
        ("Compute FLOPs", "compute_flops"),
    ]),
    ("Kernels", [
        ("CUDA kernel count", "cuda_kernel_count"),
        ("Mean kernel duration (us)", "mean_kernel_duration_us"),
    ]),
    ("Communication (NCCL)", [
        ("Theoretical volume (GB)", "theoretical_comm_volume_gb"),
        ("Measured volume (GB)", "measured_comm_volume_gb"),
        ("Total comm time (ms)", "total_nccl_comm_time_ms"),
        ("AllReduce time (ms)", "nccl_allreduce_time_ms"),
        ("AllGather time (ms)", "nccl_allgather_time_ms"),
        ("ReduceScatter time (ms)", "nccl_reducescatter_time_ms"),
        ("AllReduce count", "nccl_allreduce_count"),
        ("AllGather count", "nccl_allgather_count"),
        ("ReduceScatter count", "nccl_reducescatter_count"),
        ("Mean collective size (bytes)", "mean_collective_size_bytes"),
        ("Achieved BW (Gbps)", "achieved_bw_gbps"),
        ("Bandwidth efficiency", "bandwidth_efficiency"),
        ("Overlap % (trace)", "overlap_percentage_trace"),
        ("Overlap % (ablation)", "overlap_percentage_ablation"),
    ]),
    ("Run Status", [
        ("Outlier steps", "num_outlier_steps"),
        ("Status", "status"),
        ("Timestamp", "timestamp"),
    ]),
]


def cell(row: dict, col: str) -> str:
    """Verbatim value, or em dash for blank/absent."""
    v = row.get(col)
    if v is None:
        return EM
    v = v.strip()
    return v if v else EM


def render_row(row: dict, idx: int, total: int) -> str:
    name = cell(row, "experiment_name")
    status = cell(row, "status")
    tag = f"[{idx}/{total}]"
    out = [
        "#" * WIDTH,
        f"# START {tag}  {name}",
        f"#        status={status}",
        "#" * WIDTH,
        "",
    ]
    for title, fields in SECTIONS:
        out.append(f"  {title}")
        out.append("  " + "-" * (WIDTH - 2))
        for label, col in fields:
            out.append(f"    {label:<{LABEL_W}} : {cell(row, col)}")
        out.append("")
    out.append(f"# END {tag}  {name}")
    out.append("=" * WIDTH)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="outputs/all_experiments_summary.csv")
    ap.add_argument("--out", default=None,
                    help="output .txt (default: <csv> with .txt extension)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    out_path = args.out or os.path.splitext(args.csv)[0] + ".txt"

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {args.csv}")

    total = len(rows)
    blocks = [render_row(r, i, total) for i, r in enumerate(rows, start=1)]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")

    print(f"wrote {out_path}: {total} runs")


if __name__ == "__main__":
    main()
