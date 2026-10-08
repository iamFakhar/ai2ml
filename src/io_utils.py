# src/io_utils.py
"""CSV / JSON / TensorBoard writers + the frozen master CSV schema (design doc §12).

The MASTER_FIELDS list is the single source of truth for the master CSV. Any
field not present in an aggregate dict is written as empty; extra keys are
ignored. This keeps the CSV schema stable across all ~155 runs.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Frozen master CSV schema (design doc §12). Order matters for readability.
# --------------------------------------------------------------------------
MASTER_FIELDS: List[str] = [
    # identity
    "experiment_name", "model", "strategy", "world_size",
    "per_device_batch_size", "seq_length", "gradient_accumulation_steps",
    "precision", "grad_checkpoint", "num_measured_steps",
    # iter_time
    "iter_time_ms_mean", "iter_time_ms_std", "iter_time_ms_min",
    "iter_time_ms_max", "iter_time_ms_median", "iter_time_ms_cv",
    # phase means
    "fwd_time_ms_mean", "bwd_time_ms_mean", "opt_time_ms_mean",
    "compute_time_ms_mean",
    "bwd_compute_ms", "recompute_ms", "bwd_exposed_comm_ms",
    # attribution
    "pct_forward", "pct_backward", "pct_optimizer",
    "pct_exposed_comm", "pct_residual",
    "exposed_comm_time_ms", "straggler_wait_ms", "residual_ms",
    # throughput
    "throughput_tokens_per_sec_mean", "throughput_tokens_per_sec_std",
    "loss_mean",
    # memory
    "peak_memory_allocated_gb", "peak_memory_reserved_gb",
    "peak_memory_semantics", "headroom_to_80gb_gb", "fragmentation_gb",
    "num_alloc_retries", "num_ooms", "transient_peak_gb",
    "per_rank_peak_memory_gb", "per_rank_iter_time_ms_mean",
    "straggler_ratio",
    # efficiency
    "scaling_efficiency", "mfu", "compute_flops",
    "cuda_kernel_count", "mean_kernel_duration_us",
    # comm
    "theoretical_comm_volume_gb", "measured_comm_volume_gb",
    "total_nccl_comm_time_ms",
    "nccl_allreduce_time_ms", "nccl_allgather_time_ms",
    "nccl_reducescatter_time_ms",
    "nccl_allreduce_count", "nccl_allgather_count", "nccl_reducescatter_count",
    "mean_collective_size_bytes",
    "achieved_bw_gbps", "bandwidth_efficiency",
    "overlap_percentage_trace", "overlap_percentage_ablation",
    # stability / status
    "num_outlier_steps", "status", "timestamp",
]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

def write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(o):
    # Make numpy / non-serializable objects safe.
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


# --------------------------------------------------------------------------
# Master CSV
# --------------------------------------------------------------------------

def _flatten_arrays(row: Dict) -> Dict:
    """Serialize list fields (per-rank arrays) as JSON strings for CSV."""
    out = dict(row)
    for k, v in list(out.items()):
        if isinstance(v, (list, tuple)):
            out[k] = json.dumps([round(float(x), 4) for x in v])
    return out


def append_master_row(csv_path: str, row: Dict) -> None:
    """Append one aggregated row to the master CSV. Rank-0 only.

    Missing keys become "". If the CSV already exists, rows are aligned to the
    file's ACTUAL header rather than to MASTER_FIELDS — otherwise columns that
    post-hoc tools (e.g. augment_recompute_calibration.py) inserted into the
    file would shift every subsequent value, because DictWriter aligns by the
    fieldnames it is given, not by matching the on-disk header. Extra keys not
    in the header are dropped (extrasaction="ignore").
    """
    ensure_dir(os.path.dirname(csv_path))
    row = _flatten_arrays(row)

    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), list(MASTER_FIELDS))
    else:
        header = list(MASTER_FIELDS)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


def experiment_completed(csv_path: str, experiment_name: str) -> bool:
    """Resume support: True if a row for this experiment already exists."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("experiment_name") == experiment_name:
                return True
    return False


# --------------------------------------------------------------------------
# Provenance (design doc §9 group H)
# --------------------------------------------------------------------------

def capture_provenance(cfg: dict) -> Dict:
    """Runtime environment capture for metadata.json + anomaly forensics."""
    import torch

    prov: Dict = {
        "git_commit_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": _nccl_version(),
        "driver_version": _nvidia_smi("--query-gpu=driver_version"),
        "gpu_name": _nvidia_smi("--query-gpu=name"),
        "gpu_clock_mhz_start": _nvidia_smi("--query-gpu=clocks.sm"),
        "gpu_temp_c_start": _nvidia_smi("--query-gpu=temperature.gpu"),
        "alloc_conf_effective": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "topo_matrix": _run(["nvidia-smi", "topo", "-m"]),
    }
    try:
        import transformers
        prov["transformers_version"] = transformers.__version__
    except Exception:
        prov["transformers_version"] = "unknown"
    return prov


def snapshot_gpu_end(prov: Dict) -> None:
    """Add end-of-run thermal/clock snapshot for throttling forensics."""
    prov["gpu_clock_mhz_end"] = _nvidia_smi("--query-gpu=clocks.sm")
    prov["gpu_temp_c_end"] = _nvidia_smi("--query-gpu=temperature.gpu")


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       text=True, timeout=15).strip()
    except Exception:
        return ""


def _nvidia_smi(query: str) -> str:
    return _run(["nvidia-smi", query, "--format=csv,noheader,nounits"]).split("\n")[0]


def _nccl_version() -> str:
    try:
        import torch
        v = torch.cuda.nccl.version()
        return ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# TensorBoard (design doc §12 Sink 3)
# --------------------------------------------------------------------------

class TBLogger:
    """Thin wrapper; no-op when disabled or on non-main ranks."""

    def __init__(self, enabled: bool, log_dir: Optional[str]) -> None:
        self.writer = None
        if enabled and log_dir:
            from torch.utils.tensorboard import SummaryWriter
            ensure_dir(log_dir)
            self.writer = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_step(self, step: int, rec, throughput: float) -> None:
        if self.writer is None:
            return
        self.scalar("time/iter_ms", rec.iter_time_ms, step)
        self.scalar("time/fwd_ms", rec.fwd_time_ms, step)
        self.scalar("time/bwd_ms", rec.bwd_time_ms, step)
        self.scalar("time/opt_ms", rec.opt_time_ms, step)
        self.scalar("time/idle_sync_ms", rec.idle_sync_time_ms, step)
        self.scalar("throughput/tokens_per_s", throughput, step)
        self.scalar("mem/peak_reserved_gb", rec.peak_mem_reserved_gb, step)
        self.scalar("mem/headroom_gb", rec.headroom_to_80gb_gb, step)
        self.scalar("loss", rec.loss, step)

    def hparams(self, hparam_dict: Dict, metric_dict: Dict) -> None:
        if self.writer is not None:
            self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()