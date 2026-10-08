# src/instrumentation/memory.py
"""GPU memory instrumentation (design doc §9 group B).

Includes memory-SEMANTICS tagging and allocator-pressure signals so that
RQ2 memory-pressure claims are falsifiable (full_replica vs sharded).
"""
from __future__ import annotations

from typing import Dict

import torch

_MB = 1024 ** 2
_GB = 1024 ** 3


def reset_peak_memory_stats(device: torch.device) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def get_memory_stats(device: torch.device, gpu_mem_gb: float = 80.0) -> Dict[str, float]:
    stats = torch.cuda.memory_stats(device)
    max_alloc_gb = torch.cuda.max_memory_allocated(device) / _GB
    max_resv_gb = torch.cuda.max_memory_reserved(device) / _GB

    return {
        "mem_allocated_gb": torch.cuda.memory_allocated(device) / _GB,
        "mem_reserved_gb": torch.cuda.memory_reserved(device) / _GB,
        "peak_mem_allocated_gb": max_alloc_gb,
        "peak_mem_reserved_gb": max_resv_gb,
        "headroom_to_80gb_gb": gpu_mem_gb - max_resv_gb,
        "fragmentation_gb": max_resv_gb - max_alloc_gb,
        # Allocator-pressure signals (nonzero precedes OOM) — RQ2 evidence.
        "num_alloc_retries": float(stats.get("num_alloc_retries", 0)),
        "num_ooms": float(stats.get("num_ooms", 0)),
    }


def memory_semantics(strategy_name: str) -> str:
    """DDP holds a full replica; FSDP holds sharded state (design doc §9)."""
    return "full_replica" if strategy_name in ("ddp", "single") else "sharded"


def estimate_transient_peak_gb(peak_allocated_gb: float,
                               sharded_steady_gb: float) -> float:
    """FSDP-only: isolate all-gather buffer cost from sharded state."""
    return max(0.0, peak_allocated_gb - sharded_steady_gb)