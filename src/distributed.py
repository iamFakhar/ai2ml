# src/distributed.py
"""Distributed setup / teardown and cross-rank helpers.

CRITICAL (design doc §8 timing hygiene): none of the reduction helpers here
may be called INSIDE the timed iteration region. They are for post-step /
end-of-run aggregation only.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


def enable_nccl_tracing() -> None:
    """Must be set before init_process_group (design doc §3/§9 provenance)."""
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "COLL,INIT")
    os.environ.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int, torch.device]:
    """Supports torchrun launch and single-process (single-GPU) fallback."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        rank, world_size, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


# --- Post-step / end-of-run reductions (NEVER inside timed region) ----------

def all_reduce_mean(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return t.item()


def gather_scalar(value: float, device: torch.device) -> List[float]:
    """Gather one scalar per rank to rank 0 (returns list on rank 0, [] else)."""
    if not is_dist():
        return [value]
    t = torch.tensor([value], dtype=torch.float64, device=device)
    world = get_world_size()
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t)
    return [x.item() for x in out]


def gather_dict(local: Dict[str, float], device: torch.device) -> List[Dict[str, float]]:
    """Gather a flat dict of scalars from every rank to every rank."""
    if not is_dist():
        return [local]
    gathered: List[Dict[str, float]] = [None] * get_world_size()  # type: ignore
    dist.all_gather_object(gathered, local)
    return gathered