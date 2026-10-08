# src/strategies/base.py
"""Strategy abstraction (design doc §8).

Only the genuinely strategy-specific ~15% lives behind this interface:
wrapping, optimizer construction timing, no_sync, grad clipping,
mixed-precision handling, and memory semantics. Everything else (timing,
aggregation, protocol) is shared in runner.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


class DistStrategy(ABC):
    name: str = "base"
    uses_external_autocast: bool = True   # DDP/single True; FSDP overrides False

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def wrap_model(self, model: torch.nn.Module, local_rank: int,
                   decoder_layer_cls) -> torch.nn.Module: ...

    @abstractmethod
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Called AFTER wrap_model (critical for FSDP)."""

    @contextmanager
    def maybe_no_sync(self, model: torch.nn.Module, is_last_micro: bool) -> Iterator[None]:
        """Skip grad sync on non-final micro-steps (grad accumulation)."""
        if is_last_micro:
            yield
        else:
            with self._no_sync(model):
                yield

    @abstractmethod
    def _no_sync(self, model: torch.nn.Module): ...

    @abstractmethod
    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float) -> torch.Tensor: ...

    def autocast_context(self, dtype: torch.dtype):
        if self.uses_external_autocast:
            return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
        return nullcontext()

    @abstractmethod
    def peak_memory_semantics(self) -> str: ...

    def theoretical_comm_volume_bytes(self, num_params: int, world_size: int) -> float:
        """bf16 = 2 bytes. Factors per design doc §7 Type 2."""
        if world_size <= 1:
            return 0.0
        n = world_size
        factor = 3 * (n - 1) / n if self.name == "fsdp_zero3" else 2 * (n - 1) / n
        return factor * num_params * 2

    # Optimizer helper shared by all strategies.
    def _make_adamw(self, params) -> torch.optim.AdamW:
        opt = self.cfg["optimizer"]
        return torch.optim.AdamW(
            params,
            lr=opt["lr"],
            betas=tuple(opt["betas"]),
            eps=opt["eps"],
            weight_decay=opt["weight_decay"],
            fused=opt.get("fused", True) and torch.cuda.is_available(),
        )