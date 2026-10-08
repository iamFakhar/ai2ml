# src/strategies/ddp.py
from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import DistStrategy
from ..distributed import is_dist


class DDPStrategy(DistStrategy):
    name = "ddp"
    uses_external_autocast = True

    def wrap_model(self, model, local_rank, decoder_layer_cls):
        if not is_dist():
            return model  # single-process safety
        d = self.cfg["distributed"]["ddp"]
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=d["broadcast_buffers"],
            find_unused_parameters=d["find_unused_parameters"],
            gradient_as_bucket_view=d["gradient_as_bucket_view"],
            bucket_cap_mb=d["bucket_cap_mb"],
            static_graph=d["static_graph"],
        )

    def build_optimizer(self, model):
        return self._make_adamw(model.parameters())

    def _no_sync(self, model):
        return model.no_sync()

    def clip_grad_norm(self, model, max_norm):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def peak_memory_semantics(self):
        return "full_replica"


class SingleGPUStrategy(DDPStrategy):
    """Group 2 baseline: no distributed wrapper (design doc §6)."""
    name = "single"

    def wrap_model(self, model, local_rank, decoder_layer_cls):
        return model  # no wrapper at all

    def _no_sync(self, model):
        from contextlib import nullcontext
        return nullcontext()