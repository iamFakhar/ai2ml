# src/strategies/fsdp_zero3.py
from __future__ import annotations

import functools

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .base import DistStrategy


class FSDPFullShardStrategy(DistStrategy):
    name = "fsdp_zero3"
    uses_external_autocast = False        # MixedPrecision policy handles casting
    _sharding = ShardingStrategy.FULL_SHARD

    def wrap_model(self, model, local_rank, decoder_layer_cls):
        f = self.cfg["distributed"]["fsdp"]
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={decoder_layer_cls},
        )
        prefetch = (BackwardPrefetch.BACKWARD_PRE
                    if f["backward_prefetch"] == "BACKWARD_PRE"
                    else BackwardPrefetch.BACKWARD_POST)
        return FSDP(
            model,
            sharding_strategy=self._sharding,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp,
            backward_prefetch=prefetch,
            forward_prefetch=f["forward_prefetch"],
            limit_all_gathers=f["limit_all_gathers"],
            use_orig_params=f["use_orig_params"],
            device_id=torch.cuda.current_device(),
        )

    def build_optimizer(self, model):
        # MUST be after FSDP wrap; use_orig_params=True keeps this clean.
        return self._make_adamw(model.parameters())

    def _no_sync(self, model):
        return model.no_sync()

    def clip_grad_norm(self, model, max_norm):
        # FSDP method correctly reduces across sharded grads.
        return model.clip_grad_norm_(max_norm)

    def peak_memory_semantics(self):
        return "sharded"