# src/strategies/fsdp_zero2.py
from __future__ import annotations

from torch.distributed.fsdp import ShardingStrategy

from .fsdp_zero3 import FSDPFullShardStrategy


class FSDPShardGradOpStrategy(FSDPFullShardStrategy):
    """ZeRO-2: shard grads + optimizer states, keep params replicated."""
    name = "fsdp_zero2"
    _sharding = ShardingStrategy.SHARD_GRAD_OP