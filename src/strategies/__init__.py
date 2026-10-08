# src/strategies/__init__.py
from .base import DistStrategy
from .ddp import DDPStrategy, SingleGPUStrategy
from .fsdp_zero2 import FSDPShardGradOpStrategy
from .fsdp_zero3 import FSDPFullShardStrategy

_REGISTRY = {
    "ddp": DDPStrategy,
    "single": SingleGPUStrategy,
    "fsdp_zero2": FSDPShardGradOpStrategy,
    "fsdp_zero3": FSDPFullShardStrategy,
}


def build_strategy(name: str, cfg: dict) -> DistStrategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Options: {list(_REGISTRY)}")
    return _REGISTRY[name](cfg)