# orchestration/quick_probe_fsdp.py
"""FSDP (ZeRO-3) probe to confirm all-gather + reduce-scatter kernel names.

Run from repo root:
    cd /home/paperspace/main
    torchrun --standalone --nproc_per_node=2 orchestration/quick_probe_fsdp.py

FULL_SHARD fires:
  - all-gather      (params, during forward AND backward)
  - reduce-scatter  (gradients, during backward)
So this single run reveals both ncclDevKernel_AllGather_* and
ncclDevKernel_ReduceScatter_* name strings for the corrected regex.
"""
import functools
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.profiler import profile, ProfilerActivity


# A tiny transformer-ish block so auto_wrap has a layer class to shard.
class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden, 4 * hidden)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(4 * hidden, hidden)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x + self.fc2(self.act(self.fc1(x))))


class TinyModel(torch.nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(hidden) for _ in range(n_layers)])
        self.head = torch.nn.Linear(hidden, hidden)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _self_cuda_us(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return float(v)
    return 0.0


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    hidden, n_layers = 2048, 6
    model = TinyModel(hidden, n_layers).to(device)  # keep fp32; FSDP MP casts

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TinyBlock},   # <-- shard each TinyBlock
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        device_id=local_rank,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
    target = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
    loss_fn = torch.nn.MSELoss()

    # warmup
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # profiled
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)   # <-- all-gather (fwd)
            loss.backward()                        # <-- all-gather + reduce-scatter (bwd)
            optimizer.step()
        torch.cuda.synchronize()

    if rank == 0:
        print("\n" + "=" * 95)
        print("FSDP (ZeRO-3) EVENTS WITH SELF DEVICE TIME > 0  (sorted)")
        print("=" * 95)
        events = sorted(prof.key_averages(), key=_self_cuda_us, reverse=True)

        ag_ms = rs_ms = other_nccl_ms = 0.0
        printed = 0
        for e in events:
            us = _self_cuda_us(e)
            if us <= 0:
                continue
            name = e.key
            low = name.lower()
            tag = ""
            if "nccldevkernel" in low or "ncclkernel" in low:
                if "allgather" in low:
                    tag = "  <== ALL-GATHER"
                    ag_ms += us / 1e3
                elif "reducescatter" in low:
                    tag = "  <== REDUCE-SCATTER"
                    rs_ms += us / 1e3
                elif "allreduce" in low:
                    tag = "  <== ALL-REDUCE (unexpected in ZeRO-3)"
                    other_nccl_ms += us / 1e3
                else:
                    tag = "  <== NCCL (other)"
                    other_nccl_ms += us / 1e3
            print(f"{us/1e3:9.3f} ms  count={e.count:4d}  {name[:72]}{tag}")
            printed += 1
            if printed >= 45:
                print("... (truncated)")
                break

        print("=" * 95)
        print(f"  all-gather      device time: {ag_ms:8.3f} ms")
        print(f"  reduce-scatter  device time: {rs_ms:8.3f} ms")
        print(f"  other NCCL      device time: {other_nccl_ms:8.3f} ms")
        print("=" * 95)
        print("\nCopy the exact ncclDevKernel_AllGather_* and")
        print("ncclDevKernel_ReduceScatter_* strings to confirm the regex.\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()