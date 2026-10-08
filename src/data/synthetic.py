# src/data/synthetic.py
"""Synthetic in-memory causal-LM data (design doc §11).

Design goals:
  - Zero disk I/O; data contributes zero variance to timing.
  - Precompute enough samples for num_steps × global_batch + buffer.
  - Fixed seed per (model, strategy) so the token stream is identical across
    reps and strategies (removes data as a variance source in comparisons).

We keep this self-contained (no external synthetic_data import) so the repo
is reproducible from a single tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass
class SyntheticDataConfig:
    vocab_size: int
    seq_len: int
    total_samples: int
    seed: int = 42
    pad_token_id: int = 0
    causal_lm: bool = True


class SyntheticCausalLMDataset(Dataset):
    """Precomputed random token IDs; labels = input_ids for causal LM.

    Precomputation happens once on CPU with a fixed generator, so every rank
    (given the same seed) materializes the identical corpus, and the
    DistributedSampler slices disjoint, deterministic shards.
    """

    def __init__(self, cfg: SyntheticDataConfig) -> None:
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        # int32 storage halves RAM vs int64; cast to long at fetch time.
        self.input_ids = torch.randint(
            low=1,                       # avoid pad_token_id=0 in inputs
            high=cfg.vocab_size,
            size=(cfg.total_samples, cfg.seq_len),
            generator=g,
            dtype=torch.int32,
        )

    def __len__(self) -> int:
        return self.cfg.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx].to(torch.long)
        attn = torch.ones_like(ids)
        labels = ids.clone()
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def compute_total_samples(cfg: dict, world_size: int) -> int:
    """num_steps × grad_accum × per_device_bs × world_size + buffer."""
    w = cfg["workload"]
    p = cfg["protocol"]
    ga = cfg.get("grad_accum_steps", 1)
    buffer_batches = 8
    per_step = w["per_device_batch_size"] * max(1, world_size) * ga
    return p["num_steps"] * per_step + buffer_batches * per_step


def build_dataloader(cfg: dict, world_size: int, rank: int,
                     vocab_size: int) -> DataLoader:
    w = cfg["workload"]
    data_cfg = SyntheticDataConfig(
        vocab_size=vocab_size,
        seq_len=w["seq_len"],
        total_samples=compute_total_samples(cfg, world_size),
        seed=cfg["experiment"]["seed"],
        causal_lm=True,
    )
    dataset = SyntheticCausalLMDataset(data_cfg)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=w["per_device_batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,          # data is in RAM; workers add jitter
        pin_memory=True,
    )


class MicroBatchFeeder:
    """Yields grad_accum micro-batches per optimizer step from a looping loader."""

    def __init__(self, loader: DataLoader, device: torch.device,
                 grad_accum_steps: int) -> None:
        self.loader = loader
        self.device = device
        self.ga = grad_accum_steps
        self._iter = iter(loader)
        self._epoch = 0

    def _next_batch(self) -> Dict[str, torch.Tensor]:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._epoch += 1
            if hasattr(self.loader, "sampler") and \
               isinstance(self.loader.sampler, DistributedSampler):
                self.loader.sampler.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            batch = next(self._iter)
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def next_step(self):
        """Return a list of ga micro-batches for one optimizer step."""
        return [self._next_batch() for _ in range(self.ga)]