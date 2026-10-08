# src/instrumentation/cuda_timers.py
"""CUDA-event-based phase timing (design doc §9 group A).

Usage:
    timers = CudaTimers()
    timers.start("iter")
    ...
    timers.start("fwd"); ...; timers.stop("fwd")
    ...
    timers.stop("iter")
    torch.cuda.synchronize()
    result = timers.read()   # dict of ms; must be called AFTER synchronize

ACCUMULATING semantics: a name may be start/stop'd multiple times within one
CudaTimers instance and read() returns the SUM of all its intervals. This is
required for gradient accumulation, where fwd/bwd are timed once per micro-batch
inside the step loop — the timer must report the whole iteration's fwd/bwd
(sum over all `ga` micro-batches), not just the last one. Timers called once
(iter, opt) are unaffected: one interval == its elapsed time.

(Previously each name reused a single start/end Event pair, so repeated
start/stop overwrote earlier micro-batches and read() returned only the LAST
micro-batch — under-counting compute by a factor of ~ga and inflating the
idle/exposed-comm attribution at ga>1.)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

_Pair = Tuple[torch.cuda.Event, torch.cuda.Event]


class CudaTimers:
    def __init__(self) -> None:
        # name -> list of (start_event, end_event) closed intervals
        self._intervals: Dict[str, List[_Pair]] = {}
        # name -> start_event awaiting its stop
        self._open: Dict[str, torch.cuda.Event] = {}

    def start(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._open[name] = ev  # a re-start before stop simply supersedes

    def stop(self, name: str) -> None:
        start = self._open.pop(name, None)
        if start is None:
            raise RuntimeError(f"CudaTimers.stop({name!r}) called without start")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._intervals.setdefault(name, []).append((start, end))

    def read(self) -> Dict[str, float]:
        """Return summed elapsed ms per timer. Caller MUST have synchronized."""
        out: Dict[str, float] = {}
        for name, pairs in self._intervals.items():
            out[f"{name}_time_ms"] = sum(s.elapsed_time(e) for s, e in pairs)
        return out

    def reset(self) -> None:
        self._intervals.clear()
        self._open.clear()
