# analysis/parse_nccl_tests.py
"""Parse nccl-tests raw logs into a size-resolved BW curve (design doc §7).

nccl-tests output columns (typical):
   size   count  type  redop  root  time  algbw  busbw  #wrong  ...
  (bytes)                            (us)  (GB/s) (GB/s)

We record BUSBW (bus bandwidth) as the achievable-bandwidth reference, since
it accounts for the collective's data-movement factor and is the standard
NVSwitch ceiling metric. Output format:

  {
    "8": {
      "allreduce":     [[size_bytes, busbw_gbps], ...],
      "allgather":     [[...]],
      "reducescatter": [[...]]
    },
    "4": {...}, "2": {...}
  }
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

# Matches a data row: leading size in bytes, ... two BW columns near the end.
# We anchor on the first integer (size) and capture the two float BW columns.
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+.*?(\d+\.\d+)\s+(\d+\.\d+)\s*(?:\d+)?\s*$"
)
_FNAME_RE = re.compile(r"(allreduce|allgather|reducescatter)_g(\d+)\.log$")


def parse_log(path: str) -> List[Tuple[int, float]]:
    """Return [(size_bytes, busbw_gbps), ...] from one nccl-tests log."""
    points: List[Tuple[int, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if not m:
                continue
            size = int(m.group(1))
            # group(2)=algbw, group(3)=busbw  -> we take busbw as the ceiling
            busbw = float(m.group(3))
            if size > 0 and busbw > 0:
                points.append((size, busbw))
    # Deduplicate on size, keep max busbw (best observed at that size).
    best: Dict[int, float] = {}
    for size, bw in points:
        best[size] = max(best.get(size, 0.0), bw)
    return sorted(best.items())


def build_curves(raw_dir: str) -> Dict[str, Dict[str, List[List[float]]]]:
    curves: Dict[str, Dict[str, List[List[float]]]] = {}
    for path in glob.glob(os.path.join(raw_dir, "*.log")):
        m = _FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f"  skip (unrecognized name): {path}")
            continue
        coll, g = m.group(1), m.group(2)
        pts = parse_log(path)
        if not pts:
            print(f"  WARNING: no data parsed from {path}")
            continue
        curves.setdefault(g, {})[coll] = [[float(s), float(b)] for s, b in pts]
        peak = max(b for _, b in pts)
        print(f"  {coll} @ {g} GPUs: {len(pts)} points, peak busbw={peak:.1f} GB/s")
    return curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    curves = build_curves(args.raw_dir)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()