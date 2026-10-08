#!/usr/bin/env python3
"""Print achieved_bw vs gradient-accumulation depth, per (model, ws, strategy).

Diagnostic for notes/08 §4b: once the boundary-truncation fix (notes/07) is
applied and the ga>1 cells are re-profiled + merged, achieved_bw should be
~ga-INDEPENDENT for every strategy. Before the fix, ZeRO-2/DDP achieved_bw
collapses ~ga x (flat collective count + inflated comm-time). Run this before and
after the re-run to see whether 4b resolved.

    python -m scripts.check_ga_bandwidth [--csv outputs/all_experiments_summary.csv]
                                         [--metric achieved_bw_gbps] [--stable_tol 0.85]

A group is "stable" if min/max of the metric across ga >= stable_tol (i.e. it
does not drift by more than (1-tol)). ZeRO-3 already passes; ZeRO-2/DDP are the
ones to watch.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

GAS = [1, 2, 4, 8]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/all_experiments_summary.csv")
    ap.add_argument("--metric", default="achieved_bw_gbps",
                    help="column to track across ga (default achieved_bw_gbps)")
    ap.add_argument("--stable_tol", type=float, default=0.85,
                    help="min/max ratio across ga to call a group stable")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    # group[(model, ws, strat)][ga] = metric value
    group = defaultdict(dict)
    for r in rows:
        if r.get("status", "").lower() != "ok":
            continue
        try:
            ga = int(float(r.get("gradient_accumulation_steps") or 0))
            ws = int(float(r.get("world_size") or 0))
        except (TypeError, ValueError):
            continue
        val = _num(r.get(args.metric))
        if ga in GAS and val is not None and val > 0:
            group[(r.get("model"), ws, r.get("strategy"))][ga] = val

    hdr = f"{'model':12s} {'ws':>2s} {'strategy':11s}  " + \
        "  ".join(f"ga{g:<6d}" for g in GAS) + f"  {'min/max':>7s}  verdict"
    print(f"\n{args.metric} vs ga  (should be ~flat across ga once notes/08 §4b resolves)\n")
    print(hdr)
    print("-" * len(hdr))

    per_strat = defaultdict(list)
    for (model, ws, strat) in sorted(group,
                                     key=lambda k: (k[2], str(k[0]), k[1])):
        vals = group[(model, ws, strat)]
        present = [vals[g] for g in GAS if g in vals]
        if len(present) < 2:
            continue
        ratio = min(present) / max(present)
        stable = ratio >= args.stable_tol
        per_strat[strat].append(ratio)
        cells = "  ".join((f"{vals[g]:7.1f}" if g in vals else f"{'—':>7s}")
                          for g in GAS)
        verdict = "stable" if stable else "DRIFT (ga-artifact?)"
        print(f"{str(model):12s} {ws:>2d} {strat:11s}  {cells}  {ratio:6.2f}   {verdict}")

    print("\nPer-strategy mean(min/max across ga)  [1.00 = ga-independent = good]:")
    for strat in sorted(per_strat):
        r = per_strat[strat]
        flag = "OK" if sum(r) / len(r) >= args.stable_tol else "<-- still drifting"
        print(f"  {strat:12s} {sum(r)/len(r):.2f}  (n={len(r)})  {flag}")


if __name__ == "__main__":
    raise SystemExit(main())
