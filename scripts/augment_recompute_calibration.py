#!/usr/bin/env python3
"""Persist calibrated recompute columns into the master CSV, in place.

Adds `recompute_ms_calibrated` and `bwd_compute_ms_calibrated` next to their raw
analytical counterparts, WITHOUT re-running any GPU work and without reformatting
untouched columns (csv-module round-trip). Both raw and calibrated are kept.

  recompute_ms_calibrated = ratio * fwd_time_ms_mean        (ratio from group4a)
  bwd_compute_ms_calibrated = max(0, bwd - recompute_cal - bwd_exposed_comm)

Idempotent: re-running recomputes/overwrites the calibrated columns in place.

Usage:
  python -m scripts.augment_recompute_calibration [outputs/all_experiments_summary.csv]
"""
from __future__ import annotations

import csv
import sys

from analysis.build_figures import recompute_ratio_from_rows, calibrated_recompute

NEW_COLS = ["recompute_ms_calibrated", "bwd_compute_ms_calibrated"]


def num(r, k):
    try:
        return float(r.get(k) or "")
    except (TypeError, ValueError):
        return None


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/all_experiments_summary.csv"
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise SystemExit(f"empty CSV: {csv_path}")
    fields = list(rows[0].keys())

    ratio = recompute_ratio_from_rows(rows)
    n_pairs = sum(1 for r in rows if str(r["experiment_name"]).endswith("_gcoff"))
    print(f"calibration ratio = {ratio:.4f}  (from {n_pairs} group4a _gcoff pair(s))")

    for r in rows:
        fwd, bwd, bexp = num(r, "fwd_time_ms_mean"), num(r, "bwd_time_ms_mean"), num(r, "bwd_exposed_comm_ms")
        gc = str(r.get("grad_checkpoint")) == "True"
        if fwd is None or bwd is None:          # OOM / no-step rows: leave blank
            r["recompute_ms_calibrated"] = ""
            r["bwd_compute_ms_calibrated"] = ""
            continue
        rc, bc = calibrated_recompute(fwd, bwd, bexp, gc, ratio)
        r["recompute_ms_calibrated"] = str(rc)
        r["bwd_compute_ms_calibrated"] = str(bc)

    # Insert each calibrated column right after its raw counterpart.
    out_fields, seen = [], set()
    for f in fields:
        if f not in seen:
            out_fields.append(f); seen.add(f)
        if f == "recompute_ms" and "recompute_ms_calibrated" not in seen:
            out_fields.append("recompute_ms_calibrated"); seen.add("recompute_ms_calibrated")
        if f == "bwd_compute_ms" and "bwd_compute_ms_calibrated" not in seen:
            out_fields.append("bwd_compute_ms_calibrated"); seen.add("bwd_compute_ms_calibrated")
    for c in NEW_COLS:                          # fallback if raw cols absent
        if c not in seen:
            out_fields.append(c); seen.add(c)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in out_fields})
    print(f"wrote {csv_path}: {len(rows)} rows, {len(out_fields)} columns "
          f"(added {', '.join(NEW_COLS)})")


if __name__ == "__main__":
    main()
