#!/usr/bin/env python3
"""Rebuild the master CSV from per-run summary.json, fixing column misalignment.

Root cause of the misalignment
------------------------------
`io_utils.append_master_row` writes with `fieldnames=MASTER_FIELDS`, which does
NOT contain the two columns `bwd_compute_ms_calibrated` / `recompute_ms_calibrated`
that `scripts/augment_recompute_calibration.py` later inserted into the CSV. Every
row appended AFTER that augment step therefore wrote 69 values (MASTER_FIELDS order)
into a 71-column file, so every field from `bwd_compute_ms_calibrated` onward is
shifted right by up to two positions (throughput lands in straggler_wait_ms, loss in
throughput, and status runs off the end -> blank status). The per-run summary.json
files are intact and authoritative.

This script rebuilds each row's canonical fields from summary.json BY NAME, recomputes
the two calibrated columns exactly as the augment script does, preserves rows that have
no summary.json (e.g. OOM 32B/DDP), keeps the original row order and column order, and
writes to a NEW file by default so the original is preserved.

Usage:
  python -m scripts.rebuild_master_csv                 # -> outputs/all_experiments_summary_fixed.csv
  python -m scripts.rebuild_master_csv --inplace       # overwrite, keeping a .bak
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil

from src.io_utils import MASTER_FIELDS
from analysis.build_figures import recompute_ratio_from_rows, calibrated_recompute

CALIBRATED = ["recompute_ms_calibrated", "bwd_compute_ms_calibrated"]


def _flat(v):
    """Match io_utils._flatten_arrays: list -> rounded JSON string."""
    if isinstance(v, (list, tuple)):
        return json.dumps([round(float(x), 4) for x in v])
    return "" if v is None else v


def _num(r, k):
    try:
        return float(r.get(k) or "")
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/all_experiments_summary.csv")
    ap.add_argument("--exp_root", default="outputs/experiments")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the input CSV (a .bak copy is kept)")
    args = ap.parse_args()

    cur = list(csv.DictReader(open(args.csv)))
    header = list(cur[0].keys())  # preserve existing column set + order (incl. calibrated)

    # index summary.json by experiment_name
    summ = {}
    for p in glob.glob(os.path.join(args.exp_root, "*", "summary.json")):
        s = json.load(open(p))
        summ[s["experiment_name"]] = s

    rebuilt, fixed, kept, no_summary, deduped = [], 0, 0, 0, 0
    seen = set()
    for row in cur:
        e = row["experiment_name"]
        if e in seen:                       # --force re-runs append a 2nd row;
            deduped += 1                     # collapse to one (summary.json is
            continue                         # authoritative, so content is identical)
        seen.add(e)
        if e not in summ:
            rebuilt.append(dict(row))       # no summary.json -> keep verbatim
            no_summary += 1
            continue
        s = summ[e]
        # authoritative name-based values for every canonical field
        new = {k: _flat(s.get(k, "")) for k in MASTER_FIELDS}
        # carry over any extra columns the CSV has that aren't canonical or calibrated
        for k in header:
            if k not in MASTER_FIELDS and k not in CALIBRATED:
                new[k] = row.get(k, "")
        was_misaligned = not (row.get("status") or "").strip()
        rebuilt.append(new)
        fixed += was_misaligned
        kept += (not was_misaligned)

    # recompute calibrated columns exactly as augment_recompute_calibration does
    ratio = recompute_ratio_from_rows(rebuilt)
    for r in rebuilt:
        fwd, bwd, bexp = _num(r, "fwd_time_ms_mean"), _num(r, "bwd_time_ms_mean"), _num(r, "bwd_exposed_comm_ms")
        gc = str(r.get("grad_checkpoint")) == "True"
        if fwd is None or bwd is None:
            r["recompute_ms_calibrated"] = r["bwd_compute_ms_calibrated"] = ""
        else:
            rc, bc = calibrated_recompute(fwd, bwd, bexp, gc, ratio)
            r["recompute_ms_calibrated"], r["bwd_compute_ms_calibrated"] = str(rc), str(bc)

    out = args.csv if args.inplace else args.csv.replace(".csv", "_fixed.csv")
    if args.inplace:
        shutil.copy2(args.csv, args.csv + ".bak")
        print(f"backed up original -> {args.csv}.bak")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rebuilt:
            w.writerow({k: r.get(k, "") for k in header})

    print(f"calibration ratio = {ratio:.4f}")
    print(f"rebuilt {len(rebuilt)} rows: fixed {fixed} misaligned, "
          f"{kept} already-correct unchanged, {no_summary} kept verbatim (no summary.json), "
          f"{deduped} duplicate row(s) collapsed")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
