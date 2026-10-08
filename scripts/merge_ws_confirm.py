#!/usr/bin/env python3
"""Fold the ws_confirm re-run means into the master CSV, superseding the
low-SNR / corrupt small-model cells (and optionally the 32B/ws4 replicate).

Each target cell is replaced by the MEAN across its re-run replicates
(outputs/ws_confirm/experiments/<name>_rep*/summary.json). Derived columns are
recomputed exactly as the project pipeline does:
  recompute_ms_calibrated  = ratio * fwd_time_ms_mean        (ratio=0.7230, gc on)
  bwd_compute_ms_calibrated= max(0, bwd - recompute_cal - bwd_exposed_comm)
  scaling_efficiency       = 100 * throughput / (ws * T1_single)
Columns not re-measured (transient_peak_gb, overlap_percentage_ablation,
peak_memory_semantics) are preserved from the canonical row.

Preview (default): writes outputs/all_experiments_summary_PREVIEW.csv, prints diff.
Commit: --inplace  -> backs up canonical to .bak_<ts>, archives superseded rows.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, statistics as st, sys

RATIO = 0.7230
MASTER = "outputs/all_experiments_summary.csv"
RERUN_GLOB = "outputs/ws_confirm/experiments/%s_rep*/summary.json"  # %s = stem w/o _repN

# canonical experiment_name (rep0) of each cell to supersede
TARGETS = [
    "qwen3-0.6b_fsdp_zero3_ws2_ga1_bs2_seq2048_rep0",
    "qwen3-1.7b_fsdp_zero3_ws2_ga1_bs2_seq2048_rep0",
    "qwen3-0.6b_ddp_ws2_ga1_bs2_seq2048_rep0",
    "qwen3-0.6b_ddp_ws8_ga1_bs2_seq2048_rep0",
    "qwen3-1.7b_ddp_ws8_ga1_bs2_seq2048_rep0",
]

# columns preserved verbatim from the canonical row (not re-measured here)
PRESERVE = {"transient_peak_gb", "overlap_percentage_ablation", "peak_memory_semantics"}
NON_NUMERIC = {"experiment_name", "model", "strategy", "precision", "status",
               "peak_memory_semantics", "timestamp"}


def load_baselines(rows):
    t1 = {}
    for r in rows:
        if r.get("strategy") == "single" and r.get("status") == "ok":
            try:
                t1.setdefault(r["model"], []).append(float(r["throughput_tokens_per_sec_mean"]))
            except (KeyError, ValueError):
                pass
    return {m: sum(v) / len(v) for m, v in t1.items()}


def mean_rerun(stem):
    """stem = experiment_name WITHOUT _rep0 suffix. Average all rep summary.json."""
    base = stem.rsplit("_rep", 1)[0]
    files = sorted(glob.glob(RERUN_GLOB % base))
    if not files:
        return None, []
    dicts = [json.load(open(f)) for f in files]
    agg = {}
    keys = set().union(*[d.keys() for d in dicts])
    for k in keys:
        vals = [d.get(k) for d in dicts]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == len(vals) and nums:
            agg[k] = sum(nums) / len(nums)
        else:
            agg[k] = dicts[0].get(k)
    return agg, [os.path.basename(os.path.dirname(f)) for f in files]


def build_row(canon, agg, ws, model, t1):
    row = dict(canon)  # start from canonical, preserve column set + PRESERVE cols
    for k, v in agg.items():
        if k in PRESERVE or k not in canon:
            continue
        row[k] = v
    # keep identity columns from canonical (verbatim strings: preserve int dtype
    # of the column across the CSV round-trip; averaging would yield e.g. "2.0")
    for k in ("experiment_name", "model", "strategy", "precision",
              "world_size", "per_device_batch_size", "seq_length",
              "gradient_accumulation_steps", "grad_checkpoint"):
        row[k] = canon[k]
    row["status"] = "ok"
    row["num_measured_steps"] = int(round(agg.get("num_measured_steps", canon.get("num_measured_steps") or 0)))
    # derived: calibrated (gc assumed on for these cells)
    gc = str(agg.get("grad_checkpoint", canon.get("grad_checkpoint"))).lower() in ("true", "1", "1.0")
    fwd = float(agg["fwd_time_ms_mean"]); bwd = float(agg["bwd_time_ms_mean"])
    bexp = float(agg.get("bwd_exposed_comm_ms") or 0.0)
    rc = RATIO * fwd if gc else 0.0
    row["recompute_ms_calibrated"] = rc
    row["bwd_compute_ms_calibrated"] = max(0.0, bwd - rc - bexp)
    # derived: scaling efficiency
    if model in t1 and t1[model] > 0:
        row["scaling_efficiency"] = 100.0 * float(agg["throughput_tokens_per_sec_mean"]) / (ws * t1[model])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra canonical rep0 names to supersede (e.g. 32B/ws4)")
    args = ap.parse_args()

    with open(MASTER) as f:
        rd = csv.DictReader(f); rows = list(rd); hdr = rd.fieldnames
    t1 = load_baselines(rows)
    targets = TARGETS + args.extra
    by_name = {r["experiment_name"]: r for r in rows}

    changes = []
    for name in targets:
        canon = by_name.get(name)
        if canon is None:
            print("  WARN: canonical row not found:", name); continue
        agg, used = mean_rerun(name)
        if agg is None:
            print("  WARN: no re-runs for", name); continue
        ws = int(float(canon["world_size"])); model = canon["model"]
        new = build_row(canon, agg, ws, model, t1)
        changes.append((name, canon, new, used))

    # report
    def pe(r):
        try: return 100 * float(r["bwd_exposed_comm_ms"]) / float(r["iter_time_ms_mean"])
        except: return float("nan")
    print("\n%-46s %8s %8s   %8s %8s   reps" % ("cell", "OLD e%", "NEW e%", "OLD ov%", "NEW ov%"))
    print("-" * 100)
    for name, old, new, used in changes:
        print("%-46s %8.2f %8.2f   %8.2f %8.2f   %d" % (
            name.replace("_ga1_bs2_seq2048_rep0", ""), pe(old), pe(new),
            float(old["overlap_percentage_trace"]), float(new["overlap_percentage_trace"]), len(used)))

    # apply
    new_by = {n: nw for n, _, nw, _ in changes}
    out_rows = [new_by.get(r["experiment_name"], r) for r in rows]
    assert len(out_rows) == len(rows)

    if args.inplace:
        ts = os.environ.get("MERGE_TS", "backup")
        os.rename(MASTER, MASTER.replace(".csv", ".bak_%s.csv" % ts))
        with open("outputs/superseded_lowSNR_rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
            for _, old, _, _ in changes: w.writerow(old)
        target = MASTER
    else:
        target = MASTER.replace(".csv", "_PREVIEW.csv")
    with open(target, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out_rows)
    print("\nwrote %s (%d rows, %d cells superseded)" % (target, len(out_rows), len(changes)))


if __name__ == "__main__":
    main()
