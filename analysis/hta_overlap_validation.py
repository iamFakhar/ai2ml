#!/usr/bin/env python3
"""Cross-validate the custom trace-based comm/compute overlap against HTA.

RQ2a claims "two methods": (a) our own timeline overlap
(`overlap_percentage_trace`, computed in src/instrumentation/profiler.py as
    overlap = |comm_busy ∩ compute_busy| / |comm_busy|
) and (b) an independent tool. This script IS method (b): it runs Meta's
Holistic Trace Analysis (HTA) `get_comm_comp_overlap` on the *same* Kineto
traces and reports HTA overlap vs our number, with the per-cell delta.

Why this is a fair comparison
-----------------------------
HTA's `get_comm_comp_overlap` uses the identical definition — it sums the time
where a communication and a computation kernel run simultaneously and divides
by the merged communication-busy time (see
hta/analyzers/communication_analysis.py). The one definitional difference to
keep in mind when reading the delta:

  * OUR tool classifies compute as "every non-NCCL GPU kernel", so a memcpy that
    overlaps a collective counts as overlap.
  * HTA classifies memcpy/memset as a separate MEMORY kernel type (not
    COMPUTATION), so such overlap does NOT count.

Net effect: HTA overlap is expected to be equal-or-slightly-LOWER than ours.
A few points-per-cent gap in that direction is a classification artifact, not a
measurement disagreement.

No new GPU work
---------------
This is pure post-hoc trace analysis. It only reads traces already kept on disk
(the cells that were run with --keep_traces). By default those are the three
4B / 8-GPU anchors, one per strategy.

Usage
-----
    python -m analysis.hta_overlap_validation \
        --csv outputs/all_experiments_summary.csv \
        --exp_root outputs/experiments \
        --out outputs/figures/hta_overlap_validation.csv

    # limit memory/time by analysing fewer ranks per cell (overlap is
    # near-uniform across ranks on a single NVSwitch node):
    python -m analysis.hta_overlap_validation --max_ranks 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import warnings
from pathlib import Path

# pandas emits a noisy FutureWarning from inside HTA's .replace() calls.
warnings.filterwarnings("ignore", category=FutureWarning)

# Default anchor cells (experiment_name, human label). These are the only cells
# whose traces survive on disk; override with --cells if you keep more.
DEFAULT_CELLS = [
    ("qwen3-4b_ddp_ws8_ga1_bs2_seq2048_rep0", "DDP"),
    ("qwen3-4b_fsdp_zero2_ws8_ga1_bs2_seq2048_rep0", "ZeRO-2"),
    ("qwen3-4b_fsdp_zero3_ws8_ga1_bs2_seq2048_rep0", "ZeRO-3"),
]

CUSTOM_COL = "overlap_percentage_trace"
# distributedInfo sits in the first few KB of the Kineto JSON; never parse the
# whole ~200 MB file just to recover the rank.
_HEADER_BYTES = 65536
_DIST_RE = re.compile(r'"distributedInfo"\s*:\s*\{([^}]*)\}')
_RANK_RE = re.compile(r'"rank"\s*:\s*(\d+)')
_WS_RE = re.compile(r'"world_size"\s*:\s*(\d+)')


def _read_rank_and_ws(path: Path) -> tuple[int | None, int | None]:
    """Cheaply extract (rank, world_size) from a Kineto trace's header."""
    try:
        with open(path, "rb") as f:
            head = f.read(_HEADER_BYTES).decode("utf-8", errors="ignore")
    except OSError:
        return None, None
    m = _DIST_RE.search(head)
    if not m:
        return None, None
    block = m.group(1)
    r = _RANK_RE.search(block)
    w = _WS_RE.search(block)
    return (int(r.group(1)) if r else None, int(w.group(1)) if w else None)


def select_run(traces_dir: Path, gap_s: float = 180.0):
    """Pick one coherent run (rank -> trace file) from a traces/ directory.

    A single cell's traces/ dir often holds files from several re-profiling
    runs (each run = one file per rank). We cluster files by modification time
    (files from the same torchrun are written within seconds of each other),
    then choose the cluster covering the most ranks (tie-break: most recent).
    Within the chosen run, if a rank appears twice we keep the larger file.

    Returns (rank_to_path: dict[int, str], world_size: int | None, mtime: float).
    """
    # Prefer the current per-rank export: trace_rank{n}.json (rank in filename,
    # one file per exported rank — the boundary-fixed traces). Only fall back to
    # legacy *.pt.trace.json + mtime clustering if no per-rank export is present.
    rank_files = sorted(traces_dir.glob("trace_rank*.json"))
    if rank_files:
        rank_map, ws_seen, mtime = {}, None, 0.0
        for p in rank_files:
            m = re.search(r"trace_rank(\d+)\.json$", p.name)
            if not m:
                continue
            rank = int(m.group(1))
            _, ws = _read_rank_and_ws(p)
            if ws is not None:
                ws_seen = ws
            rank_map[rank] = str(p)
            mtime = max(mtime, p.stat().st_mtime)
        if rank_map:
            return rank_map, ws_seen, mtime

    entries = []  # (mtime, rank, size, path) -- legacy tensorboard traces
    ws_seen = None
    for p in sorted(traces_dir.glob("*.pt.trace.json")):
        rank, ws = _read_rank_and_ws(p)
        if rank is None:
            continue
        if ws is not None:
            ws_seen = ws
        entries.append((p.stat().st_mtime, rank, p.stat().st_size, str(p)))
    if not entries:
        return {}, ws_seen, 0.0

    entries.sort(key=lambda e: e[0])
    # Cluster by mtime gap.
    clusters: list[list] = [[entries[0]]]
    for e in entries[1:]:
        if e[0] - clusters[-1][-1][0] > gap_s:
            clusters.append([e])
        else:
            clusters[-1].append(e)

    def rank_map(cluster):
        m: dict[int, tuple[int, str]] = {}  # rank -> (size, path)
        for mtime, rank, size, path in cluster:
            if rank not in m or size > m[rank][0]:
                m[rank] = (size, path)
        return {r: sp[1] for r, sp in m.items()}

    # Most ranks covered wins; tie-break by most recent (last cluster).
    best = max(
        range(len(clusters)),
        key=lambda i: (len(rank_map(clusters[i])), clusters[i][-1][0]),
    )
    chosen = clusters[best]
    return rank_map(chosen), ws_seen, chosen[-1][0]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def overlap_for_cell(traces_dir: Path, max_ranks: int):
    """Run BOTH overlap methods on the *same* trace files, per rank.

    Running the custom method (`compute_overlap_from_trace`, the exact
    production code) and HTA's `get_comm_comp_overlap` over an identical set of
    per-rank trace files isolates the pure method difference: same run, same
    ranks, same steps. (The CSV's `overlap_percentage_trace` is only rank-0 of
    whichever run last wrote summary.json, so it is kept separately, as a
    reference, not as the head-to-head number.)

    Returns dict with per-rank pairs and summary stats, or None if no traces.
    """
    rank_map, ws, mtime = select_run(traces_dir)
    if not rank_map:
        return None
    if max_ranks > 0:
        keep = sorted(rank_map)[:max_ranks]
        rank_map = {r: rank_map[r] for r in keep}
    # Absolute paths are safe: HTA joins with trace_dir, and os.path.join with
    # an absolute second arg returns that absolute path unchanged.
    rank_map = {r: os.path.abspath(p) for r, p in rank_map.items()}

    # Import lazily so --help works without the deps installed.
    from hta.trace_analysis import TraceAnalysis
    from src.instrumentation.profiler import compute_overlap_from_trace

    # HTA (mean of per-rank values it computes over the given files).
    analyzer = TraceAnalysis(trace_files=rank_map, trace_dir=str(traces_dir))
    hta_df = analyzer.get_comm_comp_overlap(visualize=False)
    hta_by_rank = {int(r): float(v)
                   for r, v in zip(hta_df["rank"], hta_df["comp_comm_overlap_pctg"])}

    # Custom method on the identical files.
    custom_by_rank = {}
    for rank, path in rank_map.items():
        res = compute_overlap_from_trace(path)
        if res and "overlap_percentage_trace" in res:
            custom_by_rank[rank] = float(res["overlap_percentage_trace"])

    ranks = sorted(set(hta_by_rank) & set(custom_by_rank))
    per_rank = [(r, custom_by_rank[r], hta_by_rank[r]) for r in ranks]
    custom_vals = [c for _, c, _ in per_rank]
    hta_vals = [h for _, _, h in per_rank]
    return {
        "ranks_used": len(ranks),
        "ws": ws,
        "run_mtime": mtime,
        "custom_mean": _mean(custom_vals),
        "hta_mean": _mean(hta_vals),
        "hta_min": min(hta_vals) if hta_vals else float("nan"),
        "hta_max": max(hta_vals) if hta_vals else float("nan"),
        "per_rank": per_rank,
    }


def load_custom_overlap(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(CUSTOM_COL, "")
            try:
                out[row["experiment_name"]] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/all_experiments_summary.csv",
                    help="master CSV holding the custom overlap_percentage_trace column")
    ap.add_argument("--exp_root", default="outputs/experiments",
                    help="root dir containing <experiment_name>/traces/*.json")
    ap.add_argument("--out", default="outputs/figures/hta_overlap_validation.csv",
                    help="where to write the comparison table (CSV)")
    ap.add_argument("--max_ranks", type=int, default=4,
                    help="max ranks to analyse per cell (0 = all). Overlap is "
                         "near-uniform across ranks; fewer ranks = less RAM/time.")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="override experiment_names to validate (default: the "
                         "three 4B/8-GPU anchors)")
    ap.add_argument("--quiet_hta", action="store_true", default=True,
                    help="suppress HTA's INFO logging (default on)")
    args = ap.parse_args(argv)

    if args.quiet_hta:
        logging.getLogger("hta").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.WARNING)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2
    custom = load_custom_overlap(csv_path)

    if args.cells:
        cells = [(name, name) for name in args.cells]
    else:
        cells = DEFAULT_CELLS

    rows = []
    for exp_name, label in cells:
        traces_dir = Path(args.exp_root) / exp_name / "traces"
        if not traces_dir.is_dir():
            print(f"  [skip] {label}: no traces dir at {traces_dir}")
            continue
        print(f"  [run ] {label}: both methods on same traces in {traces_dir} ...", flush=True)
        try:
            res = overlap_for_cell(traces_dir, args.max_ranks)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  [fail] {label}: raised {type(e).__name__}: {e}")
            continue
        if res is None:
            print(f"  [skip] {label}: no usable trace files found")
            continue
        delta = res["custom_mean"] - res["hta_mean"]
        rows.append({
            "experiment_name": exp_name,
            "strategy": label,
            "custom_overlap_pct_mean": round(res["custom_mean"], 2),
            "hta_overlap_pct_mean": round(res["hta_mean"], 2),
            "hta_overlap_pct_min": round(res["hta_min"], 2),
            "hta_overlap_pct_max": round(res["hta_max"], 2),
            "delta_pp": round(delta, 2),
            "csv_recorded_rank0_pct": (round(custom[exp_name], 2)
                                       if exp_name in custom else None),
            "ranks_used": res["ranks_used"],
            "world_size": res["ws"],
        })

    if not rows:
        print("No cells validated (no traces on disk?).", file=sys.stderr)
        return 1

    # ---- pretty table (custom vs HTA on identical trace files) ----
    hdr = ["Strategy", "Custom %", "HTA % (mean)", "HTA min/max", "Δ (pp)", "ranks", "CSV(r0)"]
    w = [10, 9, 13, 13, 8, 6, 8]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print("\n  Same trace files, per rank — custom vs HTA:")
    print("\n" + line)
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for r in rows:
        csvv = "n/a" if r["csv_recorded_rank0_pct"] is None else f"{r['csv_recorded_rank0_pct']:.2f}"
        cells_txt = [
            r["strategy"].ljust(w[0]),
            f"{r['custom_overlap_pct_mean']:.2f}".ljust(w[1]),
            f"{r['hta_overlap_pct_mean']:.2f}".ljust(w[2]),
            f"{r['hta_overlap_pct_min']:.1f}/{r['hta_overlap_pct_max']:.1f}".ljust(w[3]),
            f"{r['delta_pp']:+.2f}".ljust(w[4]),
            str(r["ranks_used"]).ljust(w[5]),
            csvv.ljust(w[6]),
        ]
        print("  ".join(cells_txt))

    deltas = [abs(r["delta_pp"]) for r in rows]
    print(f"\n  max |Δ| = {max(deltas):.2f} pp,  mean |Δ| = {sum(deltas)/len(deltas):.2f} pp"
          "   (Δ = custom − HTA, both on the same files)")
    print("  HTA buckets memcpy/memset as MEMORY (not compute); the custom method treats\n"
          "  every non-NCCL GPU kernel as compute — so small positive Δ is a classification\n"
          "  artifact. 'CSV(r0)' is the rank-0 value already in the master CSV, for reference.")

    # ---- CSV out ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
