# analysis/build_figures.py
"""Figure + table generation from the master CSV (design doc §13).

Produces all RQ1-RQ3 figures and reference tables. RQ4 (decision flowchart)
is a manual synthesis, not generated here — but we emit the recommendation
TABLE that seeds it.

Inputs:
  outputs/all_experiments_summary.csv   (primary source for ALL figures)

Outputs (default outputs/figures/):
  rq1_decomposition_ws{N}.png           attributed stacked bars
  rq1_residual_quality.png              residual % per config (methodology)
  rq2_throughput_ws{N}.png              throughput vs model size + crossover
  rq2_mechanism_ws{N}.png               overlap / exposed-comm / memory panels
  rq2_driver_waterfall_{model}_ws{N}.png  DDP->ZeRO-2->ZeRO-3 iter-time attribution
  rq2_collective_breakdown.csv          per-collective reference table
  rq3_gradaccum_throughput.png          throughput vs accum (4B, 8 GPU)
  rq3_crossover_vs_ga_ws{N}.png         does grad-accum shift the DDP->FSDP crossover
  rq3_gradaccum_decomposition.png       decomposition shift across accum (trace split)
  rq3_gradaccum_exposed_comm_by_ws.png  exposed comm + overlap vs ga, by world size
  rq3_gradaccum_exposed_comm_by_model.png  exposed comm + overlap vs ga, by model
  rq3_gradaccum_strategy_contrast.png   exposed comm + overlap vs ga, by strategy
  scaling_efficiency.png                efficiency vs GPU count
  recommendation_table.csv              seeds RQ4 decision procedure

Usage:
  python -m analysis.build_figures
  python -m analysis.build_figures --csv path/to.csv --out_dir path/to/figs
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Constants / styling
# --------------------------------------------------------------------------

MODEL_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
               "qwen3-14b", "qwen3-32b"]
MODEL_PARAMS_B = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7,
                  "qwen3-4b": 4.0, "qwen3-8b": 8.0,
                  "qwen3-14b": 14.8, "qwen3-32b": 32.8}
STRATEGY_ORDER = ["ddp", "fsdp_zero2", "fsdp_zero3"]
STRATEGY_LABEL = {"ddp": "DDP", "fsdp_zero2": "ZeRO-2", "fsdp_zero3": "ZeRO-3"}
STRATEGY_COLOR = {"ddp": "#1f77b4", "fsdp_zero2": "#ff7f0e", "fsdp_zero3": "#2ca02c"}

# Attributed decomposition components (design doc §9).
# RQ3 (grad-accum) uses the lumped backward + the idle-based exposed_comm, which
# is the REAL large exposed-comm value there (final all-reduce after no_sync).
DECOMP_COMPONENTS = [
    ("fwd_time_ms_mean", "Forward", "#4c72b0"),
    ("bwd_time_ms_mean", "Backward", "#dd8452"),
    ("opt_time_ms_mean", "Optimizer", "#55a868"),
    ("exposed_comm_time_ms", "Exposed Comm", "#c44e52"),
    ("straggler_wait_ms", "Straggler", "#8172b3"),
    ("residual_ms", "Residual", "#cccccc"),
]

# RQ1 uses the MEASURED backward split (bwd = bwd_compute + recompute +
# bwd_exposed_comm), so the comm component is the trace-measured exposed comm
# rather than the ~0 idle proxy. Sums to ~iter (idle proxy is negligible at ga=1).
DECOMP_COMPONENTS_RQ1 = [
    ("fwd_time_ms_mean", "Forward", "#4c72b0"),
    ("bwd_compute_ms_calibrated", "Backward compute", "#dd8452"),
    ("recompute_ms_calibrated", "Recompute (ckpt, cal.)", "#f0c8a0"),
    ("bwd_exposed_comm_ms", "Exposed comm", "#c44e52"),
    ("opt_time_ms_mean", "Optimizer", "#55a868"),
    ("straggler_wait_ms", "Straggler", "#8172b3"),
    ("residual_ms", "Residual", "#cccccc"),
]


def _savefig(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    # bbox_inches="tight" grows the canvas to include outside legends;
    # pad_inches adds a uniform breathing margin around everything.
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"  wrote {path}")


def _legend_right(ax, **kw):
    """Place a single-axes legend just outside the plot, on the right."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("framealpha", 0.9)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0, **kw)


def _fig_legend_right(fig, handles, labels, **kw):
    """Single shared legend to the right of a multi-panel figure."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("framealpha", 0.9)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               borderaxespad=0.0, **kw)


# --------------------------------------------------------------------------
# Data loading + prep
# --------------------------------------------------------------------------

def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Average over reps for headline figures (keep std where needed).
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric


# --------------------------------------------------------------------------
# Recompute calibration (shared by build_figures + augment script)
# --------------------------------------------------------------------------
# recompute_ms is the analytical estimate (= forward compute). The group4a
# GC on/off ablation shows it overestimates the true recompute, so we calibrate
# recompute_ms_calibrated = ratio * fwd, where ratio = mean((bwd_on - bwd_off)/
# fwd_on) over the group4a twin pairs. Both raw and calibrated are kept.

def recompute_ratio_from_rows(rows) -> float:
    """Derive the recompute calibration ratio from group4a (_gcoff) twin pairs."""
    by = {str(r.get("experiment_name")): r for r in rows}

    def num(r, k):
        try:
            return float(r.get(k) or "")
        except (TypeError, ValueError):
            return None

    ratios = []
    for r in rows:
        n = str(r.get("experiment_name", ""))
        if not n.endswith("_gcoff"):
            continue
        on = by.get(n[: -len("_gcoff")])
        if not on:
            continue
        fwd, b_on, b_off = num(on, "fwd_time_ms_mean"), num(on, "bwd_time_ms_mean"), num(r, "bwd_time_ms_mean")
        if fwd and b_on and b_off and fwd > 0 and (b_on - b_off) > 0:
            ratios.append((b_on - b_off) / fwd)
    return sum(ratios) / len(ratios) if ratios else 1.0


def calibrated_recompute(fwd, bwd, bexp, gc, ratio):
    """Return (recompute_ms_calibrated, bwd_compute_ms_calibrated)."""
    rc = ratio * fwd if gc else 0.0
    bc = max(0.0, bwd - rc - (bexp or 0.0))
    return rc, bc


def add_recompute_calibration(df: pd.DataFrame):
    """Add recompute_ms_calibrated + bwd_compute_ms_calibrated columns.

    No-op (uses existing values) if the CSV already carries them (e.g. persisted
    by scripts/augment_recompute_calibration). Must run BEFORE ablation rows are
    excluded, since the ratio is derived from the _gcoff pairs.
    """
    if ("recompute_ms_calibrated" in df.columns
            and pd.to_numeric(df["recompute_ms_calibrated"], errors="coerce").notna().any()):
        return df, None
    ratio = recompute_ratio_from_rows(df.to_dict("records"))
    df = df.copy()

    def row(r):
        fwd = pd.to_numeric(r.get("fwd_time_ms_mean"), errors="coerce")
        bwd = pd.to_numeric(r.get("bwd_time_ms_mean"), errors="coerce")
        bexp = pd.to_numeric(r.get("bwd_exposed_comm_ms"), errors="coerce")
        gc = str(r.get("grad_checkpoint")) == "True"
        if pd.isna(fwd) or pd.isna(bwd):
            return pd.Series([np.nan, np.nan])
        rc, bc = calibrated_recompute(
            float(fwd), float(bwd), None if pd.isna(bexp) else float(bexp), gc, ratio)
        return pd.Series([rc, bc])

    df[["recompute_ms_calibrated", "bwd_compute_ms_calibrated"]] = df.apply(row, axis=1)
    print(f"  recompute calibration ratio = {ratio:.3f} (from group4a GC on/off pairs)")
    return df, ratio


def rep_mean(df: pd.DataFrame, filt: Dict, value_cols: List[str]) -> pd.DataFrame:
    """Filter, then mean over reps grouped by (model, strategy, world_size)."""
    sub = df.copy()
    for k, v in filt.items():
        sub = sub[sub[k] == v]
    sub = sub[sub["status"] == "ok"]
    if sub.empty:
        return sub
    group_keys = ["model", "strategy", "world_size"]
    keep = group_keys + [c for c in value_cols if c in sub.columns]
    return sub[keep].groupby(group_keys, as_index=False).mean(numeric_only=True)


# --------------------------------------------------------------------------
# RQ1: attributed decomposition
# --------------------------------------------------------------------------

def fig_rq1_decomposition(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ1: iteration decomposition (measured backward split)")
    components = DECOMP_COMPONENTS_RQ1
    comp_cols = [c[0] for c in components]
    for ws in sorted(df["world_size"].unique()):
        if ws < 2:
            continue
        agg = rep_mean(df, {"world_size": ws}, comp_cols)
        if agg.empty:
            continue

        # x positions: grouped by model, sub-grouped by strategy
        models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
        strategies = [s for s in STRATEGY_ORDER if s in agg["strategy"].unique()]
        n_s = len(strategies)
        bar_w = 0.8 / n_s

        fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(models)), 6.5))
        for si, strat in enumerate(strategies):
            xs, bottoms = [], []
            for mi, model in enumerate(models):
                row = agg[(agg["model"] == model) & (agg["strategy"] == strat)]
                x = mi + (si - (n_s - 1) / 2) * bar_w
                xs.append(x)
                bottoms.append(0.0)
            bottoms = np.zeros(len(models))
            for col, label, color in components:
                vals = []
                for model in models:
                    row = agg[(agg["model"] == model) & (agg["strategy"] == strat)]
                    vals.append(float(row[col].iloc[0]) if (not row.empty and col in row) else 0.0)
                vals = np.array(vals)
                ax.bar(xs, vals, bar_w, bottom=bottoms, color=color,
                       edgecolor="white", linewidth=0.3,
                       label=label if si == 0 else None)
                bottoms += vals

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([f"{m}\n({STRATEGY_LABEL_JOIN(strategies)})"
                            for m in models], fontsize=8)
        ax.set_ylabel("Time per iteration (ms)")
        ax.set_title(f"RQ1 — Attributed Iteration Decomposition ({ws} GPUs)\n"
                     f"bars L→R per model: {', '.join(STRATEGY_LABEL[s] for s in strategies)}")
        _legend_right(ax, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _savefig(fig, out_dir, f"rq1_decomposition_ws{ws}.png")


def STRATEGY_LABEL_JOIN(strategies: List[str]) -> str:
    return "/".join(STRATEGY_LABEL[s] for s in strategies)


def fig_rq1_residual_quality(df: pd.DataFrame, out_dir: str,
                             gate_pct: float = 5.0) -> None:
    """Residual % per config — the methodology quality metric (design doc §10)."""
    print("RQ1: residual quality gate")
    ok = df[(df["status"] == "ok") & (df["world_size"] >= 2)]
    if ok.empty or "pct_residual" not in ok.columns:
        print("  (no pct_residual data; skipping)")
        return
    agg = ok.groupby(["model", "strategy", "world_size"],
                     as_index=False)["pct_residual"].mean()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels, vals, colors = [], [], []
    for _, r in agg.iterrows():
        labels.append(f"{r['model']}\n{STRATEGY_LABEL[r['strategy']]}·{int(r['world_size'])}g")
        vals.append(r["pct_residual"])
        colors.append(STRATEGY_COLOR[r["strategy"]])
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.axhline(gate_pct, color="red", linestyle="--",
               label=f"quality gate ({gate_pct}%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Unexplained residual (% of iter_time)")
    ax.set_title("RQ1 — Decomposition Quality: residual vs 5% gate")
    _legend_right(ax)
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir, "rq1_residual_quality.png")


def fig_rq1_recompute_tax(df: pd.DataFrame, out_dir: str) -> None:
    """Validate analytical recompute_ms against the GC on/off ablation (group4a).

    measured recompute tax = bwd_time(GC on) - bwd_time(GC off)
    should approximate recompute_ms (the analytical estimate ~= forward compute).
    """
    print("RQ1: recompute-tax ablation (GC on vs off)")
    off = df[df["experiment_name"].astype(str).str.contains("_gcoff", na=False)]
    if off.empty:
        print("  (no GC-off ablation rows; skipping)")
        return
    rows = []
    for _, r in off.iterrows():
        twin = df[df["experiment_name"] == str(r["experiment_name"]).replace("_gcoff", "")]
        if twin.empty:
            continue
        on = twin.iloc[0]
        rows.append({
            "model": r["model"],
            "measured_tax": float(on["bwd_time_ms_mean"]) - float(r["bwd_time_ms_mean"]),
            "analytical_recompute": float(on.get("recompute_ms", 0.0) or 0.0),
        })
    if not rows:
        print("  (no matched GC on/off pairs; skipping)")
        return
    t = pd.DataFrame(rows)
    x = np.arange(len(t)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, t["measured_tax"], w, color="#dd8452",
           label="measured tax (bwd_on − bwd_off)")
    ax.bar(x + w / 2, t["analytical_recompute"], w, color="#4c72b0",
           label="analytical recompute_ms (≈ forward)")
    ax.set_xticks(x); ax.set_xticklabels(t["model"], fontsize=9)
    ax.set_ylabel("Recompute time (ms)")
    ax.set_title("RQ1 — Recompute tax: analytical estimate vs GC on/off ablation")
    _legend_right(ax, fontsize=8); ax.grid(axis="y", alpha=0.3)
    _savefig(fig, out_dir, "rq1_recompute_tax.png")


# --------------------------------------------------------------------------
# RQ2: throughput crossover + mechanism
# --------------------------------------------------------------------------

def fig_rq2_throughput(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ2: throughput vs model size + crossover")
    for ws in sorted(df["world_size"].unique()):
        if ws < 2:
            continue
        agg = rep_mean(df, {"world_size": ws},
                       ["throughput_tokens_per_sec_mean"])
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 6))
        for strat in STRATEGY_ORDER:
            s = agg[agg["strategy"] == strat].copy()
            if s.empty:
                continue
            s["px"] = s["model"].map(MODEL_PARAMS_B)
            s = s.sort_values("px")
            ax.plot(s["px"], s["throughput_tokens_per_sec_mean"],
                    marker="o", color=STRATEGY_COLOR[strat],
                    label=STRATEGY_LABEL[strat])
        _annotate_crossovers(ax, agg)
        ax.set_xlabel("Model size (B params)")
        ax.set_ylabel("Throughput (tokens/sec, global)")
        ax.set_title(f"RQ2 — Throughput vs Model Size ({ws} GPUs)")
        ax.set_xscale("log")
        ax.set_xticks(list(MODEL_PARAMS_B.values()))
        ax.set_xticklabels([str(v) for v in MODEL_PARAMS_B.values()])
        _legend_right(ax)
        ax.grid(alpha=0.3)
        _savefig(fig, out_dir, f"rq2_throughput_ws{ws}.png")


def _annotate_crossovers(ax, agg: pd.DataFrame) -> None:
    """Mark model sizes where the best-FSDP overtakes DDP."""
    pivot = agg.pivot_table(index="model", columns="strategy",
                            values="throughput_tokens_per_sec_mean")
    for model in pivot.index:
        if "ddp" not in pivot.columns:
            continue
        ddp = pivot.loc[model, "ddp"]
        best_fsdp = np.nanmax([pivot.loc[model].get(s, np.nan)
                               for s in ("fsdp_zero2", "fsdp_zero3")])
        if np.isnan(ddp) or np.isnan(best_fsdp):
            continue
        if best_fsdp > ddp:
            px = MODEL_PARAMS_B.get(model)
            ax.axvline(px, color="gray", linestyle=":", alpha=0.5)
            ax.text(px, ax.get_ylim()[1] * 0.95, "crossover",
                    rotation=90, fontsize=7, va="top", color="gray")


def _set_model_size_xticks(axes) -> None:
    """Replace the log-axis 10^0/10^1 ticks with actual model sizes (B)."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [MODEL_PARAMS_B[m] for m in MODEL_ORDER]           # 0.6 .. 32.8
    labels = [("%g" % round(t)) if t >= 4 else ("%g" % t) for t in ticks]  # 0.6,1.7,4,8,15,33
    for ax in axes:
        ax.set_xlim(0.5, 40)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Model size (B params)")


def fig_rq2_mechanism(df: pd.DataFrame, out_dir: str) -> None:
    """Three-panel mechanism analysis: overlap, exposed comm, memory pressure."""
    print("RQ2: mechanism panels")
    for ws in sorted(df["world_size"].unique()):
        if ws < 2:
            continue
        cols = ["overlap_percentage_trace", "bwd_exposed_comm_ms",
                "iter_time_ms_mean", "mean_collective_size_bytes",
                "nccl_collective_count", "bandwidth_efficiency",
                "headroom_to_80gb_gb", "num_alloc_retries"]
        agg = rep_mean(df, {"world_size": ws}, cols)
        if agg.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
        _panel_overlap(axes[0], agg)
        _panel_exposed_comm(axes[1], agg)
        _panel_memory(axes[2], agg)
        # Label the log x-axis with actual model sizes (B) instead of 10^0/10^1.
        _set_model_size_xticks(axes)
        # Single shared legend to the right of all three panels.
        import matplotlib.lines as mlines
        handles = [mlines.Line2D([], [], color=STRATEGY_COLOR[s], marker="o",
                                 label=STRATEGY_LABEL[s]) for s in STRATEGY_ORDER]
        handles += [
            mlines.Line2D([], [], color="gray", marker="x", linestyle="--",
                          label="mean collective size (b)"),
            mlines.Line2D([], [], color="red", linestyle="--", label="80GB wall (c)"),
        ]
        _fig_legend_right(fig, handles, [h.get_label() for h in handles], fontsize=9)
        fig.suptitle(f"RQ2 — Crossover Mechanisms ({ws} GPUs): "
                     f"overlap · exposed comm/granularity · memory pressure",
                     fontsize=12)
        # Extra padding so twin-axis labels / titles don't overlap.
        fig.subplots_adjust(left=0.05, right=0.90, bottom=0.14, top=0.86, wspace=0.45)
        _savefig(fig, out_dir, f"rq2_mechanism_ws{ws}.png")


def _panel_overlap(ax, agg: pd.DataFrame) -> None:
    for strat in STRATEGY_ORDER:
        s = agg[agg["strategy"] == strat].copy()
        if s.empty or "overlap_percentage_trace" not in s.columns:
            continue
        s["px"] = s["model"].map(MODEL_PARAMS_B)
        s = s.sort_values("px")
        ax.plot(s["px"], s["overlap_percentage_trace"], marker="o",
                color=STRATEGY_COLOR[strat], label=STRATEGY_LABEL[strat])
    ax.set_xlabel("Model size (B)")
    ax.set_ylabel("Comm overlap (%)")
    ax.set_title("(a) Overlap efficiency")
    ax.set_xscale("log"); ax.grid(alpha=0.3)


def _panel_exposed_comm(ax, agg: pd.DataFrame) -> None:
    ax2 = ax.twinx()
    for strat in STRATEGY_ORDER:
        s = agg[agg["strategy"] == strat].copy()
        if s.empty:
            continue
        s["px"] = s["model"].map(MODEL_PARAMS_B)
        s = s.sort_values("px")
        # Exposed comm as % of iteration = the throughput penalty. Plotted
        # instead of absolute ms: ms has a confusing U-shape (small models are
        # comm-latency-bound, large models grow comm volume); the %-of-iter
        # collapses that into a monotone comm-bound->compute-bound trend.
        if {"bwd_exposed_comm_ms", "iter_time_ms_mean"} <= set(s.columns):
            pct = 100.0 * s["bwd_exposed_comm_ms"] / s["iter_time_ms_mean"]
            ax.plot(s["px"], pct, marker="o",
                    color=STRATEGY_COLOR[strat], label=STRATEGY_LABEL[strat])
        if "mean_collective_size_bytes" in s.columns:
            ax2.plot(s["px"], s["mean_collective_size_bytes"] / 1e6,
                     marker="x", linestyle="--", alpha=0.5,
                     color=STRATEGY_COLOR[strat])
    ax.set_xlabel("Model size (B)")
    ax.set_ylabel("Exposed comm (% of iteration, solid)")
    ax2.set_ylabel("Mean collective size (MB, dashed)")
    ax.set_title("(b) Exposed comm + granularity")
    ax.set_xscale("log"); ax.grid(alpha=0.3)


def _panel_memory(ax, agg: pd.DataFrame) -> None:
    """Headroom to 80GB — the operative, strategy-agnostic pressure metric.

    NOTE: we deliberately plot HEADROOM (distance to the 80GB wall), NOT raw
    peak memory, because DDP (full_replica) and FSDP (sharded) peak numbers
    are not directly comparable (design doc §9/§13).
    """
    for strat in STRATEGY_ORDER:
        s = agg[agg["strategy"] == strat].copy()
        if s.empty or "headroom_to_80gb_gb" not in s.columns:
            continue
        s["px"] = s["model"].map(MODEL_PARAMS_B)
        s = s.sort_values("px")
        ax.plot(s["px"], s["headroom_to_80gb_gb"], marker="o",
                color=STRATEGY_COLOR[strat], label=STRATEGY_LABEL[strat])
    ax.axhline(0, color="red", linestyle="--", alpha=0.6, label="80GB wall")
    ax.set_xlabel("Model size (B)")
    ax.set_ylabel("Headroom to 80GB (GB)")
    ax.set_title("(c) Memory pressure (headroom)")
    ax.set_xscale("log"); ax.grid(alpha=0.3)


def table_rq2_collective_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    """Per-collective reference table for Chapter 4."""
    print("RQ2: collective breakdown table")
    cols = ["nccl_allreduce_time_ms", "nccl_allgather_time_ms",
            "nccl_reducescatter_time_ms", "total_nccl_comm_time_ms",
            "nccl_collective_count", "mean_collective_size_bytes",
            "achieved_bw_gbps", "bandwidth_efficiency",
            "theoretical_comm_volume_gb", "measured_comm_volume_gb"]
    ok = df[df["status"] == "ok"]
    have = [c for c in cols if c in ok.columns]
    if not have:
        print("  (no comm data; skipping)")
        return
    tbl = ok.groupby(["model", "strategy", "world_size"],
                     as_index=False)[have].mean()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rq2_collective_breakdown.csv")
    tbl.to_csv(path, index=False)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# RQ3: gradient accumulation
# --------------------------------------------------------------------------

def fig_rq3_gradaccum(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ3: gradient accumulation (Qwen3-4B, 8 GPUs)")
    sub = df[(df["model"] == "qwen3-4b") & (df["world_size"] == 8)
             & (df["status"] == "ok")]
    if sub.empty:
        print("  (no 4B/8GPU data; skipping)")
        return

    agg = sub.groupby(["strategy", "gradient_accumulation_steps"],
                      as_index=False).mean(numeric_only=True)

    # throughput vs accum
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for strat in STRATEGY_ORDER:
        s = agg[agg["strategy"] == strat].sort_values("gradient_accumulation_steps")
        if s.empty:
            continue
        ax.plot(s["gradient_accumulation_steps"],
                s["throughput_tokens_per_sec_mean"],
                marker="o", color=STRATEGY_COLOR[strat],
                label=STRATEGY_LABEL[strat])
    ax.set_xlabel("Gradient accumulation steps")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("RQ3 — Throughput vs Grad Accumulation (Qwen3-4B, 8 GPUs)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    _legend_right(ax); ax.grid(alpha=0.3)
    _savefig(fig, out_dir, "rq3_gradaccum_throughput.png")

    # decomposition shift (stacked bars per strategy across accum).
    # Trace-based split (DECOMP_COMPONENTS_RQ1): at ga>1 the deferred collective
    # folds into backward and the idle-based exposed_comm proxy is ~0, so the
    # RQ3 stack must use the measured bwd_exposed_comm (methodology 3.4.2/3.9.1).
    comp_cols = [c[0] for c in DECOMP_COMPONENTS_RQ1]
    accums = sorted(agg["gradient_accumulation_steps"].unique())
    fig, axes = plt.subplots(1, len(STRATEGY_ORDER),
                             figsize=(5.5 * len(STRATEGY_ORDER), 5.5), sharey=True)
    if len(STRATEGY_ORDER) == 1:
        axes = [axes]
    for ax, strat in zip(axes, STRATEGY_ORDER):
        s = agg[agg["strategy"] == strat]
        bottoms = np.zeros(len(accums))
        for col, label, color in DECOMP_COMPONENTS_RQ1:
            vals = [float(s[s["gradient_accumulation_steps"] == a][col].iloc[0])
                    if not s[s["gradient_accumulation_steps"] == a].empty else 0.0
                    for a in accums]
            vals = np.array(vals)
            ax.bar(range(len(accums)), vals, bottom=bottoms, color=color,
                   label=label)
            bottoms += vals
        ax.set_title(STRATEGY_LABEL[strat])
        ax.set_xticks(range(len(accums)))
        ax.set_xticklabels([str(int(a)) for a in accums])
        ax.set_xlabel("Accum steps")
    axes[0].set_ylabel("Time per optimizer step (ms)")
    import matplotlib.patches as mpatches
    dh = [mpatches.Patch(color=c, label=lbl) for _, lbl, c in DECOMP_COMPONENTS_RQ1]
    _fig_legend_right(fig, dh, [h.get_label() for h in dh], fontsize=8)
    fig.suptitle("RQ3 — Decomposition Shift Across Accumulation Depth "
                 "(trace-based split)")
    _savefig(fig, out_dir, "rq3_gradaccum_decomposition.png")


# --------------------------------------------------------------------------
# RQ3 (extended): communication mechanism across the accumulation sweep.
# Uses ONLY profiled cells (Pass-2 trace metrics) and the trace-based exposed
# comm — the idle-based proxy is ~0 at ga>1 (methodology 3.4.2). Achieved
# bandwidth is deliberately NOT plotted vs ga (analytical volume does not scale
# per micro-batch; see limitations 3.10).
# --------------------------------------------------------------------------

GA_TICKS = [1, 2, 4, 8]
GA_COLOR = {1: "#4c72b0", 2: "#dd8452", 4: "#55a868", 8: "#c44e52"}
WS_COLOR = {2: "#4c72b0", 4: "#dd8452", 8: "#55a868"}
MODEL_COLOR = {"qwen3-4b": "#4c72b0", "qwen3-8b": "#dd8452", "qwen3-14b": "#55a868"}
MODEL_LABEL = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
               "qwen3-14b": "Qwen3-14B"}


def _profiled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry Pass-2 communication data (trace metrics populated)."""
    t = pd.to_numeric(df.get("total_nccl_comm_time_ms"), errors="coerce")
    return df[t.fillna(0) > 0]


def _exposed_overlap_vs_ga(df, fixed, vary_col, vary_values, color_map,
                           label_map, title, out_name, out_dir):
    """Twin-axis line plot vs gradient accumulation: trace-based exposed comm
    (solid, left axis) and comm-compute overlap % (dashed, right axis), one
    colour per `vary_col` value. `df` should be pre-filtered to profiled rows."""
    sub = df.copy()
    for k, v in fixed.items():
        sub = sub[sub[k] == v]
    sub = sub[sub["status"] == "ok"]
    for c in ("bwd_exposed_comm_ms", "overlap_percentage_trace",
              "gradient_accumulation_steps"):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if sub.empty:
        print(f"  ({out_name}: no profiled data; skipping)")
        return

    fig, axL = plt.subplots(figsize=(9.5, 6))
    axR = axL.twinx()
    plotted = 0
    for val in vary_values:
        s = sub[sub[vary_col] == val].sort_values("gradient_accumulation_steps")
        if s.empty:
            continue
        color = color_map.get(val, "#333333")
        axL.plot(s["gradient_accumulation_steps"], s["bwd_exposed_comm_ms"],
                 marker="o", color=color, label=label_map.get(val, str(val)))
        axR.plot(s["gradient_accumulation_steps"], s["overlap_percentage_trace"],
                 marker="s", ls="--", color=color, alpha=0.55)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print(f"  ({out_name}: no series to plot; skipping)")
        return
    axL.set_xlabel("Gradient accumulation steps")
    axL.set_ylabel("Exposed comm — bwd_exposed_comm (ms/iter, solid)")
    axR.set_ylabel("Comm–compute overlap (%, dashed)")
    axR.set_ylim(0, 100)
    axL.set_xscale("log", base=2)
    axL.set_xticks(GA_TICKS)
    axL.set_xticklabels([str(t) for t in GA_TICKS])
    axL.grid(alpha=0.3)
    axL.set_title(title)
    _legend_right(axL)
    _savefig(fig, out_dir, out_name)


def fig_rq3_exposed_comm_by_ws(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ3: exposed comm + overlap vs ga, by world size (4B, ZeRO-3)")
    _exposed_overlap_vs_ga(
        _profiled(df),
        fixed={"model": "qwen3-4b", "strategy": "fsdp_zero3"},
        vary_col="world_size", vary_values=[2, 4, 8], color_map=WS_COLOR,
        label_map={2: "2 GPUs", 4: "4 GPUs", 8: "8 GPUs"},
        title="RQ3 — Exposed comm & overlap vs accumulation, by world size\n"
              "(Qwen3-4B, ZeRO-3)",
        out_name="rq3_gradaccum_exposed_comm_by_ws.png", out_dir=out_dir)


def fig_rq3_exposed_comm_by_model(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ3: exposed comm + overlap vs ga, by model (ZeRO-3, 8 GPUs)")
    _exposed_overlap_vs_ga(
        _profiled(df),
        fixed={"strategy": "fsdp_zero3", "world_size": 8},
        vary_col="model", vary_values=["qwen3-4b", "qwen3-8b", "qwen3-14b"],
        color_map=MODEL_COLOR, label_map=MODEL_LABEL,
        title="RQ3 — Exposed comm & overlap vs accumulation, by model size\n"
              "(ZeRO-3, 8 GPUs)",
        out_name="rq3_gradaccum_exposed_comm_by_model.png", out_dir=out_dir)


def fig_rq3_strategy_contrast(df: pd.DataFrame, out_dir: str) -> None:
    print("RQ3: exposed comm + overlap vs ga, by strategy (4B, 8 GPUs)")
    _exposed_overlap_vs_ga(
        _profiled(df),
        fixed={"model": "qwen3-4b", "world_size": 8},
        vary_col="strategy", vary_values=STRATEGY_ORDER,
        color_map=STRATEGY_COLOR, label_map=STRATEGY_LABEL,
        title="RQ3 — Exposed comm & overlap vs accumulation, by strategy\n"
              "(Qwen3-4B, 8 GPUs)",
        out_name="rq3_gradaccum_strategy_contrast.png", out_dir=out_dir)


def _infer_n_prof(df: pd.DataFrame) -> int:
    """Profiler active-window length (constant): recovered from ga=1 profiled
    rows where measured_comm_volume_gb == theoretical_comm_volume_gb x n_prof."""
    ga = pd.to_numeric(df["gradient_accumulation_steps"], errors="coerce")
    theo = pd.to_numeric(df["theoretical_comm_volume_gb"], errors="coerce")
    meas = pd.to_numeric(df["measured_comm_volume_gb"], errors="coerce")
    r = (meas / theo)[(ga == 1) & (theo > 0)].round()
    return int(r.mode().iloc[0]) if not r.empty else 1


def fig_rq3_comm_bw(df: pd.DataFrame, out_dir: str) -> None:
    """Per-token communication and achieved bus bandwidth vs accumulation,
    strategy contrast at 4B/8-GPU. Requires the ga-aware volume fix (step 6);
    achieved_bw is meaningless at ga>1 without it (see metrics._bus_volume_gb).

    DEFERRED (notes/08 §4b): the analytical bus-volume and the measured
    collective time do not track across ga (ZeRO-3 AG scales ga x while RS is
    flat; DDP/ZeRO-2 comm-time grows at flat count), so achieved_bw at ga>1 is
    NOT trustworthy. A faithful fix needs measured bytes from record_param_comms
    (requires kept traces). This figure is intentionally NOT emitted; report
    ga>1 communication via measured time / exposed-comm / overlap instead."""
    print("RQ3: per-token comm / achieved BW vs ga — SKIPPED (deferred, notes/08 §4b)")
    return
    sub = _profiled(df)
    sub = sub[(sub["model"] == "qwen3-4b") & (sub["world_size"] == 8)
              & (sub["status"] == "ok")].copy()
    if sub.empty:
        print("  (rq3_gradaccum_comm_bw: no profiled data; skipping)")
        return
    n_prof = _infer_n_prof(_profiled(df))
    for c in ("gradient_accumulation_steps", "measured_comm_volume_gb",
              "achieved_bw_gbps", "per_device_batch_size", "world_size",
              "seq_length"):
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    # bytes communicated per token = per-step bus volume / tokens per step
    toks_step = (sub["per_device_batch_size"] * sub["world_size"]
                 * sub["gradient_accumulation_steps"] * sub["seq_length"])
    sub["per_token_comm_kb"] = (sub["measured_comm_volume_gb"] * 1e6
                                / (toks_step * n_prof))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))
    for strat in STRATEGY_ORDER:
        s = sub[sub["strategy"] == strat].sort_values("gradient_accumulation_steps")
        if s.empty:
            continue
        c = STRATEGY_COLOR[strat]
        axA.plot(s["gradient_accumulation_steps"], s["per_token_comm_kb"],
                 marker="o", color=c, label=STRATEGY_LABEL[strat])
        axB.plot(s["gradient_accumulation_steps"], s["achieved_bw_gbps"],
                 marker="o", color=c, label=STRATEGY_LABEL[strat])
    for ax in (axA, axB):
        ax.set_xlabel("Gradient accumulation steps")
        ax.set_xscale("log", base=2)
        ax.set_xticks(GA_TICKS); ax.set_xticklabels([str(t) for t in GA_TICKS])
        ax.grid(alpha=0.3)
    axA.set_ylabel("Communication per token (KB/token)")
    axA.set_title("(a) Per-token comm cost — amortization")
    axB.set_ylabel("Achieved bus bandwidth (GB/s)")
    axB.set_title("(b) Achieved bandwidth")
    axA.legend(fontsize=8, framealpha=0.9)
    fig.suptitle("RQ3 — Communication cost & bandwidth vs accumulation "
                 "(Qwen3-4B, 8 GPUs)")
    _savefig(fig, out_dir, "rq3_gradaccum_comm_bw.png")


# --------------------------------------------------------------------------
# RQ3 (crossover shift): does gradient accumulation move the DDP->FSDP
# crossover, and in which direction?
# --------------------------------------------------------------------------

def fig_rq3_crossover_shift(df: pd.DataFrame, out_dir: str) -> None:
    """Does gradient accumulation shift the DDP->FSDP crossover, and which way?

    Two complementary views in one figure, per GPU count:
      (top)  throughput vs model size, one small-multiple per ga, with the
             DDP->best-FSDP crossover marked (reuses _annotate_crossovers).
      (bottom) best-FSDP / DDP throughput RATIO vs model size, one line per ga.
             Ratio > 1 => FSDP wins; the model size where each ga-line crosses
             1.0 is the crossover. A ratio that sinks toward/below 1.0 as ga
             grows => GA pushes the crossover to LARGER models (DDP's no_sync
             defers its gradient collective and amortizes it over micro-steps,
             while ZeRO-3's per-micro-batch all-gather cannot be deferred).

    Near-1.0 margins at the crossover model sit within the +/-CV band (design
    doc 6.3, ~2-3%); the evidence is the monotone TREND across ga, not any
    single cell. Uses the FULL grad-accum sweep (not df_core).
    """
    print("RQ3: does grad-accum shift the DDP->FSDP crossover?")
    gas = [1, 2, 4, 8]
    metric = "throughput_tokens_per_sec_mean"
    for ws in sorted({int(w) for w in df["world_size"].unique() if int(w) >= 2}):
        aggs = {}
        for ga in gas:
            a = rep_mean(df[df["gradient_accumulation_steps"] == ga],
                         {"world_size": ws}, [metric])
            if not a.empty:
                aggs[ga] = a
        if not aggs:
            continue

        fig = plt.figure(figsize=(17, 9))
        gs = fig.add_gridspec(2, len(gas), height_ratios=[1.0, 1.15],
                              hspace=0.42, wspace=0.28)

        # ---- top row: throughput vs model size, one panel per ga ----------
        top_axes = []
        for j, ga in enumerate(gas):
            ax = fig.add_subplot(gs[0, j])
            top_axes.append(ax)
            a = aggs.get(ga)
            if a is None:
                ax.set_title(f"ga = {ga} (no data)")
                ax.set_xscale("log")
                continue
            for strat in STRATEGY_ORDER:
                s = a[a["strategy"] == strat].copy()
                if s.empty:
                    continue
                s["px"] = s["model"].map(MODEL_PARAMS_B)
                s = s.sort_values("px")
                ax.plot(s["px"], s[metric], marker="o", markersize=4,
                        color=STRATEGY_COLOR[strat], label=STRATEGY_LABEL[strat])
            _annotate_crossovers(ax, a)
            ax.set_title(f"ga = {ga}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.set_ylabel("Throughput (tok/s, log)")
        _set_model_size_xticks(top_axes)
        top_axes[0].legend(fontsize=7, framealpha=0.9, loc="lower left")

        # ---- bottom row: best-FSDP / DDP throughput ratio vs model size ---
        axr = fig.add_subplot(gs[1, :])
        plotted = False
        for ga in gas:
            a = aggs.get(ga)
            if a is None:
                continue
            piv = a.pivot_table(index="model", columns="strategy", values=metric)
            if "ddp" not in piv.columns:
                continue
            rows = []
            for model in piv.index:
                ddp = piv.loc[model].get("ddp", np.nan)
                best = np.nanmax([piv.loc[model].get(s, np.nan)
                                  for s in ("fsdp_zero2", "fsdp_zero3")])
                px = MODEL_PARAMS_B.get(model)
                if np.isnan(ddp) or np.isnan(best) or px is None:
                    continue
                rows.append((px, best / ddp))
            rows = sorted(rows)
            if len(rows) < 2:
                continue
            axr.plot([r[0] for r in rows], [r[1] for r in rows], marker="o",
                     color=GA_COLOR[ga], label=f"ga = {ga}")
            plotted = True
        axr.axhline(1.0, color="black", linestyle="--", alpha=0.7)
        axr.axhspan(0.97, 1.03, color="gray", alpha=0.12)
        axr.text(0.995, 1.031, "+/-3% CV band", transform=axr.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color="gray")
        axr.set_xscale("log")
        _set_model_size_xticks([axr])
        axr.set_ylabel("Best-FSDP / DDP throughput\n(> 1 = FSDP wins)")
        axr.grid(alpha=0.3, which="both")
        if plotted:
            axr.legend(fontsize=8, framealpha=0.9, title="grad-accum")
        axr.set_title("Crossover shift: FSDP/DDP throughput ratio vs model size, "
                      "per grad-accum depth\n(ratio sinking below 1 as ga grows "
                      "=> GA pushes the DDP->FSDP crossover to larger models)")

        fig.suptitle(f"RQ3 - Does gradient accumulation shift the DDP->FSDP "
                     f"crossover?  ({ws} GPUs)", fontsize=13)
        _savefig(fig, out_dir, f"rq3_crossover_vs_ga_ws{ws}.png")


# --------------------------------------------------------------------------
# RQ2 (driver): what mechanism pushes the win DDP -> ZeRO-2 -> ZeRO-3?
# --------------------------------------------------------------------------

def fig_rq2_driver_waterfall(df: pd.DataFrame, out_dir: str,
                             model: str = "qwen3-4b") -> None:
    """What DRIVES the crossover as you move DDP -> ZeRO-2 -> ZeRO-3?

    Two iteration-time attribution waterfalls (per GPU count) at the crossover
    model. Each decomposes the change in iter time between adjacent strategies
    into the attributed components (RQ1 trace split): green bar = that component
    gets FASTER across the transition, red = slower; the net verdict is the
    throughput delta in the panel title. The exposed-comm bar is annotated with
    the collective-count x message-granularity mechanism (total collectives,
    mean message size, bus-BW efficiency) so the 'collective count x granularity'
    driver is explicit. Lower iter time = higher throughput; the y-axis is
    zoomed to the transition region (start/end bars are the full iter times).
    """
    print(f"RQ2: strategy-transition driver waterfall ({model})")
    comp = [(c, lbl) for c, lbl, _ in DECOMP_COMPONENTS_RQ1 if c != "residual_ms"]
    comm_cols = ["nccl_allreduce_count", "nccl_allgather_count",
                 "nccl_reducescatter_count", "mean_collective_size_bytes",
                 "bandwidth_efficiency"]
    value_cols = ([c for c, _ in comp]
                  + ["iter_time_ms_mean", "throughput_tokens_per_sec_mean"]
                  + comm_cols)
    transitions = [("ddp", "fsdp_zero2"), ("fsdp_zero2", "fsdp_zero3")]

    for ws in sorted({int(w) for w in df["world_size"].unique() if int(w) >= 2}):
        agg = rep_mean(df[df["model"] == model], {"world_size": ws}, value_cols)
        if agg.empty:
            continue
        agg = agg.set_index("strategy")
        if not all(s in agg.index for s in STRATEGY_ORDER):
            print(f"  (ws{ws}: {model} missing a strategy; skipping)")
            continue

        # NOT sharey: each transition is zoomed to its own delta region so the
        # component bars are legible (the two transitions differ ~50 vs ~40 ms).
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
        for ax, (a, b) in zip(axes, transitions):
            _draw_transition_waterfall(ax, agg, a, b, comp)
            ax.set_ylabel("Iteration time (ms)  -  lower = faster (y zoomed)")
        pretty = model.replace("qwen3-", "Qwen3-").upper()
        fig.suptitle(
            f"RQ2 - What drives the strategy crossover?  {pretty} / {ws} GPUs\n"
            "iteration-time attribution per transition (green = speedup, "
            "red = slowdown); comm bar annotated with collective count x granularity",
            fontsize=12)
        fig.subplots_adjust(top=0.80, wspace=0.10, bottom=0.22)
        _savefig(fig, out_dir, f"rq2_driver_waterfall_{model}_ws{ws}.png")


def _draw_transition_waterfall(ax, agg, a, b, comp) -> None:
    """One A->B iteration-time waterfall on `ax`. comp = [(col, label), ...]."""
    ta = float(agg.loc[a, "iter_time_ms_mean"])
    tb = float(agg.loc[b, "iter_time_ms_mean"])
    deltas, labels = [], []
    for col, lbl in comp:
        va = float(agg.loc[a, col]) if col in agg.columns else 0.0
        vb = float(agg.loc[b, col]) if col in agg.columns else 0.0
        deltas.append(vb - va); labels.append(lbl)
    # closure term so the bars land exactly on tb (residual_ms is ~0 in data)
    deltas.append((tb - ta) - sum(deltas)); labels.append("Residual/other")

    # running trajectory -> zoomed y-limits
    traj, r = [ta], ta
    for d in deltas:
        r += d; traj.append(r)
    lo, hi = min(traj + [ta, tb]), max(traj + [ta, tb])
    floor = lo - max(6.0, (hi - lo) * 0.15)
    # extra top headroom so the mechanism annotation clears the tallest bar
    top = hi + max(24.0, (hi - lo) * 0.55)
    ax.set_ylim(floor, top)

    n = len(deltas)
    # start / end anchor bars (full iter time, drawn from the zoomed floor)
    ax.bar(0, ta - floor, bottom=floor, color=STRATEGY_COLOR[a], width=0.7)
    ax.text(0, ta, f"{ta:.0f}", ha="center", va="bottom", fontsize=8)
    running = ta
    for i, (d, lbl) in enumerate(zip(deltas, labels), start=1):
        prev, nxt = running, running + d
        color = "#2ca02c" if d < 0 else "#c44e52"   # green speedup / red slowdown
        ax.bar(i, abs(d), bottom=min(prev, nxt), color=color, width=0.7,
               edgecolor="white", linewidth=0.4)
        ax.text(i, max(prev, nxt), f"{d:+.0f}", ha="center", va="bottom",
                fontsize=7)
        running = nxt
    ax.bar(n + 1, tb - floor, bottom=floor, color=STRATEGY_COLOR[b], width=0.7)
    ax.text(n + 1, tb, f"{tb:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(n + 2))
    ax.set_xticklabels([STRATEGY_LABEL.get(a, a)] + labels
                       + [STRATEGY_LABEL.get(b, b)],
                       rotation=40, ha="right", fontsize=7)

    # net throughput verdict in the title
    thra = float(agg.loc[a, "throughput_tokens_per_sec_mean"])
    thrb = float(agg.loc[b, "throughput_tokens_per_sec_mean"])
    pct = 100.0 * (thrb - thra) / thra
    verdict = "wins" if pct > 0 else "loses"
    ax.set_title(f"{STRATEGY_LABEL.get(a, a)} -> {STRATEGY_LABEL.get(b, b)}:  "
                 f"{STRATEGY_LABEL.get(b, b)} {verdict}  ({pct:+.1f}% throughput)",
                 fontsize=10)

    # annotate the exposed-comm bar with the count x granularity mechanism
    def _cc(strat):
        return sum(float(agg.loc[strat, c]) for c in
                   ("nccl_allreduce_count", "nccl_allgather_count",
                    "nccl_reducescatter_count") if c in agg.columns)
    labs = [l for _, l in comp]
    if "Exposed comm" in labs and "mean_collective_size_bytes" in agg.columns:
        na, nb = _cc(a), _cc(b)
        sza = float(agg.loc[a, "mean_collective_size_bytes"]) / 1e6
        szb = float(agg.loc[b, "mean_collective_size_bytes"]) / 1e6
        ea = float(agg.loc[a, "bandwidth_efficiency"])
        eb = float(agg.loc[b, "bandwidth_efficiency"])
        if np.isfinite(sza) and np.isfinite(szb):
            note = (f"comm mechanism (count x granularity):\n"
                    f"collectives {na:.0f} -> {nb:.0f}\n"
                    f"mean msg {sza:.0f} -> {szb:.0f} MB\n"
                    f"bus-eff {ea:.0f} -> {eb:.0f}%")
            ax.annotate(note, xy=(0.5, 0.985), xycoords="axes fraction",
                        fontsize=7, va="top", ha="center", color="#333333",
                        bbox=dict(boxstyle="round", fc="#f4f7ff", ec="#4c72b0",
                                  alpha=0.95))
    ax.axhline(ta, color="gray", lw=0.5, alpha=0.4)
    ax.grid(alpha=0.3, axis="y")


# --------------------------------------------------------------------------
# Scaling efficiency (uses Group 2 baselines)
# --------------------------------------------------------------------------

def fig_scaling_efficiency(df: pd.DataFrame, out_dir: str) -> None:
    print("Scaling efficiency vs GPU count")
    ok = df[(df["status"] == "ok") & (df["world_size"] >= 2)]
    if ok.empty or "scaling_efficiency" not in ok.columns:
        print("  (no scaling_efficiency; skipping)")
        return
    agg = ok.groupby(["model", "strategy", "world_size"],
                     as_index=False)["scaling_efficiency"].mean()

    fig, axes = plt.subplots(1, len(STRATEGY_ORDER),
                             figsize=(5.5 * len(STRATEGY_ORDER), 5.5), sharey=True)
    if len(STRATEGY_ORDER) == 1:
        axes = [axes]
    for ax, strat in zip(axes, STRATEGY_ORDER):
        s = agg[agg["strategy"] == strat]
        for model in MODEL_ORDER:
            ms = s[s["model"] == model].sort_values("world_size")
            if ms.empty:
                continue
            ax.plot(ms["world_size"], ms["scaling_efficiency"],
                    marker="o", label=model)
        ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(STRATEGY_LABEL[strat])
        ax.set_xlabel("GPUs"); ax.set_xticks([2, 4, 8])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Scaling efficiency (%)")
    seen = {}
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    _fig_legend_right(fig, list(seen.values()), list(seen.keys()), fontsize=8)
    fig.suptitle("Scaling Efficiency vs GPU Count (baseline: single-GPU)")
    _savefig(fig, out_dir, "scaling_efficiency.png")


# Models that cannot have a single-GPU (ws=1) baseline: the full replica does
# not fit on one 80 GB A100 (DDP ws2 — identical per-GPU footprint — already
# OOMs). Their scaling is expressed RELATIVE to the smallest feasible GPU count.
LARGE_NO_BASELINE_MODELS = ["qwen3-14b", "qwen3-32b"]


def fig_scaling_efficiency_large(df: pd.DataFrame, out_dir: str) -> None:
    """Relative scaling efficiency for models with no single-GPU baseline.

    For each (model, strategy) the baseline is the smallest world_size that ran
    OK (w0), and efficiency at world_size w is

        eff(w) = throughput(w) / (throughput(w0) * (w / w0)) * 100

    i.e. speedup relative to w0 divided by the ideal linear speedup w/w0.
    eff(w0) == 100% by construction. This is NOT comparable in absolute terms
    to the 1-GPU-normalized scaling_efficiency.png; it measures how well each
    strategy keeps scaling ABOVE the point where it first becomes feasible.
    """
    print("Scaling efficiency (large models, relative to smallest feasible ws)")
    ok = df[(df["status"] == "ok")
            & (df["model"].isin(LARGE_NO_BASELINE_MODELS))
            & (df["throughput_tokens_per_sec_mean"] > 0)].copy()
    if ok.empty:
        print("  (no feasible large-model cells yet; skipping)")
        return

    rows = []
    for (model, strat), g in ok.groupby(["model", "strategy"]):
        g = g.sort_values("world_size")
        w0 = g["world_size"].iloc[0]
        t0 = g["throughput_tokens_per_sec_mean"].iloc[0]
        for _, r in g.iterrows():
            w = r["world_size"]
            ideal = t0 * (w / w0)
            rows.append({
                "model": model, "strategy": strat,
                "world_size": int(w), "baseline_ws": int(w0),
                "throughput_tokens_per_sec_mean": r["throughput_tokens_per_sec_mean"],
                "scaling_efficiency_relative": r["throughput_tokens_per_sec_mean"]
                / ideal * 100.0,
            })
    rel = pd.DataFrame(rows)
    rel.to_csv(os.path.join(out_dir, "scaling_efficiency_relative_large.csv"),
               index=False)

    models = [m for m in MODEL_ORDER if m in rel["model"].unique()]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5.5 * len(models), 5.0), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        mdf = rel[rel["model"] == model]
        for strat in STRATEGY_ORDER:
            s = mdf[mdf["strategy"] == strat].sort_values("world_size")
            if s.empty:
                continue
            w0 = int(s["baseline_ws"].iloc[0])
            ax.plot(s["world_size"], s["scaling_efficiency_relative"],
                    marker="o", label=f"{STRATEGY_LABEL[strat]} (base {w0} GPU)")
        ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{model}  ({MODEL_PARAMS_B.get(model)}B)")
        ax.set_xlabel("GPUs"); ax.set_xticks([2, 4, 8])
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("Relative scaling efficiency (%)")
    fig.suptitle("Large-model scaling efficiency "
                 "(baseline = smallest feasible GPU count; single-GPU infeasible)")
    _savefig(fig, out_dir, "scaling_efficiency_large.png")


# --------------------------------------------------------------------------
# RQ4 seed: recommendation table
# --------------------------------------------------------------------------

def table_recommendation(df: pd.DataFrame, out_dir: str) -> None:
    """Best strategy per (model, GPU count) by throughput — seeds RQ4."""
    print("RQ4: recommendation table (seed)")
    ok = df[(df["status"] == "ok") & (df["world_size"] >= 2)]
    if ok.empty:
        return
    agg = ok.groupby(["model", "world_size", "strategy"],
                     as_index=False)["throughput_tokens_per_sec_mean"].mean()

    rows = []
    for model in MODEL_ORDER:
        for ws in sorted(agg["world_size"].unique()):
            cell = agg[(agg["model"] == model) & (agg["world_size"] == ws)]
            if cell.empty:
                continue
            best = cell.loc[cell["throughput_tokens_per_sec_mean"].idxmax()]
            # feasibility: did DDP OOM here?
            ddp_oom = not df[(df["model"] == model) & (df["world_size"] == ws)
                             & (df["strategy"] == "ddp")
                             & (df["status"] == "oom")].empty
            rows.append({
                "model": model,
                "world_size": ws,
                "best_strategy": STRATEGY_LABEL[best["strategy"]],
                "best_throughput": round(best["throughput_tokens_per_sec_mean"], 1),
                "ddp_oom": ddp_oom,
            })
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "recommendation_table.csv")
    out.to_csv(path, index=False)
    print(f"  wrote {path}")
    print(out.to_string(index=False))


# --------------------------------------------------------------------------
# Feasibility matrix (design doc §4.1)
# --------------------------------------------------------------------------

def table_feasibility(df: pd.DataFrame, out_dir: str) -> None:
    print("Feasibility matrix")
    rows = []
    for model in MODEL_ORDER:
        for strat in STRATEGY_ORDER:
            for ws in sorted(df["world_size"].unique()):
                if ws < 2:
                    continue
                cell = df[(df["model"] == model) & (df["strategy"] == strat)
                          & (df["world_size"] == ws)]
                if cell.empty:
                    status = "missing"
                elif (cell["status"] == "ok").any():
                    status = "ok"
                elif (cell["status"] == "oom").any():
                    status = "oom"
                else:
                    status = "unknown"
                rows.append({"model": model, "strategy": STRATEGY_LABEL[strat],
                             "world_size": ws, "status": status})
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "feasibility_matrix.csv")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# Extra figures (paper idioms): phase diagram, timeline, BW curve, Pareto
# --------------------------------------------------------------------------

def fig_phase_diagram(df: pd.DataFrame, out_dir: str) -> None:
    """model x GPU grid colored by winning strategy; infeasible cells greyed.

    The single figure that answers RQ2 'where does crossover happen' AND seeds
    RQ4: DDP region (small), ZeRO-2 region (mid), ZeRO-3-only / no-DDP region
    (large). (Alpa / auto-parallel cost-model idiom.)
    """
    print("RQ2/RQ4: strategy phase diagram")
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    ok = df[df["status"] == "ok"]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    wss = sorted({int(w) for w in df["world_size"].unique() if int(w) >= 2})
    if not models or not wss:
        return
    sidx = {"ddp": 0, "fsdp_zero2": 1, "fsdp_zero3": 2}
    colors = [STRATEGY_COLOR[s] for s in STRATEGY_ORDER]
    grid = np.full((len(models), len(wss)), np.nan)
    labels = [["" for _ in wss] for _ in models]
    for i, m in enumerate(models):
        for j, ws in enumerate(wss):
            cell = ok[(ok["model"] == m) & (ok["world_size"] == ws)]
            if cell.empty:
                labels[i][j] = "infeasible"
                continue
            best = cell.loc[cell["throughput_tokens_per_sec_mean"].astype(float).idxmax()]
            grid[i][j] = sidx.get(best["strategy"], np.nan)
            ddp_ok = "ddp" in set(cell["strategy"])
            labels[i][j] = (f"{STRATEGY_LABEL[best['strategy']]}\n"
                            f"{float(best['throughput_tokens_per_sec_mean'])/1e3:.1f}k tok/s"
                            + ("" if ddp_ok else "\n(no DDP)"))
    cmap = ListedColormap(colors); cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(1.9 * len(wss) + 2.5, 1.1 * len(models) + 1.5))
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(wss))); ax.set_xticklabels([f"{w} GPU" for w in wss])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m}\n({MODEL_PARAMS_B[m]}B)" for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(wss)):
            dark = np.isnan(grid[i][j])
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.5,
                    color="black" if dark else "white",
                    fontweight="bold")
    handles = [mpatches.Patch(color=colors[sidx[s]], label=STRATEGY_LABEL[s])
               for s in STRATEGY_ORDER]
    handles.append(mpatches.Patch(color="#dddddd", label="infeasible (all OOM)"))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Best strategy by model size x GPU count\n"
                 "(cell = winner + throughput; '(no DDP)' = DDP infeasible here)")
    _savefig(fig, out_dir, "phase_diagram_best_strategy.png")


def fig_bw_curve(df: pd.DataFrame, out_dir: str,
                 curve_path: str = "outputs/nccl_tests/bw_curves.json") -> None:
    """nccl-tests busbw-vs-size curve with each config's operating point overlaid.

    Shows WHERE each strategy operates on the bandwidth curve (granularity ->
    efficiency), RQ2b. (NCCL / collective-BW benchmarking idiom.)
    """
    print("RQ2b: bandwidth-vs-size curve + operating points")
    import json
    if not os.path.exists(curve_path):
        print("  (no bw curve; skipping)"); return
    curve = json.load(open(curve_path))
    ok = df[df["status"] == "ok"]
    ccolor = {"allreduce": "#1f77b4", "allgather": "#ff7f0e",
              "reducescatter": "#2ca02c"}
    for ws in sorted({int(w) for w in df["world_size"].unique() if int(w) >= 2}):
        wk = str(ws)
        if wk not in curve:
            continue
        fig, ax = plt.subplots(figsize=(10.5, 6))
        for cname, pts in curve[wk].items():
            pts = sorted(pts)
            ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "-",
                    color=ccolor.get(cname, "gray"), alpha=0.5,
                    label=f"{cname} (nccl-tests)")
        sub = ok[ok["world_size"] == ws].copy()
        for strat in STRATEGY_ORDER:
            s = sub[sub["strategy"] == strat].copy()
            if s.empty:
                continue
            s["sz"] = s["mean_collective_size_bytes"].astype(float)
            s["bw"] = s["achieved_bw_gbps"].astype(float)
            s = s[(s["sz"] > 0) & (s["bw"] > 0)]
            if s.empty:
                continue
            ax.scatter(s["sz"] / 1e6, s["bw"], color=STRATEGY_COLOR[strat],
                       s=70, edgecolor="black", zorder=5,
                       label=f"{STRATEGY_LABEL[strat]} (operating pts)")
        ax.set_xscale("log")
        ax.set_xlabel("Mean collective message size (MB)")
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title(f"RQ2b — Achieved busbw vs NVSwitch bandwidth curve ({ws} GPUs)")
        _legend_right(ax, fontsize=8); ax.grid(alpha=0.3, which="both")
        _savefig(fig, out_dir, f"rq2_bw_curve_ws{ws}.png")


def fig_pareto(df: pd.DataFrame, out_dir: str) -> None:
    """Throughput vs peak memory, Pareto frontier highlighted (RQ4 tradeoff)."""
    print("RQ4: throughput-memory Pareto frontier")
    import matplotlib.lines as mlines
    ok = df[(df["status"] == "ok") & (df["world_size"] >= 2)].copy()
    if ok.empty:
        return
    ok["mem"] = ok["peak_memory_reserved_gb"].astype(float)
    ok["thr"] = ok["throughput_tokens_per_sec_mean"].astype(float)
    ok = ok[(ok["mem"] > 0) & (ok["thr"] > 0)]
    markers = {2: "o", 4: "s", 8: "^"}
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for strat in STRATEGY_ORDER:
        for ws, mk in markers.items():
            s = ok[(ok["strategy"] == strat) & (ok["world_size"] == ws)]
            if s.empty:
                continue
            ax.scatter(s["mem"], s["thr"], color=STRATEGY_COLOR[strat],
                       marker=mk, s=55, alpha=0.85, edgecolor="white", linewidth=0.4)
    pts = ok[["mem", "thr"]].values
    pareto = []
    for m, t in pts:
        dominated = np.any((pts[:, 1] >= t) & (pts[:, 0] <= m)
                           & ((pts[:, 1] > t) | (pts[:, 0] < m)))
        if not dominated:
            pareto.append((m, t))
    pareto = sorted(set(pareto))
    if pareto:
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "k--",
                alpha=0.6, label="Pareto frontier")
    ax.axvline(80, color="red", linestyle=":", alpha=0.6, label="80GB wall")
    ax.set_yscale("log")
    ax.set_xlabel("Peak memory reserved (GB)")
    ax.set_ylabel("Throughput (tokens/sec, log)")
    ax.set_title("RQ4 — Throughput vs Memory Pareto\n(color = strategy, marker = GPU count)")
    handles = [mlines.Line2D([], [], color=STRATEGY_COLOR[s], marker="o",
                             linestyle="", label=STRATEGY_LABEL[s])
               for s in STRATEGY_ORDER]
    handles += [mlines.Line2D([], [], color="gray", marker=mk, linestyle="",
                              label=f"{w} GPU") for w, mk in markers.items()]
    handles += [mlines.Line2D([], [], color="black", linestyle="--", label="Pareto")]
    _legend_right(ax, handles=handles, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    _savefig(fig, out_dir, "rq4_throughput_memory_pareto.png")


_COMM_COLORS = {"allreduce": "#1f77b4", "allgather": "#ff7f0e",
                "reducescatter": "#2ca02c", "other_comm": "#9467bd"}


def _timeline_lanes(trace_path):
    """Parse a Kineto trace -> (compute_union, {collective: union}) in us."""
    import json
    import re
    nccl = re.compile(r"ncclDevKernel|ncclKernel", re.IGNORECASE)
    rs = re.compile(r"reducescatter", re.IGNORECASE)
    ag = re.compile(r"allgather", re.IGNORECASE)
    ar = re.compile(r"allreduce", re.IGNORECASE)

    def classify(name):
        if not nccl.search(name):
            return None
        if rs.search(name):
            return "reducescatter"
        if ag.search(name):
            return "allgather"
        if ar.search(name):
            return "allreduce"
        return "other_comm"

    with open(trace_path) as f:
        data = json.load(f)
    comp, comm = [], {}
    for e in data.get("traceEvents", []):
        if e.get("ph") != "X" or str(e.get("cat", "")).lower() != "kernel":
            continue
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or not dur:
            continue
        c = classify(e.get("name", ""))
        if c is None:
            comp.append((ts, ts + dur))
        else:
            comm.setdefault(c, []).append((ts, ts + dur))

    def merge(iv):
        if not iv:
            return []
        iv = sorted(iv); out = [list(iv[0])]
        for a, b in iv[1:]:
            if a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    return merge(comp), {k: merge(v) for k, v in comm.items()}


def fig_iteration_timelines(out_dir: str) -> None:
    """One compute-vs-collectives timeline per strategy over ~one iteration.

    Comm bars are colored by collective type, so the strategy's collective
    'combination' is visible: DDP = all-reduce only; ZeRO-2 = all-gather (fwd) +
    reduce-scatter (bwd); ZeRO-3 = 2x all-gather (fwd+bwd) + reduce-scatter.
    (HTA / Nsight timeline idiom.)"""
    print("RQ1/RQ2a: per-strategy single-iteration timelines")
    import glob
    strategies = [("ddp", "DDP"), ("fsdp_zero2", "ZeRO-2"), ("fsdp_zero3", "ZeRO-3")]
    for skey, slabel in strategies:
        trace = None
        # Prefer the current per-rank export (trace_rank{n}.json, rank 0 first),
        # fall back to the legacy tensorboard name (*.pt.trace.json).
        for base in ("outputs/experiments", "outputs_ddp_decomp/experiments"):
            hits = (sorted(glob.glob(f"{base}/qwen3-4b_{skey}_ws8*/traces/trace_rank0.json"))
                    or sorted(glob.glob(f"{base}/qwen3-4b_{skey}_ws8*/traces/trace_rank*.json"))
                    or sorted(glob.glob(f"{base}/qwen3-4b_{skey}_ws8*/traces/*.pt.trace.json")))
            if hits:
                trace = hits[0]
                break
        if not trace:
            print(f"  ({skey}: no anchor trace; skipping)"); continue
        comp_u, comm_by = _timeline_lanes(trace)
        if not comm_by:
            print(f"  ({skey}: no NCCL kernels; skipping)"); continue

        all_iv = comp_u + [iv for v in comm_by.values() for iv in v]
        t0 = min(a for a, _ in all_iv)
        end = max(b for _, b in all_iv)
        span = end - t0
        w0 = t0 + span * 0.45
        w1 = w0 + min(span * 0.18, 1_300_000)   # ~one 4B iteration

        def bars(iv):
            return [((a - t0) / 1e3, (b - a) / 1e3) for a, b in iv if b > w0 and a < w1]

        fig, ax = plt.subplots(figsize=(13, 3.6))
        ax.broken_barh(bars(comp_u), (11, 8), facecolors="#4c72b0")
        for coll, iv in comm_by.items():
            ax.broken_barh(bars(iv), (1, 8), facecolors=_COMM_COLORS.get(coll, "#9467bd"))
        ax.set_xlim((w0 - t0) / 1e3, (w1 - t0) / 1e3)
        ax.set_ylim(0, 20)
        ax.set_yticks([5, 15]); ax.set_yticklabels(["NCCL (comm)", "Compute"])
        ax.set_xlabel("Time within iteration (ms)")
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color="#4c72b0", label="compute")]
        handles += [mpatches.Patch(color=_COMM_COLORS[c], label=c)
                    for c in ("allreduce", "allgather", "reducescatter") if c in comm_by]
        _legend_right(ax, handles=handles, fontsize=8)
        ax.set_title(f"Single-iteration timeline — Qwen3-4B / {slabel} / 8 GPUs\n"
                     "comm hides behind compute where the lanes align in time")
        _savefig(fig, out_dir, f"iteration_timeline_{skey}.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build all thesis figures/tables")
    p.add_argument("--csv", default="outputs/all_experiments_summary.csv")
    p.add_argument("--out_dir", default="outputs/figures")
    p.add_argument("--gate_pct", type=float, default=5.0)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Master CSV not found: {args.csv}")

    df_all, _ = load_df(args.csv)
    print(f"Loaded {len(df_all)} rows from {args.csv}")
    # Recompute calibration (needs the _gcoff pairs, so BEFORE ablation exclusion).
    df_all, _ = add_recompute_calibration(df_all)
    # Ablation cells carry a name suffix (_gcoff = GC off, _noovlp = suppress
    # overlap). Exclude them from the MAIN figures, else rep_mean() would average
    # them into the normal (model, strategy, ws) cells. They're used only by
    # their dedicated comparison (fig_rq1_recompute_tax).
    is_ablation = df_all["experiment_name"].astype(str).str.contains(
        "_gcoff|_noovlp", regex=True, na=False)
    df = df_all[~is_ablation].copy()
    # CORE figures (RQ1/RQ2/RQ4, scaling, phase, BW, Pareto) must be ga=1 only.
    # The grad-accum sweep (group3: 4B/ws8, ga 2/4/8) otherwise gets averaged into
    # the 4B/ws8 cell by rep_mean's (model,strategy,ws) grouping — inflating its
    # exposed-comm and throughput. Only fig_rq3_gradaccum uses all ga.
    ga = pd.to_numeric(df["gradient_accumulation_steps"], errors="coerce")
    df_core = df[ga == 1].copy()
    print(f"  {int(is_ablation.sum())} ablation rows held out; "
          f"{int((ga != 1).sum())} grad-accum rows held out of core figures\n")

    # Feasibility + baselines first
    table_feasibility(df_core, args.out_dir)

    # RQ1
    fig_rq1_decomposition(df_core, args.out_dir)
    fig_rq1_residual_quality(df_core, args.out_dir, gate_pct=args.gate_pct)
    fig_rq1_recompute_tax(df_all, args.out_dir)

    # RQ2
    fig_rq2_throughput(df_core, args.out_dir)
    fig_rq2_mechanism(df_core, args.out_dir)
    fig_rq2_driver_waterfall(df_core, args.out_dir)   # DDP->ZeRO-2->ZeRO-3 driver
    table_rq2_collective_breakdown(df_core, args.out_dir)

    # RQ3 (needs the full grad-accum sweep)
    fig_rq3_gradaccum(df, args.out_dir)
    fig_rq3_crossover_shift(df, args.out_dir)          # does GA shift the crossover?
    # RQ3 communication mechanism across the sweep (profiled cells only)
    fig_rq3_exposed_comm_by_ws(df, args.out_dir)
    fig_rq3_exposed_comm_by_model(df, args.out_dir)
    fig_rq3_strategy_contrast(df, args.out_dir)
    fig_rq3_comm_bw(df, args.out_dir)

    # Cross-cutting
    fig_scaling_efficiency(df_core, args.out_dir)
    fig_scaling_efficiency_large(df_core, args.out_dir)

    # Extra paper-idiom figures
    fig_phase_diagram(df_core, args.out_dir)
    fig_bw_curve(df_core, args.out_dir)
    fig_pareto(df_core, args.out_dir)
    fig_iteration_timelines(args.out_dir)

    # RQ4 seeds
    table_recommendation(df_core, args.out_dir)

    print(f"\nAll figures/tables written to {args.out_dir}/")


if __name__ == "__main__":
    main()