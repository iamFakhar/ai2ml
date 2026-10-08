"""Sample analysis of a subset of the results (default: Qwen3-4B, GA=1).

Prints a throughput table and writes one figure with (a) throughput vs. GPU
count per strategy (error bars = step-level std within the single run) and
(b) the attributed iteration decomposition at the largest GPU count.

Usage:
  python analysis/sample_analysis.py                       # released dataset
  python analysis/sample_analysis.py --csv my_outputs/all_experiments_summary.csv \
      --model qwen3-0.6b --ws 2
"""
import argparse, csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRATS = [("ddp", "DDP", "#0072B2"), ("fsdp_zero2", "FSDP-Z2", "#E69F00"), ("fsdp_zero3", "FSDP-Z3", "#009E73")]
COMPS = [("fwd_time_ms_mean", "Forward", "#4C78A8"), ("bwd_compute_ms_calibrated", "Bwd compute", "#F58518"),
         ("recompute_ms_calibrated", "Recompute", "#EECA3B"), ("bwd_exposed_comm_ms", "Exposed comm", "#E45756"),
         ("opt_time_ms_mean", "Optimizer", "#54A24B"), ("straggler_wait_ms", "Straggler", "#B279A2")]


RATIO = 0.7532   # recompute/forward ratio calibrated on the released group-4a ablation


def val(r, k):
    """Component value; derive the calibrated backward split if the CSV has not been
    augmented yet (fresh runs: python -m scripts.augment_recompute_calibration)."""
    if k in ("recompute_ms_calibrated", "bwd_compute_ms_calibrated") and not r.get(k):
        rec = RATIO * float(r["fwd_time_ms_mean"])
        if k == "recompute_ms_calibrated": return rec
        return max(0.0, float(r["bwd_time_ms_mean"]) - rec - float(r["bwd_exposed_comm_ms"]))
    return float(r[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "all_experiments_summary.csv"))
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--ga", default="1")
    ap.add_argument("--out", default=str(ROOT / "docs" / "img" / "sample_subset.png"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r["model"] == a.model and r["gradient_accumulation_steps"] == a.ga
            and r["strategy"] in {s[0] for s in STRATS}]
    wss = sorted({int(r["world_size"]) for r in rows})
    get = lambda s, w: next((r for r in rows if r["strategy"] == s and int(r["world_size"]) == w), None)

    print(f"\n{a.model}, GA={a.ga}: throughput (tokens/s) [step-level CV %]")
    print("GPUs  " + "  ".join(f"{n:>18}" for _, n, _ in STRATS))
    for w in wss:
        cells = []
        for s, _, _ in STRATS:
            r = get(s, w)
            if r is None: cells.append("n/a")
            elif r["status"] != "ok": cells.append("OOM")
            else:
                t, sd = float(r["throughput_tokens_per_sec_mean"]), float(r["throughput_tokens_per_sec_std"])
                cells.append(f"{t:,.0f} [{100*sd/t:.1f}]")
        print(f"{w:>4}  " + "  ".join(f"{c:>18}" for c in cells))

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    bw = 0.26
    for i, (s, n, c) in enumerate(STRATS):
        xs, ys, es = [], [], []
        for j, w in enumerate(wss):
            r = get(s, w)
            if r and r["status"] == "ok":
                xs.append(j + (i - 1) * bw); ys.append(float(r["throughput_tokens_per_sec_mean"]) / 1000)
                es.append(float(r["throughput_tokens_per_sec_std"]) / 1000)
        ax1.bar(xs, ys, bw * 0.92, yerr=es, color=c, label=n, capsize=1.5, error_kw={"lw": 0.6})
    ax1.set_xticks(range(len(wss))); ax1.set_xticklabels([f"{w} GPUs" for w in wss])
    ax1.set_ylabel("Throughput (ktokens/s)"); ax1.set_title(f"(a) {a.model}, GA={a.ga}", fontsize=8)
    ax1.legend(frameon=False, loc="upper left"); ax1.grid(axis="y", alpha=.25, lw=.4)
    wmax = wss[-1]
    for i, (s, n, _) in enumerate(STRATS):
        r = get(s, wmax)
        if not r or r["status"] != "ok": continue
        left = 0
        for k, lab, col in COMPS:
            v = val(r, k); ax2.barh(i, v, left=left, color=col, edgecolor="white", lw=.4, label=lab if i == 0 else None); left += v
        ax2.text(left + 8, i, f"{float(r['iter_time_ms_mean']):.0f} ms", va="center", fontsize=7)
    ax2.set_yticks(range(3)); ax2.set_yticklabels([n for _, n, _ in STRATS]); ax2.invert_yaxis()
    ax2.set_xlabel("Attributed time per iteration (ms)"); ax2.set_title(f"(b) decomposition, {wmax} GPUs", fontsize=8)
    ax2.legend(ncol=3, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.28), columnspacing=1, handlelength=1.2)
    ax2.set_xlim(0, ax2.get_xlim()[1] * 1.12)
    fig.tight_layout(); pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); print("\nwrote", a.out)


if __name__ == "__main__":
    main()
