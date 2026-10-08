# orchestration/launch_matrix.py
"""Subprocess-per-run orchestrator (design doc §6, §12).

Why subprocess-per-run:
  A CUDA/NCCL crash or OOM in one run must NOT kill the remaining matrix.
  Each run is an isolated `torchrun` process; failures are recorded and the
  driver continues.

Responsibilities:
  1. Expand matrix.yaml groups into concrete run specs.
  2. Skip runs already completed (resume via master CSV).
  3. Launch each via torchrun with the correct nproc_per_node.
  4. Enforce per-run timeout and force env (alloc conf, NCCL tracing).
  5. Record hard failures (non-OOM crashes) so they can be retried.

Usage:
  python -m orchestration.launch_matrix \
      --matrix configs/matrix.yaml \
      --base configs/base.yaml \
      --models configs/models.yaml \
      [--groups group1_core group2_baseline] \
      [--dry_run] [--only_validated] [--profile_all] [--max_runs N]
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from .resume import load_completed, summarize_progress


# ==========================================================================
# Run spec
# ==========================================================================

@dataclass
class RunSpec:
    group: str
    model: str
    strategy: str
    world_size: int
    grad_accum_steps: int
    rep: int
    per_device_bs: int
    seq_len: int
    overrides: Dict = field(default_factory=dict)
    is_validated: bool = False

    @property
    def name(self) -> str:
        base = (f"{self.model}_{self.strategy}"
                f"_ws{self.world_size}_ga{self.grad_accum_steps}"
                f"_bs{self.per_device_bs}_seq{self.seq_len}_rep{self.rep}")
        # Disambiguate ablation cells whose (model,strategy,ws,ga,rep) otherwise
        # COLLIDES with a group1 twin — without this suffix they share a name and
        # get resume-skipped (group4a/group4b never ran). Suffix is added ONLY
        # when an override is active, so normal cells keep their existing names
        # (and stay resume-matched against the master CSV).
        suffix = ""
        if self.overrides.get("workload", {}).get("gradient_checkpointing") is False:
            suffix += "_gcoff"
        if self.overrides.get("ablation", {}).get("suppress_overlap"):
            suffix += "_noovlp"
        return base + suffix


# ==========================================================================
# Matrix expansion
# ==========================================================================

def expand_matrix(matrix: dict, base: dict,
                  selected_groups: Optional[List[str]]) -> List[RunSpec]:
    per_device_bs = base["workload"]["per_device_batch_size"]
    seq_len = base["workload"]["seq_len"]
    validated = set(matrix.get("validated_configs", []))

    specs: List[RunSpec] = []
    for group_name, gcfg in matrix["groups"].items():
        if not gcfg.get("enabled", True):
            continue
        if selected_groups and group_name not in selected_groups:
            continue

        combos = itertools.product(
            gcfg["models"],
            gcfg["strategies"],
            gcfg["world_sizes"],
            gcfg["grad_accum_steps"],
            range(gcfg["reps"]),
        )
        for model, strat, ws, ga, rep in combos:
            spec = RunSpec(
                group=group_name,
                model=model,
                strategy=strat,
                world_size=ws,
                grad_accum_steps=ga,
                rep=rep,
                per_device_bs=per_device_bs,
                seq_len=seq_len,
                overrides=gcfg.get("overrides", {}),
            )
            spec.is_validated = spec.name in validated
            specs.append(spec)

    return specs


def order_runs(specs: List[RunSpec]) -> List[RunSpec]:
    """Order to make resume/analysis-friendly and fail fast on the risky cells.

    Ordering rationale:
      1. Single-GPU baselines FIRST (Group 2) so scaling_efficiency can be
         computed inline for later distributed runs.
      2. Then ascending model size and GPU count so cheap runs validate the
         pipeline before expensive 8B/8-GPU runs.
      3. Largest/most-OOM-prone cells last.
    """
    def key(s: RunSpec):
        baseline_first = 0 if s.strategy == "single" else 1
        model_rank = {"qwen3-0.6b": 0, "qwen3-1.7b": 1,
                      "qwen3-4b": 2, "qwen3-8b": 3,
                      "qwen3-14b": 4, "qwen3-32b": 5}.get(s.model, 9)
        return (baseline_first, model_rank, s.world_size,
                s.grad_accum_steps, s.strategy, s.rep)
    return sorted(specs, key=key)


# ==========================================================================
# Launch
# ==========================================================================

def build_env(base: dict) -> Dict[str, str]:
    env = dict(os.environ)
    for k, v in base.get("env", {}).items():
        env[str(k)] = str(v)
    return env


def build_command(spec: RunSpec, base_path: str, models_path: str,
                  profile_all: bool = False, keep_traces: bool = False,
                  profile_only: bool = False) -> List[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={spec.world_size}",
        "-m", "src.runner",
        "--config", base_path,
        "--models_config", models_path,
        "--model", spec.model,
        "--strategy", spec.strategy,
        "--world_size", str(spec.world_size),
        "--grad_accum_steps", str(spec.grad_accum_steps),
        "--rep", str(spec.rep),
        "--experiment_name", spec.name,
    ]
    # Group-4 ablation flags carried via overrides.
    wl = spec.overrides.get("workload", {})
    if wl.get("gradient_checkpointing") is False:
        cmd.append("--no_grad_checkpoint")
    if spec.overrides.get("ablation", {}).get("suppress_overlap"):
        cmd.append("--suppress_overlap")
    # --profile_only: skip PASS 1, run PASS 2 only, emit a comm_only.json sidecar
    # (implies profiling; keeps the existing clean timing). Merge afterwards with
    # scripts.merge_comm_only. Otherwise PASS 2 profiling runs when a cell is
    # Nsight-validated OR --profile_all forces it matrix-wide.
    if profile_only:
        cmd.append("--profile_only")   # runner turns on enable_profiler_trace
    elif spec.is_validated or profile_all:
        cmd.append("--enable_profiler_trace")
    # Keep heavy traces only where asked (e.g. Nsight/HTA anchors); the runner
    # otherwise computes overlap then discards them to bound disk at scale.
    if keep_traces or spec.is_validated:
        cmd.append("--keep_traces")
    return cmd


def launch_run(spec: RunSpec, cmd: List[str], env: Dict[str, str],
               timeout_s: int, log_dir: str) -> str:
    """Run one experiment as an isolated subprocess. Returns a status string.

    Status semantics:
      'ok'          -> runner exited 0 (row written; may be status=ok or oom
                       inside the CSV; OOM is a clean exit-0 in the runner)
      'crash'       -> non-zero exit (hard failure, candidate for retry)
      'timeout'     -> exceeded timeout_s (likely hang; needs investigation)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{spec.name}.log")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# CMD: {' '.join(cmd)}\n")
        logf.write(f"# START: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n")
        logf.flush()
        try:
            result = subprocess.run(
                cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            elapsed = time.time() - start
            logf.write(f"\n# EXIT: {result.returncode} in {elapsed:.1f}s\n")
            return "ok" if result.returncode == 0 else "crash"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logf.write(f"\n# TIMEOUT after {elapsed:.1f}s (limit {timeout_s}s)\n")
            return "timeout"


def timeout_for(spec: RunSpec) -> int:
    """Per-run wall-clock ceiling. Scales with model size, GPU count, accum.

    Generous by design: num_steps (12) x up to 2 passes (clean + profiled) of
    the largest cell should complete well within these bounds; anything longer
    indicates a hang. Large models (14B/32B) get bigger factors accordingly.
    """
    base = 600  # 10 min floor
    model_factor = {"qwen3-0.6b": 1, "qwen3-1.7b": 1.5,
                    "qwen3-4b": 3, "qwen3-8b": 5,
                    "qwen3-14b": 8, "qwen3-32b": 16}.get(spec.model, 3)
    accum_factor = max(1, spec.grad_accum_steps)
    return int(base * model_factor * accum_factor)


# ==========================================================================
# Driver
# ==========================================================================

def run_matrix(args) -> None:
    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)
    with open(args.base) as f:
        base = yaml.safe_load(f)

    out_root = base["experiment"]["output_root"]
    master_csv = os.path.join(out_root, "all_experiments_summary.csv")
    log_dir = os.path.join(out_root, "run_logs")
    failure_log = os.path.join(out_root, "failures.log")

    specs = expand_matrix(matrix, base, args.groups)
    # De-duplicate by experiment_name: a cell can be declared in more than one
    # group (normally harmless — resume skips the repeat — but with --profile_only
    # or --force the repeat would run twice). Keep the first occurrence.
    _seen = set()
    specs = [s for s in specs if not (s.name in _seen or _seen.add(s.name))]
    if args.only_validated:
        specs = [s for s in specs if s.is_validated]
    # Restrict to an explicit set of experiment_names (--only / --only_file),
    # e.g. to re-profile exactly the RQ3 sweep cells.
    only = set(args.only or [])
    if args.only_file:
        with open(args.only_file) as f:
            only |= {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    if only:
        specs = [s for s in specs if s.name in only]
        missing = only - {s.name for s in specs}
        if missing:
            print(f"WARNING: {len(missing)} --only name(s) not in the expanded "
                  f"matrix (check spelling/groups): {sorted(missing)[:5]}...")
    specs = order_runs(specs)
    if args.max_runs:
        specs = specs[:args.max_runs]

    planned_names = [s.name for s in specs]
    prog = summarize_progress(master_csv, planned_names)
    print("=" * 70)
    print(f"MATRIX PLAN: {prog['planned']} runs "
          f"| done={prog['completed']} (ok={prog['ok']}, oom={prog['oom']}) "
          f"| remaining={prog['remaining']}")
    print("=" * 70)

    if args.dry_run:
        for s in specs:
            done = "DONE" if s.name in load_completed(master_csv) else "TODO"
            tags = []
            if s.is_validated:
                tags.append("validated")
            if s.is_validated or args.profile_all or args.profile_only:
                tags.append("profiled")
            if args.profile_only:
                tags.append("profile_only")
            tag = f" [{','.join(tags)}]" if tags else ""
            print(f"  [{done}] {s.name}{tag}")
        print(f"\nDry run: {len(specs)} runs planned. No execution.")
        return

    env = build_env(base)
    completed = load_completed(master_csv)

    launched = crashed = timed_out = skipped = 0
    t0 = time.time()

    for i, spec in enumerate(specs, 1):
        # --profile_only refreshes comm/overlap for ALREADY-complete cells
        # (it writes a sidecar, not a CSV row), so it must not be skipped by resume.
        if spec.name in completed and not args.force and not args.profile_only:
            skipped += 1
            print(f"[{i}/{len(specs)}] SKIP (done={completed[spec.name]}) {spec.name}")
            continue

        cmd = build_command(spec, args.base, args.models,
                            profile_all=args.profile_all,
                            keep_traces=args.keep_traces,
                            profile_only=args.profile_only)
        to = timeout_for(spec)
        print(f"[{i}/{len(specs)}] RUN {spec.name} "
              f"(ws={spec.world_size}, timeout={to}s)", flush=True)

        status = launch_run(spec, cmd, env, to, log_dir)
        launched += 1

        if status == "ok":
            # Refresh completed set from CSV (runner just wrote its row).
            completed = load_completed(master_csv)
            csv_status = completed.get(spec.name, "unknown")
            print(f"    -> completed (csv_status={csv_status})")
        else:
            crashed += (status == "crash")
            timed_out += (status == "timeout")
            _record_failure(failure_log, spec, status)
            print(f"    -> FAILURE ({status}); recorded, continuing")

        elapsed_h = (time.time() - t0) / 3600.0
        print(f"    progress: launched={launched} crashed={crashed} "
              f"timeout={timed_out} skipped={skipped} | elapsed={elapsed_h:.2f}h",
              flush=True)

    _print_final_report(master_csv, planned_names, launched, crashed,
                        timed_out, skipped, time.time() - t0)


def _record_failure(failure_log: str, spec: RunSpec, status: str) -> None:
    os.makedirs(os.path.dirname(failure_log), exist_ok=True)
    with open(failure_log, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{status}\t{spec.name}\n")


def _print_final_report(master_csv, planned_names, launched, crashed,
                        timed_out, skipped, elapsed_s) -> None:
    prog = summarize_progress(master_csv, planned_names)
    print("\n" + "=" * 70)
    print("MATRIX COMPLETE")
    print("=" * 70)
    print(f"  planned:      {prog['planned']}")
    print(f"  ok:           {prog['ok']}")
    print(f"  oom:          {prog['oom']}")
    print(f"  remaining:    {prog['remaining']}")
    print(f"  launched:     {launched}")
    print(f"  crashed:      {crashed}")
    print(f"  timed_out:    {timed_out}")
    print(f"  skipped:      {skipped}")
    print(f"  wall time:    {elapsed_s / 3600.0:.2f}h")
    if prog["remaining"] > 0:
        print(f"\n  {prog['remaining']} runs still incomplete — re-run this "
              f"command to resume (crashes/timeouts will be retried).")
    print("=" * 70)


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Experiment matrix orchestrator")
    p.add_argument("--matrix", default="configs/matrix.yaml")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--models", default="configs/models.yaml")
    p.add_argument("--groups", nargs="*", default=None,
                   help="Subset of group names to run (default: all enabled)")
    p.add_argument("--only_validated", action="store_true",
                   help="Run only Nsight/profiler-validated configs")
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to these exact experiment_names (after group "
                        "expansion). Combine with --profile_only to re-profile a "
                        "specific cell set.")
    p.add_argument("--only_file", default=None,
                   help="File with one experiment_name per line to restrict to "
                        "(merged with --only).")
    p.add_argument("--profile_all", action="store_true",
                   help="Enable the profiler comm pass (PASS 2) for EVERY run, "
                        "not just validated_configs. Populates NCCL comm metrics "
                        "matrix-wide; ~2x wall-clock per run. PASS 1 timing "
                        "stays clean.")
    p.add_argument("--keep_traces", action="store_true",
                   help="Keep Kineto traces for all profiled runs (default: "
                        "discard after overlap is computed; validated_configs "
                        "are always kept for Nsight/HTA).")
    p.add_argument("--profile_only", action="store_true",
                   help="Skip PASS 1 and run PASS 2 only for the selected cells "
                        "(refreshes comm/overlap, keeps existing timing). Writes "
                        "<exp>/comm_only.json sidecars; merge with "
                        "scripts.merge_comm_only then rebuild_master_csv. "
                        "Implies re-running already-complete cells (no resume skip).")
    p.add_argument("--max_runs", type=int, default=None,
                   help="Cap number of runs (for smoke tests)")
    p.add_argument("--dry_run", action="store_true",
                   help="List planned runs without executing")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if already completed in master CSV")
    return p.parse_args()


def main():
    args = parse_args()
    run_matrix(args)


if __name__ == "__main__":
    main()