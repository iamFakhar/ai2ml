# orchestration/resume.py
"""Resume support: determine which runs are already complete (design doc §12).

A run is 'complete' if a row with its experiment_name exists in the master
CSV with a terminal status (ok | oom). This lets the orchestrator skip
finished work after an interruption, which is essential for the ~48h budget
spread over 7-9 days.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Set


TERMINAL_STATUSES = {"ok", "oom"}


def load_completed(master_csv: str) -> Dict[str, str]:
    """Return {experiment_name: status} for all rows with terminal status."""
    completed: Dict[str, str] = {}
    if not os.path.exists(master_csv):
        return completed
    with open(master_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("experiment_name", "")
            status = (row.get("status", "") or "").strip().lower()
            if name and status in TERMINAL_STATUSES:
                completed[name] = status
    return completed


def completed_names(master_csv: str) -> Set[str]:
    return set(load_completed(master_csv).keys())


def is_complete(master_csv: str, experiment_name: str) -> bool:
    return experiment_name in completed_names(master_csv)


def summarize_progress(master_csv: str, planned_names: list[str]) -> Dict[str, int]:
    """Human-readable progress snapshot for orchestrator logging."""
    completed = load_completed(master_csv)
    ok = sum(1 for n in planned_names if completed.get(n) == "ok")
    oom = sum(1 for n in planned_names if completed.get(n) == "oom")
    done = ok + oom
    return {
        "planned": len(planned_names),
        "completed": done,
        "ok": ok,
        "oom": oom,
        "remaining": len(planned_names) - done,
    }