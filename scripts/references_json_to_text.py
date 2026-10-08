#!/usr/bin/env python3
"""Render references.json as a heading-delimited .txt for section-based chunking.

Consumes the JSON produced by scripts/bib_to_json.py (either the array form or
the --by-key object form) and emits one labelled block per reference:

    ##############################################################################
    # START [12/52]  zhao2023fsdp
    ##############################################################################

        Citekey       : zhao2023fsdp
        Author(s)     : Zhao, Yanli; Gu, Andrew; ...
        Title         : PyTorch FSDP: Experiences on Scaling Fully Sharded ...
        Year          : 2023
        arXiv ID      : 2304.11277
        Subject class : cs.DC
        URL           : https://arxiv.org/abs/2304.11277

    # END [12/52]  zhao2023fsdp
    ==============================================================================

Each `# START ... citekey` line is a stable section heading, so a downstream
chunker gets one clean chunk per reference. The block format is byte-compatible
with split.py (same 78-char borders and START/END markers), so the output can be
fanned out into per-reference files.

Usage:
    python scripts/references_json_to_text.py
    python scripts/references_json_to_text.py references.json -o references.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

WIDTH = 78
LABEL_W = 13
HASH = "#" * WIDTH
RULE = "=" * WIDTH

# Fields printed AFTER Citekey / Author(s) / Title / Year, in this order.
# (label, json field key). Absent fields are skipped.
FIELD_ORDER = [
    ("Journal", "journal"),
    ("Published in", "booktitle"),
    ("Howpublished", "howpublished"),
    ("Volume", "volume"),
    ("Number", "number"),
    ("Pages", "pages"),
    ("DOI", "doi"),
    ("arXiv ID", "eprint"),
    ("Subject class", "primaryclass"),
    ("Note", "note"),
    ("URL", "url"),
]


def _line(label: str, value: str) -> str:
    return f"    {label:<{LABEL_W}} : {value}"


def _entries_from_json(data) -> List[Dict]:
    """Accept either the array form or the --by-key object form."""
    if isinstance(data, dict):
        return [{"key": k, **v} for k, v in data.items()]
    if isinstance(data, list):
        return data
    raise SystemExit("unexpected JSON shape: expected a list or an object")


def render_entry(entry: Dict, idx: int, total: int) -> str:
    key = entry.get("key", "?")
    fields: Dict = entry.get("fields", {})
    tag = f"[{idx}/{total}]"

    out: List[str] = [HASH, f"# START {tag}  {key}", HASH, ""]

    out.append(_line("Citekey", key))
    if fields.get("file"):
        out.append(_line("Source File", fields["file"]))

    if entry.get("authors"):
        out.append(_line("Author(s)", "; ".join(entry["authors"])))
    elif entry.get("editors"):
        out.append(_line("Editor(s)", "; ".join(entry["editors"])))

    if fields.get("title"):
        out.append(_line("Title", fields["title"]))
    if fields.get("year"):
        out.append(_line("Year", fields["year"]))

    for label, fkey in FIELD_ORDER:
        val = fields.get(fkey)
        if not val:
            continue
        if fkey == "pages":
            val = val.replace("--", "–")  # 3005--3018 -> 3005–3018
        out.append(_line(label, val))

    out += ["", f"# END {tag}  {key}", RULE]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render references.json as chunkable .txt.")
    ap.add_argument("json", nargs="?", default=None,
                    help="input references.json (default: references.json next to repo root)")
    ap.add_argument("-o", "--out", default=None,
                    help="output .txt (default: <input> with .txt extension)")
    args = ap.parse_args(argv)

    in_path = Path(args.json) if args.json else Path(__file__).resolve().parent.parent / "references.json"
    if not in_path.is_file():
        print(f"error: JSON not found: {in_path}\n"
              f"       run scripts/bib_to_json.py first to generate it.")
        return 1

    entries = _entries_from_json(json.loads(in_path.read_text(encoding="utf-8")))
    if not entries:
        print(f"error: no entries in {in_path}")
        return 1

    total = len(entries)
    blocks = [render_entry(e, i, total) for i, e in enumerate(entries, start=1)]

    out_path = Path(args.out) if args.out else in_path.with_suffix(".txt")
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {total} references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
