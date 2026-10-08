#!/usr/bin/env python3
"""Parse a BibTeX (.bib) file into JSON.

Dependency-free (Python stdlib only). Handles the conventions used in this
project's references.bib: multi-line brace-delimited field values, nested
braces (e.g. {{PyTorch}}), "% ..." section-header comment lines, and common
LaTeX accents / escapes (K{\\"o}pf -> Köpf, {\\L}ukasz -> Łukasz, \\& -> &).

Output is a JSON array (default) of objects:

    {
      "key": "zhao2023fsdp",
      "type": "article",
      "fields": {"author": "...", "title": "...", ...},
      "authors": ["Zhao, Yanli", "Gu, Andrew", ...]   # only if author present
    }

Usage:
    # to stdout
    python -m scripts.bib_to_json                       # defaults to references.bib
    python scripts/bib_to_json.py references.bib

    # to a file
    python scripts/bib_to_json.py references.bib -o references.json

    # object keyed by cite key instead of an array
    python scripts/bib_to_json.py --by-key -o refs.json

    # keep values verbatim (no LaTeX -> Unicode cleaning, keep braces)
    python scripts/bib_to_json.py --raw

Re-run whenever references.bib changes; the conversion is deterministic and
idempotent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

# Entry types that are not bibliographic records.
_SKIP_TYPES = {"comment", "preamble", "string"}

# ---------------------------------------------------------------------------
# LaTeX -> Unicode cleaning
# ---------------------------------------------------------------------------

# Accent command -> combining diacritic (applied to the following base letter).
_COMBINING = {
    "`": "̀",  # grave
    "'": "́",  # acute
    "^": "̂",  # circumflex
    '"': "̈",  # diaeresis / umlaut
    "~": "̃",  # tilde
    "=": "̄",  # macron
    ".": "̇",  # dot above
    "u": "̆",  # breve      \u
    "v": "̌",  # caron      \v
    "H": "̋",  # double acute \H
    "r": "̊",  # ring above \r
    "c": "̧",  # cedilla    \c
    "k": "̨",  # ogonek     \k
    "d": "̣",  # dot below  \d
    "b": "̱",  # macron below \b
}

# Standalone special glyph commands (no base letter).
_SPECIAL = {
    "L": "Ł", "l": "ł", "O": "Ø", "o": "ø", "AA": "Å", "aa": "å",
    "AE": "Æ", "ae": "æ", "OE": "Œ", "oe": "œ", "ss": "ß",
    "i": "ı", "j": "ȷ",
}

# \"o  \"{o}  \c c  \c{c}  \v s  -> base + combining diacritic
_ACCENT_RE = re.compile(r"\\([`'\"^~=.uvHrckdb])\s*\{?\\?([A-Za-z])\}?")
# {\L}  \O  \ss  (longest first so \AE beats \A-ish matches)
_SPECIAL_RE = re.compile(
    r"\\(" + "|".join(sorted(_SPECIAL, key=len, reverse=True)) + r")(?![A-Za-z])"
)
# \&  \%  \#  \_  \$  ->  literal char
_ESCAPED_RE = re.compile(r"\\([&%#_$])")


def _apply_accent(m: "re.Match") -> str:
    base = m.group(2)
    combined = base + _COMBINING[m.group(1)]
    return unicodedata.normalize("NFC", combined)


def clean_value(text: str) -> str:
    """Collapse whitespace and convert LaTeX accents/escapes to Unicode."""
    s = _ESCAPED_RE.sub(lambda m: m.group(1), text)
    s = _ACCENT_RE.sub(_apply_accent, s)
    s = _SPECIAL_RE.sub(lambda m: _SPECIAL[m.group(1)], s)
    s = s.replace("~", " ")            # LaTeX non-breaking space
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collapse_ws(text: str) -> str:
    """Whitespace-only normalization (used in --raw mode)."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

def _split_top_level(rest: str) -> List[str]:
    """Split field assignments on commas that are outside braces/quotes."""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_quote = False
    for c in rest:
        if c == "{" and not in_quote:
            depth += 1
            buf.append(c)
        elif c == "}" and not in_quote:
            depth -= 1
            buf.append(c)
        elif c == '"' and depth == 0:
            in_quote = not in_quote
            buf.append(c)
        elif c == "," and depth == 0 and not in_quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    tail = "".join(buf)
    if tail.strip():
        parts.append(tail)
    return parts


def _unwrap(raw: str) -> str:
    """Strip one layer of {...} or "..." delimiters from a field value."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == "{" and raw[-1] == "}":
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1]
    return raw


def _parse_entry(etype: str, body: str, clean: bool) -> Optional[Dict]:
    comma = body.find(",")
    if comma == -1:
        return None
    key = body[:comma].strip()
    if not key:
        return None
    fields: Dict[str, str] = {}
    for chunk in _split_top_level(body[comma + 1:]):
        eq = chunk.find("=")
        if eq == -1:
            continue
        name = chunk[:eq].strip().lower()
        inner = _unwrap(chunk[eq + 1:])
        fields[name] = clean_value(inner) if clean else collapse_ws(inner)
    entry: Dict = {"key": key, "type": etype, "fields": fields}
    for role in ("author", "editor"):
        if role in fields:
            names = [n.strip() for n in re.split(r"\s+and\s+", fields[role]) if n.strip()]
            entry[role + "s"] = names  # "authors" / "editors"
    return entry


def parse_bib(text: str, clean: bool = True) -> List[Dict]:
    """Parse BibTeX source text into a list of entry dicts."""
    entries: List[Dict] = []
    n = len(text)
    i = 0
    while True:
        at = text.find("@", i)
        if at == -1:
            break
        j = at + 1
        while j < n and text[j].isalpha():
            j += 1
        etype = text[at + 1:j].strip().lower()
        while j < n and text[j] in " \t\r\n":
            j += 1
        if j >= n or text[j] != "{":
            i = at + 1
            continue
        depth = 0
        k = j
        while k < n:
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[j + 1:k]
        i = k + 1
        if etype in _SKIP_TYPES or not etype:
            continue
        entry = _parse_entry(etype, body, clean)
        if entry:
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_bib() -> Path:
    # scripts/ lives inside the repo root that also holds references.bib
    return Path(__file__).resolve().parent.parent / "references.bib"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a .bib file into JSON.")
    ap.add_argument("bib", nargs="?", default=None,
                    help="input .bib (default: references.bib next to the repo root)")
    ap.add_argument("-o", "--output", default=None,
                    help="output .json path (default: stdout)")
    ap.add_argument("--by-key", action="store_true",
                    help="emit an object keyed by cite key instead of an array")
    ap.add_argument("--raw", action="store_true",
                    help="keep values verbatim (no LaTeX->Unicode cleaning, keep braces)")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    ap.add_argument("--ascii", action="store_true",
                    help="escape non-ASCII in output (default: keep Unicode)")
    args = ap.parse_args(argv)

    bib_path = Path(args.bib) if args.bib else _default_bib()
    if not bib_path.is_file():
        print(f"error: bib file not found: {bib_path}", file=sys.stderr)
        return 1

    entries = parse_bib(bib_path.read_text(encoding="utf-8"), clean=not args.raw)

    if args.by_key:
        payload: object = {}
        for e in entries:
            key = e["key"]
            if key in payload:
                print(f"warning: duplicate cite key '{key}' (last one wins)",
                      file=sys.stderr)
            payload[key] = {k: v for k, v in e.items() if k != "key"}
    else:
        payload = entries

    text = json.dumps(payload, indent=args.indent, ensure_ascii=args.ascii)

    if args.output:
        out = Path(args.output)
        out.write_text(text + "\n", encoding="utf-8")
        dest = str(out)
    else:
        sys.stdout.write(text + "\n")
        dest = "stdout"

    by_type: Dict[str, int] = {}
    for e in entries:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    summary = ", ".join(f"{t}={c}" for t, c in sorted(by_type.items()))
    print(f"parsed {len(entries)} entries ({summary}) from {bib_path.name} -> {dest}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
