"""Pipe tables, parsed positionally.

The format is markdown-ish: |a|b|c| per row, sometimes with a |---|---|
separator, sometimes with leading or trailing empty cells.

The trap is `.strip("|")`, which is greedy. On `||Amount|Registration|`
it eats the empty first cell, every column shifts left by one, and each
header then labels its neighbour's value:

    Registration=$25-$200      <- wrong column
    ?=No                       <- fell off the end

That is worse than losing the header, because it reads as confirmation.
Strip exactly one pipe from each end and an empty cell keeps its place.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..domain.models import Table

PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# a tab-delimited row with at least two columns. Cheap to support and it
# is what a copy out of a spreadsheet produces.
TAB_ROW = re.compile(r"^[^\t\n]*\t[^\t\n]*(?:\t|$)")
# A table shape this parser cannot structure: HTML, or space-aligned
# columns. It still reaches the judge as prose - which is the right
# fallback, since reading it unaided beats not seeing it - but the judge
# is told so, because an unstructured table is where a misread hides.
UNSTRUCTURED_TABLE = re.compile(
    r"<\s*table\b|<\s*t[rdh]\b"                    # html
    r"|^[^\n|\t]{2,30}\s{3,}[^\n|\t]{2,30}\s*$",   # space aligned
    re.I | re.M)
# |---|:--:|---| and friends: separator chars only, and at least one dash
_SEP_ROW = re.compile(r"^[\s|:-]*-[\s|:-]*$")
_ESCAPED = "\x00"


def is_pipe_row(line: str) -> bool:
    return bool(PIPE_ROW.match(line))


def is_table_row(line: str) -> bool:
    """A pipe row, or a tab-delimited one."""
    return bool(PIPE_ROW.match(line)) or bool(TAB_ROW.match(line))


def split_row(line: str) -> List[str]:
    """One pipe row -> cells, position preserved.

    An escaped `\\|` inside a cell is data, not a boundary.
    """
    raw = str(line)
    if "|" not in raw and "\t" in raw:
        return [c.strip() for c in raw.rstrip("\n").split("\t")]
    s = raw.strip().replace(r"\|", _ESCAPED)
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.replace(_ESCAPED, "|").strip() for c in s.split("|")]


def parse_table(lines: Sequence[str]) -> Table | None:
    """Pipe rows -> a Table with every row the same width.

    Rows are padded, never truncated: a short row keeps its cells in
    their original columns instead of sliding left. A blank header
    becomes col1, col2 so the position stays visible.
    """
    rows = [split_row(l) for l in lines if not _SEP_ROW.match(str(l).strip())]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    padded = [tuple(r + [""] * (width - len(r))) for r in rows]
    header = tuple(h if h else f"col{i + 1}"
                   for i, h in enumerate(padded[0]))
    return Table(header=header, rows=tuple(padded[1:]))


def split_segments(value: str) -> List[Tuple[str, object]]:
    """Walk a field value and return ordered ('text', s) / ('table', Table).

    Order matters. A sentence printed after a table must stay after it,
    or a qualifier such as "the APAC row is superseded from July" lands
    against the wrong rows and the audit inherits the misreading it is
    looking for.
    """
    segs: List[Tuple[str, object]] = []
    run: List[str] = []
    buf: List[str] = []

    def flush_table():
        table = parse_table(run)
        if table is not None:
            segs.append(("table", table))
        else:
            buf.extend(run)          # a lone pipe row is not a table
        run.clear()

    def flush_text():
        s = "\n".join(buf).strip()
        if s:
            segs.append(("text", s))
        buf.clear()

    lines = str(value).split("\n")
    for i, line in enumerate(lines):
        if is_table_row(line):
            if buf:
                flush_text()
            run.append(line)
            continue
        # A blank line inside a table does not end it. Treating it as a
        # boundary split the header off the body, and a header alone is
        # not a table, so the whole thing fell through to prose and the
        # judge lost the column bindings.
        if run and not line.strip():
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if is_table_row(nxt):
                continue
        if run:
            flush_table()
        buf.append(line)
    if run:
        flush_table()
    flush_text()
    return segs


def render_table(table: Table) -> str:
    """Re-emit a table so a quote of one row carries its own meaning."""
    head = " | ".join(table.header)
    lines = [f"TABLE  columns: {head}",
             "  (a value belongs ONLY to the row it is printed under)"]
    lines.extend("  " + r for r in table.as_rows())
    return "\n".join(lines)


def looks_like_unstructured_table(text: str) -> bool:
    """A table we could not parse, so the judge should read it carefully.

    Two space-aligned-looking lines in a row, or any HTML table markup.
    One such line is a coincidence; two is a layout.
    """
    body = str(text or "")
    if re.search(r"<\s*table\b|<\s*t[rdh]\b", body, re.I):
        return True
    runs = 0
    for line in body.split("\n"):
        if re.match(r"^[^\n|\t]{2,30}\s{3,}[^\n|\t]{2,30}\s*$", line):
            runs += 1
            if runs >= 2:
                return True
        else:
            runs = 0
    return False
