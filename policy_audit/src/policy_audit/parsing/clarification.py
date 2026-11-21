"""Clarification template parsing, and askback detection.

The live template is:

    <preamble, containing a marker such as "please clarify">

    1. Question ?

    option a: ...
    option b: ...
    option c: ...

So a NUMBERED line is a question and an `option x:` line is an option.
That precedence matters: treating any `1.` as a possible option marker
swallows a question whose text does not end in `?`, and the question
list comes back empty.

Askback matters more than it looks. When a turn answers the agent's own
clarifying question, the user has NARROWED the request - they have just
said which case they want. Reading the original broad question as still
in scope then produces a false "incomplete answer" on exactly the rows
where the agent did the right thing.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

from ..domain.models import Clarification, Turn

_Q_LINE = re.compile(
    r"^\s*(?:\(?(\d{1,2})\s*[\).:]|[-*•●])\s*(.+?)\s*$")
_O_LINE = re.compile(
    r"^\s*(?:option\s*([0-9a-zA-Z])\s*[:.\)]|\(?([a-hA-H])\s*[\).]|"
    r"[-*•●])\s*(.*)$")


def parse_clarification(response: str,
                        markers: Sequence[str]) -> Clarification:
    """Pull the template apart. An option's text may be empty."""
    text = str(response or "")
    low = text.lower()
    marker = next((m for m in markers if m.lower() in low), "")

    questions: list[tuple[str, list[str]]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        om = _O_LINE.match(line)
        is_option = bool(om) and bool(
            om.group(1) or om.group(2) or line[:1] in "-*•●")
        qm = _Q_LINE.match(line)
        # a numbered line is a question in this template, never an option
        if qm and qm.group(1):
            questions.append((qm.group(2).strip(), []))
            continue
        if is_option:
            if not questions:
                questions.append(("", []))
            questions[-1][1].append((om.group(3) or "").strip())
            continue
        if line.endswith("?"):
            questions.append((line, []))

    return Clarification(
        marker=marker,
        questions=tuple((q, tuple(o)) for q, o in questions),
    )


def previous_assistant_turn(history: Sequence[Turn]) -> str:
    for turn in reversed(history):
        if turn.role == "assistant":
            return turn.content
    return ""


def detect_askback(history: Sequence[Turn],
                   markers: Sequence[str]) -> Tuple[bool, str]:
    """Is this turn the user answering the agent's clarifying question?

    Entirely deterministic - it reads the previous assistant turn for the
    clarification marker. No judge involvement, so the result is stable
    and can be fed to the judge as a fact rather than inferred by it.

    Returns (is_askback, the question that was asked).
    """
    prev = previous_assistant_turn(history)
    if not prev:
        return False, ""
    parsed = parse_clarification(prev, markers)
    if not parsed.is_clarification:
        return False, ""
    asked = " / ".join(q for q, _ in parsed.questions if q)
    return True, asked


def answers_an_option(request: str, history: Sequence[Turn],
                      markers: Sequence[str]) -> Optional[str]:
    """Which offered option this turn appears to pick, if any.

    Raises confidence in the askback reading from "the previous turn was
    a clarification" to "and this turn names one of the choices".
    """
    prev = previous_assistant_turn(history)
    if not prev:
        return None
    parsed = parse_clarification(prev, markers)
    if not parsed.is_clarification:
        return None
    req = re.sub(r"[^a-z0-9 ]+", " ", str(request).lower())
    req_words = set(req.split())
    best, best_score = None, 0.0
    for _, options in parsed.questions:
        for opt in options:
            words = {w for w in re.sub(r"[^a-z0-9 ]+", " ", opt.lower()).split()
                     if len(w) > 2}
            if not words:
                continue
            score = len(words & req_words) / len(words)
            if score > best_score:
                best, best_score = opt, score
    return best if best_score >= 0.5 else None
