"""Turn raw rows into Interactions. All deterministic, no model calls.

Everything the agents need is computed here: history per conversation,
parsed chunks, the ontology, the clarification template, the route, and
whether this turn answers the agent's own question.

Doing it up front means the agents are handed facts rather than asked to
infer them, and it means the whole preparation layer is testable without
an endpoint.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..config import Settings
from ..domain.models import Interaction, Turn
from ..parsing.chunks import parse_chunks
from ..parsing.clarification import (answers_an_option, detect_askback,
                                     parse_clarification)
from ..parsing.ontology import build_ontology
from ..parsing.routing import detect_route


def _col(df: pd.DataFrame, name: str) -> str | None:
    """Hive lowercases column names; match case-insensitively."""
    lookup = {str(c).lower(): c for c in df.columns}
    return lookup.get(str(name).lower())


def build_histories(df: pd.DataFrame,
                    settings: Settings) -> Dict[str, List[Turn]]:
    """History per row: strictly EARLIER turns of the same conversation.

    Strictly earlier matters. Including the current turn lets the judge
    see the response it is meant to be auditing as context, and including
    later turns lets it see information the agent did not have.
    """
    tid = _col(df, settings.col_transaction_id)
    cid = _col(df, settings.col_conversation_id)
    created = _col(df, settings.col_created_on)
    req = _col(df, settings.col_request)
    resp = _col(df, settings.col_response)

    if not tid or not req:
        raise ValueError(
            f"input needs at least {settings.col_transaction_id} and "
            f"{settings.col_request}; found {list(df.columns)}")

    work = df.copy()
    if cid is None:
        # no conversation id: every row stands alone, which is safer than
        # guessing a grouping and inventing continuity that was not there
        work["__cid"] = work[tid].astype(str)
        cid = "__cid"
    else:
        # a blank conversation id is a single-turn conversation, not a
        # group of all the blanks
        blank = work[cid].isna() | (work[cid].astype(str).str.strip() == "")
        work[cid] = work[cid].astype(str)
        work.loc[blank, cid] = "__solo_" + work.loc[blank, tid].astype(str)

    if created is not None:
        work["__order"] = pd.to_datetime(work[created], errors="coerce")
    else:
        work["__order"] = range(len(work))

    histories: Dict[str, List[Turn]] = {}
    for _, group in work.groupby(cid, sort=False):
        group = group.sort_values("__order", kind="stable")
        turns: List[Turn] = []
        for _, row in group.iterrows():
            histories[str(row[tid])] = list(turns)
            turns.append(Turn("user", str(row[req] or "")))
            if resp is not None:
                turns.append(Turn("assistant", str(row[resp] or "")))
    return histories


def prepare(df: pd.DataFrame, settings: Settings) -> List[Interaction]:
    tid = _col(df, settings.col_transaction_id)
    cid = _col(df, settings.col_conversation_id)
    req = _col(df, settings.col_request)
    resp = _col(df, settings.col_response)
    chunks_col = _col(df, settings.col_chunks)

    histories = build_histories(df, settings)
    out: List[Interaction] = []

    for raw in df.to_dict("records"):
        transaction_id = str(raw.get(tid) or "")
        request = str(raw.get(req) or "")
        response = str(raw.get(resp) or "") if resp else ""
        history = tuple(histories.get(transaction_id, ()))

        chunks = tuple(parse_chunks(raw.get(chunks_col))
                       if chunks_col else ())
        ontology = build_ontology(chunks)
        clarification = parse_clarification(response, settings.clarify_markers)
        route = detect_route(response, clarification,
                             getattr(settings, "static_responses", ()))

        is_askback, asked = detect_askback(history, settings.clarify_markers)
        picked = (answers_an_option(request, history, settings.clarify_markers)
                  if is_askback else None)
        if picked:
            asked = f"{asked}  (the user appears to have chosen: {picked})"

        out.append(Interaction(
            transaction_id=transaction_id,
            conversation_id=str(raw.get(cid) or "") if cid else "",
            request=request,
            response=response,
            history=history,
            chunks=chunks,
            ontology=ontology,
            route=route,
            clarification=clarification,
            is_askback_answer=is_askback,
            askback_question=asked,
            raw=raw,
        ))
    return out


def preflight(interactions: Sequence[Interaction],
              settings: Settings) -> Dict[str, object]:
    """What to look at before spending a single token.

    Each of these has silently invalidated a run before: a clarify-marker
    list that matched nothing emptied a whole theme, and chunk parsing
    that lost every title took the document name out of every finding.
    """
    from ..parsing.routing import route_diagnostics

    total = max(len(interactions), 1)
    with_chunks = sum(1 for i in interactions if i.chunks)
    titled = sum(1 for i in interactions
                 if i.chunks and any(c.doc_title != "?" for c in i.chunks))
    with_tables = sum(1 for i in interactions
                      if any(c.tables for c in i.chunks))
    clarifications = sum(1 for i in interactions
                         if i.clarification
                         and i.clarification.is_clarification)
    askbacks = sum(1 for i in interactions if i.is_askback_answer)
    conditional = sum(1 for i in interactions
                      if i.ontology.conditional_dimensions)

    warnings: List[str] = []
    if with_chunks and titled == 0:
        warnings.append(
            "no chunk resolved a document title. The title keys in the "
            "live format are not being recognised, which strips the "
            "document name out of every finding.")
    if not clarifications:
        warnings.append(
            f"no response matched clarify_markers={list(settings.clarify_markers)}. "
            f"If the agent does ask clarifying questions, every "
            f"clarification check will be skipped and askback turns will "
            f"be misread as fresh requests.")
    if with_chunks / total < 0.3:
        warnings.append(
            f"only {100 * with_chunks / total:.0f}% of rows have retrieved "
            f"text. Checks that rest on policy text cannot be confirmed on "
            f"the rest, so they will come out as expected behaviour.")

    diag = route_diagnostics([i.route for i in interactions])
    warnings.extend(diag["warnings"])            # type: ignore[arg-type]

    return {
        "rows": len(interactions),
        "with_retrieval": with_chunks,
        "with_resolved_title": titled,
        "with_tables": with_tables,
        "clarification_responses": clarifications,
        "askback_turns": askbacks,
        "rows_with_conditional_policy": conditional,
        "routes": diag["counts"],
        "warnings": warnings,
    }
