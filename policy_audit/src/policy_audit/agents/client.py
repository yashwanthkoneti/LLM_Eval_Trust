"""One LLM client, with the failure modes we have actually hit.

Three of them cost real time before:

  Truncated JSON. A reply cut off mid-object is not a model failure, it
  is a budget failure, and 42% of one run died this way. The fix is to
  repair what arrived and retry with a larger budget rather than discard
  the row.

  Parameter naming. Some endpoints take max_tokens, some
  max_completion_tokens. Probing once at startup beats failing on row 1.

  Admission-time reservation. Databricks reserves max_tokens against the
  per-minute budget when the call is admitted, not when it completes, so
  a generous ceiling throttles throughput even when replies are short.
  Budgets here start tight and escalate only on truncation.

The Protocol at the bottom is what the tests use: the pipeline depends on
the interface, not on any vendor, so an end-to-end test needs no endpoint.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class LLM(Protocol):
    """What the agents need. Anything satisfying this will do."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Recover an object from a reply that was cut off.

    Walks the text tracking brace depth while ignoring braces inside
    strings, truncates at the last balanced point and closes what is
    still open. A reply that stopped halfway through row 40 of 46 still
    carries 39 usable decisions, and throwing it away was costing more
    than the truncation.
    """
    if not text:
        return None
    s = _FENCE.sub("", str(text)).strip()
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    try:
        return json.loads(s)
    except Exception:
        pass

    depth, in_str, esc, last_good = 0, False, False, None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = i + 1

    if last_good:
        try:
            return json.loads(s[:last_good])
        except Exception:
            pass

    # close what is open: drop any partial trailing entry first
    head = s[:last_good] if last_good else s
    head = re.sub(r",\s*\"[^\"]*\"\s*:\s*(?:\{[^{}]*)?$", "", head)
    head = re.sub(r",\s*$", "", head)
    for closing in ("}", "}}", "}}}", "]}", "}]}"):
        try:
            return json.loads(head + closing)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# throttling and retry
# --------------------------------------------------------------------------
@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    truncations: int = 0
    failures: int = 0


class JsonAgent:
    """Wraps an LLM to return parsed JSON, escalating budget on truncation."""

    def __init__(self, llm: LLM, max_tokens: int, *, max_retries: int = 3,
                 retry_base_delay: float = 2.0,
                 min_seconds_between_calls: float = 0.0,
                 name: str = "agent") -> None:
        self.llm = llm
        self.base_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_gap = min_seconds_between_calls
        self.name = name
        self.stats = CallStats()
        self._last_call = 0.0

    def _wait(self) -> None:
        if self.min_gap <= 0:
            return
        gap = time.monotonic() - self._last_call
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)

    def ask(self, system: str, user: str,
            expect: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return parsed JSON, or {} with the reason under "_error".

        expect lists keys that must be present; a reply missing them is
        treated as truncated and retried with a larger budget, which is
        what distinguishes "the model disagreed" from "the reply was cut".
        """
        budget = self.base_tokens
        last_error = "no attempt made"

        for attempt in range(self.max_retries):
            self._wait()
            try:
                raw = self.llm.complete(system, user, budget)
                self._last_call = time.monotonic()
                self.stats.calls += 1
            except Exception as exc:                 # transport, rate limit
                last_error = f"{type(exc).__name__}: {exc}"
                self.stats.retries += 1
                time.sleep(self.retry_base_delay * (2 ** attempt))
                continue

            parsed = repair_truncated_json(raw)
            if parsed is None:
                last_error = "reply contained no recoverable JSON"
                self.stats.truncations += 1
                budget = min(budget * 2, self.base_tokens * 4)
                continue

            if expect:
                missing = [k for k in expect if k not in parsed]
                if missing:
                    last_error = f"missing keys: {missing}"
                    self.stats.truncations += 1
                    budget = min(budget * 2, self.base_tokens * 4)
                    # keep what arrived: a partial decision set still beats
                    # discarding the row entirely
                    if attempt == self.max_retries - 1:
                        parsed["_partial"] = missing
                        return parsed
                    continue

            return parsed

        self.stats.failures += 1
        return {"_error": f"{self.name}: {last_error}"}


# --------------------------------------------------------------------------
# Databricks
# --------------------------------------------------------------------------
class DatabricksLLM:
    """ChatDatabricks, with the token-parameter name probed once."""

    def __init__(self, endpoint: str, temperature: float = 0.0) -> None:
        try:
            from databricks_langchain import ChatDatabricks
        except ImportError:                          # older package name
            from langchain_databricks import ChatDatabricks  # type: ignore
        self._ChatDatabricks = ChatDatabricks
        self.endpoint = endpoint
        self.temperature = temperature
        self._token_kw: Optional[str] = None

    def _build(self, max_tokens: int):
        kws = ([self._token_kw] if self._token_kw
               else ["max_tokens", "max_completion_tokens"])
        last: Exception | None = None
        for kw in kws:
            try:
                model = self._ChatDatabricks(
                    endpoint=self.endpoint,
                    temperature=self.temperature,
                    **{kw: max_tokens},
                )
                self._token_kw = kw
                return model
            except TypeError as exc:
                last = exc
        raise RuntimeError(
            f"endpoint {self.endpoint} accepted neither max_tokens nor "
            f"max_completion_tokens: {last}")

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        model = self._build(max_tokens)
        reply = model.invoke([("system", system), ("user", user)])
        return getattr(reply, "content", str(reply))

    def smoke_test(self) -> None:
        """Fail at startup, not on row 1."""
        out = self.complete("Reply with JSON only.",
                            'Return exactly {"ok": true}', 64)
        if repair_truncated_json(out) is None:
            raise RuntimeError(
                f"endpoint {self.endpoint} did not return parseable JSON: "
                f"{out[:200]}")


# --------------------------------------------------------------------------
# testing
# --------------------------------------------------------------------------
@dataclass
class ScriptedLLM:
    """Returns canned replies in order. Lets the pipeline be tested whole."""

    replies: List[str] = field(default_factory=list)
    seen: List[str] = field(default_factory=list)
    default: str = "{}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.seen.append(user)
        if self.replies:
            return self.replies.pop(0)
        return self.default
