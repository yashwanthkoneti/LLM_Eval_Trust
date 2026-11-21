"""Every tunable in one place, overridable by environment variable.

Nothing in the package reads os.environ directly; it reads these. That
keeps a notebook, a test and a scheduled job on the same settings
object instead of three copies drifting apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

ENV_FILE = os.environ.get("PA_ENV_FILE", ".env")


def load_env_file(path: str = ENV_FILE, *, override: bool = False) -> int:
    """Read KEY=VALUE lines into os.environ. Returns how many were set.

    A deliberately small parser rather than a dependency: comments, blank
    lines, optional `export`, and quotes stripped. Values already in the
    real environment win unless override is asked for, so a one-off
    PA_LIMIT=10 on the command line still works without editing the file.
    """
    file = Path(path)
    if not file.is_file():
        return 0
    count = 0
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def _env(name: str, default):
    raw = os.environ.get(f"PA_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, tuple):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return raw


@dataclass
class Settings:
    # ---- data -------------------------------------------------------
    input_path: str = "data/audit_input.csv"
    output_dir: str = "out"
    checkpoint_path: str = "out/checkpoint.jsonl"

    col_transaction_id: str = "transaction_id"
    col_conversation_id: str = "conversation_id"
    col_created_on: str = "request_created_on"
    col_request: str = "request"
    col_response: str = "response"
    col_chunks: str = "document_chunks"

    # ---- models -----------------------------------------------------
    # The screener runs on every row and does the heavy lifting, so it is
    # the small model. The adjudicator runs only on escalations, which
    # should be a minority of rows, so it can afford to be the large one.
    screener_model: str = "gpt-5-mini"
    adjudicator_model: str = "gpt-5"
    screener_max_tokens: int = 1200
    adjudicator_max_tokens: int = 1400
    temperature: float = 0.0

    # OPENAI_API_KEY is read from the environment by the client library
    # itself, so it belongs in .env but not in this dataclass - keeping a
    # key out of a settings object that gets printed and serialised is
    # worth the small inconsistency.
    openai_base_url: str = ""            # for a proxy or Azure gateway
    request_timeout: int = 120
    # both agents must return a JSON object; where the model supports it,
    # asking for JSON mode removes most malformed replies at the source
    json_mode: bool = True

    # Budgets start tight and the client doubles them only on a truncated
    # reply, rather than reserving a generous ceiling on every call.
    max_retries: int = 3
    retry_base_delay: float = 2.0
    min_seconds_between_calls: float = 0.0

    # ---- clarification template -------------------------------------
    # The live template's own wording. The loader reports how many rows
    # carry one, so a wrong list shows up as a count of zero before a
    # single token is spent rather than as a silently empty theme.
    clarify_markers: Tuple[str, ...] = ("please clarify",)
    clarify_expected_questions: int = 3
    clarify_expected_options: int = 3

    # ---- static responses -------------------------------------------
    # The canned replies that end a conversation. These are dead ends and
    # are worth evaluating separately for whether the guardrail was too
    # eager, so they get their own route rather than being read as
    # refusals. Replace with the live wording.
    static_responses: Tuple[str, ...] = (
        "I am here to assist with policy related documents",
        "The question could directly determine some outcome",
    )

    # ---- spans ------------------------------------------------------
    # How much text travels to the adjudicator around each pointer. The
    # whole point of pointers is that this stays small.
    span_window_chars: int = 480
    max_spans_per_check: int = 3
    span_min_match: float = 0.55
    # Semantic span location, OFF by default. It is a FALLBACK for when
    # the lexical locator cannot find the screener's pointer, so the
    # embedding bill tracks the failure rate rather than the row count.
    # Locating by meaning is safe; VERIFYING by meaning is not, and
    # grounding stays lexical whatever this is set to.
    semantic_spans: bool = False
    embedding_model: str = "text-embedding-3-small"

    # ---- adjudication -----------------------------------------------
    # A confirmed finding whose quote cannot be located in the source it
    # claims is downgraded. Set strict_grounding False to record those as
    # ungrounded rather than dropping them, for one calibration run.
    strict_grounding: bool = True
    require_readable_evidence: bool = True

    # ---- review sample ----------------------------------------------
    # A defect the audit cannot substantiate is recorded as expected
    # behaviour, on purpose - an unverifiable finding costs a reviewer
    # more than the defect it might have caught. The consequence is that
    # every miss leaves silently, so a slice of the clean rows is sampled
    # into the queue to measure how often that happens.
    #
    # Stratified, because "expected behaviour" is not one thing: a row
    # nothing was suspected on and a row where a suspicion could not be
    # settled carry very different doubt. Sampling each stratum gives a
    # miss rate per stratum, which is actionable; a flat sample does not.
    sample_rate: float = 0.10
    # at 10% a stratum of six rows draws nothing, and the strata carrying
    # most doubt are usually the small ones. The floor makes each one
    # measurable.
    sample_min_per_stratum: int = 2
    sample_max_per_stratum: int = 0      # 0 = uncapped

    # ---- run --------------------------------------------------------
    batch_size: int = 25
    limit: int = 0                     # 0 = all rows
    dry_run: bool = False
    preflight_only: bool = False
    resume: bool = True

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Settings":
        """Build from the .env file plus the real environment."""
        load_env_file(env_file or ENV_FILE)
        kw = {}
        for f in fields(cls):
            kw[f.name] = _env(f.name, getattr(cls, f.name))
        return cls(**kw)


def reload(env_file: Optional[str] = None) -> "Settings":
    """Re-read configuration. Useful in a notebook after editing .env."""
    global SETTINGS
    SETTINGS = Settings.from_env(env_file)
    return SETTINGS


SETTINGS = Settings.from_env()
