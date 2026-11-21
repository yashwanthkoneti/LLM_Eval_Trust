"""Entry point. No arguments - everything comes from .env.

    policy-audit
    python -m policy_audit.pipeline.cli

Preflight runs before any model call and prints what it found. Every
warning it prints has silently invalidated a run before, so it is worth
reading rather than scrolling past: a clarify_markers list that matches
nothing empties a whole category and makes every askback turn look like
a fresh request, and chunk parsing that loses every title takes the
document name out of every finding.
"""

from __future__ import annotations

import sys

import pandas as pd

from ..agents.client import OpenAILLM
from ..config import ENV_FILE, SETTINGS, Settings
from ..reporting.export import export, print_summary
from .prepare import prepare, preflight
from .runner import Runner, read_checkpoint_ids


def load(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run(settings: Settings = SETTINGS) -> int:
    print(f"config from {ENV_FILE}")
    print(f"reading     {settings.input_path}")
    df = load(settings.input_path)

    # Rows already in the checkpoint need no preparation. History is
    # still built from the whole frame; only their chunk parsing and
    # ontology are skipped, which is the part that costs anything.
    only = None
    if settings.resume:
        done = read_checkpoint_ids(settings.checkpoint_path)
        if done:
            tid = settings.col_transaction_id
            col = next((c for c in df.columns
                        if str(c).lower() == tid.lower()), tid)
            todo = {str(v) for v in df[col]} - done
            only = todo
            print(f"checkpoint  {len(done)} row(s) already audited, "
                  f"{len(todo)} to go")

    interactions = prepare(df, settings, only=only)
    if only is not None and not interactions:
        print("\nnothing left to audit; delete the checkpoint to start over")
        return 0

    report = preflight(interactions, settings)
    print(f"\nrows                      {report['rows']}")
    print(f"with retrieval            {report['with_retrieval']}")
    print(f"  of those, titled        {report['with_resolved_title']}")
    print(f"  with a table            {report['with_tables']}")
    print(f"clarification responses   {report['clarification_responses']}")
    print(f"askback turns             {report['askback_turns']}")
    print(f"conditional policy rows   "
          f"{report['rows_with_conditional_policy']}")
    print(f"routes                    {report['routes']}")
    for warning in report["warnings"]:                  # type: ignore
        print(f"\nWARNING: {warning}")

    if settings.preflight_only:
        print("\nPA_PREFLIGHT_ONLY is set; no model calls made")
        return 0

    if report["warnings"]:
        print("\nproceeding with the warnings above. Fix them first if any "
              "look serious - they are cheaper to fix than a run.")

    print("\nchecking the models before spending anything on rows:")
    screener = OpenAILLM(settings.screener_model, settings.temperature,
                         json_mode=settings.json_mode,
                         timeout=settings.request_timeout,
                         base_url=settings.openai_base_url or None)
    adjudicator = OpenAILLM(settings.adjudicator_model, settings.temperature,
                            json_mode=settings.json_mode,
                            timeout=settings.request_timeout,
                            base_url=settings.openai_base_url or None)
    screener.smoke_test()
    if not settings.dry_run:
        adjudicator.smoke_test()

    runner = Runner(screener, adjudicator, settings)
    results = runner.run(interactions, resume=settings.resume)

    out = export(results, interactions, settings.output_dir)
    print_summary(out["summary"])                       # type: ignore
    print(f"\ncall stats {runner.call_stats}")
    for path in out["paths"].values():                  # type: ignore
        print(f"wrote {path}")
    return 0


def main() -> int:
    return run(SETTINGS)


if __name__ == "__main__":
    sys.exit(main())
