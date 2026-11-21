"""The labels an audit can produce, and nothing else.

Two layers, and keeping them apart is the whole point:

  THEME   what the USER experienced. Five of them, mutually comprehensible
          to anyone, defensible by quoting the request and the response.
          This is what goes in front of stakeholders.

  FACTOR  what the SYSTEM did that produced it. An inference about
          mechanism, which routes the fix to an owner.

A theme is an observation; a factor is an explanation. The two were
previously fused into one list of names, which meant a label like
"intent misinterpretation" asserted a mechanism while resting on evidence
that could only support an observation - and could not be defended in
review. Themes now carry no mechanism claim, and factors are never the
headline.

Every finding carries exactly one theme and zero or more factors.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Tuple


class Theme(str, Enum):
    """What the user experienced. The primary label."""

    INCORRECT = "Incorrect"
    INCOMPLETE = "Incomplete"
    IRRELEVANT = "Irrelevant"
    INCONSISTENT = "Inconsistent"
    UNSUBSTANTIATED = "Unsubstantiated"


THEME_MEANING: Dict[Theme, str] = {
    Theme.INCORRECT: "The answer states something the policy contradicts",
    Theme.INCOMPLETE: "The answer leaves out something the user asked for",
    Theme.IRRELEVANT: "The user did not get their question answered",
    Theme.INCONSISTENT: (
        "The answer conflicts with the policy set or with an earlier turn"
    ),
    Theme.UNSUBSTANTIATED: "The answer is not supported by the policy text",
}

# How to say it to someone who was not in the room. A theme that cannot be
# explained in one sentence has no business being the headline.
THEME_SENTENCE: Dict[Theme, str] = {
    Theme.INCORRECT: "The answer says {claim}; the policy says {source}.",
    Theme.INCOMPLETE: "The user asked about {asked}; the answer covers {covered}.",
    Theme.IRRELEVANT: "The user asked {asked}; the response {did} instead.",
    Theme.INCONSISTENT: "The answer says {claim}, which conflicts with {source}.",
    Theme.UNSUBSTANTIATED: (
        "The answer states {claim} with no policy text behind it."
    ),
}


class Factor(str, Enum):
    """Why it happened. Secondary, and never reported on its own."""

    # request understanding
    INTENT_MISREAD = "intent_misread"
    SCOPE_TOO_NARROW = "scope_too_narrow"
    ACTION_DETECTOR_MISFIRE = "action_detector_misfire"

    # context
    CONTEXT_CONTINUITY_FAILURE = "context_continuity_failure"
    ASKBACK_CONTEXT_LOST = "askback_context_lost"

    # retrieval and corpus
    RETRIEVAL_COVERAGE_GAP = "retrieval_coverage_gap"
    RETRIEVAL_PRECISION_MISS = "retrieval_precision_miss"
    RETRIEVAL_APPLICABILITY_MISS = "retrieval_applicability_miss"
    NO_RETRIEVAL = "no_retrieval"

    # routing decision
    UNNECESSARY_CLARIFICATION = "unnecessary_clarification"
    EXCESSIVE_GUARDRAIL = "excessive_guardrail"
    MISSING_CLARIFICATION = "missing_clarification"

    # generation
    TABLE_MISREAD = "table_misread"
    CONDITION_DROPPED = "condition_dropped"
    UNGROUNDED_GENERATION = "ungrounded_generation"
    SINGLE_BRANCH_PRESENTED_AS_RULE = "single_branch_presented_as_rule"


FACTOR_OWNER: Dict[Factor, str] = {
    Factor.INTENT_MISREAD: "intent detection / routing prompt",
    Factor.SCOPE_TOO_NARROW: "scope definition",
    Factor.ACTION_DETECTOR_MISFIRE: "action detector",
    Factor.CONTEXT_CONTINUITY_FAILURE: "conversation memory",
    Factor.ASKBACK_CONTEXT_LOST: "askback handling",
    Factor.RETRIEVAL_COVERAGE_GAP: "corpus owners",
    Factor.RETRIEVAL_PRECISION_MISS: "retrieval tuning",
    Factor.RETRIEVAL_APPLICABILITY_MISS: "retrieval tuning",
    Factor.NO_RETRIEVAL: "retrieval pipeline",
    Factor.UNNECESSARY_CLARIFICATION: "routing prompt",
    Factor.EXCESSIVE_GUARDRAIL: "guardrail prompt",
    Factor.MISSING_CLARIFICATION: "routing prompt",
    Factor.TABLE_MISREAD: "response prompt",
    Factor.CONDITION_DROPPED: "response prompt",
    Factor.UNGROUNDED_GENERATION: "response prompt",
    Factor.SINGLE_BRANCH_PRESENTED_AS_RULE: "response prompt",
}


class Gate(str, Enum):
    """The four questions, asked in this order.

    Each gate can end the audit. A request that was correctly refused
    never reaches the answer gate, so the answer gate cannot invent a
    defect about an answer that does not exist - which is how
    "answer_incomplete" used to attach itself to clarifications.
    """

    REQUEST = "request"      # what was asked
    CONTEXT = "context"      # what the agent had to work with
    ROUTE = "route"          # what the agent chose to do
    ANSWER = "answer"        # whether the answer holds up


GATE_ORDER: Tuple[Gate, ...] = (Gate.REQUEST, Gate.CONTEXT, Gate.ROUTE,
                                Gate.ANSWER)

# Which gates share a call. Four gates, three calls: context and route
# reason over the same sources - the conversation, the request and the
# response, and neither needs the policy text - so asking them separately
# bought a second system prompt and a second copy of the response for
# nothing. The gates stay distinct in the taxonomy and in the output; only
# the call is shared.
CALL_PLAN: Tuple[Tuple[Gate, ...], ...] = (
    (Gate.REQUEST,),
    (Gate.CONTEXT, Gate.ROUTE),
    (Gate.ANSWER,),
)


class Route(str, Enum):
    ANSWERED = "answered"
    CLARIFIED = "clarified"
    REFUSED_SCOPE = "refused_scope"
    REFUSED_ACTION = "refused_action"
    STATIC = "static"


# Routes where no answer was given, so the answer gate does not run.
NON_ANSWER_ROUTES: FrozenSet[Route] = frozenset({
    Route.CLARIFIED, Route.REFUSED_SCOPE, Route.REFUSED_ACTION, Route.STATIC,
})


class Suspicion(str, Enum):
    """What the cheap screener can say about a check.

    Deliberately three-valued, and deliberately asymmetric in meaning:

      CLEAR     nothing here. Final - this is where the cost saving comes
                from, so the screener's bar for CLEAR is high.
      SUSPECT   something might be wrong. NOT a finding. Escalates.
      UNSURE    cannot tell from what I was given. Escalates, because
                "I could not tell" must never silently become "clean".
    """

    CLEAR = "clear"
    SUSPECT = "suspect"
    UNSURE = "unsure"


ESCALATING: FrozenSet[Suspicion] = frozenset({Suspicion.SUSPECT,
                                              Suspicion.UNSURE})


class Adjudication(str, Enum):
    """What the costly adjudicator can say. Its word is final.

      CONFIRMED  a defect, with a quoted span on both sides
      REJECTED   the screener was wrong, and here is why
      UNPROVEN   may well be real, but it cannot be evidenced from the
                 text, so it is not reported as a defect
    """

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNPROVEN = "unproven"


# Only CONFIRMED produces a finding. UNPROVEN is recorded for calibration
# but never reported as a defect, because a finding a reviewer cannot
# verify costs more than the defect it might have caught.
REPORTABLE: FrozenSet[Adjudication] = frozenset({Adjudication.CONFIRMED})
