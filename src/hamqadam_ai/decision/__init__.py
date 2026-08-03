"""MODULE 10 - turning evidence into a recommendation.

A recommendation, never an outcome. Approval needs every condition; rejection
needs any one; everything between is a human's problem. The middle is wide on
purpose, because this project cannot validate its own thresholds and automating
a rejection on an uncalibrated one is how an honest applicant gets locked out
with no recourse.
"""

from hamqadam_ai.decision.engine import (
    BLOCKING_CONDITIONS,
    DecisionEngine,
    DecisionOutcome,
    DecisionReason,
)

__all__ = [
    "BLOCKING_CONDITIONS",
    "DecisionEngine",
    "DecisionOutcome",
    "DecisionReason",
]
