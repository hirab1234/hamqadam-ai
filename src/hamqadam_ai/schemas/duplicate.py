"""MODULE 8 response contract - duplicate face detection.

Answers "is this person already enrolled under another account?" - a 1:N
identification, which is a different statistical problem from the 1:1
verification Module 4 performs. See
:mod:`hamqadam_ai.duplicate_detection.calibration` for why the operating point
cannot simply be inherited.

Reading a hit correctly
-----------------------
``thresholds_validated`` is ``false`` and will stay false until somebody
derives the threshold against a real gallery. A hit is evidence for a human to
weigh, not a finding to act on automatically - and the strongest reason is not
statistical but biological: face recognition cannot separate identical twins at
any threshold, and siblings and cousins sit close behind them.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field

from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    Similarity,
)


class DuplicateCandidate(OutputModel):
    """One enrolled template that resembled the query."""

    reference: str = Field(
        description=(
            "The Backend's own identifier for the matched record, as it was "
            "supplied at enrolment. This service stores it opaquely and "
            "cannot resolve it to an account."
        )
    )
    similarity: Similarity = Field(description="Raw cosine similarity.")
    match_score: PercentScore = Field(
        description=(
            "Calibrated 0-100 score, interpolated through this module's own "
            "decision boundaries. Not comparable with Module 4's scores: the "
            "boundaries differ because the problem does."
        )
    )
    is_duplicate: bool = Field(
        description="Whether this candidate cleared the duplicate threshold."
    )
    needs_review: bool = Field(
        description="Whether it fell in the band between review and duplicate."
    )
    enrolled_at: dt.datetime | None = Field(
        default=None, description="When the matched template was stored."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Whatever the Backend attached at enrolment.",
    )


class DuplicateCheckResult(OutputModel):
    """MODULE 8 output for one query."""

    # -- Headline ------------------------------------------------------- #
    duplicate_found: bool = Field(
        description=(
            "Whether any enrolled template cleared the duplicate threshold. "
            "Evidence for the Backend's rules engine, not a decision."
        )
    )
    needs_review: bool = Field(
        default=False,
        description=(
            "Whether something fell in the review band: close enough to be "
            "worth a human look, not close enough to call."
        ),
    )
    best_similarity: Similarity | None = Field(
        default=None,
        description="Highest similarity found, or null on an empty gallery.",
    )
    best_match_score: PercentScore | None = Field(
        default=None, description="The same, calibrated to 0-100."
    )

    candidates: list[DuplicateCandidate] = Field(
        default_factory=list,
        description="Nearest templates, best first, including sub-threshold ones.",
    )

    # -- Context the caller needs to interpret a hit -------------------- #
    gallery_size: int = Field(
        default=0,
        description=(
            "Comparable templates searched. Not a curiosity: a 1:N false-match "
            "rate compounds with gallery size, so a similarity cannot be "
            "interpreted without knowing how many entries it beat."
        ),
    )
    searched: bool = Field(
        default=False,
        description=(
            "Whether a search actually ran. False means the gallery was empty "
            "or unavailable - which is not the same as finding nothing."
        ),
    )
    self_excluded: bool = Field(
        default=False,
        description=(
            "Whether the querying user's own template was excluded. Without "
            "this a re-verifying user matches themselves at cosine 1.0."
        ),
    )

    duplicate_threshold: float = Field(
        description="Cosine at or above which a hit counts as a duplicate."
    )
    review_threshold: float = Field(
        description="Cosine at or above which a hit is worth a human look."
    )
    thresholds_validated: bool = Field(
        default=False,
        description=(
            "Whether the operating point has been derived against a real "
            "gallery. Always false here. A simulation shows the correct "
            "threshold swings from 0.26 to 0.96 across plausible values of a "
            "parameter this project cannot measure - see the module doc."
        ),
    )

    recommended_action: str = Field(
        default="proceed",
        description="proceed, manual_review or reject. A recommendation only.",
    )

    # -- Provenance ----------------------------------------------------- #
    store: str = Field(default="", description="Which gallery adapter answered.")
    model_version: str = Field(
        default="",
        description=(
            "Recogniser version the query and every candidate share. A search "
            "never crosses versions - the scores would be meaningless."
        ),
    )

    error_code: ErrorCode | None = Field(default=None, description="Failure code.")
    error_message: str | None = Field(default=None, description="What went wrong.")
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal findings."
    )
    duration_ms: float = Field(default=0.0, description="Wall-clock time.")

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging.

        Carries no vector and no candidate reference: a reference identifies an
        account, and a log line saying which account a face matched is a
        linkage nobody asked for.
        """
        return {
            "duplicate": self.duplicate_found,
            "review": self.needs_review,
            "best_similarity": (
                round(self.best_similarity, 4)
                if self.best_similarity is not None
                else None
            ),
            "candidates": len(self.candidates),
            "gallery_size": self.gallery_size,
            "searched": self.searched,
            "store": self.store,
            "action": self.recommended_action,
            "error_code": str(self.error_code) if self.error_code else None,
            "duration_ms": round(self.duration_ms, 1),
        }


class EnrolmentResult(OutputModel):
    """Outcome of storing one template."""

    enrolled: bool = Field(description="Whether the template was stored.")
    reference: str = Field(description="The identifier it was stored under.")
    replaced: bool = Field(
        default=False,
        description=(
            "Whether an existing template under this reference was replaced. "
            "Replacement rather than accumulation is deliberate: several "
            "templates per person would each match the next query."
        ),
    )
    gallery_size: int = Field(default=0, description="Templates after the write.")
    store: str = Field(default="", description="Which gallery adapter stored it.")
    model_version: str = Field(default="", description="Recogniser version.")
    error_code: ErrorCode | None = Field(default=None, description="Failure code.")
    error_message: str | None = Field(default=None, description="What went wrong.")
    duration_ms: float = Field(default=0.0, description="Wall-clock time.")

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "enrolled": self.enrolled,
            "replaced": self.replaced,
            "gallery_size": self.gallery_size,
            "store": self.store,
            "error_code": str(self.error_code) if self.error_code else None,
        }


__all__ = [
    "DuplicateCandidate",
    "DuplicateCheckResult",
    "EnrolmentResult",
]
