"""Combining detector signals into one authenticity verdict.

Why the strongest signal wins, not the average
----------------------------------------------
Each detector looks for a *different* way an image can fail to be a genuine
capture, and the failures are alternatives rather than contributions. An image
is a screenshot **or** a photograph of a screen **or** rendered artwork; it is
essentially never a bit of each. Averaging four signals of which one is
strongly positive and three are correctly silent would dilute the one that
found something into a shrug.

So the score is driven by the **most confident finding**, and the others only
sharpen it. Concretely: a screenshot detected at confidence 0.9 should produce
a low authenticity score whatever the moire detector thought, because the moire
detector was not looking for screenshots and its silence says nothing about
them.

Silence is not evidence of authenticity
---------------------------------------
The corollary matters more than the rule. Four detectors finding nothing does
**not** establish that an image is a genuine capture - it establishes that
these four particular tests did not fire. A cropped screenshot with the chrome
removed, a screen capture taken far enough away to lose the moire, a competent
composite: all of them score 100 here. The score is an upper bound on
suspicion, not a measure of trust, and the docstrings and the response schema
both say so rather than leaving the caller to infer it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.authenticity.base import AuthenticitySignal

#: Human-readable explanation per detector, keyed by detector name. Kept here
#: rather than in the detectors so the wording a user eventually sees lives in
#: one place, and so a detector cannot quietly change what it accuses somebody
#: of.
FINDING_MESSAGES: dict[str, str] = {
    "screenshot": (
        "This looks like a screenshot of an app rather than a photograph. "
        "Upload the original photo from your gallery."
    ),
    "screen_recapture": (
        "This looks like a photograph of a screen. Upload the original image "
        "file rather than a picture of it displayed on another device."
    ),
    "print_recapture": (
        "This looks like a photograph of a printed photo. Upload the original "
        "digital image if you have it."
    ),
    "synthetic_image": (
        "This looks like an illustration or avatar rather than a photograph. "
        "Upload a real photo of yourself."
    ),
}

#: Stable codes for the fraud engine and the Backend.
FINDING_CODES: dict[str, str] = {
    "screenshot": "PROFILE_IMAGE_IS_SCREENSHOT",
    "screen_recapture": "PROFILE_IMAGE_IS_SCREEN_RECAPTURE",
    "print_recapture": "PROFILE_IMAGE_IS_PRINT_RECAPTURE",
    "synthetic_image": "PROFILE_IMAGE_IS_SYNTHETIC",
}


@dataclass(frozen=True, slots=True)
class AuthenticityFinding:
    """One thing found wrong with an image, ready for a caller to act on.

    Attributes:
        code: Stable machine-readable identifier.
        detector: Which detector produced it.
        confidence: Strength of the evidence, in ``[0, 1]``.
        message: What the user should do about it.
    """

    code: str
    detector: str
    confidence: float
    message: str

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "code": self.code,
            "detector": self.detector,
            "confidence": round(self.confidence, 4),
            "message": self.message,
        }


@dataclass(slots=True)
class AuthenticityAssessment:
    """Every detector's verdict on one image, and the conclusion drawn.

    Attributes:
        score: 0-100, where 100 means no detector found anything. **Not** a
            probability that the image is genuine - see the module docstring.
        signals: Every detector's raw signal, including the silent ones.
        findings: The detectors that triggered, strongest first.
        unmeasured: Detectors that could not run, and why.
    """

    score: float
    signals: list[AuthenticitySignal] = field(default_factory=list)
    findings: list[AuthenticityFinding] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Whether no detector triggered.

        Deliberately *not* named ``genuine``. Nothing here can establish that
        an image is a genuine capture; this says only that four specific tests
        found nothing.
        """
        return not self.findings

    @property
    def strongest(self) -> AuthenticityFinding | None:
        """The finding with the most evidence behind it."""
        return self.findings[0] if self.findings else None

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "score": round(self.score, 2),
            "clean": self.clean,
            "findings": [finding.code for finding in self.findings],
            "strongest": self.strongest.code if self.strongest else None,
            "unmeasured": list(self.unmeasured),
        }


def aggregate(signals: list[AuthenticitySignal]) -> AuthenticityAssessment:
    """Combine detector signals into one assessment.

    Args:
        signals: One per detector, including those that found nothing.

    Returns:
        The assessment. The score is driven by the strongest finding; a second
        agreeing finding lowers it further, but by less than the first, since
        two detectors firing on one image usually means one artefact tripped
        both rather than two independent problems.
    """
    findings = [
        AuthenticityFinding(
            code=FINDING_CODES.get(signal.name, signal.name.upper()),
            detector=signal.name,
            confidence=signal.confidence,
            message=FINDING_MESSAGES.get(
                signal.name, "This image does not look like a genuine photograph."
            ),
        )
        for signal in signals
        if signal.triggered
    ]
    findings.sort(key=lambda finding: finding.confidence, reverse=True)

    unmeasured = [
        f"{signal.name}: {signal.note}" for signal in signals if not signal.measured
    ]

    if not findings:
        return AuthenticityAssessment(
            score=100.0, signals=list(signals), unmeasured=unmeasured
        )

    # The strongest finding sets the score. Additional findings contribute at
    # a steeply diminishing rate.
    penalty = findings[0].confidence
    for index, finding in enumerate(findings[1:], start=1):
        penalty += finding.confidence / (2.0 ** (index + 1))
    penalty = min(penalty, 1.0)

    return AuthenticityAssessment(
        score=round(100.0 * (1.0 - penalty), 2),
        signals=list(signals),
        findings=findings,
        unmeasured=unmeasured,
    )


__all__ = [
    "FINDING_CODES",
    "FINDING_MESSAGES",
    "AuthenticityAssessment",
    "AuthenticityFinding",
    "aggregate",
]
