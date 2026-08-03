"""Combining fraud signals into one risk score.

Two rules, in this order
------------------------
**Strongest within a family.** Findings in one family are one fact seen several
ways, so the family contributes its strongest member and nothing more. This is
what stops an out-of-focus photograph accumulating four quality sub-scores into
a fraud accusation.

**Noisy-OR across families.** Families are independent evidence, so they
combine as ``1 - prod(1 - w)``. That is the standard combination for
independent indicators and it has the three properties wanted here: two
moderate findings exceed either alone, nothing ever exceeds 1.0, and no single
finding can be diluted by the presence of others - which a weighted mean would
do, and which would let an attacker reduce their score by adding innocent
noise.

Measured against the alternatives:

    case                                   additive   naive OR   family + OR
    one fact: CNIC shot off a screen          100.0       77.0          69.4
    one fact: a blurry photograph              65.0       51.0          20.0
    three genuinely independent facts         100.0       94.6          94.6

Caps, floors and the difference between them
--------------------------------------------
A **cap** bounds how far a family can push the score on its own. Capture
quality is capped low: most bad photographs are just bad photographs, and no
amount of blur should on its own read as fraud.

A **floor** is the opposite and is reserved for findings that are not "some
risk" but a conclusion. Nothing in the shipped catalogue is marked decisive -
the two candidates, a confirmed duplicate and a face mismatch, are exactly the
findings whose limits are documented most heavily elsewhere (identical twins,
uncalibrated 1:N thresholds), and quietly hard-coding them to HIGH here would
undo that care. The mechanism exists because a deployment with its own data may
reasonably decide otherwise.

What a missing check does
-------------------------
Nothing, to the score. If the duplicate gallery was unreachable, that is not
evidence of innocence and it is not evidence of guilt; it is an absence. It
lowers ``assessment_confidence`` and is listed by name, so a reviewer can see
that the number in front of them was computed without one of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.constants import RiskLevel
from hamqadam_ai.fraud_detection.signals import FraudSignal, SignalFamily


@dataclass(frozen=True, slots=True)
class FamilyContribution:
    """What one kind of evidence contributed, and which finding drove it.

    Attributes:
        family: The evidence family.
        contribution: Its strength after capping, in ``[0, 1]``.
        raw_contribution: Before capping, so a reader can see when a cap bit.
        driver: The code of the strongest finding in the family.
        signal_count: How many findings the family held. More than one means
            the others were treated as the same fact seen again.
    """

    family: SignalFamily
    contribution: float
    raw_contribution: float
    driver: str
    signal_count: int

    @property
    def capped(self) -> bool:
        """Whether a cap reduced this family's contribution."""
        return self.contribution < self.raw_contribution - 1e-9

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "family": str(self.family),
            "contribution": round(self.contribution, 4),
            "raw_contribution": round(self.raw_contribution, 4),
            "capped": self.capped,
            "driver": self.driver,
            "signal_count": self.signal_count,
        }


@dataclass(slots=True)
class RiskAssessment:
    """The combined verdict and everything behind it.

    Attributes:
        score: Fraud risk on 0-100.
        level: The band the score falls in.
        signals: Every scored finding.
        families: Per-family contributions, strongest first.
        unavailable: Checks that could not run, by name.
        unrecognised: Codes seen but absent from the catalogue. A gap in the
            engine rather than a property of the request, surfaced rather than
            swallowed.
        assessment_confidence: How complete the evidence was, in ``[0, 1]``.
            Not a confidence in the *score* being right - a measure of how many
            of the intended checks actually ran.
        floored_by: The decisive finding that raised the band, if any.
    """

    score: float
    level: RiskLevel
    signals: list[FraudSignal] = field(default_factory=list)
    families: list[FamilyContribution] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)
    assessment_confidence: float = 1.0
    floored_by: str | None = None

    @property
    def top_factors(self) -> list[str]:
        """The finding codes driving the score, strongest first."""
        return [family.driver for family in self.families]

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "score": round(self.score, 2),
            "level": str(self.level),
            "factors": self.top_factors[:5],
            "signals": len(self.signals),
            "unavailable": len(self.unavailable),
            "unrecognised": self.unrecognised,
            "confidence": round(self.assessment_confidence, 3),
            "floored_by": self.floored_by,
        }


def aggregate(
    signals: list[FraudSignal],
    *,
    low_max: float,
    medium_max: float,
    family_caps: dict[str, float] | None = None,
    unavailable: list[str] | None = None,
    unrecognised: list[str] | None = None,
    expected_checks: int = 0,
) -> RiskAssessment:
    """Combine signals into a risk score and band.

    Args:
        signals: Scored findings for this request.
        low_max: Upper bound of the LOW band.
        medium_max: Upper bound of the MEDIUM band.
        family_caps: Per-family ceilings, keyed by family value. A family
            without an entry is uncapped.
        unavailable: Names of checks that could not run.
        unrecognised: Finding codes with no catalogue entry.
        expected_checks: How many checks the caller intended to run, used to
            derive ``assessment_confidence``.

    Returns:
        The assessment.
    """
    caps = family_caps or {}
    missing = list(unavailable or [])
    unknown = list(unrecognised or [])

    grouped: dict[SignalFamily, list[FraudSignal]] = {}
    for signal in signals:
        grouped.setdefault(signal.family, []).append(signal)

    families: list[FamilyContribution] = []
    for family, members in grouped.items():
        if family is SignalFamily.UNAVAILABLE:
            continue
        strongest = max(members, key=lambda item: item.contribution)
        raw = strongest.contribution
        cap = caps.get(str(family))
        contribution = min(raw, cap) if cap is not None else raw
        families.append(
            FamilyContribution(
                family=family,
                contribution=contribution,
                raw_contribution=raw,
                driver=strongest.code,
                signal_count=len(members),
            )
        )

    families.sort(key=lambda item: item.contribution, reverse=True)

    # Noisy-OR across families.
    survival = 1.0
    for entry in families:
        survival *= 1.0 - entry.contribution
    score = round(100.0 * (1.0 - survival), 2)

    level = _band(score, low_max=low_max, medium_max=medium_max)

    # A decisive finding raises the band without inventing a score for it: the
    # arithmetic still says what the evidence weighed, and the band says what
    # to do about it.
    floored_by = None
    for signal in signals:
        if signal.definition.decisive and signal.confidence > 0.0:
            floored_by = signal.code
            level = RiskLevel.HIGH
            break

    confidence = 1.0
    if expected_checks > 0:
        confidence = max(0.0, 1.0 - len(missing) / float(expected_checks))

    return RiskAssessment(
        score=score,
        level=level,
        signals=list(signals),
        families=families,
        unavailable=missing,
        unrecognised=unknown,
        assessment_confidence=round(confidence, 4),
        floored_by=floored_by,
    )


def _band(score: float, *, low_max: float, medium_max: float) -> RiskLevel:
    """Map a score onto its risk band.

    Boundaries are inclusive at the top of each band, matching the config's
    names: ``low_max`` is the highest score that still counts as LOW.
    """
    if score <= low_max:
        return RiskLevel.LOW
    if score <= medium_max:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


__all__ = ["FamilyContribution", "RiskAssessment", "aggregate"]
