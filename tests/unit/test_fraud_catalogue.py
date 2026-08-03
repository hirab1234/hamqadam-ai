"""Catalogue completeness: every code the service can emit must be classified.

This is the most valuable test in Module 9, and the reason is that the failure
it prevents is invisible. If a module gains a warning and nobody scores it, the
risk engine ignores it, every request still produces a plausible number, and
nothing anywhere goes red. The gap is discovered when somebody wonders why a
fraudulent account scored 12.

So the codes are enumerated from the source and each one must be *deliberately*
placed: scored, benign, or infrastructure. "Not mentioned anywhere" is a
failure, not a default.

The sweep found forty-seven unclassified codes when it was first written,
including every infrastructure error - which meant an unreachable vector
database or an unloaded model was about to be scored as silence rather than as
absence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.fraud_detection.signals import (
    BENIGN_CODES,
    CATALOGUE,
    INFRASTRUCTURE_CODES,
    SignalFamily,
    classify,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "hamqadam_ai"

#: Codes emitted as ``AnalysisWarning(code="...")`` or as validation findings.
_WARNING_PATTERN = re.compile(r'code="([A-Z][A-Z0-9_]{4,})"')
_FINDING_PATTERN = re.compile(r'"(CNIC_[A-Z_]+|PROFILE_[A-Z_]+|DUPLICATE_[A-Z_]+)":')


def reachable_codes() -> set[str]:
    """Every code the service can put on a response.

    Read from the source rather than from a hand-maintained list, because a
    hand-maintained list is exactly the thing that goes stale and takes the
    guarantee with it.
    """
    found: set[str] = {str(code) for code in ErrorCode}
    for path in SOURCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= set(_WARNING_PATTERN.findall(text))
        found |= set(_FINDING_PATTERN.findall(text))
    return found


@pytest.mark.unit
def test_every_reachable_code_is_classified() -> None:
    """No code may be merely absent.

    An unclassified code contributes nothing and fails nothing, which is the
    one failure mode this engine cannot detect about itself at runtime.
    """
    unknown = sorted(
        code for code in reachable_codes() if classify(code) == "unknown"
    )

    assert not unknown, (
        f"{len(unknown)} code(s) reachable but not classified. Add each to "
        f"CATALOGUE, BENIGN_CODES or INFRASTRUCTURE_CODES - deliberately, "
        f"deciding what it means: {unknown}"
    )


@pytest.mark.unit
def test_the_sweep_actually_finds_things() -> None:
    """Guards the guard. A regex that silently stopped matching would make the
    test above pass for the wrong reason."""
    codes = reachable_codes()

    assert len(codes) > 60
    assert "CNIC_GENDER_MISMATCH" in codes
    assert "PROFILE_IMAGE_IS_SCREENSHOT" in codes
    assert "VECTOR_DB_ERROR" in codes


@pytest.mark.unit
def test_the_three_classifications_are_disjoint() -> None:
    """A code in two sets means two different intentions were recorded, and
    which one wins is an implementation detail nobody should rely on."""
    scored = set(CATALOGUE)

    assert not scored & BENIGN_CODES
    assert not scored & INFRASTRUCTURE_CODES
    assert not BENIGN_CODES & INFRASTRUCTURE_CODES


@pytest.mark.unit
def test_infrastructure_codes_are_never_scored() -> None:
    """An outage is not evidence about a person."""
    for code in INFRASTRUCTURE_CODES:
        assert classify(code) == "infrastructure"
        assert code not in CATALOGUE


@pytest.mark.unit
@pytest.mark.parametrize("code", sorted(CATALOGUE))
def test_every_definition_is_coherent(code: str) -> None:
    definition = CATALOGUE[code]

    assert definition.code == code, "the key must match the definition"
    assert 0.0 <= definition.weight <= 1.0
    assert definition.family is not SignalFamily.UNAVAILABLE
    assert len(definition.message) > 30, "a reviewer needs more than a label"


@pytest.mark.unit
@pytest.mark.parametrize("code", sorted(CATALOGUE))
def test_no_message_states_a_conclusion_it_cannot_support(code: str) -> None:
    """These messages are shown to somebody deciding whether a person is
    dishonest. A finding is evidence; none of it proves anything on its own,
    and the wording must not pretend otherwise."""
    message = CATALOGUE[code].message.lower()

    for forbidden in ("proves", "confirms fraud", "definitely", "certainly"):
        assert forbidden not in message


@pytest.mark.unit
def test_the_gravest_findings_carry_their_caveats() -> None:
    """The two findings most likely to end somebody's application are the two
    whose limits are documented most heavily elsewhere. Those limits have to
    travel with the finding, not stay in a design document."""
    duplicate = CATALOGUE["DUPLICATE_FACE_DETECTED"].message.lower()
    assert "twin" in duplicate

    foreign = CATALOGUE["CNIC_FOREIGN_FACE_PRESENT"].message.lower()
    assert "held up" in foreign


@pytest.mark.unit
def test_capture_quality_signals_are_individually_weak() -> None:
    """No single quality finding may on its own reach the MEDIUM band. Most
    bad photographs are just bad photographs."""
    quality = [
        definition
        for definition in CATALOGUE.values()
        if definition.family is SignalFamily.CAPTURE_QUALITY
    ]

    assert quality
    for definition in quality:
        assert definition.weight <= 0.30, definition.code


@pytest.mark.unit
def test_identity_and_document_findings_outweigh_quality_ones() -> None:
    """The ordering the whole design depends on: evidence about the person and
    the document must count for more than evidence about the camera."""
    def heaviest(family: SignalFamily) -> float:
        return max(
            d.weight for d in CATALOGUE.values() if d.family is family
        )

    assert heaviest(SignalFamily.IDENTITY_CONSISTENCY) > heaviest(
        SignalFamily.CAPTURE_QUALITY
    )
    assert heaviest(SignalFamily.DOCUMENT_INTEGRITY) > heaviest(
        SignalFamily.CAPTURE_QUALITY
    )


@pytest.mark.unit
def test_every_family_except_unavailable_has_at_least_one_signal() -> None:
    """A family with no members is a category nobody filled in."""
    populated = {definition.family for definition in CATALOGUE.values()}
    expected = set(SignalFamily) - {
        SignalFamily.UNAVAILABLE,
        # No detector reports document recapture separately yet: Module 7 runs
        # on user photographs, and applying it to the CNIC image is Module 10's
        # composition to make. The family exists so that when it does, the
        # finding lands somewhere other than image_authenticity - where it
        # would wrongly merge with a screenshot of a selfie.
        SignalFamily.DOCUMENT_AUTHENTICITY,
    }

    assert expected <= populated


@pytest.mark.unit
def test_benign_codes_are_a_decision_not_an_oversight() -> None:
    """Every benign code must be one the service actually emits. A stale entry
    means somebody classified a code that no longer exists, and the real one
    may be going unscored under a new name."""
    codes = reachable_codes()
    stale = sorted(BENIGN_CODES - codes)

    assert not stale, f"benign codes no longer emitted anywhere: {stale}"
