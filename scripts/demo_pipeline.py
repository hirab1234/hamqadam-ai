"""Run one end-to-end verification and print what the Backend would receive.

    python scripts/demo_pipeline.py --scenario coherent
    python scripts/demo_pipeline.py --scenario impostor --json
    python scripts/demo_pipeline.py --list

Every image is synthetic or a print-degraded public-domain reference portrait.
No real identity document is used, and none may be added to this repository.

The scenarios are the ones worth watching, not a happy path plus noise:

``coherent``
    One person, consistently. The approval case.
``impostor``
    A stranger's selfie against someone else's card. The rejection this whole
    service exists for.
``held_up_card``
    The card photographed while held in front of the user's own face, so the
    frame contains two faces. A naive extractor picks the large live one, after
    which the selfie is compared against itself and passes whoever the card
    belongs to.
``unreadable_cnic``
    Noise where the card should be. Demonstrates degrade-never-abort: the face
    comparison is still reported.
``selfie_only``
    A single image. Scores well on the little that was checked, and is still
    not approved - the evidence floor.
``duplicate``
    The same face enrolled under a different account reference first.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hamqadam_ai.core.config import get_settings  # noqa: E402
from hamqadam_ai.logging.setup import configure_logging  # noqa: E402
from hamqadam_ai.pipelines import (  # noqa: E402
    VerificationImages,
    build_pipeline,
)
from hamqadam_ai.schemas.verification import VerificationRequest  # noqa: E402

BgrImage = npt.NDArray[np.uint8]

#: Wrap the summary at a width that stays readable in a terminal.
_WIDTH = 78


def _fixtures() -> Any:
    """Import the synthetic card fixtures, or explain why they are missing."""
    try:
        from tests.fixtures import cnic_portrait
    except ImportError as exc:  # pragma: no cover - a developer-environment issue
        raise SystemExit(
            f"could not import the synthetic fixtures ({exc}). Run this from "
            f"the repository root."
        ) from exc
    return cnic_portrait


def _portraits() -> tuple[BgrImage, BgrImage]:
    """The two reference portraits, or a clear failure."""
    fixtures = _fixtures()
    first = fixtures.reference_portrait()
    second = fixtures.alternate_portrait()
    if first is None or second is None:
        raise SystemExit(
            "no public-domain reference portraits are installed. See "
            "tests/fixtures/README for how to add them."
        )
    return first, second


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def _coherent() -> tuple[VerificationImages, str]:
    face, _ = _portraits()
    card = _fixtures().render_cnic_with_portrait(face=face)
    return (
        VerificationImages(live_selfie=face, profile=face, cnic=card),
        "one person, consistently - selfie, profile and card all agree",
    )


def _impostor() -> tuple[VerificationImages, str]:
    face, stranger = _portraits()
    card = _fixtures().render_cnic_with_portrait(face=stranger)
    return (
        VerificationImages(live_selfie=face, profile=face, cnic=card),
        "a stranger's selfie against someone else's card",
    )


def _held_up_card() -> tuple[VerificationImages, str]:
    face, stranger = _portraits()
    fixtures = _fixtures()
    card = fixtures.render_cnic_with_portrait(face=stranger)
    if card is None:
        raise SystemExit("could not render the card fixture")
    frame = fixtures.card_held_in_front_of_face(card, face=face, card_scale=0.70)
    return (
        VerificationImages(live_selfie=face, profile=face, cnic=frame),
        "a stolen card held up in front of the user's own face - two faces "
        "in one frame",
    )


def _unreadable_cnic() -> tuple[VerificationImages, str]:
    face, _ = _portraits()
    noise = np.random.default_rng(7).integers(
        0, 255, (420, 660, 3), dtype=np.uint8
    )
    return (
        VerificationImages(live_selfie=face, profile=face, cnic=noise),
        "noise where the card should be - degrade, never abort",
    )


def _selfie_only() -> tuple[VerificationImages, str]:
    face, _ = _portraits()
    return (
        VerificationImages(live_selfie=face),
        "a single image - scores well on the little that was checked",
    )


SCENARIOS: dict[str, tuple[Callable[[], tuple[VerificationImages, str]], str]] = {
    "coherent": (_coherent, "the approval case"),
    "impostor": (_impostor, "the rejection this service exists for"),
    "held_up_card": (_held_up_card, "two faces in one frame"),
    "unreadable_cnic": (_unreadable_cnic, "partial evidence"),
    "selfie_only": (_selfie_only, "the evidence floor"),
    "duplicate": (_coherent, "the same face already enrolled elsewhere"),
}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _rule(title: str = "") -> None:
    if title:
        print(f"\n{title}\n{'-' * min(len(title), _WIDTH)}")
    else:
        print("=" * _WIDTH)


def _report(result: Any, description: str) -> None:
    """Print the parts of the response a reviewer actually reads."""
    _rule()
    print(f"scenario: {description}")
    _rule()

    verdict = result.recommendation
    confidence = result.identity_confidence_score
    print(f"  recommendation        {verdict}")
    print(
        "  identity confidence   "
        + ("not established" if confidence is None else f"{confidence:.1f} / 100")
    )
    print(
        f"  fraud risk            {result.fraud_risk_score:.1f} / 100 "
        f"({result.fraud_risk_level})"
    )
    print(f"  evidence available    {result.assessment_confidence:.0%}")
    print(f"  complete              {result.complete}")
    print(f"  human review          {result.requires_human_review}")
    if result.processing_time is not None:
        print(f"  elapsed               {result.processing_time.total:.0f} ms")

    _rule("why")
    for reason in result.recommendation_reasons:
        mark = "+" if reason.get("satisfied") else "-"
        print(f"  {mark} {reason['code']}")
        print(f"      {reason['message']}")

    if result.fraud is not None and result.fraud.signals:
        _rule("fraud signals")
        for signal in result.fraud.signals:
            print(
                f"  {signal.code:<40} weight {signal.weight:.2f} "
                f"confidence {signal.confidence:.2f}"
            )

    _rule("stages")
    for stage in result.stages:
        state = (
            "ok    " if stage.succeeded else ("FAILED" if stage.ran else "skipped")
        )
        print(f"  {state}  {stage.stage:<20} {stage.duration_ms:>8.1f} ms")
        if stage.error:
            print(f"          {stage.error}")

    if result.warnings:
        _rule("warnings")
        for warning in result.warnings:
            print(f"  {warning.code}")
            print(f"      {warning.message}")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run one verification through the full pipeline."
    )
    parser.add_argument(
        "--scenario",
        default="coherent",
        choices=sorted(SCENARIOS),
        help="Which submission to build.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the scenarios and exit."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response the Backend would receive.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the service's own logging."
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, (_, note) in sorted(SCENARIOS.items()):
            print(f"  {name:<18} {note}")
        return 0

    settings = get_settings()
    if args.quiet:
        settings = settings.model_copy(deep=True)
        settings.logging.level = "ERROR"
    configure_logging(settings)

    print("loading models...", flush=True)
    pipeline = build_pipeline(settings)

    build, _ = SCENARIOS[args.scenario]
    images, description = build()

    # The duplicate scenario needs the face in the gallery under a *different*
    # reference first, or there is nothing to collide with.
    if args.scenario == "duplicate":
        services = pipeline._services  # noqa: SLF001
        from hamqadam_ai.core.constants import ImageRole

        detection = services["detection"].detect(
            images.live_selfie, role=ImageRole.LIVE_SELFIE
        )
        embedding = services["embedding"].embed_to_vector(
            images.live_selfie, role=ImageRole.LIVE_SELFIE, detection=detection
        )
        services["duplicate"].enrol(embedding, reference="demo-other-account")
        print("enrolled the same face under 'demo-other-account' first")

    result = pipeline.verify_sync(
        VerificationRequest(
            verification_id=f"demo-{args.scenario}",
            user_reference="demo-account",
        ),
        images,
    )

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        _report(result, description)

    if args.scenario == "duplicate":
        # Leave the gallery as it was found. A demo that permanently enrols a
        # template is a demo that changes every subsequent run's answer.
        pipeline._services["duplicate"].forget("demo-other-account")  # noqa: SLF001
        print("\nerased 'demo-other-account' from the gallery")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
