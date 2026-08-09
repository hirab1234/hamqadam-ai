"""Run every known attack scenario and assert the outcome. Exits non-zero on a
regression.

    python scripts/verify_accuracy.py
    python scripts/verify_accuracy.py --json

What this is for
----------------
A pre-deployment gate. The unit suite pins each rule in isolation against
synthetic outcomes; this drives the whole pipeline with real images and checks
the answer that actually reaches the Backend. Both matter, and they fail
differently: a rule can be correct and still be bypassed by an input nobody
routed through it, which is how every defect below was found.

Each case names the false approval it exists to prevent. `genuine` is the
control - without it a service that refused everything would score perfectly
here.

This needs model weights and a reachable Qdrant. It enrols nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from hamqadam_ai.core.constants import Recommendation  # noqa: E402
from hamqadam_ai.pipelines.verification import (  # noqa: E402
    VerificationImages,
    build_pipeline,
)
from hamqadam_ai.schemas.verification import VerificationRequest  # noqa: E402

APPROVE = Recommendation.APPROVE


def _fixtures() -> Any:
    try:
        from tests.fixtures import cnic_portrait
    except ImportError as exc:  # pragma: no cover - developer environment
        raise SystemExit(
            f"could not import the synthetic fixtures ({exc}). Run from the "
            f"repository root."
        ) from exc
    return cnic_portrait


def _cases() -> list[dict[str, Any]]:
    """Every scenario, with the outcome each one must not produce."""
    fixtures = _fixtures()
    me = fixtures.reference_portrait()
    stranger = fixtures.alternate_portrait()
    if me is None or stranger is None:
        raise SystemExit("no reference portraits installed; see tests/fixtures")

    my_card = fixtures.render_cnic_with_portrait(face=me)
    stranger_card = fixtures.render_cnic_with_portrait(face=stranger)

    # Small enough that the detector finds no face - the shape of the reported
    # bypass, where degrading an image beat submitting an honest one.
    tiny = cv2.resize(stranger, (185, 272), interpolation=cv2.INTER_AREA)

    return [
        {
            "name": "genuine",
            "guards": "the control - a real applicant must still get through",
            "expect": "APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=me, cnic=my_card, secondaries=[me]
            ),
        },
        {
            "name": "impostor_cnic",
            "guards": "somebody else's identity document",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=me, cnic=stranger_card
            ),
        },
        {
            "name": "wrong_profile",
            "guards": "a profile photograph of a different person",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=stranger, cnic=my_card
            ),
        },
        {
            "name": "wrong_secondary",
            "guards": "a stranger among the secondary photographs",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=me, cnic=my_card, secondaries=[stranger]
            ),
        },
        {
            "name": "profile_too_small_to_read",
            "guards": "shrinking a stranger's photo so it is never compared",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=tiny, cnic=my_card
            ),
        },
        {
            "name": "secondary_too_small_to_read",
            "guards": "the same bypass through the secondary slot",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(
                live_selfie=me, profile=me, cnic=my_card, secondaries=[tiny]
            ),
        },
        {
            "name": "no_profile_supplied",
            "guards": "approving on the document alone",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(live_selfie=me, cnic=my_card),
        },
        {
            "name": "no_cnic_supplied",
            "guards": "approving without examining a document at all",
            "expect": "NOT_APPROVE",
            "images": VerificationImages(live_selfie=me, profile=me),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Assert verification outcomes.")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    pipeline = build_pipeline()
    rows: list[dict[str, Any]] = []

    for case in _cases():
        result = pipeline.verify_sync(
            # No user_reference: enrolment is keyed off it, so leaving it out
            # keeps this script read-only against the gallery.
            VerificationRequest(verification_id=f"accuracy-{case['name']}"),
            case["images"],
        )
        got = result.recommendation
        ok = (got is APPROVE) if case["expect"] == "APPROVE" else (got is not APPROVE)
        rows.append(
            {
                "case": case["name"],
                "guards": case["guards"],
                "expected": case["expect"],
                "got": str(got),
                "identity": result.identity_confidence_score,
                "capped_by": getattr(result.matching, "capped_by", None),
                "passed": ok,
            }
        )

    if args.as_json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        print(f"{'case':<30} {'expected':<12} {'got':<15} {'identity':<9} result")
        print("-" * 80)
        for row in rows:
            identity = row["identity"]
            shown = f"{identity:.2f}" if isinstance(identity, float) else "-"
            print(
                f"{row['case']:<30} {row['expected']:<12} {row['got']:<15} "
                f"{shown:<9} {'PASS' if row['passed'] else '*** FAIL ***'}"
            )
        print()
        for row in rows:
            if not row["passed"]:
                print(f"FAILED {row['case']}: guards against {row['guards']}")

    failures = sum(1 for row in rows if not row["passed"])
    print(f"\n{len(rows) - failures}/{len(rows)} scenarios behaved correctly.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
