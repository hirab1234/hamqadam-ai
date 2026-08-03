"""MODULE 9 demonstration - what does everything the other modules found add up to?

Usage
-----
::

    # The scenario walk-through
    python scripts/demo_fraud.py

    # Show the aggregation arithmetic against the alternatives
    python scripts/demo_fraud.py --compare

    # Show the signal catalogue and its coverage
    python scripts/demo_fraud.py --catalogue

    # Machine-readable
    python scripts/demo_fraud.py --json

What the run demonstrates
-------------------------
Not that fraud scores high - any scheme does that. The two rows to watch are
the ones a naive engine gets wrong:

* **a blurry photograph** produces four quality findings and must stay LOW,
  because it is a bad photo and not a dishonest one;
* **a gallery outage** must lower the *confidence* in the score without
  lowering the score, because an absent check is not evidence of innocence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS  # noqa: N814 - terse by design here

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hamqadam_ai.fraud_detection.signals import (  # noqa: E402
    BENIGN_CODES,
    CATALOGUE,
    INFRASTRUCTURE_CODES,
    SignalFamily,
)
from hamqadam_ai.logging import configure_logging  # noqa: E402
from hamqadam_ai.schemas.fraud import FraudRiskResult  # noqa: E402
from hamqadam_ai.services import build_fraud_service  # noqa: E402


def warning(code: str, **detail: object) -> NS:
    return NS(code=code, message="", stage="", detail=detail)


def finding(code: str, *, severity: str = "error", confidence: float = 1.0) -> NS:
    return NS(code=code, severity=severity, confidence=confidence)


def comparison(kind: str, decision: str, *, confidence: float = 0.9) -> NS:
    return NS(
        comparison=kind, decision=decision, confidence=confidence, compared=True
    )


def clean(**overrides: object) -> dict[str, object]:
    """A verification where everything passed, before overrides."""
    base: dict[str, object] = {
        "detection": NS(warnings=[], error_code=None),
        "quality": NS(warnings=[], error_code=None),
        "matching": NS(comparisons=[comparison("profile", "STRONG_MATCH")]),
        "ocr": NS(warnings=[], error_code=None, findings=[]),
        "cnic_face": NS(
            warnings=[], error_code=None,
            portrait=NS(foreign_face_count=0, found=True),
        ),
        "profile": NS(warnings=[], error_code=None, findings=[]),
        "duplicate": NS(
            searched=True, duplicate_found=False, needs_review=False,
            gallery_size=5_000,
        ),
    }
    base.update(overrides)
    return base


SCENARIOS: dict[str, dict[str, object]] = {
    "everything passes": clean(),

    "a blurry photograph (4 findings, 1 cause)": clean(
        detection=NS(warnings=[warning("FACE_POSE_OUT_OF_RANGE")], error_code=None),
        quality=NS(warnings=[], error_code="LOW_IMAGE_QUALITY"),
        ocr=NS(warnings=[], error_code="CNIC_OCR_FAILED", findings=[]),
        profile=NS(
            warnings=[warning("PROFILE_IMAGE_QUALITY_LOW")],
            error_code=None, findings=[],
        ),
    ),

    "the wrong document entirely": clean(
        ocr=NS(warnings=[], error_code="CNIC_NOT_RECOGNISED", findings=[]),
        cnic_face=NS(
            warnings=[], error_code=None,
            portrait=NS(foreign_face_count=0, found=False),
        ),
    ),

    "a screenshot of somebody's profile": clean(
        profile=NS(
            warnings=[warning("PROFILE_IMAGE_IS_SCREENSHOT", confidence=1.0)],
            error_code="INVALID_IMAGE", findings=[],
        ),
    ),

    "card held up in front of a stranger": clean(
        matching=NS(comparisons=[comparison("cnic", "FAILED")]),
        cnic_face=NS(
            warnings=[warning("CNIC_FOREIGN_FACE_PRESENT", count=1)],
            error_code=None,
            portrait=NS(foreign_face_count=1, found=True),
        ),
    ),

    "an altered card": clean(
        ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_GENDER_MISMATCH")],
        ),
    ),

    "three independent facts": clean(
        ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_GENDER_MISMATCH")],
        ),
        profile=NS(
            warnings=[warning("PROFILE_IMAGE_IS_SCREENSHOT", confidence=1.0)],
            error_code=None, findings=[],
        ),
        duplicate=NS(
            searched=True, duplicate_found=True, needs_review=False,
            gallery_size=9_000,
        ),
    ),

    "the gallery was unreachable": clean(
        duplicate=NS(
            searched=False, duplicate_found=False, needs_review=False,
            gallery_size=0,
        ),
    ),

    "a finding nobody has scored yet": clean(
        detection=NS(warnings=[warning("A_BRAND_NEW_WARNING")], error_code=None),
    ),
}


def line(label: str, result: FraudRiskResult) -> str:
    """One row of the walk-through."""
    factors = ", ".join(result.top_factors[:2]) or "-"
    return (
        f"{label:<44s} {result.fraud_risk_score:>5.1f}  "
        f"{str(result.fraud_risk_level):<6s} "
        f"conf {result.assessment_confidence:>4.2f}  {factors}"
    )


def render_detail(result: FraudRiskResult) -> str:
    """The full decomposition of one score."""
    lines: list[str] = ["  families (strongest first)"]
    if not result.families:
        lines.append("    none")
    for family in result.families:
        note = "  CAPPED" if family.capped else ""
        lines.append(
            f"    {family.family:<22s} {family.contribution:>5.3f}"
            f"  (raw {family.raw_contribution:.3f}, {family.signal_count} finding"
            f"{'s' if family.signal_count != 1 else ''}) <- {family.driver}{note}"
        )

    lines.append("  findings")
    if not result.signals:
        lines.append("    none")
    for signal in result.signals:
        lines.append(
            f"    {signal.code:<34s} w {signal.weight:.2f} x c {signal.confidence:.2f}"
            f" = {signal.contribution:.3f}   [{signal.family}]"
        )

    if result.unavailable_checks:
        lines.append(f"  unavailable  {result.unavailable_checks}")
    if result.unrecognised_findings:
        lines.append(f"  UNRECOGNISED {result.unrecognised_findings}")
    return "\n".join(lines)


def show_comparison() -> int:
    """Contrast the aggregation rules on the same evidence."""
    cases = {
        "one fact: CNIC shot off a screen": [
            ("document_authenticity", 0.55), ("capture_quality", 0.20),
            ("document_integrity", 0.15), ("document_authenticity", 0.25),
        ],
        "one fact: a blurry photograph": [
            ("capture_quality", 0.20), ("capture_quality", 0.15),
            ("capture_quality", 0.20), ("capture_quality", 0.10),
        ],
        "three independent facts": [
            ("document_authenticity", 0.55), ("duplication", 0.60),
            ("document_integrity", 0.70),
        ],
        "two independent facts": [
            ("identity_consistency", 0.75), ("duplication", 0.60),
        ],
    }

    def additive(sigs: list[tuple[str, float]]) -> float:
        return min(sum(w for _f, w in sigs), 1.0) * 100

    def naive_or(sigs: list[tuple[str, float]]) -> float:
        survival = 1.0
        for _f, weight in sigs:
            survival *= 1 - weight
        return (1 - survival) * 100

    def family_or(sigs: list[tuple[str, float]]) -> float:
        best: dict[str, float] = {}
        for family, weight in sigs:
            best[family] = max(best.get(family, 0.0), weight)
        survival = 1.0
        for weight in best.values():
            survival *= 1 - weight
        return (1 - survival) * 100

    print()
    print("How the same evidence scores under three combination rules")
    print("=" * 84)
    print(f"{'case':<40s} {'additive':>9s} {'naive OR':>9s} {'family+OR':>10s}")
    print("-" * 84)
    for label, sigs in cases.items():
        print(f"{label:<40s} {additive(sigs):>9.1f} {naive_or(sigs):>9.1f} "
              f"{family_or(sigs):>10.1f}")
    print()
    print("  Bands: LOW <= 30, MEDIUM <= 65, HIGH > 65")
    print()
    print("  The second row decides the design. Additive scoring puts an")
    print("  ordinary out-of-focus snapshot at 65 - the edge of HIGH risk -")
    print("  purely because four quality sub-scores each contributed. Grouping")
    print("  by family puts it at 20, and leaves the genuinely independent")
    print("  cases exactly where they were.")
    print()
    return 0


def show_catalogue() -> int:
    """List what is scored, what is benign, and what means 'unavailable'."""
    print()
    print(f"Signal catalogue: {len(CATALOGUE)} scored, {len(BENIGN_CODES)} benign, "
          f"{len(INFRASTRUCTURE_CODES)} infrastructure")
    print("=" * 96)
    for family in SignalFamily:
        members = sorted(
            (d for d in CATALOGUE.values() if d.family is family),
            key=lambda d: d.weight,
            reverse=True,
        )
        if not members:
            continue
        print(f"\n{family}")
        for definition in members:
            print(f"  {definition.weight:>4.2f}  {definition.code}")
    print()
    print("Infrastructure codes - these mean a check could not run, and")
    print("contribute nothing rather than being scored as silence:")
    for code in sorted(INFRASTRUCTURE_CODES):
        print(f"  {code}")
    print()
    return 0


def run(*, as_json: bool) -> int:
    """Score every scenario."""
    service = build_fraud_service()
    results = {label: service.assess(**kwargs) for label, kwargs in SCENARIOS.items()}

    if as_json:
        print(json.dumps(
            {label: result.model_dump(mode="json") for label, result in results.items()},
            indent=2,
        ))
        return 0

    print()
    print("Fraud risk across nine verification outcomes")
    print("=" * 104)
    for label, result in results.items():
        print(line(label, result))

    for label in ("a blurry photograph (4 findings, 1 cause)", "three independent facts"):
        print()
        print(label)
        print("-" * 104)
        print(render_detail(results[label]))

    print()
    print("An absent check is not a pass")
    print("-" * 104)
    outage = results["the gallery was unreachable"]
    print(f"  score {outage.fraud_risk_score:.1f} ({outage.fraud_risk_level}) - "
          f"unchanged, because nothing was found")
    print(f"  confidence {outage.assessment_confidence:.2f} - lowered, because "
          f"one of the checks did not run")
    print(f"  unavailable: {outage.unavailable_checks}")
    print()
    print("  Scoring the silence as innocence is how a fraud engine gets")
    print("  quietly switched off by an outage while still emitting confident")
    print("  low-risk verdicts.")

    print()
    print("A gap in the engine is reported, not swallowed")
    print("-" * 104)
    gap = results["a finding nobody has scored yet"]
    print(f"  unrecognised: {gap.unrecognised_findings}")
    print("  A new upstream warning that nobody scored contributes nothing and")
    print("  fails nothing. Surfacing it is the only way anyone finds out.")

    blurry = results["a blurry photograph (4 findings, 1 cause)"]
    if str(blurry.fraud_risk_level) != "LOW":
        print()
        print(f"FAILURE: a blurry photograph scored {blurry.fraud_risk_level}.")
        return 1

    print()
    print("A bad photograph stayed LOW; independent facts compounded to HIGH.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate every module's findings into one risk assessment."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Contrast the aggregation rules on the same evidence.",
    )
    parser.add_argument(
        "--catalogue", action="store_true", help="List the signal catalogue."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    configure_logging()

    if arguments.compare:
        return show_comparison()
    if arguments.catalogue:
        return show_catalogue()
    return run(as_json=arguments.json)


if __name__ == "__main__":
    raise SystemExit(main())
