"""MODULE 6 demonstration - find the CNIC portrait, match it to the selfie.

Usage
-----
::

    # Built-in scenarios, including the held-up-card attack
    python scripts/demo_cnic_face.py

    # Your own images
    python scripts/demo_cnic_face.py --cnic card.jpg --selfie me.jpg

    # Write annotated overlays showing what was chosen and what was refused
    python scripts/demo_cnic_face.py --overlays out/

    # Machine-readable
    python scripts/demo_cnic_face.py --json

Every built-in card is synthetic and carries a public-domain reference
portrait, print-degraded. No real identity document and no private
individual's photograph appears in this repository.

What the run demonstrates
-------------------------
The genuine cases match and the impostor cases do not - but the point of the
run is the block in the middle. A card held up in front of a face puts two
faces in one frame, and the large live one matches the selfie almost perfectly
because it *is* the selfie. Picking it would pass whoever the card belongs to.
The demo holds a **stolen** card at three distances and shows that the answer
is never a match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hamqadam_ai.core.constants import ImageRole  # noqa: E402
from hamqadam_ai.documents.portrait import portrait_regions  # noqa: E402
from hamqadam_ai.embeddings.base import FaceEmbedding  # noqa: E402
from hamqadam_ai.logging import configure_logging  # noqa: E402
from hamqadam_ai.schemas.cnic_face import CnicFaceMatchResult  # noqa: E402
from hamqadam_ai.services import (  # noqa: E402
    build_cnic_face_service,
    build_embedding_service,
    build_face_detection_service,
)
from hamqadam_ai.services.cnic_face_service import CnicFaceService  # noqa: E402
from hamqadam_ai.utils.geometry import BoundingBox  # noqa: E402
from hamqadam_ai.utils.image_io import load_image  # noqa: E402

BgrImage = npt.NDArray[np.uint8]


def build_scenarios() -> tuple[list[tuple[str, BgrImage]], BgrImage, BgrImage]:
    """The built-in cards, plus the two reference faces they use."""
    from tests.fixtures.cnic_portrait import (
        alternate_portrait,
        blank_card,
        card_beside_a_bystander,
        card_held_in_front_of_face,
        reference_portrait,
        render_cnic_with_portrait,
    )

    face = reference_portrait()
    stranger = alternate_portrait()
    if face is None or stranger is None:
        raise SystemExit(
            "No public-domain reference portrait is installed.\n"
            "Install one with:  pip install matplotlib scikit-image"
        )

    mine = render_cnic_with_portrait(face=face)
    ghosted = render_cnic_with_portrait(face=face, ghost=True)
    theirs = render_cnic_with_portrait(face=stranger)

    scenarios: list[tuple[str, BgrImage]] = [
        ("my card, flat", mine),
        ("my card, with ghost print", ghosted),
        ("a stranger's card", theirs),
        ("my card, bystander in shot", card_beside_a_bystander(mine)),
    ]
    for scale in (0.52, 0.70, 0.88):
        scenarios.append(
            (
                f"ATTACK stolen card held @{scale:.2f}",
                card_held_in_front_of_face(theirs, face=face, card_scale=scale),
            )
        )
    scenarios.append(
        ("my own card held up @0.70",
         card_held_in_front_of_face(mine, face=face, card_scale=0.70))
    )
    scenarios.append(("card with no portrait", blank_card()))

    return [(n, i) for n, i in scenarios if i is not None], face, stranger


def summarise(label: str, result: CnicFaceMatchResult) -> str:
    """One line per scenario."""
    portrait = result.portrait
    if result.success:
        verdict = "MATCH " if result.cnic_identity_match else "differ"
        score = f"{result.cnic_face_match_score:>5.1f}"
        similarity = f"{result.similarity:>6.3f}"
    else:
        verdict, score, similarity = "none  ", "    -", "     -"

    return (
        f"{label:<34s} {verdict}  score {score}  cos {similarity}  "
        f"faces {len(portrait.candidates)}  "
        f"ghost {'y' if portrait.has_ghost else '-'}  "
        f"foreign {portrait.foreign_face_count}  "
        f"rect {'y' if portrait.rectified else '-'}  "
        f"{result.duration_ms:>6.0f} ms"
    )


def render_detail(result: CnicFaceMatchResult) -> str:
    """Everything that went into one decision."""
    lines: list[str] = []
    portrait = result.portrait

    lines.append("  portrait search")
    lines.append(f"    found              {portrait.found}")
    lines.append(f"    usable             {portrait.usable}")
    lines.append(f"    quality            {portrait.quality_score}")
    lines.append(f"    rectified          {portrait.rectified}")
    lines.append(
        f"    searched           {portrait.searched_width}x{portrait.searched_height}"
    )

    lines.append("  candidates")
    for index, candidate in enumerate(portrait.candidates):
        chosen = (
            portrait.portrait is not None
            and candidate.box.x1 == portrait.portrait.box.x1
            and candidate.box.y1 == portrait.portrait.box.y1
        )
        marker = "->" if chosen else "  "
        reasons = ", ".join(candidate.rejections) or "admissible"
        lines.append(
            f"   {marker} [{index}] area {candidate.area_ratio:>7.4f}  "
            f"short {min(candidate.box.width, candidate.box.height):>5.0f}px  "
            f"band {candidate.band_score:>4.2f}  "
            f"conf {candidate.detector_confidence:>4.2f}  "
            f"plaus {candidate.plausibility:>5.3f}  {reasons}"
        )

    lines.append("  comparison")
    lines.append(f"    decision           {result.decision}")
    lines.append(f"    identity match     {result.cnic_identity_match}")
    lines.append(f"    strong match at    {result.strong_match_threshold}")
    lines.append(f"    review at          {result.review_threshold}")
    lines.append(f"    validated          {result.thresholds_validated}")

    if result.warnings:
        lines.append("  warnings")
        for warning in result.warnings:
            lines.append(f"    {warning.code}: {warning.message}")

    if result.error_message:
        lines.append(f"  error   {result.error_code}: {result.error_message}")

    return "\n".join(lines)


def write_overlay(
    directory: Path,
    name: str,
    service: CnicFaceService,
    image: BgrImage,
    result: CnicFaceMatchResult,
) -> None:
    """Draw what was chosen, what was refused, and where the priors looked.

    Drawn on the frame the service actually searched, which after successful
    rectification is the warped card rather than the image passed in. Drawing
    on the original would put every box in the wrong place - and produce a
    picture that contradicts a correct result.
    """
    import cv2
    from tests.fixtures.cnic_portrait import annotate

    portrait = result.portrait
    if not portrait.searched_width or not portrait.searched_height:
        return

    canvas, _rectify, _scale = service.prepare_card(image)

    boxes: list[tuple[BoundingBox, str]] = [
        (region, "band")
        for region in portrait_regions(
            portrait.searched_width, portrait.searched_height
        )
    ]
    chosen = portrait.portrait
    for candidate in portrait.candidates:
        box = BoundingBox(
            candidate.box.x1, candidate.box.y1, candidate.box.x2, candidate.box.y2
        )
        if chosen is not None and candidate.box.x1 == chosen.box.x1:
            boxes.append((box, "portrait"))
        elif candidate.rejections:
            boxes.append((box, "foreign"))
        else:
            boxes.append((box, "ghost"))

    directory.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in name)
    cv2.imwrite(str(directory / f"{safe}.png"), annotate(canvas, boxes))


def selfie_embedding(face: BgrImage) -> FaceEmbedding:
    """Embed the live selfie the cards are compared against."""
    detector = build_face_detection_service()
    embedder = build_embedding_service()
    return embedder.embed_to_vector(
        face,
        role=ImageRole.LIVE_SELFIE,
        detection=detector.detect(face, role=ImageRole.LIVE_SELFIE),
    )


def run_builtin(
    service: CnicFaceService, *, as_json: bool, overlays: Path | None
) -> int:
    """Read every built-in scenario."""
    scenarios, face, _stranger = build_scenarios()
    selfie = selfie_embedding(face)

    results = [(label, image, service.match(image, selfie))
               for label, image in scenarios]

    if as_json:
        print(json.dumps(
            {label: result.model_dump(mode="json") for label, _img, result in results},
            indent=2,
        ))
        return 0

    print()
    print("Synthetic CNIC cards, public-domain portraits, no real document used")
    print("=" * 104)
    for label, _image, result in results:
        print(summarise(label, result))

    print()
    print("The genuine case in full")
    print("-" * 104)
    print(render_detail(results[0][2]))

    attacks = [r for label, _i, r in results if label.startswith("ATTACK")]
    print()
    print("The attack: a stolen card held up in front of the attacker's own face")
    print("-" * 104)
    print(
        "  The live face behind the card matches the selfie almost perfectly -\n"
        "  it IS the selfie. An extractor that took the largest face would pass\n"
        "  this every time, whoever the card belongs to. Below, at every\n"
        "  distance, the live face is refused and the answer is never a match."
    )
    for label, _image, result in results:
        if label.startswith("ATTACK"):
            print()
            print(f"  {label}")
            print(render_detail(result))

    if overlays is not None:
        for label, image, result in results:
            write_overlay(overlays, label, service, image, result)
        print()
        print(f"Overlays written to {overlays}")

    leaked = [
        label for label, _i, r in results
        if label.startswith("ATTACK") and r.cnic_identity_match is True
    ]
    genuine_ok = results[0][2].cnic_identity_match is True

    print()
    if leaked:
        print(f"FAILURE: a stolen card matched at {', '.join(leaked)}")
        return 1
    if not genuine_ok:
        print("FAILURE: the genuine card did not match its holder.")
        return 1
    print(
        f"Genuine card matched; {len(attacks)} stolen-card attacks all refused "
        f"or failed on identity."
    )
    return 0


def run_single(
    service: CnicFaceService, cnic: Path, selfie_path: Path | None, *, as_json: bool
) -> int:
    """Read one card, optionally against one selfie."""
    card = load_image(cnic)
    probe = selfie_embedding(load_image(selfie_path)) if selfie_path else None
    result = service.match(card, probe)

    if as_json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    print()
    print(summarise(cnic.name, result))
    print(render_detail(result))
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Locate the portrait printed on a CNIC and compare it with a live "
            "selfie."
        )
    )
    parser.add_argument("--cnic", type=Path, help="Read this card image.")
    parser.add_argument("--selfie", type=Path, help="Compare against this selfie.")
    parser.add_argument(
        "--overlays", type=Path, help="Write annotated overlays to this directory."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    configure_logging()

    try:
        service = build_cnic_face_service()
    except Exception as exc:  # noqa: BLE001 - a missing model is a clean exit
        print(f"CNIC face service unavailable: {exc}", file=sys.stderr)
        print("Fetch the weights with:  python scripts/download_models.py",
              file=sys.stderr)
        return 2

    if arguments.cnic:
        return run_single(
            service, arguments.cnic, arguments.selfie, as_json=arguments.json
        )
    return run_builtin(
        service, as_json=arguments.json, overlays=arguments.overlays
    )


if __name__ == "__main__":
    raise SystemExit(main())
