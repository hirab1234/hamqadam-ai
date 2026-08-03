"""MODULE 3 demonstration - generate and compare face embeddings.

Usage
-----
::

    # Built-in demonstration: one identity degraded, plus a second person
    python scripts/demo_embeddings.py

    # Your own images - every pair is compared
    python scripts/demo_embeddings.py a.jpg b.jpg c.jpg

    # Write the aligned crops out, to see what the recogniser actually sees
    python scripts/demo_embeddings.py a.jpg --save-crops out/

    # Machine-readable
    python scripts/demo_embeddings.py a.jpg b.jpg --json

The similarity matrix is the useful output: genuine pairs should sit far above
impostor pairs, and how far is exactly what Module 4's thresholds encode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.alignment import align_face_for_recognition
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging import configure_logging
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_embedding_service, build_face_detection_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_io import load_image

BgrImage = npt.NDArray[np.uint8]

_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")


def reference_portrait() -> BgrImage:
    """Public-domain portrait bundled with matplotlib."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read the reference portrait at {path}")
    return image


def second_person() -> BgrImage | None:
    """A different identity, so impostor separation can be shown."""
    try:
        from skimage import data

        return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def built_in_cases() -> dict[str, BgrImage]:
    """One identity through several defects, plus a genuinely different person."""
    base = reference_portrait()
    rng = np.random.default_rng(5)

    def jpeg(quality: int) -> BgrImage:
        ok, buffer = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else base

    def rotated(degrees: float) -> BgrImage:
        height, width = base.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        return cv2.warpAffine(base, matrix, (width, height), borderValue=(114,) * 3)

    cases = {
        "A: original": base,
        "A: jpeg q8": jpeg(8),
        "A: blurred": cv2.GaussianBlur(base, (11, 11), 0),
        "A: dark": np.clip(base * 0.4, 0, 255).astype(np.uint8),
        "A: noisy": np.clip(
            base.astype(np.float32) + rng.normal(0, 20, base.shape), 0, 255
        ).astype(np.uint8),
        "A: rotated 25deg": rotated(25.0),
    }
    other = second_person()
    if other is not None:
        cases["B: DIFFERENT PERSON"] = other
    return cases


def geometry(detection) -> tuple[BoundingBox | None, Landmarks5 | None]:  # noqa: ANN001
    """Pull box and landmarks out of a detection result."""
    if detection.primary_face is None:
        return None, None
    bb = detection.primary_face.bounding_box
    box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
    named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
    marks = (
        Landmarks5(np.array([named[n] for n in _LANDMARK_ORDER], dtype=np.float32))
        if all(n in named for n in _LANDMARK_ORDER)
        else None
    )
    return box, marks


def print_matrix(labels: list[str], embeddings: list[FaceEmbedding]) -> None:
    """Print the pairwise cosine similarity matrix."""
    width = max(len(label) for label in labels) + 2
    print()
    print("=" * 78)
    print("PAIRWISE COSINE SIMILARITY")
    print("=" * 78)
    print(" " * width + "".join(f"{index:>8d}" for index in range(len(labels))))
    for row, (label, left) in enumerate(zip(labels, embeddings, strict=True)):
        line = f"{label:<{width}s}"
        for column, right in enumerate(embeddings):
            score = left.similarity_to(right)
            marker = " " if row == column else ("*" if score >= 0.5 else " ")
            line += f"{score:>7.3f}{marker}"
        print(line)
    print()
    print("  * marks pairs at or above 0.5. Module 4 turns these into a decision;")
    print("    its configured strong-match point for selfie-vs-profile is 0.62.")


def print_details(labels: list[str], embeddings: list[FaceEmbedding]) -> None:
    """Print per-embedding provenance."""
    width = max(len(label) for label in labels) + 2
    print()
    print("=" * 78)
    print("EMBEDDINGS")
    print("=" * 78)
    print(
        f"{'image':<{width}s}{'dim':>5s}{'raw_norm':>10s}{'conf':>7s}"
        f"{'aligned':>9s}{'residual':>10s}{'cached':>8s}"
    )
    print("-" * (width + 49))
    for label, embedding in zip(labels, embeddings, strict=True):
        residual = (
            f"{embedding.alignment_residual:.4f}"
            if embedding.alignment_residual is not None
            else "-"
        )
        print(
            f"{label:<{width}s}{embedding.dimension:>5d}{embedding.raw_norm:>10.2f}"
            f"{embedding.confidence:>7.3f}{str(embedding.aligned):>9s}"
            f"{residual:>10s}{str(embedding.cache_hit):>8s}"
        )
    print()
    print("  raw_norm is reported as a diagnostic only. Measurement on this model")
    print("  shows it does NOT track face quality - a heavily blurred face reads")
    print("  higher than a pristine one - so confidence comes from the alignment")
    print("  residual instead.")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MODULE 3 face embedding generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="*", type=Path, help="Images to embed.")
    parser.add_argument(
        "--save-crops",
        type=Path,
        metavar="DIR",
        help="Write the aligned 112x112 crops the recogniser actually sees.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    print("=" * 78)
    print("HAMQADAM AI - MODULE 3: FACE EMBEDDINGS")
    print("=" * 78)

    registry = get_registry(settings)
    try:
        detector = build_face_detection_service(settings, registry)
        service = build_embedding_service(settings, registry)
    except Exception as exc:  # noqa: BLE001 - the demo explains rather than traces
        print()
        print(f"  Could not start: {exc}")
        print("  Run: python scripts/download_models.py")
        return 1

    described = service.describe()
    print(f"  model      : {described['model_key']} ({described['model_version']})")
    print(f"  dimension  : {described['dimension']}")
    print(f"  flip aug   : {described['flip_augmentation']}")
    print(f"  max batch  : {described['max_batch']}")
    print(f"  cache      : {described['cache'].get('backend')}")

    if args.images:
        cases: dict[str, BgrImage] = {}
        for path in args.images:
            if not path.is_file():
                print(f"  skipping {path}: not a file")
                continue
            cases[path.name] = load_image(path).pixels
        if not cases:
            parser.error("none of the supplied paths could be read")
    else:
        print()
        print("  no images supplied; using the reference portrait, several")
        print("  controlled degradations of it, and a second person")
        cases = built_in_cases()

    labels: list[str] = []
    embeddings: list[FaceEmbedding] = []
    payload: dict[str, dict] = {}

    for label, image in cases.items():
        detection = detector.detect(image, role=ImageRole.LIVE_SELFIE)
        if detection.primary_face is None:
            print(f"  {label}: no face detected, skipping")
            continue
        box, marks = geometry(detection)

        embedding = service.embed_to_vector(
            image, role=ImageRole.LIVE_SELFIE, box=box, landmarks=marks
        )
        labels.append(label)
        embeddings.append(embedding)
        payload[label] = embedding.describe()

        if args.save_crops:
            args.save_crops.mkdir(parents=True, exist_ok=True)
            crop = align_face_for_recognition(
                image,
                box=box,
                landmarks=marks,
                output_size=settings.embedding.alignment.output_size,
                max_residual=settings.embedding.alignment.max_alignment_residual,
            ).crop
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
            cv2.imwrite(str(args.save_crops / f"{safe}.png"), crop)

    if not embeddings:
        print("  no faces could be embedded")
        return 1

    if args.json:
        matrix = {
            labels[i]: {
                labels[j]: round(embeddings[i].similarity_to(embeddings[j]), 6)
                for j in range(len(labels))
            }
            for i in range(len(labels))
        }
        print(json.dumps({"embeddings": payload, "similarity": matrix}, indent=2))
        return 0

    print_details(labels, embeddings)
    print_matrix(labels, embeddings)

    genuine = [
        embeddings[i].similarity_to(embeddings[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if labels[i].startswith("A:") and labels[j].startswith("A:")
    ]
    impostor = [
        embeddings[i].similarity_to(embeddings[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if labels[i][:2] != labels[j][:2]
    ]
    if genuine and impostor:
        print()
        print("=" * 78)
        print("SEPARATION")
        print("=" * 78)
        print(f"  genuine pairs  : min {min(genuine):.4f}  median {np.median(genuine):.4f}")
        print(f"  impostor pairs : max {max(impostor):.4f}")
        print(f"  margin         : {min(genuine) - max(impostor):+.4f}")
        if min(genuine) > max(impostor):
            print("  -> the two populations are cleanly separated on this sample")
        else:
            print("  -> the populations OVERLAP; no threshold separates them here")

    if args.save_crops:
        print()
        print(f"  aligned crops written to {args.save_crops}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
