"""MODULE 8 demonstration - is this face already enrolled?

Usage
-----
::

    # The full walk-through, using real ArcFace templates
    python scripts/demo_duplicate.py

    # Against the durable adapter (embedded engine, no server needed)
    python scripts/demo_duplicate.py --backend qdrant

    # Machine-readable
    python scripts/demo_duplicate.py --json

Every face is a public-domain reference photograph. No private individual's
photograph appears in this repository, and nothing this script enrols outlives
the process.

What the run demonstrates
-------------------------
The duplicate itself is the easy part. The three interesting rows are the ones
where a naive implementation goes wrong:

* a **returning user** must not be a duplicate of themselves - without
  excluding the querying reference they match at cosine 1.0, every time;
* a **rejected applicant** must not have been enrolled merely by being checked,
  or their next legitimate attempt matches the ghost of the first;
* a template from **another recogniser version** must not be compared at all.
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

from hamqadam_ai.core.config import get_settings  # noqa: E402
from hamqadam_ai.core.constants import ImageRole  # noqa: E402
from hamqadam_ai.duplicate_detection import (  # noqa: E402
    InMemoryVectorStore,
    system_far,
)
from hamqadam_ai.embeddings.base import FaceEmbedding  # noqa: E402
from hamqadam_ai.logging import configure_logging  # noqa: E402
from hamqadam_ai.schemas.duplicate import DuplicateCheckResult  # noqa: E402
from hamqadam_ai.services import (  # noqa: E402
    build_embedding_service,
    build_face_detection_service,
)
from hamqadam_ai.services.duplicate_service import DuplicateService  # noqa: E402

BgrImage = npt.NDArray[np.uint8]


def build_store(backend: str, dimension: int):  # noqa: ANN201
    """The gallery adapter named on the command line."""
    if backend == "qdrant":
        from hamqadam_ai.duplicate_detection.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=":memory:", collection="demo", dimension=dimension
        )
    return InMemoryVectorStore(max_records=1000)


def templates() -> tuple[FaceEmbedding, FaceEmbedding, FaceEmbedding]:
    """Alice, Alice re-encoded, and Bob - from public-domain photographs."""
    from tests.fixtures.profile_images import (
        _jpeg,
        alternate_photo,
        reference_photo,
    )

    first = reference_photo()
    second = alternate_photo()
    if first is None or second is None:
        raise SystemExit(
            "No public-domain reference photograph is installed.\n"
            "Install one with:  pip install matplotlib scikit-image"
        )

    detector = build_face_detection_service()
    embedder = build_embedding_service()

    def embed(image: BgrImage) -> FaceEmbedding:
        detection = detector.detect(image, role=ImageRole.LIVE_SELFIE)
        return embedder.embed_to_vector(
            image, role=ImageRole.LIVE_SELFIE, detection=detection
        )

    return embed(first), embed(_jpeg(first, quality=70)), embed(second)


def line(label: str, result: DuplicateCheckResult) -> str:
    """One row of the walk-through."""
    verdict = (
        "DUPLICATE" if result.duplicate_found
        else "review   " if result.needs_review
        else "clear    "
    )
    similarity = (
        f"{result.best_similarity:>6.3f}" if result.best_similarity is not None else "     -"
    )
    return (
        f"{label:<38s} {verdict}  cos {similarity}  "
        f"gallery {result.gallery_size:>3d}  "
        f"action {result.recommended_action:<14s} {result.duration_ms:>5.1f} ms"
    )


def run(backend: str, *, as_json: bool) -> int:
    """Walk through the scenarios."""
    alice, alice_again, bob = templates()
    settings = get_settings()
    store = build_store(backend, alice.dimension)
    service = DuplicateService(store=store, settings=settings)

    steps: list[tuple[str, DuplicateCheckResult]] = []

    steps.append(("empty gallery, first ever user", service.check(alice, reference="acct-1")))
    service.enrol(alice, reference="acct-1")

    steps.append(
        ("acct-1 re-verifies (self-excluded)", service.check(alice_again, reference="acct-1"))
    )
    steps.append(
        ("acct-2: SAME person, new account", service.check(alice_again, reference="acct-2"))
    )
    steps.append(("acct-3: a different person", service.check(bob, reference="acct-3")))

    # The check above must not have enrolled anybody.
    enrolled_after_checks = service.gallery_size()

    other_version = FaceEmbedding(
        vector=alice.vector,
        raw_norm=alice.raw_norm,
        confidence=alice.confidence,
        model_key=alice.model_key,
        model_version="arcface-vNEXT",
        role=ImageRole.LIVE_SELFIE,
    )
    steps.append(
        ("same face, another model version", service.check(other_version, reference="acct-4"))
    )
    steps.append(
        ("no reference supplied (warns)", service.check(alice_again, reference=None))
    )

    if as_json:
        print(json.dumps(
            {label: result.model_dump(mode="json") for label, result in steps}, indent=2
        ))
        service.close()
        return 0

    print()
    print(f"Gallery adapter: {store.name}   "
          f"(durable: {store.health().get('durable')})")
    print("=" * 108)
    for label, result in steps:
        print(line(label, result))

    print()
    print("Checking never enrols")
    print("-" * 108)
    print(f"  Four checks ran against a gallery of {enrolled_after_checks}.")
    print("  Only the one explicit enrol() call wrote anything. Enrolling as a")
    print("  side effect of checking would put a REJECTED applicant's face in")
    print("  the gallery, where it would match their next legitimate attempt.")

    print()
    print("Warnings the caller receives")
    print("-" * 108)
    for label, result in steps:
        for warning in result.warnings:
            print(f"  [{label}]")
            print(f"    {warning.code}: {warning.message}")

    print()
    print("Right to erasure")
    print("-" * 108)
    print(f"  forget('acct-1') -> {service.forget('acct-1')}")
    print(f"  again            -> {service.forget('acct-1')}   (idempotent)")
    after = service.check(alice_again, reference="acct-9")
    print(f"  same face now    -> duplicate={after.duplicate_found}, "
          f"gallery={after.gallery_size}")

    configured = settings.duplicate
    print()
    print("Why the threshold is reported as unvalidated")
    print("-" * 108)
    print(f"  configured duplicate threshold: {configured.similarity_threshold}")
    print(f"  thresholds_validated:           {steps[0][1].thresholds_validated}")
    print()
    print("  A 1:N search compares against every enrolled template, so its")
    print("  false-match rate compounds with gallery size. At a per-comparison")
    print("  FAR of 1e-4:")
    for size in (1_000, 10_000, 100_000):
        print(f"    gallery {size:>7,d}  ->  {system_far(1e-4, size):>6.1%} of queries "
              f"return a false duplicate")
    print()
    print("  Run scripts/calibrate_duplicate_threshold.py --explain for the")
    print("  full argument, and --vectors to derive a real threshold.")

    service.close()

    genuine_caught = steps[2][1].duplicate_found
    self_match = steps[1][1].duplicate_found
    stranger = steps[3][1].duplicate_found
    cross_version = steps[4][1].duplicate_found

    print()
    if not genuine_caught:
        print("FAILURE: the same person on a second account was not caught.")
        return 1
    if self_match:
        print("FAILURE: a returning user was flagged as their own duplicate.")
        return 1
    if stranger or cross_version:
        print("FAILURE: a false duplicate was reported.")
        return 1
    if enrolled_after_checks != 1:
        print(f"FAILURE: checking enrolled {enrolled_after_checks - 1} extra templates.")
        return 1

    print("Duplicate caught; returning user cleared; nothing enrolled by checking.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Search a gallery of enrolled faces for the same person."
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "qdrant"],
        default="memory",
        help="Gallery adapter. Qdrant uses its embedded engine; no server needed.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    configure_logging()

    try:
        return run(arguments.backend, as_json=arguments.json)
    except Exception as exc:  # noqa: BLE001 - a missing model is a clean exit
        print(f"Could not run the demonstration: {exc}", file=sys.stderr)
        print("Fetch the weights with:  python scripts/download_models.py",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
