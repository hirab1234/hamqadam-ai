"""Derive face-matching thresholds from a labelled identity corpus.

This is the script that turns the operating points in
``configs/thresholds.yaml`` from engineering defaults into validated ones.
Until it has been run against real data, the service reports
``thresholds_validated: false`` on every response and says so in a warning.

Dataset layout
--------------
One directory per identity, any number of images inside::

    corpus/
      person_0001/
        selfie.jpg
        profile.jpg
      person_0002/
        a.jpg
        b.jpg
        c.jpg
      ...

Two images in the same directory form a **genuine** pair; two in different
directories form an **impostor** pair. Nothing else is inferred, and no
filename convention is required.

Usage
-----
::

    python scripts/evaluate_matching.py --corpus path/to/corpus
    python scripts/evaluate_matching.py --corpus corpus --max-impostor-pairs 200000
    python scripts/evaluate_matching.py --corpus corpus --plot reports/
    python scripts/evaluate_matching.py --self-test

How many identities are needed
------------------------------
To resolve a FAR of 1e-4 you need at least 10,000 impostor pairs. An all-pairs
protocol over *n* identities with *k* images each yields roughly
``n(n-1)k^2/2`` of them, so 50 identities with 3 images apiece already gives
about 11,000 - enough for 1e-4 but not for 1e-5. The report states plainly
which targets the corpus can and cannot resolve rather than printing a
confident-looking number derived from too little data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.base import FaceEmbedding, cosine_similarity
from hamqadam_ai.logging import configure_logging
from hamqadam_ai.matching.evaluation import (
    EvaluationReport,
    evaluate,
    recommend_thresholds,
)
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_embedding_service, build_face_detection_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_io import load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")


@dataclass(slots=True)
class Sample:
    """One successfully embedded image."""

    identity: str
    path: Path
    embedding: FaceEmbedding


def discover(corpus: Path) -> dict[str, list[Path]]:
    """Map identity directory name to the images inside it."""
    identities: dict[str, list[Path]] = {}
    for directory in sorted(p for p in corpus.iterdir() if p.is_dir()):
        images = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            identities[directory.name] = images
    return identities


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


def embed_corpus(
    identities: dict[str, list[Path]], *, verbose: bool = True
) -> tuple[list[Sample], list[tuple[Path, str]]]:
    """Embed every image, reporting the ones that could not be processed."""
    settings = get_settings()
    registry = get_registry(settings)
    detector = build_face_detection_service(settings, registry)
    service = build_embedding_service(settings, registry)

    samples: list[Sample] = []
    failures: list[tuple[Path, str]] = []
    total = sum(len(paths) for paths in identities.values())
    done = 0

    for identity, paths in identities.items():
        for path in paths:
            done += 1
            if verbose and done % 25 == 0:
                sys.stderr.write(f"\r  embedding {done}/{total}...")
                sys.stderr.flush()
            try:
                image = load_image(path).pixels
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((path, f"could not read: {exc}"))
                continue

            detection = detector.detect(image, role=ImageRole.PROFILE_IMAGE)
            if detection.primary_face is None:
                failures.append((path, "no face detected"))
                continue

            box, marks = geometry(detection)
            try:
                embedding = service.embed_to_vector(
                    image, role=ImageRole.PROFILE_IMAGE, box=box, landmarks=marks
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((path, f"embedding failed: {exc}"))
                continue

            samples.append(Sample(identity=identity, path=path, embedding=embedding))

    if verbose:
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
    return samples, failures


def score_pairs(
    samples: list[Sample], *, max_impostor_pairs: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build the genuine and impostor similarity distributions.

    Every genuine pair is used - they are the scarce resource. Impostor pairs
    are sampled when the full set would be unmanageable, because their count
    grows quadratically and a corpus of 500 identities produces tens of
    millions.
    """
    by_identity: dict[str, list[Sample]] = {}
    for sample in samples:
        by_identity.setdefault(sample.identity, []).append(sample)

    genuine: list[float] = []
    for group in by_identity.values():
        for left, right in itertools.combinations(group, 2):
            genuine.append(
                cosine_similarity(left.embedding.vector, right.embedding.vector)
            )

    identities = list(by_identity)
    impostor: list[float] = []
    all_cross = [
        (a, b)
        for index, a in enumerate(identities)
        for b in identities[index + 1 :]
    ]

    rng = random.Random(seed)
    budget = max_impostor_pairs

    # Round-robin across identity pairs so the sample is not dominated by the
    # handful of identities that happen to have the most images.
    candidates: list[tuple[Sample, Sample]] = []
    for left_id, right_id in all_cross:
        for left in by_identity[left_id]:
            for right in by_identity[right_id]:
                candidates.append((left, right))

    if len(candidates) > budget:
        candidates = rng.sample(candidates, budget)

    for left, right in candidates:
        impostor.append(
            cosine_similarity(left.embedding.vector, right.embedding.vector)
        )

    return (
        np.asarray(genuine, dtype=np.float64),
        np.asarray(impostor, dtype=np.float64),
    )


def print_report(report: EvaluationReport, settings: Any) -> None:  # noqa: ANN401
    """Print the human-readable evaluation."""
    print()
    print("=" * 78)
    print("SCORE DISTRIBUTIONS")
    print("=" * 78)
    for label, described in (
        ("genuine", report.as_dict()["genuine_distribution"]),
        ("impostor", report.as_dict()["impostor_distribution"]),
    ):
        if not described:
            continue
        print(
            f"  {label:<9s} n={described['count']:>8d}  "
            f"min {described['min']:+.4f}  p05 {described['p05']:+.4f}  "
            f"median {described['median']:+.4f}  p95 {described['p95']:+.4f}  "
            f"max {described['max']:+.4f}"
        )
    print()
    print(f"  separation (worst genuine - best impostor): {report.separation:+.4f}")
    if report.separable:
        print("  the two populations do NOT overlap on this corpus")
    else:
        print("  the populations OVERLAP; no threshold classifies it perfectly")

    print()
    print("=" * 78)
    print("ACCURACY")
    print("=" * 78)
    print(f"  ROC AUC : {report.roc.auc:.6f}")
    print(f"  EER     : {report.equal_error_rate:.6f} at threshold "
          f"{report.eer_threshold:.4f}")
    print()
    print("  Operating points (TAR at a fixed FAR ceiling):")
    print(f"    {'target FAR':>12s}{'TAR':>10s}{'threshold':>12s}   resolvable")
    print("    " + "-" * 52)
    for entry in report.operating_points.values():
        mark = "yes" if entry["resolvable"] else f"NO (need {entry['impostor_pairs_needed']:,})"
        print(
            f"    {entry['target_far']:>12g}{entry['tar']:>10.4f}"
            f"{entry['threshold']:>12.4f}   {mark}"
        )

    print()
    print("=" * 78)
    print("CONFUSION MATRIX AT THE CURRENTLY-CONFIGURED THRESHOLDS")
    print("=" * 78)
    matching = settings.matching
    for label, thresholds in (
        ("profile", matching.selfie_vs_profile),
        ("secondary", matching.selfie_vs_secondary),
        ("cnic", matching.selfie_vs_cnic),
    ):
        matrix = report.confusion_at(thresholds.strong_match)
        print(
            f"  {label:<10s} strong_match={thresholds.strong_match:.2f}  "
            f"TP {matrix.true_positives:>6d}  FP {matrix.false_positives:>6d}  "
            f"FN {matrix.false_negatives:>6d}  TN {matrix.true_negatives:>7d}"
        )
        print(
            f"  {'':<10s} precision {matrix.precision:.4f}  "
            f"recall {matrix.recall:.4f}  F1 {matrix.f1:.4f}  "
            f"FAR {matrix.false_accept_rate:.6f}  FRR {matrix.false_reject_rate:.4f}"
        )

    print()
    print("=" * 78)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 78)
    recommended = recommend_thresholds(report)
    print(f"  strong_match : {recommended['strong_match']:.4f}   (at FAR <= 1e-4)")
    print(f"  review       : {recommended['review']:.4f}   (at FAR <= 1e-2)")
    print()
    print("  The strong-match boundary is set at a strict FAR because crossing it")
    print("  admits somebody to an account; the review boundary is looser because")
    print("  crossing it only routes the case to a human reviewer.")
    print()
    print("  To adopt these, edit configs/thresholds.yaml under `matching:` and")
    print("  set thresholds_validated appropriately in your deployment notes.")


def write_plots(report: EvaluationReport, directory: Path) -> list[Path]:
    """Write ROC, DET and score-distribution plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable; skipping plots")
        return []

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(report.roc.false_accept_rate, report.roc.true_accept_rate, linewidth=2)
    axis.set_xscale("log")
    positive_far = report.roc.false_accept_rate[report.roc.false_accept_rate > 0]
    axis.set_xlim(max(1e-6, float(positive_far.min())), 1.0)
    axis.set_ylim(0.0, 1.005)
    axis.set_xlabel("False accept rate (log)")
    axis.set_ylabel("True accept rate")
    axis.set_title(f"ROC - AUC {report.roc.auc:.5f}, EER {report.equal_error_rate:.5f}")
    axis.grid(True, which="both", alpha=0.3)
    path = directory / "roc.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(report.impostor, bins=80, alpha=0.65, label="impostor", density=True)
    axis.hist(report.genuine, bins=80, alpha=0.65, label="genuine", density=True)
    settings = get_settings()
    for label, thresholds in (
        ("profile", settings.matching.selfie_vs_profile),
        ("cnic", settings.matching.selfie_vs_cnic),
    ):
        axis.axvline(
            thresholds.strong_match, linestyle="--", linewidth=1.2,
            label=f"{label} strong={thresholds.strong_match}",
        )
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Density")
    axis.set_title("Genuine vs impostor score distributions")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    path = directory / "distributions.png"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)

    return written


def synthetic_corpus() -> tuple[np.ndarray, np.ndarray]:
    """Score distributions with the shape real ArcFace data has.

    Used by ``--self-test`` to exercise the whole report path without a
    corpus. The numbers are plausible but **invented** - they validate the
    machinery, never the thresholds.
    """
    rng = np.random.default_rng(11)
    genuine = np.clip(rng.normal(0.68, 0.13, 4000), -1.0, 1.0)
    impostor = np.clip(rng.normal(0.03, 0.10, 60000), -1.0, 1.0)
    return genuine, impostor


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Derive face-matching thresholds from a labelled corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, help="Directory of identity folders.")
    parser.add_argument(
        "--max-impostor-pairs",
        type=int,
        default=200_000,
        help="Cap on sampled impostor pairs (default 200000).",
    )
    parser.add_argument("--plot", type=Path, metavar="DIR", help="Write ROC plots here.")
    parser.add_argument("--report", type=Path, metavar="FILE", help="Write a JSON report.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Exercise the report machinery on synthetic distributions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    print("=" * 78)
    print("HAMQADAM AI - MODULE 4: MATCHING THRESHOLD EVALUATION")
    print("=" * 78)

    if args.self_test:
        print()
        print("  SELF-TEST MODE: distributions are SYNTHETIC and invented.")
        print("  This validates the evaluation machinery, never the thresholds.")
        genuine, impostor = synthetic_corpus()
    elif args.corpus:
        if not args.corpus.is_dir():
            parser.error(f"{args.corpus} is not a directory")
        identities = discover(args.corpus)
        if len(identities) < 2:
            parser.error(
                f"found {len(identities)} identity folder(s) in {args.corpus}; "
                f"at least 2 are needed to form impostor pairs"
            )
        image_count = sum(len(paths) for paths in identities.values())
        print(f"  corpus     : {args.corpus}")
        print(f"  identities : {len(identities)}")
        print(f"  images     : {image_count}")
        print()

        started = time.perf_counter()
        samples, failures = embed_corpus(identities)
        elapsed = time.perf_counter() - started
        print(f"  embedded   : {len(samples)}/{image_count} in {elapsed:.0f}s")
        if failures:
            print(f"  failed     : {len(failures)}")
            for path, reason in failures[:10]:
                print(f"      {path.name}: {reason}")
            if len(failures) > 10:
                print(f"      ... and {len(failures) - 10} more")

        usable = {s.identity for s in samples}
        if len(usable) < 2:
            print()
            print("  Fewer than two identities survived embedding; cannot continue.")
            return 1

        genuine, impostor = score_pairs(
            samples, max_impostor_pairs=args.max_impostor_pairs, seed=args.seed
        )
    else:
        print()
        print("  No corpus supplied.")
        print()
        print("  The thresholds currently in configs/thresholds.yaml are")
        print("  ENGINEERING DEFAULTS taken from the ArcFace/IJB-C literature.")
        print("  They have not been validated against this deployment's data.")
        print()
        print("  Supply a labelled corpus to derive real ones:")
        print("      python scripts/evaluate_matching.py --corpus path/to/corpus")
        print()
        print("  Layout: one directory per identity, images inside. Two images in")
        print("  the same directory are a genuine pair; two in different")
        print("  directories are an impostor pair.")
        print()
        print("  Or exercise the machinery on synthetic data:")
        print("      python scripts/evaluate_matching.py --self-test")
        return 2

    if genuine.size == 0 or impostor.size == 0:
        print()
        print(f"  Could not form both pair types: {genuine.size} genuine, "
              f"{impostor.size} impostor.")
        return 1

    print()
    print(f"  genuine pairs  : {genuine.size:,}")
    print(f"  impostor pairs : {impostor.size:,}")

    report = evaluate(genuine, impostor)
    print_report(report, settings)

    if args.plot:
        written = write_plots(report, args.plot)
        if written:
            print()
            print("  plots written:")
            for path in written:
                print(f"    {path}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = report.as_dict()
        payload["roc"] = report.roc.as_dict()
        payload["recommended"] = recommend_thresholds(report)
        payload["synthetic"] = bool(args.self_test)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"  JSON report written to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
