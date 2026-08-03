"""Measure ArcFace behaviour on real faces before trusting any threshold.

Answers the questions the Module 3 configuration depends on:

1. What is the raw-norm distribution, and does it actually track quality?
2. Does flip augmentation measurably help, or is it 2x cost for nothing?
3. How much does correct alignment matter versus the box-only fallback?
4. What does batching buy?
5. Is embedding deterministic, and is the cache key sound?

    python scripts/_calibrate_embeddings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.alignment import align_face_for_recognition
from hamqadam_ai.embeddings.base import EmbeddingRequest, cosine_similarity
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_embedding_service, build_face_detection_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


def portrait() -> BgrImage:
    """Public-domain reference portrait bundled with matplotlib."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("reference portrait unavailable")
    return image


def second_face() -> BgrImage | None:
    """A *different* person, for the impostor comparison.

    Without one, every similarity measured here is a genuine pair and the
    numbers say nothing about separability.
    """
    try:
        from skimage import data  # type: ignore[import-not-found]

        rgb = data.astronaut()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 - scikit-image is an optional dev dependency
        return None


def degrade(base: BgrImage) -> dict[str, BgrImage]:
    """Single controlled degradations of the same identity."""
    rng = np.random.default_rng(5)

    def jpeg(quality: int) -> BgrImage:
        ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else base

    def rescaled(factor: float) -> BgrImage:
        h, w = base.shape[:2]
        small = cv2.resize(
            base, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    def rotated(degrees: float) -> BgrImage:
        h, w = base.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        return cv2.warpAffine(base, matrix, (w, h), borderValue=(114, 114, 114))

    return {
        "original": base,
        "jpeg_q30": jpeg(30),
        "jpeg_q8": jpeg(8),
        "blur_5": cv2.GaussianBlur(base, (5, 5), 0),
        "blur_11": cv2.GaussianBlur(base, (11, 11), 0),
        "blur_21": cv2.GaussianBlur(base, (21, 21), 0),
        "dark_x0.4": np.clip(base * 0.4, 0, 255).astype(np.uint8),
        "bright_x1.7": np.clip(base * 1.7, 0, 255).astype(np.uint8),
        "noise_20": np.clip(
            base.astype(np.float32) + rng.normal(0, 20, base.shape), 0, 255
        ).astype(np.uint8),
        "downscaled_4x": rescaled(0.25),
        "rotated_12deg": rotated(12.0),
        "rotated_25deg": rotated(25.0),
    }


def geometry(detection) -> tuple[BoundingBox | None, Landmarks5 | None]:  # noqa: ANN001
    """Pull box and landmarks out of a detection result."""
    if detection.primary_face is None:
        return None, None
    bb = detection.primary_face.bounding_box
    box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
    named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
    order = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")
    marks = (
        Landmarks5(np.array([named[n] for n in order], dtype=np.float32))
        if all(n in named for n in order)
        else None
    )
    return box, marks


def main() -> int:  # noqa: PLR0915 - a report, linear by nature
    """Run the measurements and print the tables."""
    settings = get_settings()
    registry = get_registry(settings)
    detector = build_face_detection_service(settings, registry)
    service = build_embedding_service(settings, registry)

    base = portrait()
    variants = degrade(base)

    detections = {
        name: detector.detect(image, role=ImageRole.LIVE_SELFIE)
        for name, image in variants.items()
    }
    usable = {n: d for n, d in detections.items() if d.primary_face is not None}
    if "original" not in usable:
        raise SystemExit("no face found in the reference portrait")

    print("=" * 92)
    print("1. RAW NORM AND SELF-SIMILARITY UNDER DEGRADATION")
    print("=" * 92)
    print(
        f"{'variant':<16s}{'raw_norm':>10s}{'conf':>8s}"
        f"{'cos(orig)':>11s}{'residual':>10s}{'aligned':>9s}  note"
    )
    print("-" * 92)

    reference = None
    norms: list[tuple[str, float, float]] = []
    for name, image in variants.items():
        detection = detections[name]
        if detection.primary_face is None:
            print(f"{name:<16s}{'-':>10s}{'-':>8s}{'-':>11s}{'-':>10s}{'-':>9s}  NO FACE")
            continue
        box, marks = geometry(detection)
        embedding = service.embed_to_vector(
            image, role=ImageRole.LIVE_SELFIE, box=box, landmarks=marks
        )
        if name == "original":
            reference = embedding
        similarity = (
            embedding.similarity_to(reference) if reference is not None else float("nan")
        )
        norms.append((name, embedding.raw_norm, similarity))
        residual = (
            f"{embedding.alignment_residual:.4f}"
            if embedding.alignment_residual is not None
            else "-"
        )
        print(
            f"{name:<16s}{embedding.raw_norm:>10.2f}{embedding.confidence:>8.3f}"
            f"{similarity:>11.4f}{residual:>10s}{str(embedding.aligned):>9s}"
        )

    if norms:
        values = [n for _, n, _ in norms]
        print()
        print(
            f"  raw_norm range: {min(values):.1f} - {max(values):.1f}   "
            f"median {np.median(values):.1f}"
        )
        genuine = [s for name, _, s in norms if name != "original" and np.isfinite(s)]
        if genuine:
            print(
                f"  genuine self-similarity: min {min(genuine):.4f}  "
                f"median {np.median(genuine):.4f}"
            )

    # ---------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("2. IMPOSTOR SEPARATION (a genuinely different person)")
    print("=" * 92)
    other = second_face()
    if other is None:
        print("  scikit-image unavailable; cannot measure impostor separation.")
        print("  Every number above is a GENUINE pair and says nothing about")
        print("  how well the model separates different people.")
    else:
        other_detection = detector.detect(other, role=ImageRole.PROFILE_IMAGE)
        if other_detection.primary_face is None:
            print("  no face detected in the second image")
        else:
            box, marks = geometry(other_detection)
            impostor = service.embed_to_vector(
                other, role=ImageRole.PROFILE_IMAGE, box=box, landmarks=marks
            )
            assert reference is not None
            score = reference.similarity_to(impostor)
            print(f"  impostor cosine similarity : {score:.4f}")
            print(f"  impostor raw_norm          : {impostor.raw_norm:.2f}")
            genuine_min = min(
                s for name, _, s in norms if name != "original" and np.isfinite(s)
            )
            print(f"  worst genuine similarity   : {genuine_min:.4f}")
            print(f"  separation margin          : {genuine_min - score:.4f}")

    # ---------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("3. DOES FLIP AUGMENTATION EARN ITS 2x COST?")
    print("=" * 92)
    from hamqadam_ai.embeddings.arcface import ArcFaceEmbedder

    spec = settings.model_spec(settings.embedding.model)
    model = registry.get(settings.embedding.model)
    plain = ArcFaceEmbedder(model, spec, max_batch=16, flip_augmentation=False)
    flipped = ArcFaceEmbedder(model, spec, max_batch=16, flip_augmentation=True)

    crops = {}
    for name in ("original", "blur_5", "jpeg_q30", "rotated_12deg", "dark_x0.4"):
        if name not in usable:
            continue
        box, marks = geometry(detections[name])
        crops[name] = align_face_for_recognition(
            variants[name], box=box, landmarks=marks
        ).crop

    if "original" in crops:
        ref_plain = plain.embed_aligned([crops["original"]])[0][0]
        ref_flip = flipped.embed_aligned([crops["original"]])[0][0]
        print(f"{'variant':<16s}{'no-flip':>12s}{'flip-avg':>12s}{'delta':>10s}")
        print("-" * 52)
        deltas = []
        for name, crop in crops.items():
            if name == "original":
                continue
            a = cosine_similarity(ref_plain, plain.embed_aligned([crop])[0][0])
            b = cosine_similarity(ref_flip, flipped.embed_aligned([crop])[0][0])
            deltas.append(b - a)
            print(f"{name:<16s}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")
        if deltas:
            print("-" * 52)
            print(f"{'mean delta':<16s}{'':>12s}{'':>12s}{np.mean(deltas):>+10.4f}")

    # ---------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("4. HOW MUCH DOES ALIGNMENT MATTER?")
    print("=" * 92)
    if "original" in usable:
        box, marks = geometry(detections["original"])
        aligned = align_face_for_recognition(base, box=box, landmarks=marks)
        boxed = align_face_for_recognition(
            base, box=box, landmarks=None, allow_box_fallback=True
        )
        aligned_vec = flipped.embed_aligned([aligned.crop])[0]
        boxed_vec = flipped.embed_aligned([boxed.crop])[0]
        print(f"  template-aligned raw_norm : {aligned_vec[1]:.2f}")
        print(f"  box-only        raw_norm : {boxed_vec[1]:.2f}")
        print(
            f"  cosine(aligned, box-only) : "
            f"{cosine_similarity(aligned_vec[0], boxed_vec[0]):.4f}"
        )
        print("  (the same face through two crops; well below 1.0 means the")
        print("   fallback genuinely changes the embedding)")

        rot = usable.get("rotated_25deg")
        if rot is not None:
            rbox, rmarks = geometry(rot)
            r_aligned = align_face_for_recognition(
                variants["rotated_25deg"], box=rbox, landmarks=rmarks
            )
            r_boxed = align_face_for_recognition(
                variants["rotated_25deg"], box=rbox, landmarks=None
            )
            ra = flipped.embed_aligned([r_aligned.crop])[0][0]
            rb = flipped.embed_aligned([r_boxed.crop])[0][0]
            print()
            print("  on a 25-degree rotated capture, versus the upright original:")
            print(f"    template-aligned : {cosine_similarity(aligned_vec[0], ra):.4f}")
            print(f"    box-only         : {cosine_similarity(aligned_vec[0], rb):.4f}")

    # ---------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("5. BATCHING AND CACHING")
    print("=" * 92)
    sample = list(crops.values())[:1] * 8
    for size in (1, 4, 8):
        embedder = ArcFaceEmbedder(model, spec, max_batch=size, flip_augmentation=False)
        embedder.embed_aligned(sample[:1])
        started = time.perf_counter()
        embedder.embed_aligned(sample)
        elapsed = (time.perf_counter() - started) * 1000.0
        print(f"  8 crops, max_batch={size:<2d}: {elapsed:7.0f} ms  "
              f"({elapsed / 8:5.0f} ms/face)")

    requests = []
    for name in ("original", "blur_5", "jpeg_q30"):
        if name not in usable:
            continue
        box, marks = geometry(detections[name])
        requests.append(
            EmbeddingRequest(
                image=variants[name],
                box=box,
                landmarks=marks,
                role=ImageRole.SECONDARY_IMAGE,
            )
        )

    # The cache is already warm from section 1, which would make the "cold"
    # measurement a second set of hits and report a meaningless 1x speed-up.
    service._cache.clear()  # noqa: SLF001 - deliberate, this is a measurement
    cold = service.embed_many(requests)
    warm = service.embed_many(requests)
    print()
    print(f"  cold batch: {cold.total_duration_ms:6.0f} ms  "
          f"hits={cold.cache_hits}  passes={cold.forward_passes}")
    print(f"  warm batch: {warm.total_duration_ms:6.0f} ms  "
          f"hits={warm.cache_hits}  passes={warm.forward_passes}")
    if cold.total_duration_ms > 0:
        speedup = cold.total_duration_ms / max(warm.total_duration_ms, 0.01)
        print(f"  cache speed-up: {speedup:.0f}x")

    # ---------------------------------------------------------------- #
    print()
    print("=" * 92)
    print("6. DETERMINISM")
    print("=" * 92)
    if "original" in crops:
        first = flipped.embed_aligned([crops["original"]])[0][0]
        second = flipped.embed_aligned([crops["original"]])[0][0]
        identical = bool(np.array_equal(first, second))
        print(f"  bit-identical across two calls : {identical}")
        print(f"  cosine(self)                   : {cosine_similarity(first, second):.10f}")
        if not identical:
            print(f"  max elementwise delta          : {np.abs(first - second).max():.3e}")
            print("  NOTE: a non-deterministic embedder makes the cache unsound.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
