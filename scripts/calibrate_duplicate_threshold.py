"""Derive a 1:N duplicate threshold from a real gallery.

Usage
-----
::

    # Show why the shipped threshold cannot be trusted, using simulation
    python scripts/calibrate_duplicate_threshold.py --explain

    # Derive a threshold from a real gallery of templates
    python scripts/calibrate_duplicate_threshold.py --vectors templates.npy \\
        --gallery-size 250000 --target-far 0.001

    # Machine-readable
    python scripts/calibrate_duplicate_threshold.py --vectors templates.npy --json

Why this script exists
----------------------
The threshold shipped in ``configs/thresholds.yaml`` is **not validated**, and
the service says so in every response. It cannot be validated from this
repository: the correct value depends on the intrinsic dimension of the
embedding manifold and on the enrolled population, and neither can be estimated
from two public-domain reference faces.

``--explain`` demonstrates the problem with simulation. Everything else needs
your gallery.

Preparing the input
-------------------
``--vectors`` takes a ``.npy`` file holding an ``(n, d)`` float array: **one
row per distinct person**. That last part is the only assumption that matters.
Two templates of one person in the file would put a genuine match into the
impostor distribution and inflate the recommended threshold - which fails
silently, and in the unsafe direction.

The file is biometric data. Produce it on the machine that holds the gallery,
use it, and delete it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hamqadam_ai.core.config import get_settings  # noqa: E402
from hamqadam_ai.duplicate_detection.calibration import (  # noqa: E402
    measure_impostor_distribution,
    per_comparison_far,
    recommend_threshold,
    system_far,
)

DIMENSION = 512


def isotropic(count: int, dimension: int, *, seed: int = 0) -> np.ndarray:
    """Unit vectors filling the sphere - the naive impostor model."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dimension)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def on_manifold(
    count: int, effective_dim: int, dimension: int = DIMENSION, *, seed: int = 0
) -> np.ndarray:
    """Unit vectors confined to a lower-dimensional subspace.

    The realistic model: every enrolled template is a *face*, so they share a
    manifold whose intrinsic dimension is far below the nominal one.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(effective_dim, dimension))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    vectors = rng.normal(size=(count, effective_dim)) @ basis
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def explain() -> int:
    """Show why the shipped threshold cannot be trusted."""
    configured = get_settings().duplicate

    print()
    print("1. A 1:N search is not a 1:1 verification")
    print("=" * 74)
    print("   A query against N templates has N chances to match one by accident.")
    print()
    print(f"   {'gallery':>10s}  {'system FMR at a 1e-4 per-comparison FAR':>42s}")
    print("   " + "-" * 56)
    for size in (100, 1_000, 10_000, 100_000):
        print(f"   {size:>10d}  {system_far(1e-4, size):>41.1%}")
    print()
    print("   An unremarkable recogniser FAR becomes 'most queries' by 10,000.")
    print()
    print(f"   {'gallery':>10s}  {'FAR needed for 1% system':>26s}  {'for 0.1%':>12s}")
    print("   " + "-" * 52)
    for size in (1_000, 10_000, 100_000, 1_000_000):
        print(
            f"   {size:>10d}  {per_comparison_far(0.01, size):>26.2e}  "
            f"{per_comparison_far(0.001, size):>12.2e}"
        )

    print()
    print("2. The naive model says this is harmless. It is wrong.")
    print("=" * 74)
    print("   Modelled as isotropic random unit vectors, impostors never come")
    print("   close to the threshold:")
    print()
    statistics = measure_impostor_distribution(isotropic(3_000, DIMENSION))
    print(f"   isotropic 512-d, 3000 templates: max impostor cosine "
          f"{statistics.maximum:.3f}")
    print(f"   configured duplicate threshold:  {configured.similarity_threshold:.3f}")
    print()
    print("   But every enrolled vector is a FACE, so they share a manifold of")
    print("   far lower intrinsic dimension. Repeating the measurement with the")
    print("   population confined to a subspace:")
    print()
    print(f"   {'effective dim':>14s}  {'99.9th pct':>11s}  {'max':>7s}  verdict")
    print("   " + "-" * 58)
    for effective in (512, 128, 64, 32, 16, 8):
        vectors = (
            isotropic(2_000, DIMENSION)
            if effective >= DIMENSION
            else on_manifold(2_000, effective)
        )
        measured = measure_impostor_distribution(vectors)
        breached = measured.maximum >= configured.similarity_threshold
        verdict = "EXCEEDS the threshold" if breached else "safe"
        print(
            f"   {effective:>14d}  {measured.percentiles[99.9]:>11.3f}  "
            f"{measured.maximum:>7.3f}  {verdict}"
        )

    print()
    print("3. What follows")
    print("=" * 74)
    print("   The answer swings from 'entirely safe' to 'fails always' across a")
    print("   parameter this project cannot measure. So the shipped threshold is")
    print("   reported as unvalidated, and this script exists to replace it.")
    print()
    print("   Run it with --vectors pointing at a real gallery.")
    print()
    print("   And note what no threshold fixes: face recognition cannot separate")
    print("   identical twins, and siblings and cousins score well above chance.")
    print("   A platform serving extended families will enrol exactly that")
    print("   population. Only a second factor addresses it.")
    print()
    return 0


def calibrate(
    path: Path, *, gallery_size: int | None, target_far: float, as_json: bool
) -> int:
    """Derive a threshold from a real gallery."""
    vectors = np.load(path)
    if vectors.ndim != 2:
        raise SystemExit(f"expected an (n, d) array, got shape {vectors.shape}")

    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("normalising input vectors", file=sys.stderr)
        vectors = vectors / np.maximum(norms, 1e-12)

    recommendation = recommend_threshold(
        vectors, gallery_size=gallery_size, target_system_far=target_far
    )

    if as_json:
        print(json.dumps(recommendation.as_dict(), indent=2))
        return 0

    statistics = recommendation.statistics
    configured = get_settings().duplicate

    print()
    print(f"Gallery: {vectors.shape[0]} templates of {vectors.shape[1]} dimensions")
    print("=" * 74)
    print(f"  impostor pairs measured   {statistics.sample_count:,}")
    print(f"  mean                      {statistics.mean:.4f}")
    print(f"  standard deviation        {statistics.std:.4f}")
    for level, value in sorted(statistics.percentiles.items()):
        print(f"  {level:>6.2f}th percentile      {value:.4f}")
    print(f"  maximum observed          {statistics.maximum:.4f}")

    print()
    print("Recommendation")
    print("-" * 74)
    print(f"  sized for gallery         {recommendation.gallery_size:,}")
    print(f"  target system FMR         {recommendation.target_system_far:.2%}")
    print(f"  per-comparison FAR        {recommendation.per_comparison_far:.3e}")
    print(f"  THRESHOLD                 {recommendation.threshold:.4f}")
    print(f"  currently configured      {configured.similarity_threshold:.4f}")

    if recommendation.extrapolated:
        print()
        print("  EXTRAPOLATED. The target rate lies beyond what this many pairs")
        print("  can resolve, so the threshold comes from a Gaussian tail rather")
        print("  than from data. Real impostor distributions are right-skewed by")
        print("  look-alikes, so the true value is very likely HIGHER than this.")
        print("  Supply more templates, or accept a looser target.")

    if recommendation.threshold > configured.similarity_threshold:
        print()
        print(f"  The configured threshold of {configured.similarity_threshold:.2f} is")
        print("  LOOSER than this gallery supports. Raise it to at least")
        print(f"  {recommendation.threshold:.2f}, or expect false duplicates.")

    print()
    print("  No threshold separates identical twins, and siblings and cousins")
    print("  score well above chance. Treat every hit as evidence for a human.")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Derive a 1:N duplicate threshold from a real gallery."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show why the shipped threshold cannot be trusted, by simulation.",
    )
    parser.add_argument(
        "--vectors", type=Path, help="An (n, d) .npy file, one row per distinct person."
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        help="Size to plan for. Defaults to the number of vectors supplied.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Acceptable probability that a query returns a false duplicate.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    if arguments.vectors:
        return calibrate(
            arguments.vectors,
            gallery_size=arguments.gallery_size,
            target_far=arguments.target_far,
            as_json=arguments.json,
        )
    return explain()


if __name__ == "__main__":
    raise SystemExit(main())
