"""MODULE 8 capability - the gallery of enrolled faces.

The one stateful part of this service. Everything else forgets each request;
this keeps biometric templates, which is why the port carries erasure as a
first-class operation and why every record is keyed by an opaque reference the
Backend supplies rather than by anything derived from a face.

Two adapters. :class:`InMemoryVectorStore` does exact search over a NumPy
matrix and is the right answer for a single node - it is fast and correct, and
it loses everything on restart. :class:`QdrantVectorStore` is durable and
shared, and is what any multi-replica deployment needs.

:mod:`~hamqadam_ai.duplicate_detection.calibration` is not optional reading. A
1:N search is not a 1:1 verification, and the operating point does not carry
over.
"""

from hamqadam_ai.duplicate_detection.base import (
    SearchHit,
    VectorRecord,
    VectorStore,
    utc_now,
)
from hamqadam_ai.duplicate_detection.calibration import (
    ImpostorStatistics,
    ThresholdRecommendation,
    measure_impostor_distribution,
    per_comparison_far,
    recommend_threshold,
    system_far,
)
from hamqadam_ai.duplicate_detection.memory_store import (
    InMemoryVectorStore,
    build_memory_store,
)

__all__ = [
    "ImpostorStatistics",
    "InMemoryVectorStore",
    "SearchHit",
    "ThresholdRecommendation",
    "VectorRecord",
    "VectorStore",
    "build_memory_store",
    "build_store",
    "measure_impostor_distribution",
    "per_comparison_far",
    "recommend_threshold",
    "system_far",
    "utc_now",
]


def build_store(config: object, *, dimension: int) -> VectorStore:
    """Construct the configured gallery adapter.

    Falls back to the in-process store when Qdrant is not configured or its
    client is not installed, and says so loudly rather than quietly: an
    in-process gallery behind several replicas detects duplicates only against
    whichever pod answered, which looks exactly like a working system.

    Args:
        config: The ``duplicate`` section of the settings.
        dimension: Vector length the recogniser produces.

    Returns:
        A ready store.
    """
    from hamqadam_ai.core.exceptions import DependencyUnavailableError, VectorStoreError
    from hamqadam_ai.logging.setup import get_logger

    log = get_logger(__name__)
    backend = getattr(config, "backend", "memory")

    if backend == "qdrant":
        from hamqadam_ai.duplicate_detection.qdrant_store import QdrantVectorStore

        try:
            return QdrantVectorStore(
                url=config.qdrant.url,  # type: ignore[attr-defined]
                collection=config.qdrant.collection,  # type: ignore[attr-defined]
                dimension=dimension,
                api_key=config.qdrant.api_key,  # type: ignore[attr-defined]
                timeout_seconds=config.qdrant.timeout_seconds,  # type: ignore[attr-defined]
                create_collection=config.qdrant.create_collection,  # type: ignore[attr-defined]
            )
        except (DependencyUnavailableError, VectorStoreError) as exc:
            log.error(
                "duplicate.qdrant_unavailable",
                reason=str(exc),
                note=(
                    "falling back to an in-process gallery: it is lost on "
                    "restart and not shared between replicas, so duplicates "
                    "will be missed in any deployment running more than one"
                ),
            )

    return build_memory_store(getattr(config, "max_memory_records", 250_000))
