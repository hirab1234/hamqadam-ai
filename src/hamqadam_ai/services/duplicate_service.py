"""MODULE 8 service - has this face already been enrolled by somebody else?

The one stateful module
-----------------------
Everything else in this service forgets each request. This one keeps a gallery
of biometric templates, which makes three things design problems rather than
housekeeping:

* **Self-exclusion.** A user re-verifying matches their own enrolled template
  at cosine 1.0. Without excluding the querying reference, every returning user
  is reported as a duplicate of themselves - and the bug would look exactly
  like a working detector.
* **Model versioning.** A search never crosses recogniser versions. Two
  embeddings from different ArcFace builds are not comparable, and comparing
  them anyway produces scores that look ordinary and mean nothing.
* **Erasure.** ``forget()`` exists, is idempotent, and is part of the port
  rather than bolted on. A service that stores face templates and cannot
  delete one on request cannot lawfully be deployed.

Check before enrol, and never both automatically
------------------------------------------------
:meth:`DuplicateService.check` does not enrol, and :meth:`enrol` does not
check. That looks like an inconvenience and is a safety property: enrolling as
a side effect of checking would put a rejected applicant's face permanently in
the gallery, where it would then match their next legitimate attempt. The
Backend decides when a verification has succeeded, so the Backend calls
``enrol``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hamqadam_ai.core.config import MatchThresholds, Settings, get_settings
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError, VectorStoreError
from hamqadam_ai.duplicate_detection.base import (
    SearchHit,
    VectorRecord,
    VectorStore,
    utc_now,
)
from hamqadam_ai.duplicate_detection.calibration import system_far
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.matching.similarity import calibrate_score
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.duplicate import (
    DuplicateCandidate,
    DuplicateCheckResult,
    EnrolmentResult,
)

log = get_logger(__name__)

#: Gallery size beyond which an uncalibrated threshold stops being defensible.
#: Not a hard limit and not a tuned number - it is the point at which the
#: simulation in the calibration module shows the answer becoming sensitive to
#: the embedding manifold's effective dimension, which nobody here has measured.
UNCALIBRATED_GALLERY_WARNING = 5_000


class DuplicateService:
    """Searches a gallery of enrolled faces for the same person.

    Args:
        store: The gallery adapter.
        settings: Service configuration.
    """

    __slots__ = ("_config", "_matching", "_settings", "_store")

    def __init__(self, *, store: VectorStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._config = settings.duplicate
        self._matching = settings.matching

    # -- Public surface ------------------------------------------------------ #

    def check(
        self,
        embedding: FaceEmbedding | None,
        *,
        reference: str | None = None,
    ) -> DuplicateCheckResult:
        """Search the gallery for this face.

        Args:
            embedding: The live selfie's template. ``None`` is a normal input -
                the selfie may itself have failed - and yields a result saying
                no search was possible, which is different from finding
                nothing.
            reference: The querying user's own identifier, excluded from the
                search. **Pass it whenever it is known.** Omitting it makes
                every re-verification a self-match at cosine 1.0.

        Returns:
            The populated result. A gallery outage is reported through
            ``error_code``, never raised: a duplicate check failing must not
            take the rest of a verification with it.
        """
        started = time.perf_counter()
        thresholds = self._thresholds()

        if embedding is None or embedding.is_degenerate:
            return self._empty_result(
                thresholds=thresholds,
                started=started,
                error_code=ErrorCode.FACE_NOT_DETECTED,
                error_message=(
                    "There is no usable face template to search with. The "
                    "gallery was not queried."
                ),
            )

        try:
            hits = self._store.search(
                embedding.vector,
                model_version=embedding.model_version,
                top_k=self._config.top_k,
                exclude=reference,
            )
            gallery_size = self._store.count(model_version=embedding.model_version)
        except (VectorStoreError, HamqadamError) as exc:
            log.warning("duplicate.search_failed", reason=str(exc))
            return self._empty_result(
                thresholds=thresholds,
                started=started,
                error_code=ErrorCode.VECTOR_DB_ERROR,
                error_message=(
                    f"The duplicate-face gallery could not be searched: {exc}. "
                    f"No conclusion about duplicates can be drawn from this "
                    f"request."
                ),
                model_version=embedding.model_version,
            )

        candidates = [self._to_candidate(hit, thresholds) for hit in hits]
        duplicates = [c for c in candidates if c.is_duplicate]
        reviews = [c for c in candidates if c.needs_review]

        result = DuplicateCheckResult(
            duplicate_found=bool(duplicates),
            needs_review=bool(reviews) and not duplicates,
            best_similarity=candidates[0].similarity if candidates else None,
            best_match_score=candidates[0].match_score if candidates else None,
            candidates=candidates,
            gallery_size=gallery_size,
            searched=True,
            self_excluded=reference is not None,
            duplicate_threshold=thresholds.strong_match,
            review_threshold=thresholds.review,
            thresholds_validated=False,
            recommended_action=self._action(
                duplicate=bool(duplicates), review=bool(reviews)
            ),
            store=self._store.name,
            model_version=embedding.model_version,
            warnings=self._warnings(
                gallery_size=gallery_size,
                reference=reference,
                duplicates=bool(duplicates),
            ),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        log.info("duplicate.completed", **result.summary())
        return result

    def enrol(
        self,
        embedding: FaceEmbedding,
        *,
        reference: str,
        metadata: dict[str, Any] | None = None,
    ) -> EnrolmentResult:
        """Store this face against a reference.

        Called by the Backend once a verification has succeeded, never as a
        side effect of :meth:`check` - see the module docstring.

        Args:
            embedding: The template to store.
            reference: The Backend's opaque identifier. Re-enrolling the same
                reference replaces rather than appends.
            metadata: Free-form labels. **Must be PII-free.** This service
                cannot enforce that and does not pretend to; it stores what it
                is given, and what it is given ends up in the gallery.

        Returns:
            The outcome. A gallery outage is reported, not raised.
        """
        started = time.perf_counter()

        if not reference:
            raise ValueError("a reference is required to enrol a template")
        if embedding.is_degenerate:
            return EnrolmentResult(
                enrolled=False,
                reference=reference,
                store=self._store.name,
                model_version=embedding.model_version,
                error_code=ErrorCode.VALIDATION_ERROR,
                error_message="the template carries no usable direction",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            # A direct lookup. Inferring this from a top-1 search would report
            # "new enrolment" whenever somebody else's template happened to be
            # nearer than the one being replaced.
            replaced = self._store.exists(reference)

            self._store.enrol(
                VectorRecord(
                    reference=reference,
                    vector=embedding.vector,
                    model_version=embedding.model_version,
                    enrolled_at=utc_now(),
                    metadata=dict(metadata or {}),
                )
            )
            gallery_size = self._store.count(model_version=embedding.model_version)
        except (VectorStoreError, HamqadamError, MemoryError) as exc:
            log.warning("duplicate.enrol_failed", reason=str(exc))
            return EnrolmentResult(
                enrolled=False,
                reference=reference,
                store=self._store.name,
                model_version=embedding.model_version,
                error_code=ErrorCode.VECTOR_DB_ERROR,
                error_message=f"could not enrol the template: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        result = EnrolmentResult(
            enrolled=True,
            reference=reference,
            replaced=replaced,
            gallery_size=gallery_size,
            store=self._store.name,
            model_version=embedding.model_version,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        log.info("duplicate.enrolled", **result.summary())
        return result

    def forget(self, reference: str) -> bool:
        """Erase a stored template.

        Idempotent: erasing an absent reference succeeds. A caller retrying an
        erasure request must not be told it failed the second time, because
        that is how erasure requests get abandoned half-done.

        Args:
            reference: The identifier to erase.

        Returns:
            Whether anything was removed. ``False`` means there was nothing to
            remove, not that the request failed.
        """
        removed = self._store.delete(reference)
        log.info("duplicate.forgotten", removed=removed)
        return removed

    def gallery_size(self, *, model_version: str | None = None) -> int:
        """How many comparable templates are enrolled."""
        return self._store.count(model_version=model_version)

    async def check_async(
        self, embedding: FaceEmbedding | None, *, reference: str | None = None
    ) -> DuplicateCheckResult:
        """Search without blocking the event loop.

        An exact search over a large gallery is a matrix product of real size,
        and a network round trip to Qdrant is worse.
        """
        return await asyncio.to_thread(self.check, embedding, reference=reference)

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        thresholds = self._thresholds()
        return {
            "store": self._store.health(),
            "duplicate_threshold": thresholds.strong_match,
            "review_threshold": thresholds.review,
            "top_k": self._config.top_k,
            "on_duplicate": self._config.on_duplicate,
            "thresholds_validated": False,
            "known_limits": {
                "identical_twins": (
                    "indistinguishable to face recognition at any threshold"
                ),
                "close_relatives": "siblings and cousins score well above chance",
                "operating_point": (
                    "not derived against a real gallery; see "
                    "duplicate_detection.calibration"
                ),
            },
        }

    def close(self) -> None:
        """Release the gallery adapter."""
        self._store.close()

    # -- Internals ----------------------------------------------------------- #

    def _thresholds(self) -> MatchThresholds:
        """This module's own operating points, as a threshold pair.

        Built rather than taken from ``matching`` on purpose: reusing the 1:1
        verification thresholds is precisely the mistake the calibration module
        exists to document.
        """
        return MatchThresholds(
            strong_match=self._config.similarity_threshold,
            review=self._config.review_threshold,
        )

    def _to_candidate(
        self, hit: SearchHit, thresholds: MatchThresholds
    ) -> DuplicateCandidate:
        """Render one search hit with its verdict."""
        is_duplicate = hit.similarity >= thresholds.strong_match
        return DuplicateCandidate(
            reference=hit.reference,
            similarity=hit.similarity,
            match_score=calibrate_score(
                hit.similarity, thresholds, self._matching.calibration
            ),
            is_duplicate=is_duplicate,
            needs_review=(not is_duplicate and hit.similarity >= thresholds.review),
            enrolled_at=hit.enrolled_at,
            metadata=dict(hit.metadata),
        )

    def _action(self, *, duplicate: bool, review: bool) -> str:
        """Translate the verdict into a recommendation for the rules engine."""
        if duplicate:
            return "reject" if self._config.on_duplicate == "reject" else "manual_review"
        if review:
            return "manual_review"
        return "proceed"

    def _warnings(
        self, *, gallery_size: int, reference: str | None, duplicates: bool
    ) -> list[AnalysisWarning]:
        """Everything the caller needs to interpret the verdict honestly."""
        warnings: list[AnalysisWarning] = []

        if reference is None:
            warnings.append(
                AnalysisWarning(
                    code="DUPLICATE_SELF_NOT_EXCLUDED",
                    message=(
                        "No querying reference was supplied, so the user's own "
                        "enrolled template was not excluded. A returning user "
                        "will match themselves at a similarity of 1.0."
                    ),
                    stage="duplicate",
                    detail={},
                )
            )

        if gallery_size >= UNCALIBRATED_GALLERY_WARNING:
            warnings.append(
                AnalysisWarning(
                    code="DUPLICATE_THRESHOLD_UNCALIBRATED",
                    message=(
                        f"The gallery holds {gallery_size} templates and the "
                        f"threshold has not been derived against it. A 1:N "
                        f"false-match rate compounds with gallery size: at a "
                        f"per-comparison rate of 1e-4 a gallery this large "
                        f"produces a false duplicate on "
                        f"{system_far(1e-4, gallery_size):.0%} of queries. Run "
                        f"scripts/calibrate_duplicate_threshold.py against the "
                        f"live gallery."
                    ),
                    stage="duplicate",
                    detail={"gallery_size": gallery_size},
                )
            )

        if duplicates:
            warnings.append(
                AnalysisWarning(
                    code="DUPLICATE_MAY_BE_A_RELATIVE",
                    message=(
                        "A match is not proof of the same person. Face "
                        "recognition cannot separate identical twins at any "
                        "threshold, and siblings and cousins score well above "
                        "chance. Treat this as evidence for a human, not as a "
                        "finding."
                    ),
                    stage="duplicate",
                    detail={},
                )
            )

        return warnings

    def _empty_result(
        self,
        *,
        thresholds: MatchThresholds,
        started: float,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
        model_version: str = "",
    ) -> DuplicateCheckResult:
        """A result saying no search happened, and why."""
        result = DuplicateCheckResult(
            duplicate_found=False,
            searched=False,
            duplicate_threshold=thresholds.strong_match,
            review_threshold=thresholds.review,
            thresholds_validated=False,
            recommended_action="proceed",
            store=self._store.name,
            model_version=model_version,
            error_code=error_code,
            error_message=error_message,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        log.info("duplicate.not_searched", **result.summary())
        return result


def build_duplicate_service(
    settings: Settings | None = None, *, store: VectorStore | None = None
) -> DuplicateService:
    """Wire up a :class:`DuplicateService` from configuration.

    Falls back to the in-process gallery when no Qdrant endpoint is configured
    or the client is not installed. The fallback is loud - a warning naming the
    consequence - because an in-process gallery in a multi-replica deployment
    silently detects duplicates only against whichever pod answered.

    Args:
        settings: Service configuration.
        store: An explicit gallery adapter, overriding the configured one.

    Returns:
        A ready service.
    """
    from hamqadam_ai.duplicate_detection import build_store

    settings = settings or get_settings()

    if store is None:
        store = build_store(
            settings.duplicate, dimension=settings.embedding.dimension
        )
        if store.name == "memory":
            log.warning(
                "duplicate.using_in_process_gallery",
                note=(
                    "the gallery is lost on restart and not shared between "
                    "replicas; configure duplicate.backend=qdrant for any "
                    "deployment running more than one"
                ),
            )

    log.info(
        "duplicate.service_ready",
        store=store.name,
        duplicate_threshold=settings.duplicate.similarity_threshold,
        review_threshold=settings.duplicate.review_threshold,
    )
    return DuplicateService(store=store, settings=settings)


__all__ = ["UNCALIBRATED_GALLERY_WARNING", "DuplicateService", "build_duplicate_service"]
