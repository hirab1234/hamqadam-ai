"""MODULE 6 service - extract the CNIC portrait and match it to the selfie.

Pipeline for one card image::

    rectify  ->  upscale  ->  detect every face
                                    |
                          locate_portrait: which face is printed on the card?
                                    |
                    +---------------+---------------+
                    |                               |
              portrait found                  nothing admissible
                    |                               |
          quality (CNIC scale)              CNIC_FACE_NOT_FOUND
                    |
              embed (CNIC role)
                    |
        compare against the selfie embedding
                    |
             CnicFaceMatchResult

Ordering that is not arbitrary
------------------------------
**Rectify before detecting**, not after. Warping the card to its own
quadrilateral removes everything that is not on it, so a face held behind the
card never reaches the detector at all. Detecting first and filtering later
would work too, but it leaves the containment argument resting on a threshold
rather than on the pixels no longer existing.

**Upscale before detecting.** The portrait is a small, low-DPI print. On a
1012-pixel-wide rectified card the face inside the photo box is perhaps 120 px
across, which is close to where detectors start missing. Enlarging costs
milliseconds and recovers real portraits.

**Quality before embedding**, and quality is allowed to say no. An illegible
print embeds to a vector that is not wrong so much as meaningless, and a
meaningless vector produces a match score somebody will act on. Reporting the
portrait unusable is the honest outcome.

What this module does not decide
--------------------------------
The operating point. Module 4 owns ``selfie_vs_cnic`` and its calibration, and
this service feeds it through ``compare_pair`` rather than re-deriving a
threshold. Re-litigating it here would give two places to change one number.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError
from hamqadam_ai.documents.portrait import (
    REJECT_OFF_TEMPLATE,
    REJECT_TOO_LARGE,
    PortraitCandidate,
    PortraitLocation,
    locate_portrait,
)
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.matching.comparator import ComparisonType, MatchOutcome
from hamqadam_ai.ocr.preprocessing import RectifyResult, rectify_document
from hamqadam_ai.schemas.cnic_face import (
    CnicFaceMatchResult,
    CnicPortraitResult,
    PortraitCandidateModel,
)
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.detection import BoundingBoxModel, FaceDetectionResult
from hamqadam_ai.services.embedding_service import EmbeddingService
from hamqadam_ai.services.face_detection_service import FaceDetectionService
from hamqadam_ai.services.matching_service import MatchingService
from hamqadam_ai.services.quality_service import QualityService
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]

#: A face occupying more than this share of the frame is too big to be any
#: print on a card, whatever else is wrong with its placement. Used only to
#: pick the right guidance message, never to accept or reject.
_LARGE_FACE = 0.06


class CnicFaceService:
    """Locates the portrait on a CNIC and compares it with the live selfie.

    Args:
        detection: Module 1, used to find every face on the card.
        quality: Module 2, used on the CNIC-portrait scale.
        embedding: Module 3.
        matching: Module 4, which owns the CNIC operating point.
        settings: Service configuration.
    """

    __slots__ = (
        "_config",
        "_detection",
        "_embedding",
        "_matching",
        "_quality",
        "_settings",
    )

    def __init__(
        self,
        *,
        detection: FaceDetectionService,
        quality: QualityService,
        embedding: EmbeddingService,
        matching: MatchingService,
        settings: Settings,
    ) -> None:
        self._detection = detection
        self._quality = quality
        self._embedding = embedding
        self._matching = matching
        self._settings = settings
        self._config = settings.cnic_face

    # -- Public surface ------------------------------------------------------ #

    def extract_portrait(
        self, cnic_image: BgrImage
    ) -> tuple[CnicPortraitResult, FaceEmbedding | None]:
        """Find the portrait on a card and embed it.

        Separated from :meth:`match` because the pipeline in Module 10 embeds
        the portrait once and may compare it against more than one thing, and
        because a caller may legitimately want the portrait without a selfie
        to compare it to.

        Args:
            cnic_image: The photograph of the card, BGR uint8.

        Returns:
            ``(result, embedding)``. The embedding is ``None`` whenever the
            portrait was not found or was judged unusable - which the result
            explains.
        """
        if cnic_image.ndim != 3 or cnic_image.shape[2] != 3:
            raise ValueError(
                f"CNIC face extraction expects a (H, W, 3) BGR array; "
                f"got {cnic_image.shape}"
            )

        card, rectify, scale = self.prepare_card(cnic_image)
        height, width = card.shape[:2]

        # Strict mode. Without the card's own bounds, "this face is on the
        # card" rests on a geometric prior rather than on the pixels outside
        # the card no longer existing - and a deployment may reasonably decide
        # that is not good enough to verify an identity against. Refuse here
        # rather than searching anyway, which is what makes the setting mean
        # something.
        if not rectify.rectified and not self._config.allow_unrectified:
            refused = self._portrait_schema(
                location=PortraitLocation(portrait=None),
                rectify=rectify,
                width=width,
                height=height,
            )
            refused.error_code = ErrorCode.CNIC_FACE_NOT_FOUND
            refused.error_message = (
                "The edges of the card could not be found, and this deployment "
                "requires them before trusting a portrait. Photograph the whole "
                "card against a plain, contrasting background."
            )
            log.info("cnic_face.rectification_required", **refused.summary())
            return refused, None

        detection = self._detection.detect(card, role=ImageRole.CNIC_IMAGE)
        location = locate_portrait(
            [
                (_to_box(face.bounding_box), face.confidence)
                for face in detection.faces
            ],
            width=width,
            height=height,
            geometry=self._config.geometry,
            min_detector_confidence=self._config.min_detector_confidence,
        )

        result = self._portrait_schema(
            location=location, rectify=rectify, width=width, height=height
        )

        if not location.found or location.portrait is None:
            result.error_code = ErrorCode.CNIC_FACE_NOT_FOUND
            result.error_message = self._not_found_message(location)
            log.info("cnic_face.portrait_missing", **result.summary())
            return result, None

        chosen = location.portrait
        landmarks = self._landmarks_for(detection, chosen)

        quality = self._quality.assess(
            card,
            role=ImageRole.CNIC_PORTRAIT,
            face_box=chosen.box,
            landmarks=landmarks,
        )
        result.quality_score = quality.image_quality_score

        # `usable` already encodes the CNIC-portrait role policy - its own
        # composite floor and its own critical components. Re-deriving a
        # second threshold here would give two places to change one number,
        # and they would drift.
        if not quality.usable:
            result.usable = False
            result.error_code = ErrorCode.LOW_IMAGE_QUALITY
            result.error_message = (
                f"The portrait on the card was located but is too degraded to "
                f"compare (quality {quality.image_quality_score:.0f} against a "
                f"floor of {quality.min_required:.0f}"
                + (
                    f"; {', '.join(quality.critical_failures)} below the "
                    f"critical floor"
                    if quality.critical_failures
                    else ""
                )
                + "). Photograph the card in even light, holding it flat and "
                "avoiding glare on the laminate."
            )
            log.info("cnic_face.portrait_unusable", **result.summary())
            return result, None

        # ``embed_to_vector`` is the path Modules 4, 6 and 8 share, and it is
        # fail-fast: a caller asking for a vector has nothing to do with a
        # failure object. Here the failure *is* a business outcome, so it is
        # caught and reported rather than propagated.
        try:
            embedding = self._embedding.embed_to_vector(
                card,
                role=ImageRole.CNIC_PORTRAIT,
                box=chosen.box,
                landmarks=landmarks,
            )
        except HamqadamError as exc:
            result.usable = False
            result.error_code = exc.code
            result.error_message = (
                f"The portrait was located but could not be encoded: {exc}"
            )
            log.info("cnic_face.portrait_embedding_failed", **result.summary())
            return result, None

        result.usable = True
        # Scale is recorded so a caller can map the box back onto the image it
        # supplied; the boxes themselves stay in searched-image coordinates,
        # which is the only frame in which the area ratios mean anything.
        log.info("cnic_face.portrait_ready", scale=round(scale, 3), **result.summary())
        return result, embedding

    def match(
        self,
        cnic_image: BgrImage,
        selfie: FaceEmbedding | None,
    ) -> CnicFaceMatchResult:
        """Compare the live selfie against the portrait printed on the card.

        Args:
            cnic_image: The photograph of the card, BGR uint8.
            selfie: The live selfie's embedding, from Module 3. ``None`` is a
                normal input - the selfie may itself have failed - and yields
                a result that says no comparison was possible rather than a
                comparison that failed.

        Returns:
            The populated result. Never raises for a business outcome.
        """
        started = time.perf_counter()

        portrait_result, portrait_embedding = self.extract_portrait(cnic_image)
        warnings = self._warnings_for(portrait_result)

        outcome = self._matching.compare_pair(
            selfie, portrait_embedding, ComparisonType.CNIC
        )
        thresholds = self._settings.matching.thresholds_for("cnic")

        result = CnicFaceMatchResult(
            success=outcome.compared,
            cnic_identity_match=outcome.matched if outcome.compared else None,
            cnic_face_match_score=outcome.score if outcome.compared else None,
            similarity=outcome.similarity if outcome.compared else None,
            decision=str(outcome.decision) if outcome.compared else None,
            match_confidence=outcome.confidence if outcome.compared else None,
            portrait=portrait_result,
            strong_match_threshold=thresholds.strong_match,
            review_threshold=thresholds.review,
            thresholds_validated=False,
            error_code=self._match_error_code(portrait_result, selfie, outcome),
            error_message=self._match_error_message(portrait_result, selfie, outcome),
            warnings=warnings,
            model_version=(
                portrait_embedding.model_version if portrait_embedding else ""
            ),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        log.info("cnic_face.completed", **result.summary())
        return result

    async def match_async(
        self, cnic_image: BgrImage, selfie: FaceEmbedding | None
    ) -> CnicFaceMatchResult:
        """Match without blocking the event loop.

        Detection, quality and embedding are each tens of milliseconds of CPU;
        together they would stall every other in-flight verification.
        """
        return await asyncio.to_thread(self.match, cnic_image, selfie)

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        geometry = self._config.geometry
        return {
            "rectify_first": self._config.rectify_first,
            "allow_unrectified": self._config.allow_unrectified,
            "detection_min_width": self._config.detection_min_width,
            "min_detector_confidence": self._config.min_detector_confidence,
            "portrait_quality_policy": "quality.roles.cnic_portrait",
            "geometry": {
                "candidate_bands": [list(b) for b in geometry.candidate_bands],
                "vertical_band": list(geometry.vertical_band),
                "min_face_pixels": geometry.min_face_pixels,
                "max_face_area_ratio": geometry.max_face_area_ratio,
            },
            "thresholds_validated": False,
        }

    # -- Preparation --------------------------------------------------------- #

    def prepare_card(
        self, image: BgrImage
    ) -> tuple[BgrImage, RectifyResult, float]:
        """Isolate the card and enlarge it enough for the detector.

        Public because two callers outside this service need the same frame.
        The pipeline in Module 10 rectifies a card for OCR and would otherwise
        rectify it a second time here, paying twice for one Canny-and-warp;
        and anything drawing an overlay needs the frame the boxes are actually
        expressed in, which after rectification is the warped card rather than
        the image the caller passed.

        Args:
            image: The photograph of the card, BGR uint8.

        Returns:
            ``(card_image, rectify_result, cumulative_scale)``. Every box in
            :class:`~hamqadam_ai.schemas.cnic_face.CnicPortraitResult` is in
            the coordinates of ``card_image``.
        """
        if self._config.rectify_first:
            rectify = rectify_document(
                image,
                min_coverage=self._settings.ocr.preprocessing.min_card_coverage,
                aspect_tolerance=self._settings.ocr.preprocessing.aspect_tolerance,
            )
        else:
            rectify = RectifyResult(
                image=image, rectified=False, reason="rectification disabled"
            )

        if not rectify.rectified and not self._config.allow_unrectified:
            # Strict mode: the caller refuses the search entirely, so there is
            # nothing to gain from enlarging the frame first.
            return image, rectify, 1.0

        card = rectify.image
        width = card.shape[1]
        target = self._config.detection_min_width
        if width >= target:
            return card, rectify, 1.0

        scale = target / float(width)
        enlarged = cv2.resize(
            card, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        return np.asarray(enlarged, dtype=np.uint8), rectify, scale

    @staticmethod
    def _landmarks_for(
        detection: FaceDetectionResult, chosen: PortraitCandidate
    ) -> Landmarks5 | None:
        """Recover the detector's keypoints for the face that was chosen.

        Matched by index into the detector's own output rather than by
        geometry, so a card carrying a ghost reproduction cannot hand the
        primary portrait the ghost's landmarks.
        """
        if chosen.index >= len(detection.faces):
            return None
        face = detection.faces[chosen.index]
        if len(face.landmarks) != 5:
            return None
        return Landmarks5(
            points=np.array(
                [[point.x, point.y] for point in face.landmarks], dtype=np.float32
            )
        )

    # -- Result assembly ----------------------------------------------------- #

    def _portrait_schema(
        self,
        *,
        location: PortraitLocation,
        rectify: RectifyResult,
        width: int,
        height: int,
    ) -> CnicPortraitResult:
        """Render the portrait search as its response model."""
        return CnicPortraitResult(
            found=location.found,
            usable=False,
            portrait=(
                _candidate_model(location.portrait) if location.portrait else None
            ),
            has_ghost=location.has_ghost,
            foreign_face_count=location.foreign_face_count,
            candidates=[_candidate_model(c) for c in location.candidates],
            rectified=rectify.rectified,
            card_coverage=min(max(rectify.coverage, 0.0), 1.0),
            searched_width=width,
            searched_height=height,
        )

    def _warnings_for(self, portrait: CnicPortraitResult) -> list[AnalysisWarning]:
        """Non-fatal findings the Backend and the fraud engine should see."""
        warnings: list[AnalysisWarning] = []

        if portrait.foreign_face_count:
            warnings.append(
                AnalysisWarning(
                    code="CNIC_FOREIGN_FACE_PRESENT",
                    message=(
                        f"{portrait.foreign_face_count} face(s) were visible that "
                        f"are not plausibly printed on the card. The most common "
                        f"innocent cause is a person in shot behind it; the most "
                        f"common fraudulent one is a card held up in front of a "
                        f"face so the live face is compared instead of the "
                        f"portrait."
                    ),
                    stage="cnic_face",
                    detail={"count": portrait.foreign_face_count},
                )
            )

        if not portrait.rectified:
            warnings.append(
                AnalysisWarning(
                    code="CNIC_CARD_BOUNDS_NOT_FOUND",
                    message=(
                        "No card border could be located, so the portrait was "
                        "sought in the whole frame. Placement was still checked "
                        "against the card template, but the guarantee that "
                        "nothing outside the card was considered does not hold."
                    ),
                    stage="cnic_face",
                    detail={"coverage": portrait.card_coverage},
                )
            )

        if portrait.has_ghost:
            warnings.append(
                AnalysisWarning(
                    code="CNIC_GHOST_PORTRAIT_PRESENT",
                    message=(
                        "A second face was found on the card. Modern cards print "
                        "a faded reproduction of the portrait as a security "
                        "feature, so this is expected rather than suspicious."
                    ),
                    stage="cnic_face",
                    detail={},
                )
            )

        return warnings

    @staticmethod
    def _not_found_message(location: PortraitLocation) -> str:
        """Explain a failed search in terms of what to do about it."""
        if not location.candidates:
            return (
                "No face could be found on the card. Photograph the front of "
                "the CNIC - the side carrying the portrait - filling most of "
                "the frame."
            )

        # A face far bigger than any print on the card is the diagnostic
        # signature of the card being held up in front of somebody, so it gets
        # the message that actually helps. Checked before the off-template
        # case, because such a face is usually both.
        oversized = [
            c
            for c in location.candidates
            if REJECT_TOO_LARGE in c.rejections
            or (REJECT_OFF_TEMPLATE in c.rejections and c.area_ratio > _LARGE_FACE)
        ]
        if oversized:
            return (
                "A face was visible but is far too large to be the portrait "
                "printed on the card. Photograph the card on its own, lying "
                "flat, rather than holding it up in front of your face."
            )

        return (
            "A face was visible but not where the card's photograph sits. "
            "Photograph the whole front of the CNIC, flat and square to the "
            "camera."
        )

    @staticmethod
    def _match_error_code(
        portrait: CnicPortraitResult,
        selfie: FaceEmbedding | None,
        outcome: MatchOutcome,
    ) -> ErrorCode | None:
        """Which failure, if any, prevented a comparison.

        The portrait's own failure takes precedence: it is the one the user
        can act on by retaking the card photograph.
        """
        if outcome.compared:
            return None
        if portrait.error_code is not None:
            return portrait.error_code
        if selfie is None:
            return ErrorCode.FACE_NOT_DETECTED
        return ErrorCode.CNIC_FACE_NOT_FOUND

    @staticmethod
    def _match_error_message(
        portrait: CnicPortraitResult,
        selfie: FaceEmbedding | None,
        outcome: MatchOutcome,
    ) -> str | None:
        """Guidance matching the code above."""
        if outcome.compared:
            return None
        if portrait.error_message is not None:
            return portrait.error_message
        if selfie is None:
            return (
                "There is no usable live selfie to compare the card against. "
                "The card itself was read successfully."
            )
        return outcome.reason or "The comparison could not be made."


def _to_box(model: BoundingBoxModel) -> BoundingBox:
    """Convert the response model back into the geometry type."""
    return BoundingBox(x1=model.x1, y1=model.y1, x2=model.x2, y2=model.y2)


def _candidate_model(candidate: PortraitCandidate) -> PortraitCandidateModel:
    """Render one scored candidate as its response model."""
    box = candidate.box
    return PortraitCandidateModel(
        box=BoundingBoxModel(
            x1=box.x1,
            y1=box.y1,
            x2=box.x2,
            y2=box.y2,
            width=box.width,
            height=box.height,
        ),
        detector_confidence=min(max(candidate.detector_confidence, 0.0), 1.0),
        area_ratio=candidate.area_ratio,
        band_score=min(max(candidate.band_score, 0.0), 1.0),
        plausibility=min(max(candidate.plausibility, 0.0), 1.0),
        rejections=list(candidate.rejections),
    )


def build_cnic_face_service(settings: Settings | None = None) -> CnicFaceService:
    """Wire up a :class:`CnicFaceService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.

    Returns:
        A ready service, with Modules 1 through 4 constructed beneath it.
    """
    from hamqadam_ai.services import (
        build_embedding_service,
        build_face_detection_service,
        build_matching_service,
        build_quality_service,
    )

    settings = settings or get_settings()
    service = CnicFaceService(
        detection=build_face_detection_service(settings),
        quality=build_quality_service(settings),
        embedding=build_embedding_service(settings),
        matching=build_matching_service(settings),
        settings=settings,
    )

    log.info(
        "cnic_face.service_ready",
        rectify_first=settings.cnic_face.rectify_first,
        detection_min_width=settings.cnic_face.detection_min_width,
    )
    return service


__all__ = ["CnicFaceService", "build_cnic_face_service"]
