"""MODULE 5 service - orchestrates preprocessing, recognition and parsing.

Pipeline for one CNIC image::

    rectify  ->  deskew  ->  enhance  ->  upscale
                                            |
                          +-----------------+
                          |  for each candidate orientation, until accepted:
                          |      rotate -> recognise -> parse -> score
                          +-----------------+
                                            |
                                     validate -> CnicOcrResult

The orientation loop is the part worth explaining. Recognition costs roughly
four seconds a pass on CPU, so trying all four rotations exhaustively would
cost seventeen seconds per document - a third of the whole request budget for
one image. Instead the first pass is run upright, the geometry of the returned
text boxes is used to order the remaining candidates, and the loop exits as
soon as a pass yields a usable document. On an ordinary upright photograph
that is a single pass.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
    HamqadamError,
)
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.ocr.base import OcrEngine, OcrOutput
from hamqadam_ai.ocr.cnic.fields import CnicFields, FieldValue
from hamqadam_ai.ocr.cnic.parser import CnicParser, looks_like_cnic
from hamqadam_ai.ocr.cnic.validation import CnicValidator, ValidationOutcome
from hamqadam_ai.ocr.engines import build_engine
from hamqadam_ai.ocr.preprocessing import (
    Orientation,
    RectifyResult,
    describe_preprocessing,
    deskew,
    enhance_for_ocr,
    orientation_candidates,
    rectify_document,
    rotate_image,
    upscale_if_small,
)
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.ocr import (
    CnicOcrResult,
    ExtractedField,
    ValidationFindingModel,
)

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]

_FIELD_NAME_MAP = {
    "cnic_number": "cnic_number",
    "full_name": "name",
    "father_name": "father_name",
    "gender": "gender",
    "date_of_birth": "date_of_birth",
    "date_of_issue": "issue_date",
    "date_of_expiry": "expiry_date",
    "country_of_stay": "country_of_stay",
}


class OcrService:
    """Reads a Pakistani CNIC and returns its structured fields.

    Args:
        engine: The active OCR adapter.
        parser: Field extraction.
        validator: Cross-field validation.
        settings: Service configuration.
        requested_engine: The engine the configuration asked for first, so a
            fallback can be reported.
    """

    __slots__ = (
        "_config",
        "_engine",
        "_parser",
        "_requested_engine",
        "_settings",
        "_validator",
    )

    def __init__(
        self,
        *,
        engine: OcrEngine,
        parser: CnicParser,
        validator: CnicValidator,
        settings: Settings,
        requested_engine: str,
    ) -> None:
        self._engine = engine
        self._parser = parser
        self._validator = validator
        self._settings = settings
        self._config = settings.ocr
        self._requested_engine = requested_engine

    # -- Public surface ----------------------------------------------------- #

    def read(self, image: BgrImage) -> CnicOcrResult:
        """Read one CNIC image.

        Args:
            image: The photograph of the card, BGR uint8.

        Returns:
            The structured result. A failure to read is reported through
            ``success`` and ``error_code`` rather than raised - the pipeline
            still needs the face-matching findings for the same request.
        """
        started = time.perf_counter()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"CNIC OCR expects a (H, W, 3) BGR array; got {image.shape}"
            )

        prepared, rectify_result, preprocessing = self._preprocess(image)
        best = self._read_with_orientation(prepared)
        self._recover_missing(best)

        fields = best.fields
        validation = self._validator.validate(fields)
        is_cnic, document_score = looks_like_cnic(best.output)

        confidence = self._document_confidence(fields, validation)

        result = self._to_schema(
            fields=fields,
            validation=validation,
            output=best.output,
            confidence=confidence,
            is_cnic=is_cnic,
            document_score=document_score,
            rotation=best.orientation,
            attempts=best.attempts,
            rectify=rectify_result,
            preprocessing=preprocessing,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        log.info("ocr.completed", **result.summary())
        return result

    async def read_async(self, image: BgrImage) -> CnicOcrResult:
        """Read without blocking the event loop.

        Recognition is seconds of CPU, so this matters: without it a single
        CNIC would stall every other in-flight verification.
        """
        return await asyncio.to_thread(self.read, image)

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        return {
            "engine": self._engine.name,
            "engine_version": self._engine.version,
            "requested_engine": self._requested_engine,
            "using_fallback": self._engine.name != self._requested_engine,
            "language": self._config.language,
            "min_confidence": self._config.min_confidence,
            "min_required_fields": self._config.min_required_fields,
            "preprocessing": {
                "rectify": self._config.preprocessing.rectify,
                "deskew": self._config.preprocessing.deskew,
                "enhance": self._config.preprocessing.enhance,
            },
            "orientation": {
                "enabled": self._config.orientation.enabled,
                "early_exit": self._config.orientation.early_exit,
            },
        }

    def close(self) -> None:
        """Release the engine."""
        self._engine.close()

    # -- Preprocessing -------------------------------------------------------- #

    def _preprocess(
        self, image: BgrImage
    ) -> tuple[BgrImage, RectifyResult, dict[str, Any]]:
        """Rectify, deskew, enhance and upscale the card."""
        config = self._config.preprocessing

        if config.rectify:
            rectified = rectify_document(
                image,
                min_coverage=config.min_card_coverage,
                aspect_tolerance=config.aspect_tolerance,
            )
        else:
            rectified = RectifyResult(
                image=image, rectified=False, reason="rectification disabled"
            )

        working = rectified.image
        angle = 0.0
        if config.deskew:
            working, angle = deskew(working)

        if config.enhance:
            working = enhance_for_ocr(working, clip_limit=config.clahe_clip_limit)

        working, scale = upscale_if_small(working, min_width=config.min_width)

        return (
            working,
            rectified,
            describe_preprocessing(
                rectified,
                deskew_angle=angle,
                upscale=scale,
                enhanced=config.enhance,
            ),
        )

    # -- Orientation ---------------------------------------------------------- #

    class _Attempt:
        """One orientation's outcome, held so the best can be chosen."""

        __slots__ = (
            "anchors",
            "attempts",
            "fields",
            "image",
            "orientation",
            "output",
            "score",
        )

        def __init__(
            self,
            *,
            output: OcrOutput,
            fields: CnicFields,
            anchors: dict[str, Any],
            image: BgrImage,
            orientation: int,
            score: float,
            attempts: int,
        ) -> None:
            self.output = output
            self.fields = fields
            self.anchors = anchors
            self.image = image
            self.orientation = orientation
            self.score = score
            self.attempts = attempts

    def _read_with_orientation(self, image: BgrImage) -> OcrService._Attempt:
        """Try candidate rotations until one yields a usable document.

        The first pass is always upright. Its box geometry then orders the
        remaining candidates, because a page rotated a quarter turn produces
        text regions taller than they are wide - a signal available from the
        pass already paid for.
        """
        orientation_config = self._config.orientation

        first_output = self._recognise(image, Orientation.UPRIGHT)
        first_fields, first_anchors = self._parser.parse_with_anchors(first_output)
        first_score = self._candidate_score(first_output, first_fields)

        best = OcrService._Attempt(
            output=first_output,
            fields=first_fields,
            anchors=first_anchors,
            image=image,
            orientation=int(Orientation.UPRIGHT),
            score=first_score,
            attempts=1,
        )

        if not orientation_config.enabled:
            return best

        if orientation_config.early_exit and self._acceptable(
            first_output, first_fields
        ):
            return best

        candidates = [
            candidate
            for candidate in orientation_candidates(first_output.lines)
            if candidate is not Orientation.UPRIGHT
        ][: max(0, orientation_config.max_attempts - 1)]

        for candidate in candidates:
            rotated = rotate_image(image, candidate)
            output = self._recognise(rotated, candidate)
            fields, anchors = self._parser.parse_with_anchors(output)
            score = self._candidate_score(output, fields)
            attempts = best.attempts + 1

            if score > best.score:
                best = OcrService._Attempt(
                    output=output,
                    fields=fields,
                    anchors=anchors,
                    image=rotated,
                    orientation=int(candidate),
                    score=score,
                    attempts=attempts,
                )
            else:
                best.attempts = attempts

            if orientation_config.early_exit and self._acceptable(output, fields):
                break

        if best.orientation != 0:
            log.info(
                "ocr.orientation_corrected",
                rotation=best.orientation,
                attempts=best.attempts,
            )
        return best

    def _recognise(self, image: BgrImage, orientation: Orientation) -> OcrOutput:
        """Run one recognition pass, filtering weak lines."""
        output = self._engine.read(image)
        output.rotation_applied = int(orientation)
        floor = self._config.min_line_confidence
        output.lines = [line for line in output.lines if line.confidence >= floor]
        return output

    def _candidate_score(self, output: OcrOutput, fields: CnicFields) -> float:
        """Rank one orientation's result against the others.

        Weighted heavily towards *field extraction* rather than raw recognition
        confidence. A rotated page still recognises text well - the engine's
        per-line angle classifier sees to that - so confidence alone barely
        separates the candidates. What does separate them is whether the
        spatial layout made sense, which is exactly what successful field
        extraction measures.
        """
        if not output.lines:
            return 0.0

        score = 0.0
        # A valid identity number is the strongest single signal.
        if fields.cnic_number.present:
            score += 5.0
        # A label-anchored field proves the *layout* was read correctly,
        # which is the only thing that separates orientations: the
        # pattern-extracted fields succeed at any rotation.
        if fields.full_name.present:
            score += 4.0
        if fields.father_name.present:
            score += 2.0
        score += 2.0 * len(fields.present_fields)
        score += 0.5 * min(len(output.lines), 20) / 20.0
        score += output.mean_confidence
        return score

    def _acceptable(self, output: OcrOutput, fields: CnicFields) -> bool:
        """Whether this orientation is good enough to stop searching.

        Requires ``full_name``, which is the canary for orientation. The
        pattern-extracted fields - the identity number and the dates - are
        found by shape and succeed on a rotated card regardless, because the
        engine's per-line angle classifier reads the text correctly whichever
        way the page is turned. Measured on a card rotated 90 degrees, five
        of six fields still extracted and only the name failed, because it
        alone is located by looking *to the right of* its printed label.
        Accepting on field count therefore never corrected a rotation.
        """
        return (
            fields.cnic_number.present
            and fields.full_name.present
            and len(output.lines) >= self._config.orientation.min_lines_for_acceptance
            and len(fields.present_fields) >= self._config.min_required_fields
        )

    # -- Targeted recovery ------------------------------------------------------ #

    def _recover_missing(self, attempt: OcrService._Attempt) -> None:
        """Retry fields whose label was found but whose value was not.

        The full-page detector routinely misses the gender value because it is
        a single glyph in a wide margin, and text detectors are trained on
        lines. Cropping the region beside the label and recognising that alone
        removes the detection problem entirely: there is nothing else in the
        crop to compete with.

        Only labelled, currently-absent fields are retried, so this costs one
        small recognition pass at most and usually none at all.
        """
        fields = attempt.fields
        anchors = attempt.anchors
        if not anchors:
            return

        height, width = attempt.image.shape[:2]
        recoveries = (
            ("gender", "gender", self._parser.recover_gender),
            ("name", "full_name", self._parser.recover_name),
            ("father_name", "father_name", self._parser.recover_name),
        )

        for anchor_key, field_name, recover in recoveries:
            existing = fields.get(field_name)
            # A gender derived from the number is worth replacing with a real
            # reading: it restores the parity cross-check, which is the single
            # most useful validation this module has.
            if existing.present and existing.source != "derived":
                continue
            anchor = anchors.get(anchor_key)
            if anchor is None:
                continue

            region = self._parser.value_region(anchor, width=width, height=height)
            if region is None:
                continue

            x1, y1, x2, y2 = region
            crop = attempt.image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                lines = self._engine.recognise_crop(self._pad(crop))
            except HamqadamError as exc:
                log.info("ocr.recovery_failed", field=field_name, reason=str(exc))
                continue

            # Crop recognition bypasses detection, so its scores are on a
            # different scale: an isolated glyph measured 0.24 where the
            # same character in a line would score above 0.8. It gets its
            # own, lower floor rather than being judged against the
            # full-page one, which would discard every recovery.
            floor = self._config.min_crop_confidence
            recovered = recover(
                [
                    (line.text, line.confidence)
                    for line in lines
                    if line.confidence >= floor
                ]
            )
            if recovered is not None:
                fields.set(field_name, recovered)
                log.info(
                    "ocr.field_recovered",
                    field=field_name,
                    confidence=round(recovered.confidence, 3),
                )

        # Fall back to the parity-derived gender only if the crop pass also
        # found nothing.
        if not fields.gender.present and fields.implied_gender is not None:
            fields.gender = FieldValue(
                value=fields.implied_gender,
                confidence=0.6 * fields.cnic_number.confidence,
                source="derived",
            )

    @staticmethod
    def _pad(crop: BgrImage, *, margin: int = 12) -> BgrImage:
        """Surround a crop with quiet space.

        PP-OCR's detector expects text to sit inside a margin; a glyph flush
        against the edge of a tight crop is frequently missed altogether.
        """
        return np.asarray(
            cv2.copyMakeBorder(
                crop, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            ),
            dtype=np.uint8,
        )

    # -- Confidence ------------------------------------------------------------ #

    def _document_confidence(
        self, fields: CnicFields, validation: ValidationOutcome
    ) -> float:
        """Combine field confidences, completeness and validation into one score.

        Three factors, multiplied rather than averaged:

        * the weighted mean confidence of the fields that *were* read,
        * completeness, so a card yielding two perfect fields does not score
          as highly as one yielding six,
        * the validation penalty, so an internally inconsistent card is
          discounted however cleanly its individual fields were recognised.

        Multiplication rather than a weighted sum because these are not
        alternative sources of evidence that can compensate for one another -
        each is a necessary condition, and failing any one should pull the
        result down regardless of the others.
        """
        weights = self._config.field_weights
        total_weight = 0.0
        accumulated = 0.0

        for internal_name in fields.REQUIRED:
            weight = weights.get(internal_name, 0.0)
            if weight <= 0.0:
                continue
            value = fields.get(internal_name)
            if value.present:
                accumulated += weight * value.confidence
                total_weight += weight

        weighted = accumulated / total_weight if total_weight > 0.0 else 0.0

        # Completeness enters with a floor: a document that read four of six
        # fields perfectly is degraded, not worthless, so the factor runs
        # 0.55-1.0 rather than 0-1.
        completeness_factor = 0.55 + 0.45 * fields.completeness

        return float(
            max(
                0.0,
                min(
                    1.0,
                    weighted * completeness_factor * (1.0 - validation.confidence_penalty),
                ),
            )
        )

    # -- Response --------------------------------------------------------------- #

    def _to_schema(
        self,
        *,
        fields: CnicFields,
        validation: ValidationOutcome,
        output: OcrOutput,
        confidence: float,
        is_cnic: bool,
        document_score: float,
        rotation: int,
        attempts: int,
        rectify: RectifyResult,
        preprocessing: dict[str, Any],
        duration_ms: float,
    ) -> CnicOcrResult:
        """Render everything as the API response model."""
        detail = {
            external: ExtractedField(
                value=self._render(fields.get(internal).value),
                confidence=round(fields.get(internal).confidence, 4),
                source=fields.get(internal).source,
                corrected=fields.get(internal).corrected,
                present=fields.get(internal).present,
            )
            for internal, external in _FIELD_NAME_MAP.items()
        }

        warnings: list[AnalysisWarning] = []
        if not is_cnic:
            warnings.append(
                AnalysisWarning(
                    code="OCR_NOT_A_CNIC",
                    message=(
                        "The image does not appear to be a Pakistani CNIC. None "
                        "of the expected document markers were found and no "
                        "identity number was located."
                    ),
                    stage="ocr",
                    detail={"document_score": round(document_score, 3)},
                )
            )
        if rotation != 0:
            warnings.append(
                AnalysisWarning(
                    code="OCR_ROTATION_CORRECTED",
                    message=(
                        f"The card was photographed rotated {rotation} degrees "
                        f"and was corrected before reading."
                    ),
                    stage="ocr",
                    detail={"rotation": rotation, "attempts": attempts},
                )
            )
        if not rectify.rectified and rectify.reason:
            warnings.append(
                AnalysisWarning(
                    code="OCR_CARD_NOT_ISOLATED",
                    message=(
                        f"The card outline could not be located ({rectify.reason}), "
                        f"so the whole frame was read. Fields may be missed if the "
                        f"card is small in the photograph."
                    ),
                    stage="ocr",
                )
            )
        if self._engine.name != self._requested_engine:
            warnings.append(
                AnalysisWarning(
                    code="OCR_FALLBACK_ENGINE",
                    message=(
                        f"The configured OCR engine "
                        f"({self._requested_engine}) is unavailable; "
                        f"{self._engine.name} was used instead."
                    ),
                    stage="ocr",
                )
            )
        for finding in validation.findings:
            if finding.severity.value in {"error", "warning"}:
                warnings.append(
                    AnalysisWarning(
                        code=finding.code,
                        message=finding.message,
                        stage="ocr",
                        detail={"fields": list(finding.fields)},
                    )
                )

        enough_fields = len(fields.present_fields) >= self._config.min_required_fields
        success = bool(
            is_cnic and enough_fields and confidence >= self._config.min_confidence
        )

        error_code = None
        error_message = None
        if not success:
            # Separate codes because they ask the user for different things:
            # a document that is not a CNIC will never read as one however
            # many times it is rephotographed.
            error_code = (
                ErrorCode.CNIC_OCR_FAILED if is_cnic else ErrorCode.CNIC_NOT_RECOGNISED
            )
            error_message = self._failure_message(
                is_cnic=is_cnic,
                enough_fields=enough_fields,
                confidence=confidence,
                fields=fields,
                validation=validation,
            )

        return CnicOcrResult(
            success=success,
            is_cnic=is_cnic,
            cnic_number=fields.cnic_number.value,
            name=fields.full_name.value,
            father_name=fields.father_name.value,
            gender=fields.gender.value,
            date_of_birth=fields.date_of_birth.value,
            issue_date=fields.date_of_issue.value,
            expiry_date=fields.date_of_expiry.value,
            country_of_stay=fields.country_of_stay.value,
            province=fields.province,
            implied_gender=fields.implied_gender,
            is_expired=validation.expired,
            is_lifetime=fields.is_lifetime,
            ocr_confidence_score=round(confidence * 100.0, 2),
            field_confidence=round(fields.mean_confidence(), 4),
            completeness=round(fields.completeness, 4),
            fields_present=fields.present_fields,
            fields_missing=fields.missing_fields,
            consistent=validation.consistent,
            findings=[
                ValidationFindingModel(
                    code=finding.code,
                    severity=str(finding.severity),
                    message=finding.message,
                    fields=list(finding.fields),
                )
                for finding in validation.findings
            ],
            fields=detail,
            engine=self._engine.name,
            engine_version=self._engine.version,
            used_fallback_engine=self._engine.name != self._requested_engine,
            rotation_applied=rotation,
            preprocessing=preprocessing,
            lines_detected=output.line_count,
            orientation_attempts=attempts,
            error_code=error_code,
            error_message=error_message,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _render(value: Any) -> str | None:
        """Render a field value for the string-typed schema slot."""
        if value is None:
            return None
        if isinstance(value, dt.date):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _failure_message(
        *,
        is_cnic: bool,
        enough_fields: bool,
        confidence: float,
        fields: CnicFields,
        validation: ValidationOutcome,
    ) -> str:
        """Build actionable guidance from why the read failed.

        Every branch has to survive the question "what should the user do
        next?". Telling someone to retake a photograph when the card itself
        contradicts itself sends them round a loop that cannot terminate, so
        the inconsistency case is separated out and named.
        """
        if not is_cnic:
            return (
                "This does not look like a Pakistani identity card. Photograph "
                "the front of your CNIC, filling most of the frame."
            )
        if not enough_fields:
            missing = ", ".join(fields.missing_fields)
            return (
                f"Only part of the card could be read - {missing} could not be "
                f"made out. Retake the photograph in even light, with the whole "
                f"card flat and in focus."
            )
        if validation.errors:
            # The fields were read; they disagree with each other. Retaking the
            # photograph will reproduce the disagreement, because it is in the
            # document rather than in the capture.
            reasons = "; ".join(finding.message for finding in validation.errors)
            return (
                f"Every field was read, but the card is not internally "
                f"consistent: {reasons} This needs manual review rather than "
                f"another photograph."
            )
        return (
            f"The card was read but not clearly enough to rely on "
            f"(confidence {confidence * 100:.0f}%). Retake the photograph in "
            f"even light, avoiding glare on the laminate."
        )


def build_ocr_service(
    settings: Settings | None = None,
    *,
    today: dt.date | None = None,
    engine_chain: Sequence[str] | None = None,
) -> OcrService:
    """Wire up an :class:`OcrService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.
        today: Reference date for expiry and age validation. Injectable so the
            tests do not change behaviour as the calendar advances.
        engine_chain: Override the configured fallback order. Exists so one
            engine can be pinned for a comparison run or to reproduce a
            reading, without editing the deployed configuration.

    Returns:
        A ready service.

    Raises:
        DependencyUnavailableError: when no OCR engine could be constructed.
            Unlike face detection there is no bundled last resort - an OCR
            engine cannot be improvised from OpenCV primitives - so this is a
            genuine hard failure that should keep the pod out of the load
            balancer rather than let it accept traffic it cannot serve.
    """
    settings = settings or get_settings()
    config = settings.ocr
    chain = list(engine_chain) if engine_chain else list(config.engine_chain)
    if not chain:
        raise ConfigurationError("the OCR engine chain is empty")

    try:
        engine = build_engine(chain)
    except DependencyUnavailableError:
        log.error(
            "ocr.no_engine_available",
            chain=chain,
            note="the service cannot read CNIC documents",
        )
        raise

    log.info(
        "ocr.service_ready",
        engine=engine.name,
        version=engine.version,
        requested=chain[0],
        overridden=engine_chain is not None,
        min_confidence=config.min_confidence,
    )

    return OcrService(
        engine=engine,
        parser=CnicParser(),
        validator=CnicValidator(config, today=today),
        settings=settings,
        requested_engine=chain[0],
    )


__all__ = ["OcrService", "build_ocr_service"]
