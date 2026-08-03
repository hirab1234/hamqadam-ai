"""OCR engine adapters.

Three engines behind one port, in fallback order:

``onnx_ppocr``
    PP-OCR running on ONNX Runtime, via ``rapidocr-onnxruntime``. The default,
    for a reason that is architectural rather than about accuracy: it uses the
    same ONNX Runtime the detector and recogniser already run on, so it
    inherits the service's device selection and adds no second inference
    runtime to the image. The models are the same PP-OCR ones PaddleOCR ships
    and are bundled in the wheel, so there is nothing extra to download.

``paddleocr``
    The engine the specification names. Fully implemented and selected when
    installed. Held second only because ``paddlepaddle`` is a second, large
    inference runtime with its own device handling, which is a real cost in a
    container that already carries ONNX Runtime.

``easyocr``
    The specification's named fallback. Torch-based, and stronger than PP-OCR
    on some degraded captures, which is why it earns a place rather than being
    redundant.

Every adapter emits the same :class:`~hamqadam_ai.ocr.base.TextLine` in source
coordinates with a confidence in ``[0, 1]``, so the parser above them is
engine-agnostic and testable without any of them installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hamqadam_ai.core.exceptions import DependencyUnavailableError, InferenceError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.ocr.base import BgrImage, OcrEngine, TextLine, quad_from_box

log = get_logger(__name__)


def _as_quad(points: Any) -> tuple[tuple[float, float], ...]:
    """Coerce an engine's polygon into the four-corner tuple form."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if array.shape[0] < 4:
        x1, y1 = array.min(axis=0)
        x2, y2 = array.max(axis=0)
        return quad_from_box(float(x1), float(y1), float(x2), float(y2))
    return tuple((float(x), float(y)) for x, y in array[:4])


class OnnxPpOcrEngine(OcrEngine):
    """PP-OCR detection, angle classification and recognition on ONNX Runtime.

    Args:
        use_angle_classifier: Run PP-OCR's per-line angle classifier. Worth
            keeping on: it is the reason a card photographed sideways still
            yields correct text, at a measured cost of about 50 ms against the
            recognition stage's 2.8 s.
        text_score: Minimum recognition confidence for a line to be returned.
    """

    def __init__(
        self, *, use_angle_classifier: bool = True, text_score: float = 0.35
    ) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise DependencyUnavailableError(
                "rapidocr-onnxruntime",
                purpose="ONNX-based PP-OCR text recognition",
                extra="rapidocr-onnxruntime",
                cause=exc,
            ) from exc

        version = "unknown"
        try:
            from importlib.metadata import version as package_version

            version = package_version("rapidocr-onnxruntime")
        except Exception:  # noqa: BLE001 - version is diagnostic only
            pass

        super().__init__(name="onnx_ppocr", version=f"rapidocr-{version}")
        self._engine = RapidOCR(text_score=text_score)
        self._use_angle_classifier = use_angle_classifier

        log.info(
            "ocr.onnx_ppocr.ready",
            version=self.version,
            angle_classifier=use_angle_classifier,
            text_score=text_score,
        )

    def recognise(self, image: BgrImage) -> list[TextLine]:
        """Detect and recognise every text line."""
        try:
            result, _elapsed = self._engine(
                image, use_cls=self._use_angle_classifier
            )
        except Exception as exc:  # noqa: BLE001 - normalised into a typed error
            raise InferenceError(
                f"PP-OCR recognition failed: {exc}",
                details={"engine": self.name},
                cause=exc,
            ) from exc

        if not result:
            return []

        lines: list[TextLine] = []
        for entry in result:
            # (polygon, text, score)
            quad, text, score = entry[0], entry[1], entry[2]
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=_as_quad(quad),
                )
            )
        return lines

    def recognise_crop(self, image: BgrImage) -> list[TextLine]:
        """Feed the crop straight to the recognition head."""
        recogniser = getattr(self._engine, "text_recognizer", None)
        if recogniser is None:  # pragma: no cover - older rapidocr builds
            return self.recognise(image)

        try:
            output = recogniser([image])
        except Exception as exc:  # noqa: BLE001 - degrade to full detection
            log.info("ocr.crop_recognition_failed", reason=str(exc))
            return self.recognise(image)

        results = output[0] if isinstance(output, tuple) else output
        if not results:
            return []

        height, width = image.shape[:2]
        lines: list[TextLine] = []
        for text, score in results:
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=quad_from_box(0.0, 0.0, float(width), float(height)),
                )
            )
        return lines

    def close(self) -> None:
        """Release the sessions."""
        self._engine = None


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR, the engine the specification names.

    Args:
        language: Recognition language. ``en`` for the CNIC's English side.
        use_angle_classifier: Run the per-line angle classifier.
    """

    def __init__(
        self, *, language: str = "en", use_angle_classifier: bool = True
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise DependencyUnavailableError(
                "paddleocr",
                purpose="PaddleOCR text recognition",
                extra="paddleocr paddlepaddle",
                cause=exc,
            ) from exc

        version = "unknown"
        try:
            from importlib.metadata import version as package_version

            version = package_version("paddleocr")
        except Exception:  # noqa: BLE001 - version is diagnostic only
            pass

        super().__init__(name="paddleocr", version=f"paddleocr-{version}")
        # `show_log` was removed in PaddleOCR 3.x; passing it to a 3.x build
        # raises, and omitting it on a 2.x build merely leaves it verbose.
        try:
            self._engine = PaddleOCR(
                use_angle_cls=use_angle_classifier, lang=language, show_log=False
            )
        except TypeError:
            self._engine = PaddleOCR(use_angle_cls=use_angle_classifier, lang=language)
        self._use_angle_classifier = use_angle_classifier

        log.info("ocr.paddleocr.ready", version=self.version, language=language)

    def recognise(self, image: BgrImage) -> list[TextLine]:
        """Detect and recognise every text line."""
        try:
            result = self._engine.ocr(image, cls=self._use_angle_classifier)
        except Exception as exc:  # noqa: BLE001 - normalised into a typed error
            raise InferenceError(
                f"PaddleOCR recognition failed: {exc}",
                details={"engine": self.name},
                cause=exc,
            ) from exc

        if not result:
            return []

        # PaddleOCR nests per-image results; a single image yields one page.
        page = result[0] if isinstance(result[0], list) else result
        if not page:
            return []

        lines: list[TextLine] = []
        for entry in page:
            if not entry:
                continue
            quad, payload = entry[0], entry[1]
            text, score = payload[0], payload[1]
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=_as_quad(quad),
                )
            )
        return lines

    def recognise_crop(self, image: BgrImage) -> list[TextLine]:
        """Recognise without detection, via PaddleOCR's ``det=False`` mode."""
        try:
            output = self._engine.ocr(image, det=False, cls=False)
        except Exception as exc:  # noqa: BLE001 - degrade to full detection
            log.info("ocr.crop_recognition_failed", reason=str(exc))
            return self.recognise(image)

        if not output:
            return []
        results = output[0] if isinstance(output[0], list) else output

        height, width = image.shape[:2]
        lines: list[TextLine] = []
        for entry in results:
            if not entry:
                continue
            text, score = entry[0], entry[1]
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=quad_from_box(0.0, 0.0, float(width), float(height)),
                )
            )
        return lines

    def close(self) -> None:
        """Release the engine."""
        self._engine = None


class EasyOcrEngine(OcrEngine):
    """EasyOCR, the specification's named fallback.

    Args:
        languages: Recognition languages.
        use_gpu: Let EasyOCR use CUDA when torch reports it available.
    """

    def __init__(
        self, *, languages: tuple[str, ...] = ("en",), use_gpu: bool = False
    ) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise DependencyUnavailableError(
                "easyocr",
                purpose="EasyOCR text recognition",
                extra="easyocr",
                cause=exc,
            ) from exc

        version = "unknown"
        try:
            from importlib.metadata import version as package_version

            version = package_version("easyocr")
        except Exception:  # noqa: BLE001 - version is diagnostic only
            pass

        super().__init__(name="easyocr", version=f"easyocr-{version}")
        self._reader = easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)

        log.info("ocr.easyocr.ready", version=self.version, languages=list(languages))

    def recognise(self, image: BgrImage) -> list[TextLine]:
        """Detect and recognise every text line."""
        try:
            result = self._reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001 - normalised into a typed error
            raise InferenceError(
                f"EasyOCR recognition failed: {exc}",
                details={"engine": self.name},
                cause=exc,
            ) from exc

        lines: list[TextLine] = []
        for quad, text, score in result or []:
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=_as_quad(quad),
                )
            )
        return lines

    def recognise_crop(self, image: BgrImage) -> list[TextLine]:
        """Recognise without detection, via EasyOCR's ``recognize``."""
        import cv2

        height, width = image.shape[:2]
        try:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # A single box covering the whole crop, which is the contract
            # `recognize` expects when detection is skipped.
            output = self._reader.recognize(
                grey, horizontal_list=[[0, width, 0, height]], free_list=[]
            )
        except Exception as exc:  # noqa: BLE001 - degrade to full detection
            log.info("ocr.crop_recognition_failed", reason=str(exc))
            return self.recognise(image)

        lines: list[TextLine] = []
        for _quad, text, score in output or []:
            if not text:
                continue
            lines.append(
                TextLine(
                    text=str(text),
                    confidence=float(score),
                    quad=quad_from_box(0.0, 0.0, float(width), float(height)),
                )
            )
        return lines

    def close(self) -> None:
        """Release the reader."""
        self._reader = None


#: Construction order. Each is attempted in turn; the first that imports and
#: initialises becomes the active engine.
ENGINE_BUILDERS: dict[str, Any] = {
    "onnx_ppocr": OnnxPpOcrEngine,
    "paddleocr": PaddleOcrEngine,
    "easyocr": EasyOcrEngine,
}


def build_engine(chain: list[str]) -> OcrEngine:
    """Construct the first available engine from a preference chain.

    Args:
        chain: Engine names in preference order.

    Returns:
        The first engine that could be constructed.

    Raises:
        DependencyUnavailableError: when none of them could be. Unlike face
            detection there is no bundled last resort here - an OCR engine
            cannot be improvised from OpenCV primitives - so this is a genuine
            hard failure that should keep the pod out of the load balancer.
    """
    attempted: list[str] = []
    for name in chain:
        builder = ENGINE_BUILDERS.get(name)
        if builder is None:
            log.warning("ocr.unknown_engine", engine=name, known=sorted(ENGINE_BUILDERS))
            continue
        try:
            engine: OcrEngine = builder()
        except DependencyUnavailableError as exc:
            attempted.append(f"{name}: not installed")
            log.info("ocr.engine_unavailable", engine=name, reason=str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - try the next engine
            attempted.append(f"{name}: {exc}")
            log.warning("ocr.engine_init_failed", engine=name, reason=str(exc))
            continue
        if name != chain[0]:
            log.warning(
                "ocr.using_fallback_engine",
                requested=chain[0],
                active=name,
                note="OCR accuracy may differ from the calibrated engine",
            )
        return engine

    raise DependencyUnavailableError(
        "an OCR engine",
        purpose="reading the CNIC",
        extra="rapidocr-onnxruntime  (or paddleocr, or easyocr)",
    )


__all__ = [
    "ENGINE_BUILDERS",
    "EasyOcrEngine",
    "OnnxPpOcrEngine",
    "PaddleOcrEngine",
    "build_engine",
]
