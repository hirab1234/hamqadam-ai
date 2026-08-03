"""The authenticity-detector port and its shared value objects.

What this package answers
-------------------------
Not "who is this?" - Modules 3 and 4 do that - but **"is this a genuine
camera photograph, taken by the person uploading it?"** A profile photo that
is a screenshot of somebody else's social media, a photograph of a laptop
screen, or a cartoon avatar is a different problem from a photo of the wrong
person, and it needs different evidence.

Contract every detector honours
-------------------------------
* It takes an :class:`AuthenticityContext` - the shared views several
  detectors need - and returns one :class:`AuthenticitySignal`.
* It **never raises for difficult input**. A degenerate image yields a signal
  with a note explaining why, because the caller must still produce a complete
  report for the other images in the request.
* It reports every raw measurement alongside the verdict. A rejection that
  cannot be explained cannot be appealed, and these rejections accuse a user
  of uploading something dishonest.
* It says how confident it is. Every detector here has a false-positive mode,
  and a detector that cannot express doubt forces the aggregator to treat a
  marginal reading as certain.

Why not a trained classifier
----------------------------
A CNN trained on screenshots-versus-photographs would very likely beat these
measurements. There is no such model available to this project, and shipping a
stub that returns a plausible-looking constant would be worse than shipping
nothing. What is here is signal processing whose separation has been measured
against controlled fixtures, with the margins written down - see
``docs/modules/07_profile_image_analysis.md`` - so the next engineer knows
exactly how much confidence the numbers support.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

BgrImage = npt.NDArray[np.uint8]
GrayImage = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]

#: Side the image is resampled to for spectral work. A power of two keeps the
#: FFT fast, and 512 leaves the display-grid frequencies a recapture produces
#: comfortably inside the band that survives resampling.
SPECTRAL_SIZE = 512


@dataclass(frozen=True, slots=True)
class AuthenticitySignal:
    """One detector's verdict on one image.

    Attributes:
        name: Detector name, matching its configuration key.
        triggered: Whether the detector believes it has found its target.
        confidence: How strongly, in ``[0, 1]``. Reported even when
            ``triggered`` is False, where it means "confidence there is
            nothing here" is *not* implied - it is the strength of the
            evidence for the finding, which is simply low.
        measurements: Raw values behind the verdict, keyed by sub-metric.
            Reported verbatim: these are the numbers an engineer recalibrates
            against.
        note: Why the detector could not measure, when it could not.
    """

    name: str
    triggered: bool
    confidence: float
    measurements: dict[str, float] = field(default_factory=dict)
    note: str | None = None

    @property
    def measured(self) -> bool:
        """Whether the detector produced a usable reading."""
        return self.note is None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, safe for logs - carries no pixels."""
        return {
            "name": self.name,
            "triggered": self.triggered,
            "confidence": round(self.confidence, 4),
            "measurements": {
                key: round(float(value), 6) for key, value in self.measurements.items()
            },
            "note": self.note,
        }


# No ``slots=True``: every view below is a ``cached_property``, and caching
# needs an instance ``__dict__``. Matches ``QualityContext``, which is the same
# shape for the same reason.
@dataclass
class AuthenticityContext:
    """Views of one image that several detectors need in common.

    Greyscale conversion, the spectral resample and the block decomposition
    are each wanted by two or three detectors. Computing them once removes
    most of the module's arithmetic and - more importantly - guarantees every
    detector measures identical pixels, without which the measured separation
    margins would not transfer.

    Attributes:
        image: The source image, BGR uint8.
    """

    image: BgrImage

    @cached_property
    def height(self) -> int:
        """Source height in pixels."""
        return int(self.image.shape[0])

    @cached_property
    def width(self) -> int:
        """Source width in pixels."""
        return int(self.image.shape[1])

    @cached_property
    def grey(self) -> GrayImage:
        """Single-channel view of the source."""
        return np.asarray(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), dtype=np.uint8
        )

    @cached_property
    def grey_f32(self) -> FloatArray:
        """Greyscale as float, for arithmetic that would overflow uint8."""
        return self.grey.astype(np.float32)

    @cached_property
    def spectral_grey(self) -> FloatArray:
        """Greyscale resampled to a fixed square for Fourier analysis.

        Fixed size because the frequencies a display grid produces are a
        property of the *capture*, and comparing them across images requires a
        common sampling grid. ``INTER_AREA`` because it is the only OpenCV
        interpolation that low-pass filters on the way down; anything else
        aliases, which would manufacture exactly the periodic peaks the moire
        detector looks for.
        """
        return np.asarray(
            cv2.resize(
                self.grey, (SPECTRAL_SIZE, SPECTRAL_SIZE),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.float32,
        )

    @cached_property
    def log_spectrum(self) -> FloatArray:
        """Centred log power spectrum of :attr:`spectral_grey`.

        Hann-windowed on both axes. Without a window, the image's own edges
        become a discontinuity at the frame boundary and smear energy along
        the axes - which is precisely where a detector hunting for isolated
        peaks would misread it.
        """
        window = np.outer(np.hanning(SPECTRAL_SIZE), np.hanning(SPECTRAL_SIZE))
        transformed = np.fft.fftshift(np.fft.fft2(self.spectral_grey * window))
        return np.asarray(np.log1p(np.abs(transformed)), dtype=np.float32)

    @cached_property
    def radius_map(self) -> FloatArray:
        """Distance from the spectrum's centre, in pixels."""
        centre = SPECTRAL_SIZE // 2
        ys, xs = np.mgrid[0:SPECTRAL_SIZE, 0:SPECTRAL_SIZE]
        return np.asarray(
            np.sqrt((xs - centre) ** 2 + (ys - centre) ** 2), dtype=np.float32
        )

    @cached_property
    def constant_row_mask(self) -> npt.NDArray[np.bool_]:
        """Rows whose pixels are constant across the **full** image width.

        The load-bearing measurement of the screenshot detector, and the
        reason it works where a flat-block measure does not. Heavy JPEG
        quantisation flattens 8x8 blocks in smooth areas - measured at 35% of
        blocks on a quality-12 photograph, indistinguishable from a real
        screenshot - but it cannot flatten a row spanning the whole frame,
        because a photograph's content varies horizontally somewhere along it.
        """
        rows = self.grey.astype(np.int16)
        return np.asarray(rows.max(axis=1) - rows.min(axis=1) <= 2)

    def blocks(self, size: int = 16) -> FloatArray:
        """Non-overlapping square blocks of the greyscale image, flattened.

        Returns:
            ``(block_count, size * size)``. Empty when the image is smaller
            than one block.
        """
        rows, cols = self.height // size, self.width // size
        if rows == 0 or cols == 0:
            return np.zeros((0, size * size), dtype=np.float32)
        trimmed = self.grey_f32[: rows * size, : cols * size]
        stacked = trimmed.reshape(rows, size, cols, size).swapaxes(1, 2)
        return np.asarray(stacked.reshape(rows * cols, -1), dtype=np.float32)

    @cached_property
    def degenerate(self) -> bool:
        """Whether the image is too uniform or too small to analyse.

        A flat colour field satisfies almost every "this is not a photograph"
        test trivially, and a detector reporting high confidence on it is
        reporting confidence in an arithmetic artefact.
        """
        if min(self.height, self.width) < 32:
            return True
        return bool(self.grey_f32.std() < 1.0)


class AuthenticityDetector(abc.ABC):
    """Abstract detector for one way an image can fail to be a real capture.

    Args:
        name: Short identifier, matching the configuration key.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Measure one image.

        Args:
            context: Shared views of the image.

        Returns:
            The signal. Never raises for difficult input.
        """

    def safe_analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Run :meth:`analyse`, converting any fault into an unmeasured signal.

        A detector that throws must not take the whole report with it: the
        other detectors' findings are still worth having, and a profile photo
        should not be rejected because one measurement hit an edge case.
        """
        try:
            return self.analyse(context)
        except Exception as exc:  # noqa: BLE001 - normalised into a note
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note=f"{type(exc).__name__}: {exc}",
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


def ramp(value: float, *, floor: float, ceiling: float) -> float:
    """Map a measurement onto ``[0, 1]`` between two anchors.

    Linear rather than smoothstep, deliberately. These anchors sit in the wide
    empty gap between two measured populations, so the exact shape of the
    interpolation across that gap carries no information - and a straight line
    is the one a reader can invert in their head when asked why a photograph
    scored what it did.

    Args:
        value: The measurement.
        floor: Value mapping to 0.0.
        ceiling: Value mapping to 1.0. May be below ``floor``, for metrics
            where lower means more suspicious.
    """
    if ceiling == floor:
        return 0.0 if value < ceiling else 1.0
    position = (value - floor) / (ceiling - floor)
    return float(min(max(position, 0.0), 1.0))


__all__ = [
    "SPECTRAL_SIZE",
    "AuthenticityContext",
    "AuthenticityDetector",
    "AuthenticitySignal",
    "BgrImage",
    "ramp",
]
