"""The four authenticity detectors, against controlled fixtures.

No model weights: these are signal-processing measurements, and every one runs
on a bare install. What they need is the public-domain reference photograph,
so the module skips cleanly when neither matplotlib nor scikit-image is there.

Every threshold these tests pin was chosen from a measured gap between two
populations, and the populations are named in the assertions. That matters
more here than elsewhere in the codebase: a triggered detector accuses a user
of uploading something dishonest.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hamqadam_ai.authenticity import (
    AuthenticityContext,
    MoireDetector,
    PrintRecaptureDetector,
    ScreenshotDetector,
    SyntheticImageDetector,
    build_detectors,
)
from hamqadam_ai.authenticity.base import ramp
from hamqadam_ai.core.config import get_settings
from tests.fixtures.profile_images import (
    as_heavily_compressed,
    as_print_recapture,
    as_recompressed,
    as_screen_recapture,
    as_screenshot,
    as_synthetic_render,
    flat_colour,
    landscape_photo,
    reference_photo,
)


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    image = reference_photo()
    if image is None:
        pytest.skip("no public-domain reference photograph installed")
    return image


@pytest.fixture
def config():  # noqa: ANN201 - pytest fixture
    return get_settings().profile


def letterbox(
    image: np.ndarray, colour: tuple[int, int, int] = (255, 255, 255), pad: float = 0.25
) -> np.ndarray:
    """Pad top and bottom, as a social app does to square an upload."""
    rows = int(image.shape[0] * pad)
    return cv2.copyMakeBorder(
        image, rows, rows, 0, 0, cv2.BORDER_CONSTANT, value=colour
    )


def confidence(detector, image: np.ndarray) -> float:
    """Run one detector over one image."""
    return detector.safe_analyse(AuthenticityContext(image=image)).confidence


# --------------------------------------------------------------------------- #
# Screenshot
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_screenshot_is_detected(photo: np.ndarray, config) -> None:
    detector = ScreenshotDetector(config.screenshot)
    signal = detector.safe_analyse(AuthenticityContext(image=as_screenshot(photo)))

    assert signal.triggered is True
    assert signal.measurements["interior_constant_rows"] > 0.10


@pytest.mark.unit
@pytest.mark.parametrize("resolution", [(1080, 1920), (1170, 2532), (1284, 2778)])
def test_screenshots_at_several_device_sizes_are_detected(
    photo: np.ndarray, config, resolution: tuple[int, int]
) -> None:
    detector = ScreenshotDetector(config.screenshot)
    shot = as_screenshot(photo, resolution=resolution)

    assert detector.safe_analyse(AuthenticityContext(image=shot)).triggered is True


@pytest.mark.unit
def test_a_genuine_photograph_is_not_a_screenshot(
    photo: np.ndarray, config
) -> None:
    detector = ScreenshotDetector(config.screenshot)
    assert confidence(detector, photo) == pytest.approx(0.0)


@pytest.mark.unit
def test_a_heavily_compressed_photograph_is_not_a_screenshot(
    photo: np.ndarray, config
) -> None:
    """The hard negative, and the reason this detector counts constant *rows*
    rather than flat blocks.

    Measured: a quality-12 JPEG has exactly-zero-variance 8x8 blocks across
    35% of the frame, against 36% for a real screenshot. Flatness cannot
    separate them. A full-width constant row can, because the photograph's
    content varies horizontally somewhere along every row.
    """
    detector = ScreenshotDetector(config.screenshot)
    assert detector.safe_analyse(
        AuthenticityContext(image=as_heavily_compressed(photo))
    ).triggered is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("colour", "pad"), [((255, 255, 255), 0.25), ((0, 0, 0), 0.15), ((245, 245, 245), 0.2)]
)
def test_a_padded_photograph_is_not_a_screenshot(
    photo: np.ndarray, config, colour: tuple[int, int, int], pad: float
) -> None:
    """The false positive that shaped this detector.

    Social apps pad uploads to a square. On the raw measure a letterboxed
    photograph scores 0.333 - identical to a screenshot - so the first version
    would have flagged a large share of honest uploads. Stripping contiguous
    constant borders before measuring is what separates them.
    """
    detector = ScreenshotDetector(config.screenshot)
    padded = letterbox(photo, colour, pad)

    assert detector.safe_analyse(AuthenticityContext(image=padded)).triggered is False


@pytest.mark.unit
def test_padding_is_reported_rather_than_silently_discarded(
    photo: np.ndarray, config
) -> None:
    """A reviewer asking why a padded image scored as it did needs to see that
    the padding was found and stripped, not infer it."""
    detector = ScreenshotDetector(config.screenshot)
    signal = detector.safe_analyse(AuthenticityContext(image=letterbox(photo)))

    assert signal.measurements["padding_rows_top"] > 50
    assert signal.measurements["padding_rows_bottom"] > 50


@pytest.mark.unit
def test_cropping_defeats_the_screenshot_detector(
    photo: np.ndarray, config
) -> None:
    """A known and documented evasion, pinned so it cannot be forgotten.

    Cropping ~10% off a screenshot puts its flat chrome against the new frame
    edge, where it is stripped as padding. The two are geometrically identical
    and cannot be told apart without semantic understanding. The trade favours
    not accusing innocent users of dishonesty, which is the right default -
    but the limitation is real and this test is where it is written down.
    """
    detector = ScreenshotDetector(config.screenshot)
    shot = as_screenshot(photo)
    height = shot.shape[0]
    cropped = shot[int(height * 0.05):int(height * 0.95)]

    assert detector.safe_analyse(AuthenticityContext(image=shot)).triggered is True
    assert detector.safe_analyse(AuthenticityContext(image=cropped)).triggered is False


# --------------------------------------------------------------------------- #
# Screen recapture
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("pitch", [3, 4, 5, 6, 8])
def test_a_photograph_of_a_screen_is_detected(
    photo: np.ndarray, config, pitch: int
) -> None:
    """Across display pixel pitches. The interference frequency moves with the
    pitch, so a detector keyed to one spacing would miss the others."""
    detector = MoireDetector(config.moire)
    recapture = as_screen_recapture(photo, pixel_pitch=pitch)

    assert detector.safe_analyse(AuthenticityContext(image=recapture)).triggered is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "maker",
    [
        pytest.param(lambda p: p, id="genuine"),
        pytest.param(as_heavily_compressed, id="jpeg_q12"),
        pytest.param(as_recompressed, id="recompressed"),
        pytest.param(as_screenshot, id="screenshot"),
        pytest.param(as_print_recapture, id="print"),
    ],
)
def test_nothing_else_reads_as_a_screen_recapture(
    photo: np.ndarray, config, maker
) -> None:
    """Measured, everything that is not a screen capture sits in a prominence
    band 0.05 wide, against 3.4-4.2 for a capture. The widest separation any
    measurement in this module achieves."""
    detector = MoireDetector(config.moire)
    assert detector.safe_analyse(
        AuthenticityContext(image=maker(photo))
    ).triggered is False


@pytest.mark.unit
def test_the_moire_detector_survives_re_encoding(photo: np.ndarray, config) -> None:
    """Robust to compression, which matters: a recapture usually reaches the
    service through at least one messaging app."""
    from tests.fixtures.profile_images import _jpeg

    detector = MoireDetector(config.moire)
    recapture = as_screen_recapture(photo)

    assert detector.safe_analyse(
        AuthenticityContext(image=_jpeg(recapture, quality=30))
    ).triggered is True


@pytest.mark.unit
def test_heavy_downscaling_defeats_the_moire_detector(
    photo: np.ndarray, config
) -> None:
    """The documented limit, pinned. Moire depends on the display grid
    surviving into the pixels; resample hard enough and the grid frequency
    goes with it. A null result from this detector is not evidence of a
    genuine capture, and this is why."""
    detector = MoireDetector(config.moire)
    recapture = as_screen_recapture(photo)
    small = cv2.resize(recapture, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    assert detector.safe_analyse(AuthenticityContext(image=recapture)).triggered is True
    assert detector.safe_analyse(AuthenticityContext(image=small)).triggered is False


@pytest.mark.unit
def test_the_peak_location_is_reported(photo: np.ndarray, config) -> None:
    """So an engineer can check the peak sits where a display grid would put
    it, rather than trusting the verdict."""
    detector = MoireDetector(config.moire)
    signal = detector.safe_analyse(
        AuthenticityContext(image=as_screen_recapture(photo))
    )

    assert 0.0 < signal.measurements["peak_radius_fraction"] <= 1.5
    assert signal.measurements["mirror_symmetry"] > 0.3


# --------------------------------------------------------------------------- #
# Print recapture
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_photograph_of_a_print_is_detected(photo: np.ndarray, config) -> None:
    detector = PrintRecaptureDetector(config.print_recapture)
    signal = detector.safe_analyse(
        AuthenticityContext(image=as_print_recapture(photo))
    )

    assert signal.triggered is True
    assert signal.measurements["high_frequency_share"] < 0.006


@pytest.mark.unit
@pytest.mark.parametrize(
    "maker",
    [
        pytest.param(lambda p: p, id="genuine"),
        pytest.param(as_heavily_compressed, id="jpeg_q12"),
        pytest.param(as_recompressed, id="recompressed"),
    ],
)
def test_a_real_photograph_is_not_a_print(photo: np.ndarray, config, maker) -> None:
    """Including the quality-12 JPEG, which loses high-frequency content too -
    just an order of magnitude less than a print does."""
    detector = PrintRecaptureDetector(config.print_recapture)
    assert detector.safe_analyse(
        AuthenticityContext(image=maker(photo))
    ).triggered is False


# --------------------------------------------------------------------------- #
# Synthetic artwork
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_rendered_artwork_is_detected(config) -> None:
    detector = SyntheticImageDetector(config.synthetic)
    signal = detector.safe_analyse(AuthenticityContext(image=as_synthetic_render()))

    assert signal.triggered is True
    assert signal.measurements["unique_colour_ratio"] < 0.01
    assert signal.measurements["flat_block_fraction"] > 0.80


@pytest.mark.unit
@pytest.mark.parametrize(
    "maker",
    [
        pytest.param(lambda p: p, id="genuine"),
        pytest.param(as_heavily_compressed, id="jpeg_q12"),
        pytest.param(as_screenshot, id="screenshot"),
        pytest.param(as_print_recapture, id="print"),
    ],
)
def test_photographs_are_not_synthetic(photo: np.ndarray, config, maker) -> None:
    """The quality-12 JPEG is the case that forces both signals to agree: it
    has a 0.355 flat-block fraction, which alone would convict it."""
    detector = SyntheticImageDetector(config.synthetic)
    assert detector.safe_analyse(
        AuthenticityContext(image=maker(photo))
    ).triggered is False


@pytest.mark.unit
def test_both_synthetic_signals_must_agree(photo: np.ndarray, config) -> None:
    """A geometric mean, so either term at zero kills the verdict. Pinned
    because switching it to a sum would convict every heavily-compressed
    photograph."""
    detector = SyntheticImageDetector(config.synthetic)
    compressed = detector.safe_analyse(
        AuthenticityContext(image=as_heavily_compressed(photo))
    )

    assert compressed.measurements["flat_block_fraction"] > 0.20
    assert compressed.measurements["unique_colour_ratio"] > 0.09
    assert compressed.confidence == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "detector_name", ["screenshot", "moire", "print_recapture", "synthetic"]
)
def test_a_flat_image_is_reported_as_unmeasurable(config, detector_name: str) -> None:
    """A uniform field satisfies almost every "this is not a photograph" test
    trivially. A detector reporting high confidence on it would be reporting
    confidence in an arithmetic artefact."""
    detectors = {d.name: d for d in build_detectors(config)}
    lookup = {
        "screenshot": "screenshot",
        "moire": "screen_recapture",
        "print_recapture": "print_recapture",
        "synthetic": "synthetic_image",
    }
    detector = detectors[lookup[detector_name]]

    signal = detector.safe_analyse(AuthenticityContext(image=flat_colour()))

    assert signal.triggered is False
    assert signal.measured is False
    assert signal.note is not None


@pytest.mark.unit
def test_a_tiny_image_does_not_crash_any_detector(config) -> None:
    tiny = np.random.default_rng(0).integers(0, 256, (12, 14, 3), dtype=np.uint8)
    context = AuthenticityContext(image=tiny)

    for detector in build_detectors(config):
        assert detector.safe_analyse(context).triggered is False


@pytest.mark.unit
def test_a_photograph_with_no_person_is_still_authentic(config) -> None:
    """Authenticity and subject are separate questions. A landscape is a
    genuine capture; whether it works as a profile picture is Module 1's
    business, not this package's."""
    image = landscape_photo()
    if image is None:
        pytest.skip("no second reference photograph installed")

    context = AuthenticityContext(image=image)
    for detector in build_detectors(config):
        assert detector.safe_analyse(context).triggered is False


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_every_detector_reports_its_measurements(photo: np.ndarray, config) -> None:
    """A rejection that cannot be explained cannot be appealed."""
    context = AuthenticityContext(image=photo)
    for detector in build_detectors(config):
        assert detector.safe_analyse(context).measurements


@pytest.mark.unit
def test_a_signal_serialises(photo: np.ndarray, config) -> None:
    import json

    context = AuthenticityContext(image=photo)
    for detector in build_detectors(config):
        json.dumps(detector.safe_analyse(context).as_dict())


@pytest.mark.unit
def test_a_faulting_detector_is_isolated(config) -> None:
    """One detector throwing must not take the report with it: the other
    findings are still worth having."""

    class Broken(ScreenshotDetector):
        def analyse(self, context: AuthenticityContext):  # noqa: ANN201, ARG002
            raise RuntimeError("deliberate")

    signal = Broken(config.screenshot).safe_analyse(
        AuthenticityContext(image=flat_colour())
    )

    assert signal.triggered is False
    assert signal.note is not None
    assert "deliberate" in signal.note


@pytest.mark.unit
def test_the_context_is_computed_once(photo: np.ndarray) -> None:
    """Several detectors want the same views, and they must be identical
    pixels or the measured separation margins do not transfer."""
    context = AuthenticityContext(image=photo)

    assert context.spectral_grey is context.spectral_grey
    assert context.log_spectrum is context.log_spectrum
    assert context.grey is context.grey


@pytest.mark.unit
def test_detection_is_deterministic(photo: np.ndarray, config) -> None:
    """A verification score that changes between two runs of one image is not
    a score anybody can defend."""
    shot = as_screenshot(photo)
    first = [confidence(d, shot) for d in build_detectors(config)]
    second = [confidence(d, shot) for d in build_detectors(config)]

    assert first == second


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "floor", "ceiling", "expected"),
    [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (0.5, 0.0, 1.0, 0.5),
        (-1.0, 0.0, 1.0, 0.0),
        (2.0, 0.0, 1.0, 1.0),
        # Downward anchors, for metrics where less means more suspicious.
        (0.009, 0.009, 0.003, 0.0),
        (0.003, 0.009, 0.003, 1.0),
        (0.006, 0.009, 0.003, 0.5),
    ],
)
def test_the_ramp_maps_measurements_onto_a_unit_interval(
    value: float, floor: float, ceiling: float, expected: float
) -> None:
    assert ramp(value, floor=floor, ceiling=ceiling) == pytest.approx(expected)


@pytest.mark.unit
def test_a_degenerate_ramp_does_not_divide_by_zero() -> None:
    assert ramp(5.0, floor=1.0, ceiling=1.0) == pytest.approx(1.0)
    assert ramp(0.0, floor=1.0, ceiling=1.0) == pytest.approx(0.0)
