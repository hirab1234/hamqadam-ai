"""Profile-photo fixtures: a genuine capture and the things people upload instead.

Every image is either a **public-domain reference photograph** shipped with
matplotlib or scikit-image, or is synthesised here. No private individual's
photograph appears in this repository.

What each impostor reproduces
-----------------------------
``as_screenshot``
    A photo pasted into phone UI chrome - status bar, nav bar, flat
    backgrounds, caption blocks.

    An earlier version of this docstring claimed the UI regions have
    mathematically zero variance "which no camera sensor can produce, and
    neither can JPEG". **Measurement refuted that.** A quality-12 JPEG of a
    real photograph has exactly-zero-variance 8x8 blocks across 35% of the
    frame, against 36% here - JPEG quantisation zeroes every AC coefficient
    in a smooth block. What actually separates the two is a full-width
    constant *row*, which a photograph cannot have because its content varies
    horizontally somewhere along every one.

``as_screen_recapture``
    A photograph of an LCD. The camera's sensor grid beats against the
    display's pixel grid and produces moire: periodic interference that shows
    up as off-axis peaks in the Fourier transform.

``as_print_recapture``
    A photograph of a printed photograph. The same signature characterised in
    Module 6 - high frequencies gone, tonal range compressed - plus paper
    texture and a border.

``as_synthetic_render``
    Flat-shaded vector art, standing in for the avatar or cartoon people use
    as a profile picture. No noise floor anywhere.

``as_heavily_compressed``
    A genuine photograph at JPEG quality 12. The **hard negative**: it also
    destroys the noise floor, so any detector keying on "no noise" alone will
    call it a screenshot. Distinguishing the two is the interesting part.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

BgrImage = npt.NDArray[np.uint8]

#: Common phone screenshot dimensions, portrait orientation.
PHONE_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1080, 1920),
    (1170, 2532),
    (1284, 2778),
    (1440, 3120),
)


def reference_photo() -> BgrImage | None:
    """A public-domain photograph of a person, as a camera produced it."""
    try:
        import matplotlib

        path = (
            Path(matplotlib.__file__).parent
            / "mpl-data" / "sample_data" / "grace_hopper.jpg"
        )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return np.asarray(image, dtype=np.uint8)
    except Exception:  # noqa: BLE001 - fall through to the other source
        pass

    try:
        from skimage import data

        return np.asarray(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), dtype=np.uint8
        )
    except Exception:  # noqa: BLE001
        return None


def alternate_photo() -> BgrImage | None:
    """A second public-domain photograph, of somebody else."""
    try:
        from skimage import data

        return np.asarray(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), dtype=np.uint8
        )
    except Exception:  # noqa: BLE001
        return None


def landscape_photo() -> BgrImage | None:
    """A photograph with no person in it."""
    try:
        from skimage import data

        return np.asarray(
            cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR), dtype=np.uint8
        )
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Impostors
# --------------------------------------------------------------------------- #


def as_screenshot(
    photo: BgrImage,
    *,
    resolution: tuple[int, int] = (1080, 1920),
    seed: int = 0,
) -> BgrImage:
    """Compose a phone screenshot containing the photograph.

    The giveaway is the *span*, not the flatness. Chrome is written from a
    constant across the whole frame width; a photograph's content varies
    horizontally somewhere along any row, and no amount of quantisation
    flattens the entire span. Flatness alone does not separate the two - see
    the module docstring.

    Note the caption blocks below the photo. They are what keeps the signal
    alive in the image *interior* after contiguous edge padding is stripped,
    which is the measurement the detector actually uses.
    """
    width, height = resolution
    canvas = np.full((height, width, 3), (250, 250, 250), dtype=np.uint8)

    status_bar = 84
    nav_bar = 60
    # Status bar: solid dark, with blocky glyph shapes standing in for the
    # clock and battery icons.
    canvas[:status_bar] = (24, 24, 24)
    for x0 in (40, 90, 140):
        cv2.rectangle(canvas, (x0, 30), (x0 + 34, 54), (235, 235, 235), -1)
    for x0 in (width - 190, width - 130, width - 70):
        cv2.rectangle(canvas, (x0, 30), (x0 + 40, 54), (235, 235, 235), -1)

    # A flat app header and a flat card behind the photo.
    header = status_bar + 130
    canvas[status_bar:header] = (245, 245, 247)
    cv2.rectangle(canvas, (0, header), (width, header + 2), (222, 222, 224), -1)

    # The photograph itself, scaled to the content width.
    content_width = width - 80
    scale = content_width / photo.shape[1]
    content_height = int(photo.shape[0] * scale)
    resized = cv2.resize(
        photo, (content_width, content_height), interpolation=cv2.INTER_AREA
    )
    top = header + 60
    top = min(top, max(height - nav_bar - content_height - 40, header + 10))
    canvas[top:top + content_height, 40:40 + content_width] = resized

    # Flat caption blocks below it, then the nav bar.
    caption = top + content_height + 40
    for index in range(3):
        y = caption + index * 44
        if y + 26 < height - nav_bar:
            cv2.rectangle(
                canvas, (40, y), (40 + int(content_width * (0.9 - 0.2 * index)), y + 26),
                (228, 228, 232), -1,
            )
    canvas[height - nav_bar:] = (18, 18, 18)

    return canvas


def as_screen_recapture(
    photo: BgrImage,
    *,
    pixel_pitch: int = 4,
    strength: float = 0.32,
    angle_degrees: float = 7.0,
    seed: int = 1,
) -> BgrImage:
    """Photograph the image off an LCD panel.

    Two things happen and both are measurable. The display's pixel grid, seen
    through the camera's own sampling grid at a slight angle, produces
    **moire** - periodic interference that appears as symmetric off-axis peaks
    in the Fourier transform. And the panel's backlight adds a broad luminance
    gradient with a specular sheen.

    Args:
        pixel_pitch: Display pixel period in captured pixels. Small values put
            the interference near Nyquist where it is hardest to see by eye
            and easiest to see in the spectrum.
        strength: Amplitude of the grid modulation.
        angle_degrees: Angle between the display grid and the sensor grid.
    """
    rng = np.random.default_rng(seed)
    height, width = photo.shape[:2]
    working = photo.astype(np.float32)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radians = np.deg2rad(angle_degrees)
    rotated_x = xs * np.cos(radians) - ys * np.sin(radians)
    rotated_y = xs * np.sin(radians) + ys * np.cos(radians)

    # The panel grid: a product of two sinusoids at the pixel pitch. The
    # product is what puts energy at sum and difference frequencies, which is
    # what moire actually is.
    grid = (
        np.cos(2.0 * np.pi * rotated_x / pixel_pitch)
        * np.cos(2.0 * np.pi * rotated_y / pixel_pitch)
    )
    working *= 1.0 + strength * grid[:, :, None]

    # Backlight: a broad, off-centre glow.
    glow = np.exp(
        -(((xs - width * 0.34) ** 2) / (2 * (width * 0.62) ** 2)
          + ((ys - height * 0.28) ** 2) / (2 * (height * 0.62) ** 2))
    )
    working += glow[:, :, None] * 26.0

    # LCDs cannot reach black, and the capture is re-encoded.
    working = working * 0.92 + 14.0
    working += rng.normal(0.0, 2.2, working.shape)
    captured = np.clip(working, 0, 255).astype(np.uint8)

    return _jpeg(captured, quality=88)


def as_print_recapture(
    photo: BgrImage, *, seed: int = 2, border: int = 26
) -> BgrImage:
    """Photograph a printed copy of the image, border and all."""
    rng = np.random.default_rng(seed)
    height, width = photo.shape[:2]

    # Print resolution: the detail is gone and no upscale brings it back.
    small = cv2.resize(
        photo, (max(width // 3, 8), max(height // 3, 8)), interpolation=cv2.INTER_AREA
    )
    printed = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    printed = printed.astype(np.float32) * 0.76 + 40.0

    # Paper texture: low-amplitude, spatially correlated - unlike sensor noise,
    # which is per-pixel independent.
    texture = cv2.GaussianBlur(
        rng.normal(0.0, 9.0, (height, width)).astype(np.float32), (0, 0), 1.8
    )
    printed += texture[:, :, None]

    printed = np.clip(printed, 0, 255).astype(np.uint8)

    framed = cv2.copyMakeBorder(
        printed, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(238, 236, 230),
    )
    return _jpeg(framed, quality=86)


def as_synthetic_render(size: tuple[int, int] = (512, 512)) -> BgrImage:
    """Flat-shaded vector art - the avatar people use instead of a photo.

    Every region is a constant fill, so there is no noise floor anywhere and
    no high-frequency texture beyond the hard edges themselves.
    """
    width, height = size
    canvas = np.full((height, width, 3), (232, 214, 190), dtype=np.uint8)

    cv2.rectangle(canvas, (0, int(height * 0.72)), (width, height), (168, 122, 96), -1)
    cv2.circle(canvas, (width // 2, int(height * 0.44)), int(width * 0.22),
               (206, 168, 138), -1)
    cv2.ellipse(canvas, (width // 2, int(height * 0.30)),
                (int(width * 0.23), int(height * 0.16)), 0, 180, 360, (58, 44, 38), -1)
    for side in (-1, 1):
        cv2.circle(canvas,
                   (width // 2 + side * int(width * 0.08), int(height * 0.42)),
                   int(width * 0.022), (44, 38, 34), -1)
    cv2.ellipse(canvas, (width // 2, int(height * 0.52)),
                (int(width * 0.07), int(height * 0.035)), 0, 0, 180, (150, 90, 84), 3)
    return canvas


def as_heavily_compressed(photo: BgrImage, *, quality: int = 12) -> BgrImage:
    """A genuine photograph, re-encoded until the noise floor is gone.

    The hard negative for every detector in this module. Anything keying on
    "no sensor noise" alone will call this a screenshot, and it is not - it is
    a real photograph of a real person that happens to have been through a
    messaging app twice.
    """
    return _jpeg(photo, quality=quality)


def as_recompressed(photo: BgrImage, *, first: int = 92, second: int = 74) -> BgrImage:
    """A photograph JPEG-encoded twice at different quality factors.

    Produces the periodic structure in the DCT coefficient histogram that
    double-compression detection keys on. Extremely common and entirely
    innocent - every image that passes through a social app is like this -
    which is exactly why the module reports it as weak evidence.
    """
    return _jpeg(_jpeg(photo, quality=first), quality=second)


def with_sensor_noise(photo: BgrImage, *, sigma: float = 2.4, seed: int = 5) -> BgrImage:
    """Add a plausible camera noise floor to an image that lacks one."""
    rng = np.random.default_rng(seed)
    noisy = photo.astype(np.float32) + rng.normal(0.0, sigma, photo.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def flat_colour(size: tuple[int, int] = (600, 800), value: int = 128) -> BgrImage:
    """A perfectly uniform image - the degenerate case for every metric."""
    width, height = size
    return np.full((height, width, 3), value, dtype=np.uint8)


def _jpeg(image: BgrImage, *, quality: int) -> BgrImage:
    """Round-trip through JPEG at a given quality."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - only on a broken OpenCV build
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return np.asarray(decoded, dtype=np.uint8)


__all__ = [
    "PHONE_RESOLUTIONS",
    "alternate_photo",
    "as_heavily_compressed",
    "as_print_recapture",
    "as_recompressed",
    "as_screen_recapture",
    "as_screenshot",
    "as_synthetic_render",
    "flat_colour",
    "landscape_photo",
    "reference_photo",
    "with_sensor_noise",
]
