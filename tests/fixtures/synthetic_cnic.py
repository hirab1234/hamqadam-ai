"""Render synthetic Pakistani CNIC cards for testing.

**Every card produced here is entirely invented.** No real identity document
appears anywhere in this repository, and none should: a CNIC is personal data
under any reading, and a test corpus of real ones would be a liability rather
than an asset.

The renderer is deliberately faithful in *layout* rather than appearance. What
the parser depends on is the arrangement - label on the left, value to its
right, dates in ``DD.MM.YYYY``, the identity number in ``5-7-1`` form - and
that is what is reproduced. It makes no attempt to mimic NADRA's security
printing, holograms or fonts, and a card rendered here would not fool anyone.

The degradation helpers matter as much as the renderer. A verification service
sees photographs taken at an angle, in poor light, at a distance, on a desk -
never a flat scan - and the preprocessing pipeline exists entirely to cope
with that.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

BgrImage = npt.NDArray[np.uint8]

#: ID-1 at roughly 300 dpi.
CARD_WIDTH = 1012
CARD_HEIGHT = 638


def _font(size: int, *, bold: bool = False) -> Any:
    """Load a bundled TrueType face at a given size."""
    import matplotlib
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
    return ImageFont.truetype(str(path), size)


@dataclass(slots=True)
class CnicSpec:
    """The invented data a synthetic card is rendered from.

    Defaults describe a plausible but wholly fictitious holder. The identity
    number's final digit is even, which encodes female and agrees with the
    gender field - so the validator's parity cross-check passes, and a test
    wanting it to fail must break it deliberately.
    """

    name: str = "AYESHA KHAN"
    father_name: str = "MUHAMMAD KHAN"
    gender: str = "F"
    country: str = "Pakistan"
    cnic_number: str = "42101-8375926-4"
    date_of_birth: dt.date = field(default_factory=lambda: dt.date(1994, 3, 14))
    date_of_issue: dt.date = field(default_factory=lambda: dt.date(2019, 7, 22))
    date_of_expiry: dt.date = field(default_factory=lambda: dt.date(2029, 7, 22))
    lifetime: bool = False

    def rows(self) -> list[tuple[str, str]]:
        """The label/value pairs printed on the English side."""
        expiry = "Lifetime" if self.lifetime else _fmt(self.date_of_expiry)
        return [
            ("Name", self.name),
            ("Father Name", self.father_name),
            ("Gender", self.gender),
            ("Country of Stay", self.country),
            ("Identity Number", self.cnic_number),
            ("Date of Birth", _fmt(self.date_of_birth)),
            ("Date of Issue", _fmt(self.date_of_issue)),
            ("Date of Expiry", expiry),
        ]


def _fmt(value: dt.date) -> str:
    """Render a date the way the card prints it."""
    return f"{value.day:02d}.{value.month:02d}.{value.year}"


def render_cnic(spec: CnicSpec | None = None) -> BgrImage:
    """Render a clean, flat synthetic CNIC.

    Args:
        spec: The invented data. A plausible default is used when omitted.

    Returns:
        The card as a BGR uint8 array.
    """
    from PIL import Image, ImageDraw

    spec = spec or CnicSpec()
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (243, 244, 238))
    draw = ImageDraw.Draw(card)

    draw.rectangle([0, 0, CARD_WIDTH - 1, 78], fill=(28, 92, 58))
    draw.text(
        (28, 22), "ISLAMIC REPUBLIC OF PAKISTAN",
        font=_font(30, bold=True), fill=(255, 255, 255),
    )
    draw.text((28, 96), "National Identity Card", font=_font(24), fill=(40, 40, 40))

    y = 150
    for label, value in spec.rows():
        draw.text((28, y), label, font=_font(21), fill=(95, 95, 95))
        draw.text((300, y - 2), value, font=_font(24, bold=True), fill=(15, 15, 15))
        y += 52

    # A stand-in for the portrait. Deliberately not a face: the CNIC portrait
    # path is Module 6's problem, and a detectable face here would make these
    # fixtures do two jobs badly instead of one well.
    draw.rectangle(
        [CARD_WIDTH - 250, 150, CARD_WIDTH - 40, 430],
        outline=(120, 120, 120), width=2,
    )
    draw.text(
        (CARD_WIDTH - 226, 275), "PHOTO", font=_font(26), fill=(150, 150, 150)
    )

    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


def render_unrelated_document() -> BgrImage:
    """Render a document that is emphatically not a CNIC.

    A utility bill: dense printed text, a long reference number, dates in the
    same format the card uses. Chosen because it is the hard negative - a
    blank page is rejected by any rule at all, whereas this one has to be
    rejected on content, which is what the caller actually needs.

    Wholly fictitious, like every other fixture here.
    """
    from PIL import Image, ImageDraw

    page = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), (252, 252, 250))
    draw = ImageDraw.Draw(page)

    draw.text((28, 30), "ELECTRIC SUPPLY COMPANY",
              font=_font(30, bold=True), fill=(20, 20, 20))
    draw.text((28, 74), "Monthly Consumption Statement",
              font=_font(22), fill=(70, 70, 70))

    rows = [
        ("Consumer Reference", "04 11223 3445566 U"),
        ("Tariff Category", "A-1 Residential"),
        ("Billing Month", "06.2026"),
        ("Reading Date", "18.06.2026"),
        ("Units Consumed", "312"),
        ("Due Date", "05.07.2026"),
        ("Amount Payable", "PKR 8,420"),
        ("Late Payment Surcharge", "PKR 842"),
    ]
    y = 130
    for label, value in rows:
        draw.text((28, y), label, font=_font(21), fill=(95, 95, 95))
        draw.text((420, y - 2), value, font=_font(24, bold=True), fill=(15, 15, 15))
        y += 52

    return cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Degradations - what a real capture actually looks like
# --------------------------------------------------------------------------- #


def photograph_on_desk(
    card: BgrImage,
    *,
    angle: float = 8.0,
    scale: float = 0.62,
    background: tuple[int, int, int] = (92, 88, 84),
    seed: int = 0,
) -> BgrImage:
    """Place the card on a surface, rotated and perspective-skewed.

    The default case the rectifier exists for: a card lying on a desk,
    photographed slightly from one side, occupying part of the frame.
    """
    rng = np.random.default_rng(seed)
    height, width = card.shape[:2]

    frame_width = int(width / scale)
    frame_height = int(height / scale)
    frame = np.full((frame_height, frame_width, 3), background, dtype=np.uint8)
    # A little surface texture, so the edge detector faces a real boundary
    # rather than a synthetic one against flat colour.
    noise = rng.normal(0, 6, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    offset_x = (frame_width - width) // 2
    offset_y = (frame_height - height) // 2

    source = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    tilt = width * 0.035
    destination = np.array(
        [
            [offset_x + tilt, offset_y],
            [offset_x + width, offset_y + tilt * 0.5],
            [offset_x + width - tilt, offset_y + height],
            [offset_x, offset_y + height - tilt * 0.5],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        card, transform, (frame_width, frame_height),
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        transform,
        (frame_width, frame_height),
    )
    composed = frame.copy()
    composed[mask > 128] = warped[mask > 128]

    if abs(angle) > 0.01:
        centre = (frame_width / 2.0, frame_height / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        composed = cv2.warpAffine(
            composed, matrix, (frame_width, frame_height),
            borderMode=cv2.BORDER_REPLICATE,
        )

    return np.asarray(composed, dtype=np.uint8)


def dim_lighting(card: BgrImage, *, factor: float = 0.42) -> BgrImage:
    """Underexpose the card, as an indoor capture without flash."""
    return np.clip(card.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def glare(card: BgrImage, *, strength: float = 0.75, seed: int = 1) -> BgrImage:
    """Add a specular highlight across one corner.

    The characteristic failure of photographing a laminated card, and the
    reason the enhancement step uses local rather than global equalisation.
    """
    height, width = card.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    centre_x, centre_y = width * 0.72, height * 0.28
    radius = min(width, height) * 0.55
    falloff = np.exp(-(((x - centre_x) ** 2 + (y - centre_y) ** 2) / (2 * radius**2)))
    lifted = card.astype(np.float32) + (falloff * 255.0 * strength)[:, :, None]
    return np.clip(lifted, 0, 255).astype(np.uint8)


def low_resolution(card: BgrImage, *, factor: float = 0.30) -> BgrImage:
    """Shrink and re-enlarge, as a card photographed from too far away."""
    height, width = card.shape[:2]
    small = cv2.resize(
        card, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_AREA
    )
    return np.asarray(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC),
        dtype=np.uint8,
    )


def jpeg_artifacts(card: BgrImage, *, quality: int = 25) -> BgrImage:
    """Compress heavily, as an image forwarded through a chat application."""
    ok, buffer = cv2.imencode(".jpg", card, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return card
    return np.asarray(cv2.imdecode(buffer, cv2.IMREAD_COLOR), dtype=np.uint8)


def sensor_noise(card: BgrImage, *, sigma: float = 14.0, seed: int = 2) -> BgrImage:
    """Add grain, as a high-ISO capture in poor light."""
    rng = np.random.default_rng(seed)
    noisy = card.astype(np.float32) + rng.normal(0.0, sigma, card.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotated(card: BgrImage, degrees: int) -> BgrImage:
    """Rotate by a multiple of ninety degrees, as a sideways capture."""
    mapping = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    code = mapping.get(degrees % 360)
    if code is None:
        return card
    return np.asarray(cv2.rotate(card, code), dtype=np.uint8)


__all__ = [
    "CARD_HEIGHT",
    "CARD_WIDTH",
    "CnicSpec",
    "dim_lighting",
    "glare",
    "jpeg_artifacts",
    "low_resolution",
    "photograph_on_desk",
    "render_cnic",
    "rotated",
    "sensor_noise",
]
