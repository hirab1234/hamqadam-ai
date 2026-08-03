"""Synthetic CNIC cards carrying an actual face, plus the capture scenarios.

Extends :mod:`tests.fixtures.synthetic_cnic`, whose renderer deliberately draws
an empty PHOTO placeholder - locating a real portrait is this module's problem,
and a detectable face there would have made those fixtures do two jobs badly.

The face used is a **public-domain reference portrait** shipped with matplotlib
or scikit-image, degraded to look like a sub-300-dpi print behind laminate. No
real identity document, and no private individual's photograph, appears
anywhere in this repository.

The scenario that matters
-------------------------
:func:`card_held_in_front_of_face` reproduces the attack the module exists to
stop: the card photographed while held up in front of the holder's own face, so
a naive "biggest face wins" extractor compares the selfie against the live face
rather than against the printed portrait, and passes whoever the card belongs
to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from tests.fixtures.synthetic_cnic import (
    CARD_HEIGHT,
    CARD_WIDTH,
    CnicSpec,
    render_cnic,
)

BgrImage = npt.NDArray[np.uint8]

#: Where the renderer puts the photo box, matching ``render_cnic``.
PHOTO_BOX = (CARD_WIDTH - 250, 150, CARD_WIDTH - 40, 430)


def reference_portrait() -> BgrImage | None:
    """A public-domain face, or ``None`` when neither source is installed."""
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
    except Exception:  # noqa: BLE001 - no reference portrait available
        return None


def alternate_portrait() -> BgrImage | None:
    """A genuinely different person, for the impostor case."""
    try:
        from skimage import data

        return np.asarray(
            cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), dtype=np.uint8
        )
    except Exception:  # noqa: BLE001
        return None


def as_printed(face: BgrImage, *, width: int, height: int, seed: int = 0) -> BgrImage:
    """Make a photograph look like a small print behind laminate.

    Four things happen to a face between a camera and a laminated ID card, and
    all four hurt a recogniser:

    * it is reproduced at a few hundred dpi, so fine detail is simply gone;
    * the print process compresses tonal range;
    * the laminate adds a specular sheen;
    * the whole thing is then re-photographed by a phone.

    Reproduced here by downsampling hard and scaling back up, which destroys
    high-frequency detail the way printing does, then flattening contrast and
    adding a directional sheen.
    """
    rng = np.random.default_rng(seed)

    # Print resolution: throw the detail away, then restore the pixel count.
    small = cv2.resize(face, (max(width // 4, 8), max(height // 4, 8)),
                       interpolation=cv2.INTER_AREA)
    printed = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)

    # Ink cannot reach either end of the tonal range.
    printed = np.clip(printed.astype(np.float32) * 0.78 + 34.0, 0, 255)

    # Laminate sheen: a soft diagonal gradient, brightest at one corner.
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    sheen = ((xs / max(width - 1, 1)) + (1.0 - ys / max(height - 1, 1))) / 2.0
    printed += (sheen[:, :, None] ** 3) * 46.0

    # Phone sensor noise on top of all of it.
    printed += rng.normal(0.0, 3.0, printed.shape)

    return np.clip(printed, 0, 255).astype(np.uint8)


def render_cnic_with_portrait(
    spec: CnicSpec | None = None,
    *,
    face: BgrImage | None = None,
    ghost: bool = False,
    seed: int = 0,
) -> BgrImage | None:
    """Render a card with a real, print-degraded face in the photo box.

    Args:
        spec: The invented cardholder data.
        face: The portrait to print. The bundled reference is used when
            omitted.
        ghost: Also print the faded secondary reproduction that modern cards
            carry as a security feature.
        seed: Degradation seed.

    Returns:
        The card, or ``None`` when no reference portrait is installed.
    """
    source = face if face is not None else reference_portrait()
    if source is None:
        return None

    card = render_cnic(spec).copy()
    x1, y1, x2, y2 = PHOTO_BOX
    box_width, box_height = x2 - x1, y2 - y1

    portrait = _fit(source, box_width, box_height)
    card[y1:y2, x1:x2] = as_printed(
        portrait, width=box_width, height=box_height, seed=seed
    )
    cv2.rectangle(card, (x1, y1), (x2 - 1, y2 - 1), (120, 120, 120), 2)

    if ghost:
        card = _add_ghost(card, portrait, seed=seed)

    return card


def _add_ghost(card: BgrImage, portrait: BgrImage, *, seed: int) -> BgrImage:
    """Print the faded secondary reproduction modern cards carry.

    Deliberately smaller and much fainter than the primary, which is what the
    candidate ranking has to get right: a ghost outranking the real portrait
    would feed the recogniser the worse of the two images.
    """
    height = 150
    width = int(height * portrait.shape[1] / max(portrait.shape[0], 1))
    small = as_printed(_fit(portrait, width, height), width=width, height=height,
                       seed=seed + 1)

    # Directly below the main photo box, which is where cards actually put it.
    # An earlier version placed it mid-card, in the gap between the template's
    # two candidate bands, so it was correctly rejected as off-template and the
    # fixture silently stopped exercising the ghost path at all.
    box_x1, _box_y1, box_x2, box_y2 = PHOTO_BOX
    x1 = box_x1 + ((box_x2 - box_x1) - width) // 2
    y1 = box_y2 + 12
    if x1 < 0 or y1 + height > CARD_HEIGHT:
        return card

    region = card[y1:y1 + height, x1:x1 + width].astype(np.float32)
    blended = region * 0.62 + small.astype(np.float32) * 0.38
    card[y1:y1 + height, x1:x1 + width] = np.clip(blended, 0, 255).astype(np.uint8)
    return card


def card_held_in_front_of_face(
    card: BgrImage,
    *,
    face: BgrImage | None = None,
    card_scale: float = 0.52,
    seed: int = 0,
) -> BgrImage | None:
    """The attack: the card photographed while held up in front of a face.

    A large live face fills the frame; the card sits over the lower half of
    it, small. A "biggest face wins" extractor picks the live face, and the
    selfie then matches itself at near-perfect similarity **whoever the card
    belongs to** - which is the entire document check, bypassed by holding the
    card the way everybody holds it.

    Args:
        card: The card to hold up.
        face: The live face behind it. The bundled reference is used when
            omitted, which makes the attack maximally convincing: the same
            person is on the card and behind it, so only the *geometry*
            distinguishes the two, never the identity.
        card_scale: Card width as a fraction of the frame width.
        seed: Background noise seed.

    Returns:
        The composite, or ``None`` when no reference portrait is installed.
    """
    live = face if face is not None else reference_portrait()
    if live is None:
        return None

    rng = np.random.default_rng(seed)
    frame_width = 1400
    frame_height = 1050

    # The live face, filling the frame the way a held-up-phone selfie does.
    background = _fill(live, frame_width, frame_height)
    background = np.clip(
        background.astype(np.float32) + rng.normal(0.0, 3.0, background.shape), 0, 255
    ).astype(np.uint8)

    target_width = int(frame_width * card_scale)
    target_height = int(target_width * card.shape[0] / card.shape[1])
    held = cv2.resize(card, (target_width, target_height),
                      interpolation=cv2.INTER_AREA)

    offset_x = (frame_width - target_width) // 2
    offset_y = int(frame_height * 0.56)
    offset_y = min(offset_y, frame_height - target_height)

    composite = background.copy()
    composite[offset_y:offset_y + target_height,
              offset_x:offset_x + target_width] = held

    # A soft shadow under the card, so the edge detector has a real boundary
    # to find rather than a synthetic one.
    cv2.rectangle(
        composite,
        (offset_x - 3, offset_y - 3),
        (offset_x + target_width + 2, offset_y + target_height + 2),
        (40, 40, 40),
        thickness=3,
    )
    return composite


def card_beside_a_bystander(
    card: BgrImage, *, face: BgrImage | None = None, seed: int = 3
) -> BgrImage | None:
    """A card on a desk with somebody else's face also in shot.

    The innocent version of the same geometry, and the reason a foreign face
    is reported as a signal rather than treated as proof.
    """
    live = face if face is not None else alternate_portrait()
    if live is None:
        return None

    rng = np.random.default_rng(seed)
    frame = np.full((900, 1500, 3), (96, 92, 88), dtype=np.uint8)
    frame = np.clip(
        frame.astype(np.float32) + rng.normal(0.0, 5.0, frame.shape), 0, 255
    ).astype(np.uint8)

    card_width = 820
    card_height = int(card_width * card.shape[0] / card.shape[1])
    resized = cv2.resize(card, (card_width, card_height),
                         interpolation=cv2.INTER_AREA)
    frame[70:70 + card_height, 60:60 + card_width] = resized

    person = _fill(live, 420, 560)
    frame[300:860, 1030:1450] = person
    return frame


def blank_card() -> BgrImage:
    """A CNIC whose photo box is empty - no portrait to find at all."""
    return render_cnic()


def _fit(image: BgrImage, width: int, height: int) -> BgrImage:
    """Resize to exactly ``width`` x ``height``, preserving the face centre.

    Centre-crops to the target aspect first, so the face is not squashed - a
    stretched face measurably degrades recognition, which would make the
    fixture harder than reality rather than representative of it.
    """
    return _fill(image, width, height)


def _fill(image: BgrImage, width: int, height: int) -> BgrImage:
    """Centre-crop to the target aspect ratio, then resize to fill it."""
    source_height, source_width = image.shape[:2]
    target_aspect = width / max(height, 1)
    source_aspect = source_width / max(source_height, 1)

    if source_aspect > target_aspect:
        crop_width = int(source_height * target_aspect)
        x1 = (source_width - crop_width) // 2
        cropped = image[:, x1:x1 + crop_width]
    else:
        crop_height = int(source_width / max(target_aspect, 1e-9))
        # Bias upward: a portrait's face sits above centre, and cropping
        # symmetrically would cut the forehead before the chin.
        y1 = max((source_height - crop_height) // 4, 0)
        cropped = image[y1:y1 + crop_height, :]

    return np.asarray(
        cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )


def annotate(card: BgrImage, boxes: list[tuple[Any, str]]) -> BgrImage:
    """Draw labelled boxes on a copy, for the demo's overlay output."""
    canvas = card.copy()
    palette = {
        "portrait": (60, 200, 60),
        "ghost": (200, 170, 40),
        "foreign": (40, 40, 220),
        "band": (150, 150, 150),
    }
    for box, label in boxes:
        colour = palette.get(label, (200, 200, 200))
        x1, y1, x2, y2 = (int(v) for v in (box.x1, box.y1, box.x2, box.y2))
        thickness = 1 if label == "band" else 3
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        if label != "band":
            cv2.putText(
                canvas, label, (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
            )
    return canvas


__all__ = [
    "PHOTO_BOX",
    "alternate_portrait",
    "annotate",
    "as_printed",
    "blank_card",
    "card_beside_a_bystander",
    "card_held_in_front_of_face",
    "reference_portrait",
    "render_cnic_with_portrait",
]
