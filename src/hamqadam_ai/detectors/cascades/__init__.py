"""Vendored Haar cascade classifiers.

Why these XML files live in the repository
------------------------------------------
The terminal fallback in the detector chain is only worth having if it is
genuinely unconditional, and it turned out not to be: ``opencv-python-headless``
- the correct wheel for a container, since it drops the GUI and codec
dependencies - ships an **empty** ``cv2/data`` directory. The cascades are
present only in the full ``opencv-python`` wheel.

Relying on ``cv2.data.haarcascades`` therefore meant the "always available"
fallback silently disappeared in exactly the deployment target it was meant to
protect. Vendoring the three XML files (1.5 MB total) makes the guarantee real:
the service can detect a face with no model store, no network and no non-
headless OpenCV.

Provenance and licence
----------------------
Taken verbatim from ``opencv/opencv`` tag ``4.10.0``, path
``data/haarcascades/``. OpenCV is distributed under the Apache License 2.0, and
these files carry the additional Intel Corporation BSD-style notice reproduced
in their XML headers. Both permit redistribution; the notices are intact inside
each file.

Files
-----
``haarcascade_frontalface_default.xml``
    The primary detector for the fallback path.
``haarcascade_eye_tree_eyeglasses.xml``
    Used to recover an eye pair so approximate landmarks can be constructed.
    The ``tree_eyeglasses`` variant is chosen over plain ``eye`` because it was
    trained to fire through spectacles, which plain ``eye`` frequently misses.
``haarcascade_profileface.xml``
    Not used by the default chain. Retained so an operator can enable
    profile-face recovery for a use case that permits non-frontal captures.
"""

from __future__ import annotations

from pathlib import Path

#: Directory containing the vendored cascade XML files.
CASCADE_DIR: Path = Path(__file__).parent

FRONTAL_FACE = "haarcascade_frontalface_default.xml"
EYE_TREE_EYEGLASSES = "haarcascade_eye_tree_eyeglasses.xml"
PROFILE_FACE = "haarcascade_profileface.xml"


def cascade_path(filename: str) -> Path | None:
    """Resolve a cascade file, preferring the vendored copy.

    Falls back to OpenCV's own data directory so a full ``opencv-python``
    installation still works if the vendored files are ever stripped by a
    packaging step.

    Args:
        filename: Cascade XML filename, e.g. ``haarcascade_frontalface_default.xml``.

    Returns:
        A path to an existing file, or ``None`` when the cascade is unavailable
        from either source.
    """
    vendored = CASCADE_DIR / filename
    if vendored.is_file():
        return vendored

    try:
        import cv2

        # `cv2.data` is a runtime-generated submodule that the stubs do not
        # declare; the AttributeError branch below is the real guard.
        fallback = Path(cv2.data.haarcascades) / filename  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return None

    return fallback if fallback.is_file() else None


__all__ = [
    "CASCADE_DIR",
    "EYE_TREE_EYEGLASSES",
    "FRONTAL_FACE",
    "PROFILE_FACE",
    "cascade_path",
]
