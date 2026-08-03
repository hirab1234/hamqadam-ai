"""Document-image understanding that is not text recognition.

Module 5 (:mod:`hamqadam_ai.ocr`) reads what a card *says*. This package
reasons about what a card *is* - where its photo box sits, how large a printed
portrait can physically be, whether a detected face is plausibly on the
document or merely in front of it.

Kept separate from ``ocr`` because it shares none of its machinery: no
recognition engine, no language, no text. It shares only the subject.
"""

from hamqadam_ai.documents.portrait import (
    PortraitCandidate,
    PortraitLocation,
    PortraitRejection,
    band_containment,
    locate_portrait,
    portrait_regions,
)

__all__ = [
    "PortraitCandidate",
    "PortraitLocation",
    "PortraitRejection",
    "band_containment",
    "locate_portrait",
    "portrait_regions",
]
