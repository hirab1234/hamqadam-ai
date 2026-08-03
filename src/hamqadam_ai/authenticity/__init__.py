"""MODULE 7 capability - is this image a genuine camera capture?

Four detectors, each looking for a different way a profile photograph can fail
to be what it claims: a screenshot of an app, a photograph of a screen, a
photograph of a print, or rendered artwork.

Every anchor in this package sits in a measured gap between two populations,
and the measurements are written into the docstring beside the anchor they
justify. That is not decoration: these detectors accuse a user of uploading
something dishonest, and a threshold nobody can trace back to evidence is not
one anybody should be accused on.
"""

from hamqadam_ai.authenticity.aggregate import (
    FINDING_CODES,
    FINDING_MESSAGES,
    AuthenticityAssessment,
    AuthenticityFinding,
    aggregate,
)
from hamqadam_ai.authenticity.base import (
    AuthenticityContext,
    AuthenticityDetector,
    AuthenticitySignal,
    ramp,
)
from hamqadam_ai.authenticity.moire import MoireDetector
from hamqadam_ai.authenticity.recapture import (
    PrintRecaptureDetector,
    SyntheticImageDetector,
)
from hamqadam_ai.authenticity.screenshot import ScreenshotDetector

__all__ = [
    "FINDING_CODES",
    "FINDING_MESSAGES",
    "AuthenticityAssessment",
    "AuthenticityContext",
    "AuthenticityDetector",
    "AuthenticityFinding",
    "AuthenticitySignal",
    "ramp",
    "MoireDetector",
    "PrintRecaptureDetector",
    "ScreenshotDetector",
    "SyntheticImageDetector",
    "aggregate",
    "build_detectors",
]


def build_detectors(config: object) -> list[AuthenticityDetector]:
    """Construct every detector from a :class:`ProfileAnalysisConfig`.

    Args:
        config: The ``profile`` section of the settings.

    Returns:
        The detectors, in the order their findings are reported when two are
        equally confident.
    """
    return [
        ScreenshotDetector(config.screenshot),  # type: ignore[attr-defined]
        MoireDetector(config.moire),  # type: ignore[attr-defined]
        PrintRecaptureDetector(config.print_recapture),  # type: ignore[attr-defined]
        SyntheticImageDetector(config.synthetic),  # type: ignore[attr-defined]
    ]
