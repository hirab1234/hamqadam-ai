"""The fraud-signal catalogue: what counts as evidence, and of what.

Signal families
---------------
Every signal belongs to a **family**, and the family is the load-bearing part
of the design rather than a filing convenience.

A single underlying fact usually shows up as several findings. A CNIC
photographed off a laptop screen makes Module 7 report a screen recapture,
Module 2 report low quality, Module 5 report low OCR confidence, and Module 6
report a degraded portrait - four findings, one fact. Adding them up charges a
user four times for one thing, and the arithmetic is not close:

    case                                   additive   family-max + noisy-OR
    one fact: CNIC shot off a screen          100.0                    69.4
    one fact: a blurry photograph              65.0                    20.0
    three genuinely independent facts         100.0                    94.6

The blurry-photograph row is the one that matters. Additive scoring puts an
ordinary out-of-focus snapshot at **65** - the boundary of HIGH risk - purely
because four quality sub-scores each contributed. It is not fraud, it is a bad
photo, and a system that cannot tell the difference will reject honest users
all day. Taking the strongest signal *within* a family and combining *across*
families puts it at 20, while leaving three independent facts at 94.6 where
they belong.

Weights are policy, not measurement
-----------------------------------
Every weight below is a judgement about how much the business should care about
a finding, not a probability derived from labelled fraud. There is no such data
in this project. They are defaults chosen to be defensible and are overridable
per-code in ``configs/thresholds.yaml``; the response reports
``weights_validated: false`` so nobody mistakes them for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SignalFamily(StrEnum):
    """Independent kinds of evidence.

    Two findings in the same family are treated as the same fact seen twice.
    Two findings in different families are treated as separate facts and
    compound.
    """

    #: The document contradicts itself - dates out of order, gender against
    #: the identity number's parity, a number that is not a CNIC.
    DOCUMENT_INTEGRITY = "document_integrity"

    #: The document image is not a photograph of a document: a screenshot, a
    #: picture of a screen, a picture of a print.
    DOCUMENT_AUTHENTICITY = "document_authenticity"

    #: A user-supplied photograph is not a genuine capture of the user.
    IMAGE_AUTHENTICITY = "image_authenticity"

    #: The faces across the submitted images are not all the same person.
    IDENTITY_CONSISTENCY = "identity_consistency"

    #: How the submission was staged: a face held behind the card, a group
    #: photograph, a portrait that is not on the card at all.
    PRESENTATION = "presentation"

    #: This face is already enrolled under another account.
    DUPLICATION = "duplication"

    #: The captures are poor. Weak evidence of anything - most bad photographs
    #: are just bad photographs - and capped accordingly.
    CAPTURE_QUALITY = "capture_quality"

    #: A check could not be run. Never contributes to the score; it lowers
    #: confidence *in* the score instead.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SignalDefinition:
    """What a finding means, and how much it counts.

    Attributes:
        code: The stable code emitted by an upstream module.
        family: Which kind of evidence it is.
        weight: Evidence strength in ``[0, 1]`` when the finding is certain.
        decisive: Whether a confident occurrence should floor the risk band at
            HIGH regardless of what else the arithmetic says. Reserved for
            findings that are not "some risk" but a conclusion.
        message: What to tell a reviewer.
    """

    code: str
    family: SignalFamily
    weight: float
    message: str
    decisive: bool = False


#: The catalogue. Codes come from Modules 1 through 8 verbatim; a finding whose
#: code is absent here is reported as unrecognised rather than silently
#: dropped, because a new upstream warning that nobody scored is a hole in the
#: risk engine that no test would otherwise notice.
CATALOGUE: dict[str, SignalDefinition] = {
    # -- Document integrity: the card contradicts itself ------------------- #
    "CNIC_GENDER_MISMATCH": SignalDefinition(
        code="CNIC_GENDER_MISMATCH",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.70,
        message=(
            "The printed gender contradicts the parity of the identity "
            "number's final digit. On a genuine CNIC these always agree, so "
            "either a digit was misread or the card was altered."
        ),
    ),
    "CNIC_ISSUE_BEFORE_BIRTH": SignalDefinition(
        code="CNIC_ISSUE_BEFORE_BIRTH",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.65,
        message="The card was issued before the holder was born.",
    ),
    "CNIC_EXPIRY_BEFORE_ISSUE": SignalDefinition(
        code="CNIC_EXPIRY_BEFORE_ISSUE",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.65,
        message="The card expires before it was issued.",
    ),
    "CNIC_BIRTH_IN_FUTURE": SignalDefinition(
        code="CNIC_BIRTH_IN_FUTURE",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.65,
        message="The date of birth is in the future.",
    ),
    "CNIC_IMPLAUSIBLE_AGE": SignalDefinition(
        code="CNIC_IMPLAUSIBLE_AGE",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.55,
        message="The implied age is outside any plausible range.",
    ),
    "CNIC_NOT_RECOGNISED": SignalDefinition(
        code="CNIC_NOT_RECOGNISED",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.40,
        message=(
            "The image is readable but is not a Pakistani CNIC. Usually the "
            "wrong document rather than a fraudulent one."
        ),
    ),
    "CNIC_UNUSUAL_VALIDITY_TERM": SignalDefinition(
        code="CNIC_UNUSUAL_VALIDITY_TERM",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.15,
        message=(
            "The validity term is not one NADRA commonly issues. Most often a "
            "misread year digit."
        ),
    ),

    # -- Image authenticity: not a genuine capture ------------------------- #
    "PROFILE_IMAGE_IS_SCREENSHOT": SignalDefinition(
        code="PROFILE_IMAGE_IS_SCREENSHOT",
        family=SignalFamily.IMAGE_AUTHENTICITY,
        weight=0.60,
        message=(
            "A submitted photograph is a screenshot of an application rather "
            "than a photograph - commonly somebody else's social media."
        ),
    ),
    "PROFILE_IMAGE_IS_SCREEN_RECAPTURE": SignalDefinition(
        code="PROFILE_IMAGE_IS_SCREEN_RECAPTURE",
        family=SignalFamily.IMAGE_AUTHENTICITY,
        weight=0.60,
        message="A submitted photograph is a picture of a screen.",
    ),
    "PROFILE_IMAGE_IS_PRINT_RECAPTURE": SignalDefinition(
        code="PROFILE_IMAGE_IS_PRINT_RECAPTURE",
        family=SignalFamily.IMAGE_AUTHENTICITY,
        weight=0.50,
        message="A submitted photograph is a picture of a printed photograph.",
    ),
    "PROFILE_IMAGE_IS_SYNTHETIC": SignalDefinition(
        code="PROFILE_IMAGE_IS_SYNTHETIC",
        family=SignalFamily.IMAGE_AUTHENTICITY,
        weight=0.45,
        message=(
            "A submitted image is an illustration or avatar rather than a "
            "photograph. Often naivety rather than fraud."
        ),
    ),

    # -- Identity consistency ---------------------------------------------- #
    "FACE_MISMATCH": SignalDefinition(
        code="FACE_MISMATCH",
        family=SignalFamily.IDENTITY_CONSISTENCY,
        weight=0.80,
        message="The faces submitted are not all the same person.",
    ),
    "CNIC_FACE_MISMATCH": SignalDefinition(
        code="CNIC_FACE_MISMATCH",
        family=SignalFamily.IDENTITY_CONSISTENCY,
        weight=0.75,
        message=(
            "The live selfie does not match the portrait printed on the "
            "identity card."
        ),
    ),
    "SECONDARY_IMAGE_MISMATCH": SignalDefinition(
        code="SECONDARY_IMAGE_MISMATCH",
        family=SignalFamily.IDENTITY_CONSISTENCY,
        weight=0.45,
        message=(
            "One of the additional photographs shows a different person. Can "
            "be innocent - a photograph of a family member - but the account "
            "claims they are all the same person."
        ),
    ),

    # -- Presentation: how the submission was staged ----------------------- #
    "CNIC_FOREIGN_FACE_PRESENT": SignalDefinition(
        code="CNIC_FOREIGN_FACE_PRESENT",
        family=SignalFamily.PRESENTATION,
        weight=0.55,
        message=(
            "A face was visible that is not printed on the card - most often "
            "the card held up in front of somebody. It matters because a "
            "naive extractor would compare the live face instead of the "
            "portrait and pass whoever the card belongs to."
        ),
    ),
    "CNIC_FACE_NOT_FOUND": SignalDefinition(
        code="CNIC_FACE_NOT_FOUND",
        family=SignalFamily.PRESENTATION,
        weight=0.35,
        message=(
            "No portrait could be located on the card, so it could not be "
            "compared with the selfie."
        ),
    ),
    "PROFILE_IMAGE_IS_GROUP_PHOTO": SignalDefinition(
        code="PROFILE_IMAGE_IS_GROUP_PHOTO",
        family=SignalFamily.PRESENTATION,
        weight=0.20,
        message=(
            "More than one person is in a submitted photograph, so which one "
            "the account belongs to is ambiguous. Not dishonest."
        ),
    ),
    "MULTIPLE_FACES_DETECTED": SignalDefinition(
        code="MULTIPLE_FACES_DETECTED",
        family=SignalFamily.PRESENTATION,
        weight=0.20,
        message="More than one person is present in an image meant to show one.",
    ),

    # -- Duplication -------------------------------------------------------- #
    "DUPLICATE_FACE_DETECTED": SignalDefinition(
        code="DUPLICATE_FACE_DETECTED",
        family=SignalFamily.DUPLICATION,
        weight=0.70,
        message=(
            "This face is already enrolled under another account. Not proof "
            "of one person: face recognition cannot separate identical twins "
            "at any threshold, and siblings score well above chance."
        ),
    ),
    "DUPLICATE_FACE_REVIEW": SignalDefinition(
        code="DUPLICATE_FACE_REVIEW",
        family=SignalFamily.DUPLICATION,
        weight=0.30,
        message="A similar face is enrolled elsewhere, close enough to check.",
    ),

    # -- Capture quality: weak evidence of anything ------------------------ #
    "LOW_IMAGE_QUALITY": SignalDefinition(
        code="LOW_IMAGE_QUALITY",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.15,
        message="A submitted image is too degraded for reliable analysis.",
    ),
    "FACE_NOT_DETECTED": SignalDefinition(
        code="FACE_NOT_DETECTED",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.15,
        message="No face could be found in an image that should contain one.",
    ),
    "FACE_OCCLUDED": SignalDefinition(
        code="FACE_OCCLUDED",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.20,
        message=(
            "The face is substantially covered. Usually a mask or sunglasses; "
            "deliberate concealment is the less common reading."
        ),
    ),
    "FACE_POSE_OUT_OF_RANGE": SignalDefinition(
        code="FACE_POSE_OUT_OF_RANGE",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message="The head is turned too far for reliable comparison.",
    ),
    "CNIC_OCR_FAILED": SignalDefinition(
        code="CNIC_OCR_FAILED",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.15,
        message="The card's text could not be read clearly enough to rely on.",
    ),
    "EMBEDDING_LOW_CONFIDENCE": SignalDefinition(
        code="EMBEDDING_LOW_CONFIDENCE",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message=(
            "A face template was produced from a poorly aligned or degraded "
            "crop, so the comparisons using it are less reliable."
        ),
    ),

    # -- Presentation, continued ------------------------------------------- #
    "MULTIPLE_FACES_PRESENT": SignalDefinition(
        code="MULTIPLE_FACES_PRESENT",
        family=SignalFamily.PRESENTATION,
        weight=0.20,
        message="More than one person is present in an image meant to show one.",
    ),

    # -- Document integrity, continued ------------------------------------- #
    "OCR_NOT_A_CNIC": SignalDefinition(
        code="OCR_NOT_A_CNIC",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.40,
        message=(
            "The uploaded document does not read as a Pakistani CNIC. Usually "
            "the wrong document rather than a fraudulent one."
        ),
    ),
    "CNIC_TOO_FEW_FIELDS": SignalDefinition(
        code="CNIC_TOO_FEW_FIELDS",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.20,
        message="Too little of the card could be read to check it against itself.",
    ),
    "CNIC_FIELDS_MISSING": SignalDefinition(
        code="CNIC_FIELDS_MISSING",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.15,
        message="Some of the card's fields could not be read.",
    ),
    "CNIC_UNKNOWN_REGION_CODE": SignalDefinition(
        code="CNIC_UNKNOWN_REGION_CODE",
        family=SignalFamily.DOCUMENT_INTEGRITY,
        weight=0.25,
        message=(
            "The identity number's leading digit is not an allocated region "
            "code. Either a misread digit or a number that was never issued."
        ),
    ),

    # -- Image authenticity, continued ------------------------------------- #
    "DECOMPRESSION_BOMB": SignalDefinition(
        code="DECOMPRESSION_BOMB",
        family=SignalFamily.IMAGE_AUTHENTICITY,
        weight=0.75,
        message=(
            "The upload expands to far more pixels than its file size implies "
            "- a deliberately malformed image intended to exhaust the service. "
            "Nobody sends one of these by accident."
        ),
    ),

    # -- Capture quality, continued ---------------------------------------- #
    "FACE_TOO_SMALL": SignalDefinition(
        code="FACE_TOO_SMALL",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message="The face occupies too little of the frame to compare reliably.",
    ),
    "FACE_TRUNCATED": SignalDefinition(
        code="FACE_TRUNCATED",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.15,
        message="Part of the face falls outside the frame.",
    ),
    "FACE_NOT_VISIBLE": SignalDefinition(
        code="FACE_NOT_VISIBLE",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.20,
        message="Overall face visibility is below what comparison needs.",
    ),
    "PARTIAL_OCCLUSION": SignalDefinition(
        code="PARTIAL_OCCLUSION",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.15,
        message=(
            "Part of the face is covered. Usually a hand, hair or glasses; "
            "deliberate concealment is the less common reading."
        ),
    ),
    "POSE_NOT_FRONTAL": SignalDefinition(
        code="POSE_NOT_FRONTAL",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message="The head is turned away from the camera.",
    ),
    "VISIBILITY_MARGINAL": SignalDefinition(
        code="VISIBILITY_MARGINAL",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message=(
            "Overall face visibility is only just adequate, so any "
            "comparison using this image carries more uncertainty."
        ),
    ),
    "LANDMARKS_UNAVAILABLE": SignalDefinition(
        code="LANDMARKS_UNAVAILABLE",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message=(
            "No facial keypoints were found, so the recognition crop was "
            "aligned from the box alone and is materially less accurate."
        ),
    ),
    "OCR_CARD_NOT_ISOLATED": SignalDefinition(
        code="OCR_CARD_NOT_ISOLATED",
        family=SignalFamily.CAPTURE_QUALITY,
        weight=0.10,
        message="The card's edges could not be found in the photograph.",
    ),
}

#: Findings that are informational and must never contribute to risk. Listed
#: explicitly rather than simply omitted from the catalogue, so that the
#: "unrecognised code" report stays meaningful: an absent code is a gap, and a
#: code here is a decision.
BENIGN_CODES: frozenset[str] = frozenset({
    # -- Correct readings that are not adverse -------------------------- #
    "CNIC_EXPIRED",
    "CNIC_GENDER_CONSISTENT",
    "CNIC_GENDER_DERIVED",
    "CNIC_GHOST_PORTRAIT_PRESENT",
    "PROFILE_IMAGE_HAS_NO_FACE",
    "DUPLICATE_MAY_BE_A_RELATIVE",

    # -- Conclusions, not evidence (Module 10's decision engine) -------- #
    # These are what the engine *decided*, derived from the fraud score that
    # was already computed. Feeding one back in as a signal would score the
    # conclusion a second time on top of the evidence that produced it.
    "EVIDENCE_SUFFICIENT",
    "FRAUD_RISK_ACCEPTABLE",
    "FRAUD_RISK_TOO_HIGH",
    "IDENTITY_CONFIDENCE_SUFFICIENT",
    "IDENTITY_CONFIDENCE_TOO_LOW",
    # `NO_IDENTITY_COMPARISON` deliberately absent: it is a *blocking* code,
    # passed straight to the decision engine rather than through the collector,
    # so it is never classified here. The reachability scanner does not see it
    # either, which is why adding it here failed the stale-entry test.

    # -- The service explaining itself ---------------------------------- #
    "CNIC_CARD_BOUNDS_NOT_FOUND",
    "PROFILE_IMAGE_QUALITY_LOW",
    "DUPLICATE_SELF_NOT_EXCLUDED",
    "DUPLICATE_THRESHOLD_UNCALIBRATED",
    "PROFILE_DETECTOR_UNAVAILABLE",
    "MATCHING_THRESHOLDS_UNVALIDATED",
    "MATCHING_NO_REFERENCE",
    "MATCHING_OBSERVATION",
    "QUALITY_OBSERVATION",
    "QUALITY_DIMENSION_UNMEASURED",
    "BACKGROUND_FACES_IGNORED",
    "IMAGE_UPSCALED",
    "OCR_ROTATION_CORRECTED",
    "LANDMARKS_DERIVED",
    "EMBEDDING_BOX_ALIGNED",
    "FALLBACK_DETECTOR_ACTIVE",
    "OCR_FALLBACK_ENGINE",

    # -- Surfaced for a human, deliberately not scored ------------------- #
    #
    # A CNIC photographed off a screen is a genuinely useful thing to know, and
    # the moire detector is the one authenticity check that transfers from
    # photographs of people to images of documents. But it transfers with a
    # thin margin, measured:
    #
    #   card shot off a screen              1.000   (fires)
    #   card photographed honestly, on a desk 0.393   (does not fire, but close)
    #
    # A real card photographed at a different angle could cross that. Scoring
    # it would let an uncalibrated margin drive an automated rejection of a
    # legitimate identity document, so it is reported and left to a human.
    "CNIC_IMAGE_MAY_BE_SCREEN_RECAPTURE",

    # -- Generic envelopes whose specific cause is always reported too --- #
    #
    # Module 7 returns INVALID_IMAGE alongside the finding that caused it
    # (PROFILE_IMAGE_IS_SCREENSHOT and friends), so scoring the envelope as
    # well would count one fact twice - in a *different* family, which is
    # precisely what the family grouping exists to prevent.
    "INVALID_IMAGE",
    "VALIDATION_ERROR",

    # -- The user sent something the service cannot read ----------------- #
    # Bad input, not evidence about the person. A malformed upload is a
    # retry, not an accusation - with the deliberate exception of
    # DECOMPRESSION_BOMB, which is scored.
    "UNSUPPORTED_IMAGE_FORMAT",
    "IMAGE_DECODE_FAILED",
    "IMAGE_TOO_LARGE",
    "IMAGE_TOO_SMALL",
    "PAYLOAD_TOO_LARGE",

    # -- Transport and policy, which never reach this engine ------------- #
    "UNAUTHORIZED",
    "FORBIDDEN",
    "RATE_LIMITED",
})

#: Codes meaning **a check could not run**, not that anything was found.
#:
#: These are the ones it would be most damaging to get wrong. An unreachable
#: vector database or an unloaded model produces silence, and silence scored as
#: "nothing suspicious" is how a fraud engine gets quietly switched off by an
#: outage while continuing to emit confident low-risk scores. They contribute
#: nothing and lower ``assessment_confidence`` instead.
INFRASTRUCTURE_CODES: frozenset[str] = frozenset({
    "AI_SERVICE_ERROR",
    "CACHE_ERROR",
    "CONFIGURATION_ERROR",
    "DEPENDENCY_UNAVAILABLE",
    "INFERENCE_FAILED",
    "MODEL_CHECKSUM_MISMATCH",
    "MODEL_LOAD_FAILED",
    "MODEL_NOT_LOADED",
    "PROCESSING_TIMEOUT",
    "QUEUE_ERROR",
    "VECTOR_DB_ERROR",

    # Module 10's pipeline. A stage that failed and a budget that ran out are
    # both checks that could not run, so they mark evidence unavailable rather
    # than scoring as suspicion. An outage is not a fraud signal, and a slow
    # request is not a dishonest one.
    "VERIFICATION_BUDGET_EXHAUSTED",
    "VERIFICATION_STAGE_FAILED",
})


@dataclass(frozen=True, slots=True)
class FraudSignal:
    """One piece of evidence, as it applies to this request.

    Attributes:
        definition: What the finding means and what it weighs.
        confidence: How sure the upstream module was, in ``[0, 1]``. A finding
            reported without a confidence is taken as certain, because the
            module chose to report it.
        stage: Which module produced it.
        detail: Structured context. Must be PII-free.
    """

    definition: SignalDefinition
    confidence: float = 1.0
    stage: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def code(self) -> str:
        """The finding's stable code."""
        return self.definition.code

    @property
    def family(self) -> SignalFamily:
        """Which kind of evidence this is."""
        return self.definition.family

    @property
    def contribution(self) -> float:
        """Evidence strength once uncertainty is taken into account."""
        return float(
            min(max(self.definition.weight * self.confidence, 0.0), 1.0)
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form, safe for logs."""
        return {
            "code": self.code,
            "family": str(self.family),
            "weight": round(self.definition.weight, 4),
            "confidence": round(self.confidence, 4),
            "contribution": round(self.contribution, 4),
            "decisive": self.definition.decisive,
            "stage": self.stage,
            "message": self.definition.message,
            "detail": dict(self.detail),
        }


def lookup(code: str) -> SignalDefinition | None:
    """Find a definition by code, or ``None`` when it is not scored."""
    return CATALOGUE.get(code)


def is_benign(code: str) -> bool:
    """Whether a code is deliberately not scored."""
    return code in BENIGN_CODES


def is_infrastructure(code: str) -> bool:
    """Whether a code means a check could not run rather than a finding."""
    return code in INFRASTRUCTURE_CODES


def classify(code: str) -> str:
    """Return ``scored``, ``benign``, ``infrastructure`` or ``unknown``.

    Exposed so a test can assert that **every** code the service can emit falls
    into one of the first three. An unclassified code contributes nothing and
    fails nothing, which is the one failure mode this engine cannot detect
    about itself.
    """
    if code in CATALOGUE:
        return "scored"
    if code in BENIGN_CODES:
        return "benign"
    if code in INFRASTRUCTURE_CODES:
        return "infrastructure"
    return "unknown"


__all__ = [
    "BENIGN_CODES",
    "CATALOGUE",
    "INFRASTRUCTURE_CODES",
    "FraudSignal",
    "SignalDefinition",
    "SignalFamily",
    "classify",
    "is_benign",
    "is_infrastructure",
    "lookup",
]
