"""MODULE 4 - face matching.

Turns pairs of embeddings into the scores and decisions the Backend's rules
engine consumes:

* live selfie against the main profile image,
* live selfie against each secondary profile image,
* live selfie against the portrait on the CNIC.

Structure
---------
``similarity``
    Cosine comparison and the calibration that maps it onto the reported
    0-100 scale.

``comparator``
    One pair in, one :class:`~hamqadam_ai.matching.comparator.MatchOutcome`
    out. Applies the configured operating points and propagates embedding
    confidence.

``aggregator``
    Many outcomes in, one identity confidence out. Weights the CNIC most
    heavily and caps the result when the document comparison fails.

The live selfie is the reference for every comparison. That is deliberate: it
is the only image captured under the app's control, so it is the closest thing
to a trusted sample in the request.
"""

from __future__ import annotations

from hamqadam_ai.matching.aggregator import (
    IdentityAggregator,
    IdentityAssessment,
)
from hamqadam_ai.matching.comparator import (
    ComparisonType,
    FaceComparator,
    MatchOutcome,
)
from hamqadam_ai.matching.evaluation import (
    ConfusionMatrix,
    EvaluationReport,
    RocCurve,
    confusion_matrix,
    equal_error_rate,
    evaluate,
    recommend_thresholds,
    roc_curve,
)
from hamqadam_ai.matching.similarity import (
    calibrate_score,
    decide,
    uncalibrate_score,
)

__all__ = [
    "ConfusionMatrix",
    "ComparisonType",
    "FaceComparator",
    "EvaluationReport",
    "IdentityAggregator",
    "IdentityAssessment",
    "MatchOutcome",
    "RocCurve",
    "calibrate_score",
    "confusion_matrix",
    "decide",
    "equal_error_rate",
    "evaluate",
    "recommend_thresholds",
    "roc_curve",
    "uncalibrate_score",
]
