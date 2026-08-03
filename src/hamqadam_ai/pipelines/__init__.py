"""MODULE 10 - end-to-end orchestration.

One pipeline, one shared time budget, and a stage-failure policy of "record it
and continue". A verification where the CNIC was unreadable still has a face
comparison worth reporting.
"""

from hamqadam_ai.pipelines.verification import (
    VerificationImages,
    VerificationPipeline,
    build_pipeline,
)

__all__ = ["VerificationImages", "VerificationPipeline", "build_pipeline"]
