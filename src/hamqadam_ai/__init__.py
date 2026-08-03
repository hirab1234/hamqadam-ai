"""Hamqadam AI Identity Verification Service.

A production identity-verification engine that receives images from the
Hamqadam Backend API, runs face, document and fraud analysis, and returns a
standardised verification result.

The package is layered as a clean architecture:

``core`` / ``schemas``
    Domain layer. Enums, value objects, the error taxonomy and configuration
    contracts. Contains no I/O and imports nothing from the outer layers.

``detectors`` / ``quality`` / ``embeddings`` / ``matching`` / ``ocr`` /
``duplicate_detection`` / ``fraud_detection``
    Capability layer. Each package exposes an abstract port plus one or more
    concrete adapters. Adapters are interchangeable through configuration.

``models`` / ``utils`` / ``logging``
    Infrastructure layer. ONNX Runtime session management, the versioned model
    registry, image codecs, secure temporary storage and structured logging.

``services`` / ``pipelines``
    Application layer. Orchestrates the capability layer and applies the
    configurable accept / reject policy.

``api`` / ``workers``
    Delivery layer. FastAPI HTTP surface and asynchronous queue consumers.

Dependencies point strictly inward. A detector never imports a service; a
service never imports FastAPI.
"""

from __future__ import annotations

__all__ = ["__version__", "SERVICE_NAME"]

#: Version of the *service contract*. Bumped whenever the response schema or
#: the semantics of a returned score change. Reported in every API response so
#: the Backend can pin behaviour.
__version__ = "1.0.0"

SERVICE_NAME = "hamqadam-ai-verification"
