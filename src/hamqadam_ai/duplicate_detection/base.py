"""The vector-store port and its value objects.

Why this module is different from every other one
--------------------------------------------------
Everything else in this service is stateless: images arrive, scores leave,
nothing is kept. This module **stores biometric templates**, which makes it the
one place where the privacy posture is a design problem rather than a matter of
deleting temporary files.

Three consequences run through the code below:

* A record is keyed by an **opaque reference the Backend supplies**, never by
  anything this service invents or derives from a face. The Backend owns the
  mapping from reference to account; the AI service cannot resolve one.
* Every record carries the **model version** that produced its vector, and a
  search never crosses versions. Two embeddings from different ArcFace builds
  are not comparable, and silently comparing them would produce scores that
  look ordinary and mean nothing.
* Deletion is a first-class operation, not an afterthought. A stored face
  template is personal data under any reading, and a service that cannot erase
  one on request cannot be deployed.

1:N is not 1:1
--------------
Module 4 asks "are these two the same person?". This module asks "is this
person already in the gallery?", and the difference is not cosmetic - see
:mod:`hamqadam_ai.duplicate_detection.calibration` for the measurement showing
how far the operating point moves.
"""

from __future__ import annotations

import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One enrolled face template.

    Attributes:
        reference: The Backend's opaque identifier for whoever this belongs
            to. This service never interprets it, and never derives one.
        vector: The L2-normalised embedding.
        model_version: Version of the recogniser that produced it. Records are
            never compared across versions.
        enrolled_at: When it was stored, UTC.
        metadata: Free-form labels the Backend supplied. Must be PII-free -
            this service cannot enforce that and says so rather than pretending
            to.
    """

    reference: str
    vector: FloatArray
    model_version: str
    enrolled_at: dt.datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if norm <= 0.0:
            raise ValueError("cannot enrol a zero vector")
        object.__setattr__(self, "vector", array / norm)

    @property
    def dimension(self) -> int:
        """Length of the stored vector."""
        return int(self.vector.shape[0])

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging. Never includes the vector.

        An embedding is biometric data. It is not a diagnostic, and it does not
        go in a log line.
        """
        return {
            "reference": self.reference,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "enrolled_at": self.enrolled_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One gallery entry that resembled the query.

    Attributes:
        reference: The Backend's identifier for the matched record.
        similarity: Cosine similarity to the query, in ``[-1, 1]``.
        model_version: Version the matched vector was produced by. Always
            equal to the query's, because a search never crosses versions.
        enrolled_at: When the matched record was stored.
        metadata: Whatever the Backend attached at enrolment.
    """

    reference: str
    similarity: float
    model_version: str
    enrolled_at: dt.datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp into the mathematically valid range. A cosine cannot exceed 1,
        # but a backend computing one in float32 can report that it did:
        # Qdrant returned 1.0000000158616884 for a vector matched against
        # itself, which the response schema then rejected outright and took
        # the whole verification down with it. Clamping is not papering over a
        # discrepancy - the excess *is* the rounding error, and the true value
        # is the bound.
        object.__setattr__(
            self, "similarity", float(min(max(self.similarity, -1.0), 1.0))
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "reference": self.reference,
            "similarity": round(self.similarity, 6),
            "model_version": self.model_version,
            "enrolled_at": (
                self.enrolled_at.isoformat() if self.enrolled_at else None
            ),
            "metadata": dict(self.metadata),
        }


class VectorStore(abc.ABC):
    """Abstract gallery of enrolled face templates.

    Args:
        name: Short adapter identifier, reported on ``/health``.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def enrol(self, record: VectorRecord) -> None:
        """Store or replace one template.

        Replacing rather than appending on a repeated reference is deliberate:
        a user re-verifying should end with one template, not a growing pile
        that would each match the next query.
        """

    @abc.abstractmethod
    def search(
        self,
        vector: FloatArray,
        *,
        model_version: str,
        top_k: int,
        exclude: str | None = None,
    ) -> list[SearchHit]:
        """Find the nearest enrolled templates.

        Args:
            vector: The query embedding, L2-normalised.
            model_version: Only records from this recogniser version are
                considered. Not a filter for convenience - vectors from
                different versions are not comparable at all.
            top_k: How many hits to return, best first.
            exclude: A reference to omit. **Load-bearing**: without it a user
                re-verifying matches their own enrolled template at cosine 1.0
                and is reported as a duplicate of themselves, every time.

        Returns:
            Hits sorted by descending similarity. Empty when the gallery holds
            nothing comparable, which is the normal state of a new deployment.
        """

    @abc.abstractmethod
    def exists(self, reference: str) -> bool:
        """Whether a template is already stored under this reference.

        A direct lookup, not a nearest-neighbour search. An earlier version of
        the service inferred this from a top-1 search, which returns the
        *closest* record rather than the one asked for - so it reported "new
        enrolment" whenever somebody else's template happened to be nearer.
        """

    @abc.abstractmethod
    def delete(self, reference: str) -> bool:
        """Remove a template.

        Returns:
            Whether anything was removed. Idempotent: deleting an absent
            reference is a successful no-op, because a caller retrying an
            erasure request must not get an error the second time.
        """

    @abc.abstractmethod
    def count(self, *, model_version: str | None = None) -> int:
        """How many templates are enrolled.

        The gallery size is not a curiosity. The false-match rate of a 1:N
        search compounds with it, so a caller cannot interpret a hit without
        knowing how many entries it beat.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any connection held by the adapter."""

    # -- Inspection ------------------------------------------------------- #
    #
    # Non-abstract on purpose. These serve the admin/debug routes only; the
    # verification pipeline never calls them. Making them abstract would break
    # every test double that implements the six operating methods, and an
    # adapter with no efficient scan is entitled to decline.

    def list_references(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Enumerate stored references, newest-agnostic order.

        **Never returns the vectors.** A 512-float template is biometric data;
        an inspection endpoint that hands it out turns a debugging aid into an
        exfiltration route. Only the reference, model version and enrolment
        timestamp are exposed.

        Raises:
            NotImplementedError: when the adapter cannot scan.
        """
        raise NotImplementedError(
            f"the {self.name} adapter does not support enumeration"
        )

    def get(self, reference: str) -> dict[str, Any] | None:
        """One record's metadata, or ``None`` when absent.

        Vectors are omitted for the same reason as above.
        """
        raise NotImplementedError(
            f"the {self.name} adapter does not support single-record lookup"
        )

    def health(self) -> dict[str, Any]:
        """Adapter status for ``/health``."""
        return {"store": self.name, "available": True}

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


def utc_now() -> dt.datetime:
    """Current UTC time, timezone-aware.

    Centralised so the stores and the service agree, and so a test can patch
    one place rather than three.
    """
    return dt.datetime.now(dt.UTC)


__all__ = [
    "FloatArray",
    "SearchHit",
    "VectorRecord",
    "VectorStore",
    "utc_now",
]
