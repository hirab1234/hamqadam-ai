"""Exact in-process gallery, backed by NumPy.

Not a stub. Exact search over a matrix of L2-normalised vectors is a single
matrix-vector product, and at the sizes a single deployment reaches it beats
an approximate index on both latency and correctness - the matrix is kept
materialised, so a query is one BLAS call over it.

Measured on an idle development machine, 512 dimensions, median of seven:

    gallery      search
      1,000      0.18 ms
     10,000      0.96 ms
     50,000      6.9 ms
    100,000     17.1 ms

Those figures are after the matrix was materialised. Stacking it per query -
which is what the first version did - cost 285 ms at 100,000 and grew faster
than linearly, because every search copied the whole 200 MB gallery.

Where it stops being the right answer is **durability**, not speed: this
gallery lives in one process and dies with it. That is fine for tests, for a
single-node deployment that re-enrols from the Backend's own records on start,
and for a development machine. It is not fine for a cluster, where several
replicas would each hold a different partial gallery and a duplicate would be
found or missed depending on which pod answered. Use Qdrant there.

The class is deliberately thread-safe. Uvicorn dispatches inference to a
thread pool, so two requests genuinely do enrol concurrently, and a torn
read of the matrix would produce a similarity score against half-written
memory rather than an error anybody would notice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from hamqadam_ai.duplicate_detection.base import (
    FloatArray,
    SearchHit,
    VectorRecord,
    VectorStore,
)
from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class _Gallery:
    """A growable matrix of templates, plus the references its rows name.

    A dynamic array rather than a plain matrix, because the alternative was
    measurably wrong. Appending with ``np.vstack`` copies the whole buffer, so
    building a gallery of n costs O(n^2): measured, a check-then-enrol cycle
    against 20,000 templates took **60 ms**, of which 58 ms was the copy - on a
    search that costs 2 ms. Doubling the capacity makes appends amortised O(1)
    and brought the same cycle to single-digit milliseconds.

    Attributes:
        references: One per used row, in row order.
        positions: Row index by reference. Turns two O(n) operations into
            O(1): replacing a template, which would otherwise discard the
            whole index, and excluding the querying user, which is a list
            scan on **every** query.
        buffer: Backing storage. Only the first ``count`` rows are live.
        count: Rows in use.
    """

    references: list[str]
    positions: dict[str, int]
    buffer: FloatArray
    count: int

    @classmethod
    def from_rows(cls, references: list[str], vectors: list[FloatArray]) -> _Gallery:
        """Build from an existing set of templates."""
        if not vectors:
            return cls(
                references=[],
                positions={},
                buffer=np.zeros((0, 0), dtype=np.float32),
                count=0,
            )
        stacked = np.stack(vectors)
        # Room to grow, so the first few enrolments after a rebuild do not each
        # trigger another one.
        capacity = max(stacked.shape[0] * 2, 16)
        buffer = np.zeros((capacity, stacked.shape[1]), dtype=np.float32)
        buffer[: stacked.shape[0]] = stacked
        return cls(
            references=list(references),
            positions={ref: row for row, ref in enumerate(references)},
            buffer=buffer,
            count=stacked.shape[0],
        )

    @property
    def matrix(self) -> FloatArray:
        """A view of the live rows. Not a copy."""
        return self.buffer[: self.count]

    @property
    def dimension(self) -> int:
        """Row width, or zero when empty."""
        return int(self.buffer.shape[1]) if self.buffer.size else 0

    def append(self, reference: str, vector: FloatArray) -> None:
        """Add one row, growing the buffer only when it is full."""
        if self.count >= self.buffer.shape[0]:
            capacity = max(self.buffer.shape[0] * 2, 16)
            grown = np.zeros((capacity, self.buffer.shape[1]), dtype=np.float32)
            grown[: self.count] = self.matrix
            self.buffer = grown
        self.buffer[self.count] = vector
        self.references.append(reference)
        self.positions[reference] = self.count
        self.count += 1

    def replace(self, reference: str, vector: FloatArray) -> bool:
        """Overwrite an existing row in place.

        The path a **returning user** takes: re-verifying re-enrols under the
        same reference. Invalidating the index for it - which is what the first
        version did - meant rebuilding the whole gallery on every
        re-verification.

        Returns:
            Whether the reference was present and overwritten.
        """
        row = self.positions.get(reference)
        if row is None:
            return False
        self.buffer[row] = vector
        return True

    def row_of(self, reference: str) -> int | None:
        """Row index for a reference, or ``None`` when it is absent."""
        return self.positions.get(reference)



class InMemoryVectorStore(VectorStore):
    """Exact nearest-neighbour gallery held in this process.

    Args:
        max_records: Refuse to grow past this. A guard, not a policy: an
            unbounded in-process gallery is a memory leak with a plausible
            excuse, and silently evicting the oldest entry would silently stop
            detecting duplicates of long-standing users.
    """

    __slots__ = ("_index", "_lock", "_max_records", "_records")

    def __init__(self, *, max_records: int = 250_000) -> None:
        super().__init__(name="memory")
        self._records: dict[str, VectorRecord] = {}
        self._max_records = max_records
        self._lock = threading.RLock()
        # Per model version: a growable row buffer and the references its rows
        # correspond to, or None when stale. Rebuilt lazily by `_matrix_for`.
        self._index: dict[str, _Gallery | None] = {}

    def enrol(self, record: VectorRecord) -> None:
        """Store or replace one template."""
        with self._lock:
            existing = self._records.get(record.reference)
            if existing is None and len(self._records) >= self._max_records:
                raise MemoryError(
                    f"in-memory gallery is full at {self._max_records} records; "
                    f"configure a Qdrant backend for a gallery this size"
                )
            self._records[record.reference] = record

            cached = self._index.get(record.model_version)
            same_version = (
                existing is None or existing.model_version == record.model_version
            )
            if (
                cached is not None
                and same_version
                and cached.dimension == record.dimension
            ):
                if existing is None:
                    # A brand-new reference: append. The common case - the
                    # pipeline checks, then enrols - so invalidating here would
                    # make the cache worthless exactly where it is needed.
                    cached.append(record.reference, record.vector)
                    return
                if cached.replace(record.reference, record.vector):
                    # A returning user re-verifying. Overwriting the row in
                    # place avoids rebuilding the whole gallery for what is a
                    # routine event.
                    return

            # A first record, a dimension change, or a version move: the cached
            # row order no longer describes the gallery.
            self._index[record.model_version] = None
            if existing is not None and existing.model_version != record.model_version:
                self._index[existing.model_version] = None

    def _matrix_for(self, model_version: str) -> _Gallery:
        """The comparable gallery as one buffer, rebuilt only when stale.

        Materialised and kept, rather than stacked per query. Stacking 100,000
        512-dimensional float32 rows copies 200 MB, and doing it on every
        search took a query to a measured **285 ms** - and made the cost grow
        faster than linearly in gallery size, which for a structure whose whole
        justification is that exact search *is* linear was the wrong shape as
        well as the wrong number.

        Caller must hold the lock.
        """
        cached = self._index.get(model_version)
        if cached is not None:
            return cached

        vectors = [
            record.vector
            for record in self._records.values()
            if record.model_version == model_version
        ]
        references = [
            record.reference
            for record in self._records.values()
            if record.model_version == model_version
        ]

        built = _Gallery.from_rows(references, vectors)
        self._index[model_version] = built
        return built

    def search(
        self,
        vector: FloatArray,
        *,
        model_version: str,
        top_k: int,
        exclude: str | None = None,
    ) -> list[SearchHit]:
        """Exact cosine search over the comparable subset of the gallery."""
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm <= 0.0:
            return []
        query = query / norm

        with self._lock:
            gallery = self._matrix_for(model_version)
            references = gallery.references
            if gallery.count == 0 or gallery.dimension != query.shape[0]:
                return []

            similarities = gallery.matrix @ query

            # Exclusion by score rather than by rebuilding the matrix without
            # that row: -inf can never win, and rebuilding would defeat the
            # cache on the one call that always passes an exclusion.
            if exclude is not None:
                # A dict lookup, not a list scan: this runs on every query, and
                # scanning would put an O(n) Python loop in front of a BLAS call
                # that is already O(n) but hundreds of times faster per element.
                row = gallery.row_of(exclude)
                if row is not None:
                    similarities[row] = -np.inf

            # argpartition rather than a full sort: at 100k records the sort is
            # most of the query cost and only the top few are ever read.
            take = min(top_k, similarities.shape[0])
            top = np.argpartition(-similarities, take - 1)[:take]
            top = top[np.argsort(-similarities[top])]

            hits: list[SearchHit] = []
            for index in top:
                if not np.isfinite(similarities[index]):
                    continue
                record = self._records[references[index]]
                hits.append(
                    SearchHit(
                        reference=record.reference,
                        similarity=float(similarities[index]),
                        model_version=record.model_version,
                        enrolled_at=record.enrolled_at,
                        metadata=dict(record.metadata),
                    )
                )
            return hits

    def exists(self, reference: str) -> bool:
        """Whether a template is stored under this reference."""
        with self._lock:
            return reference in self._records

    def delete(self, reference: str) -> bool:
        """Remove a template. Idempotent."""
        with self._lock:
            removed = self._records.pop(reference, None)
            if removed is not None:
                self._index[removed.model_version] = None
            return removed is not None

    def count(self, *, model_version: str | None = None) -> int:
        """How many templates are enrolled."""
        with self._lock:
            if model_version is None:
                return len(self._records)
            return sum(
                1
                for record in self._records.values()
                if record.model_version == model_version
            )

    def close(self) -> None:
        """Drop every template.

        The gallery is biometric data, so releasing this store releases the
        templates with it rather than leaving them for the garbage collector
        to get to eventually.
        """
        with self._lock:
            self._records.clear()
            self._index.clear()

    def health(self) -> dict[str, Any]:
        """Adapter status, including the durability caveat."""
        with self._lock:
            size = len(self._records)
        return {
            "store": self.name,
            "available": True,
            "records": size,
            "capacity": self._max_records,
            "durable": False,
            "note": (
                "in-process gallery; lost on restart and not shared between "
                "replicas"
            ),
        }

    # -- Test and calibration support ---------------------------------------- #

    def references(self) -> list[str]:
        """Every enrolled reference. Used by the calibration script."""
        with self._lock:
            return list(self._records)

    def vectors_for(self, model_version: str) -> FloatArray:
        """Every comparable vector as one matrix.

        Exposed for :mod:`hamqadam_ai.duplicate_detection.calibration`, which
        needs the whole gallery to measure its impostor distribution. Returns a
        copy, so a caller cannot mutate the gallery through it.
        """
        with self._lock:
            return self._matrix_for(model_version).matrix.copy()


def build_memory_store(max_records: int = 250_000) -> InMemoryVectorStore:
    """Construct an in-process gallery."""
    store = InMemoryVectorStore(max_records=max_records)
    log.info("duplicate.memory_store_ready", capacity=max_records)
    return store


__all__ = ["InMemoryVectorStore", "build_memory_store"]
