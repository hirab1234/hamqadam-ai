"""Qdrant-backed gallery: the durable, shared adapter.

Chosen over the in-process store when the gallery must survive a restart or be
shared between replicas - which is every real deployment. A cluster running the
memory store would give each pod a different partial gallery, so whether a
duplicate was found would depend on which one answered.

Two decisions worth explaining
------------------------------
**Cosine distance, and vectors normalised before they are sent.** Qdrant's
cosine metric normalises internally, but the service normalises anyway: the
same vector then means the same thing in both stores, and the calibration
script can read raw vectors out of either and get comparable numbers.

**The model version is a payload filter, not part of the point id.** Making it
part of the id would let one reference hold two templates from two model
versions, which sounds harmless and is not: after an upgrade the gallery would
contain both, the old ones would never be searched, and nothing would ever say
so. A filter makes the stale records visible to ``count`` and deletable.

Point ids
---------
Qdrant requires an unsigned integer or a UUID. The Backend's reference is an
arbitrary string, so it is hashed into a UUID5 under a fixed namespace -
deterministic, so re-enrolling the same reference replaces rather than
duplicates, and the original reference is kept in the payload because a UUID5
cannot be reversed.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import uuid
from typing import Any

import numpy as np

from hamqadam_ai.core.exceptions import DependencyUnavailableError, VectorStoreError
from hamqadam_ai.duplicate_detection.base import (
    FloatArray,
    SearchHit,
    VectorRecord,
    VectorStore,
)
from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)

#: Fixed namespace for deriving point ids from Backend references. Must never
#: change: it would orphan every previously enrolled template.
REFERENCE_NAMESPACE = uuid.UUID("6f1c9a2e-4b7d-5e83-9a10-2c7f4d8b1e6a")

#: Payload keys. Named constants because the calibration script and any manual
#: Qdrant query have to agree with the writer.
FIELD_REFERENCE = "reference"
FIELD_MODEL_VERSION = "model_version"
FIELD_ENROLLED_AT = "enrolled_at"
FIELD_METADATA = "metadata"


def point_id(reference: str) -> str:
    """Derive a deterministic Qdrant point id from a Backend reference."""
    return str(uuid.uuid5(REFERENCE_NAMESPACE, reference))


class QdrantVectorStore(VectorStore):
    """Durable gallery backed by a Qdrant collection.

    Args:
        url: Qdrant endpoint, for example ``http://qdrant:6333``. Two local
            forms are also accepted and are handled by the client's embedded
            engine rather than over HTTP: ``:memory:`` for a gallery that
            lives and dies with the process, and a filesystem path for one
            persisted to local storage. The embedded engine runs the same
            query planner as the server, which is what makes it worth testing
            this adapter against rather than a hand-written fake.
        collection: Collection name.
        dimension: Vector length. Must match the recogniser's output.
        api_key: Optional API key for a managed instance.
        timeout_seconds: Per-request timeout.
        create_collection: Create the collection when it is absent. Left on by
            default so a fresh deployment works; a locked-down environment can
            turn it off and provision the collection itself.
    """

    __slots__ = ("_client", "_collection", "_dimension", "_local", "_models")

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dimension: int,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        create_collection: bool = True,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise DependencyUnavailableError(
                "qdrant-client",
                purpose="durable duplicate-face gallery",
                extra="qdrant-client",
                cause=exc,
            ) from exc

        super().__init__(name="qdrant")
        self._models = models
        self._collection = collection
        self._dimension = dimension

        try:
            # Three modes, and the client wants a different keyword for each.
            #
            # ``:memory:``     -> location=":memory:"   ephemeral, dies with the process
            # a filesystem dir -> path="/var/lib/..."   embedded and DURABLE
            # an http(s) URL   -> url="http://..."      a Qdrant server
            #
            # `path` rather than `location` for a directory is the part that was
            # wrong. `location` is parsed as a URL, so a perfectly good path was
            # rejected with "Unknown scheme: c" on Windows (`C:/data`) and would
            # have silently become a *hostname* on Linux (`/var/lib/qdrant`),
            # producing a connection error rather than an on-disk gallery. The
            # documented "or a filesystem path" mode simply did not work.
            self._local = url == ":memory:" or not url.startswith(
                ("http://", "https://")
            )
            if url == ":memory:":
                self._client = QdrantClient(location=":memory:")
            elif self._local:
                self._client = QdrantClient(path=url)
            else:
                self._client = QdrantClient(
                    url=url, api_key=api_key, timeout=int(timeout_seconds)
                )
            if create_collection:
                self._ensure_collection()
        except DependencyUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised into a typed error
            raise VectorStoreError(
                f"could not reach Qdrant at {url}: {exc}",
                details={"collection": collection},
                cause=exc,
            ) from exc

        log.info(
            "duplicate.qdrant_store_ready",
            collection=collection,
            dimension=dimension,
        )

    def _ensure_collection(self) -> None:
        """Create the collection and its payload index if they are absent.

        The index on ``model_version`` is not optional. Every search filters on
        it, and an unindexed payload filter in Qdrant is a full scan - which
        turns a millisecond query into a linear one at exactly the gallery
        size where it matters.
        """
        models = self._models
        if self._client.collection_exists(self._collection):
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )
        # The embedded engine has no payload indexes and warns if asked for
        # one. It also has no query planner to need it: local mode scans
        # regardless, so the index is a server-only concern.
        if not self._local:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=FIELD_MODEL_VERSION,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        log.info("duplicate.qdrant_collection_created", collection=self._collection)

    def enrol(self, record: VectorRecord) -> None:
        """Store or replace one template."""
        models = self._models
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=point_id(record.reference),
                        vector=record.vector.tolist(),
                        payload={
                            FIELD_REFERENCE: record.reference,
                            FIELD_MODEL_VERSION: record.model_version,
                            FIELD_ENROLLED_AT: record.enrolled_at.isoformat(),
                            FIELD_METADATA: dict(record.metadata),
                        },
                    )
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not enrol into Qdrant: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc

    def search(
        self,
        vector: FloatArray,
        *,
        model_version: str,
        top_k: int,
        exclude: str | None = None,
    ) -> list[SearchHit]:
        """Find the nearest comparable templates."""
        models = self._models
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm <= 0.0:
            return []
        query = query / norm

        conditions: list[Any] = [
            models.FieldCondition(
                key=FIELD_MODEL_VERSION,
                match=models.MatchValue(value=model_version),
            )
        ]
        must_not: list[Any] = []
        if exclude is not None:
            must_not.append(models.HasIdCondition(has_id=[point_id(exclude)]))

        try:
            found = self._client.query_points(
                collection_name=self._collection,
                query=query.tolist(),
                # One extra, so excluding the querying user by id cannot leave
                # the caller a hit short of what it asked for.
                limit=top_k + (1 if exclude else 0),
                query_filter=models.Filter(must=conditions, must_not=must_not),
                with_payload=True,
            ).points
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Qdrant search failed: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc

        hits: list[SearchHit] = []
        for point in found[:top_k]:
            payload = point.payload or {}
            reference = str(payload.get(FIELD_REFERENCE, point.id))
            if reference == exclude:
                continue
            hits.append(
                SearchHit(
                    reference=reference,
                    # Qdrant's cosine score is the similarity itself, in
                    # [-1, 1], not a distance. Converting would invert it.
                    similarity=float(point.score),
                    model_version=str(payload.get(FIELD_MODEL_VERSION, model_version)),
                    enrolled_at=_parse_timestamp(payload.get(FIELD_ENROLLED_AT)),
                    metadata=dict(payload.get(FIELD_METADATA) or {}),
                )
            )
        return hits

    def exists(self, reference: str) -> bool:
        """Whether a template is stored under this reference."""
        try:
            found = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id(reference)],
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not look up a Qdrant point: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc
        return bool(found)

    def delete(self, reference: str) -> bool:
        """Remove a template. Idempotent."""
        models = self._models
        try:
            if not self.exists(reference):
                return False
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=[point_id(reference)]),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not delete from Qdrant: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc
        return True

    def list_references(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Enumerate stored references.

        ``with_vectors=False`` is not an optimisation. A 512-float template is
        biometric data, and an inspection endpoint that returns it turns a
        debugging aid into an exfiltration route.
        """
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                limit=limit + offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not scroll Qdrant points: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc

        rows = [self._describe(point.payload or {}) for point in points]
        rows.sort(key=lambda row: str(row.get("reference") or ""))
        return rows[offset : offset + limit]

    def get(self, reference: str) -> dict[str, Any] | None:
        """One record's metadata, or ``None``. Vectors are never returned."""
        try:
            found = self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id(reference)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not look up a Qdrant point: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc
        if not found:
            return None
        return self._describe(found[0].payload or {})

    @staticmethod
    def _describe(payload: dict[str, Any]) -> dict[str, Any]:
        """Render a payload for inspection, without the vector."""
        return {
            "reference": payload.get(FIELD_REFERENCE),
            "model_version": payload.get(FIELD_MODEL_VERSION),
            "enrolled_at": payload.get(FIELD_ENROLLED_AT),
            "metadata": payload.get(FIELD_METADATA) or {},
        }

    def count(self, *, model_version: str | None = None) -> int:
        """How many templates are enrolled."""
        models = self._models
        query_filter = None
        if model_version is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=FIELD_MODEL_VERSION,
                        match=models.MatchValue(value=model_version),
                    )
                ]
            )
        try:
            return int(
                self._client.count(
                    collection_name=self._collection,
                    count_filter=query_filter,
                    exact=True,
                ).count
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"could not count Qdrant points: {exc}",
                details={"collection": self._collection},
                cause=exc,
            ) from exc

    def close(self) -> None:
        """Close the client connection."""
        # Closing must not raise: this runs from a `finally` and from
        # interpreter shutdown, where an exception loses the real error.
        with contextlib.suppress(Exception):
            self._client.close()

    def health(self) -> dict[str, Any]:
        """Adapter status, including whether the collection is reachable."""
        try:
            size = self.count()
        except VectorStoreError as exc:
            return {
                "store": self.name,
                "available": False,
                "durable": True,
                "error": str(exc),
            }
        return {
            "store": self.name,
            "available": True,
            "durable": True,
            "collection": self._collection,
            "records": size,
        }


def _parse_timestamp(value: object) -> dt.datetime | None:
    """Parse an ISO timestamp from a payload, tolerating a missing one."""
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["REFERENCE_NAMESPACE", "QdrantVectorStore", "point_id"]
