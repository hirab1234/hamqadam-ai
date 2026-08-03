"""Asynchronous verification, consumed from a queue.

Why a queue at all
------------------
A verification is seconds of CPU across seven images. Holding an HTTP
connection open for that works, and is what ``/v1/verify`` does, but it couples
the Backend's request timeout to this service's worst case and gives it nothing
to do while it waits. For bulk enrolment or a re-verification sweep, a queue is
the right shape.

Redelivery and the duplicate gallery
------------------------------------
The one thing that makes this more than a loop around the pipeline. A message
broker guarantees *at least once*, so a worker that crashes after finishing but
before acknowledging will see the same message again. For a stateless analysis
that is harmless. For enrolment it is not: the naive handling would add a
second template for one person, and every future query would then match them
twice.

Enrolment is keyed by the Backend's reference and **replaces** rather than
appends, so a redelivery is a no-op. That property lives in the store (Module
8) rather than here, which is why this consumer can be as simple as it is.

Poison messages
---------------
A message that fails repeatedly is rejected without requeue after
``max_attempts``, and the reason is published to the reply queue rather than
lost. Requeuing forever is how one malformed payload takes down a worker pool.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.context import RequestContext, pseudonymise, request_context
from hamqadam_ai.core.exceptions import DependencyUnavailableError, HamqadamError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.schemas.verification import VerificationRequest
from hamqadam_ai.utils.image_io import decode_image

log = get_logger(__name__)

#: How many times a message may be redelivered before it is treated as poison.
DEFAULT_MAX_ATTEMPTS = 3


#: Environment variables the queue reads, matching the convention the Redis
#: cache already uses (``HQ_REDIS__URL``). The broker is deployment topology
#: rather than analysis policy, so it lives here and not in `thresholds.yaml`.
_URL_ENV = "HQ_QUEUE__URL"
_REQUEST_QUEUE_ENV = "HQ_QUEUE__REQUEST_QUEUE"
_REPLY_QUEUE_ENV = "HQ_QUEUE__REPLY_QUEUE"


@dataclass(slots=True)
class QueueSettings:
    """Where the worker connects and what it listens to.

    Read from the environment, which is the whole point: the defaults below
    address ``localhost``, and a containerised worker that silently kept them
    would never reach the broker sitting beside it. The compose stack sets
    ``HQ_QUEUE__URL``; before this was wired the variable was accepted and
    discarded, because unknown ``HQ_``-prefixed settings are ignored rather
    than rejected.

    Attributes:
        url: AMQP connection string.
        request_queue: Queue carrying verification jobs.
        reply_queue: Queue results are published to.
        prefetch: How many messages one worker holds at a time. **One** by
            default, deliberately: a verification saturates the inference pool
            on its own, so prefetching more would only build a queue inside the
            worker where the broker cannot see it or redistribute it.
        max_attempts: Redeliveries before a message is treated as poison.
    """

    url: str = field(
        default_factory=lambda: os.environ.get(
            _URL_ENV, "amqp://guest:guest@localhost:5672/"
        )
    )
    request_queue: str = field(
        default_factory=lambda: os.environ.get(
            _REQUEST_QUEUE_ENV, "hamqadam.verification.requests"
        )
    )
    reply_queue: str = field(
        default_factory=lambda: os.environ.get(
            _REPLY_QUEUE_ENV, "hamqadam.verification.results"
        )
    )
    prefetch: int = 1
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


class VerificationConsumer:
    """Runs verifications from a queue.

    Args:
        pipeline: The verification pipeline.
        queue: Connection and queue names.
        settings: Service configuration.
    """

    __slots__ = ("_pipeline", "_queue", "_running", "_settings")

    def __init__(
        self, *, pipeline: Any, queue: QueueSettings, settings: Settings
    ) -> None:
        self._pipeline = pipeline
        self._queue = queue
        self._settings = settings
        self._running = False

    async def run(self) -> None:
        """Consume until cancelled.

        Raises:
            DependencyUnavailableError: when ``aio-pika`` is not installed.
                A hard failure rather than a silent no-op: a worker that starts
                and consumes nothing looks healthy while the queue grows.
        """
        try:
            import aio_pika
        except ImportError as exc:
            raise DependencyUnavailableError(
                "aio-pika",
                purpose="asynchronous verification from a message queue",
                extra="infra",
                cause=exc,
            ) from exc

        connection = await aio_pika.connect_robust(self._queue.url)
        self._running = True

        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=self._queue.prefetch)

            requests = await channel.declare_queue(
                self._queue.request_queue, durable=True
            )
            await channel.declare_queue(self._queue.reply_queue, durable=True)

            log.info(
                "worker.ready",
                queue=self._queue.request_queue,
                prefetch=self._queue.prefetch,
            )

            async with requests.iterator() as messages:
                async for message in messages:
                    await self._handle(message, channel)

    async def _handle(self, message: Any, channel: Any) -> None:
        """Process one message, acknowledging exactly once."""
        import aio_pika

        attempts = int(message.headers.get("x-attempts", 0)) + 1
        try:
            payload = json.loads(message.body)
        except (ValueError, TypeError) as exc:
            # Unparseable, and no amount of redelivery will change that.
            log.error("worker.malformed_message", reason=str(exc))
            await message.reject(requeue=False)
            return

        verification_id = str(payload.get("verification_id", "unknown"))
        context = RequestContext(
            request_id=str(message.message_id or verification_id),
            verification_id=verification_id,
            user_pseudonym=(
                pseudonymise(str(payload["user_reference"]))
                if payload.get("user_reference")
                else None
            ),
            attempt=attempts,
        )

        try:
            with request_context(context):
                result = await self._verify(payload)
                body = result.model_dump_json().encode("utf-8")
        except HamqadamError as exc:
            if attempts >= self._queue.max_attempts:
                log.error(
                    "worker.poison_message",
                    verification_id=verification_id,
                    attempts=attempts,
                    reason=str(exc),
                )
                await self._publish_failure(channel, verification_id, str(exc))
                await message.reject(requeue=False)
                return
            log.warning(
                "worker.retrying",
                verification_id=verification_id,
                attempt=attempts,
                reason=str(exc),
            )
            await message.nack(requeue=True)
            return
        except Exception as exc:  # noqa: BLE001 - one message must not kill the loop
            log.error(
                "worker.unexpected_error",
                verification_id=verification_id,
                reason=f"{type(exc).__name__}: {exc}",
            )
            await self._publish_failure(channel, verification_id, str(exc))
            await message.reject(requeue=False)
            return

        await channel.default_exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                correlation_id=message.correlation_id,
                headers={"verification_id": verification_id},
            ),
            routing_key=message.reply_to or self._queue.reply_queue,
        )
        # Acknowledged only after the result is published. The other order
        # loses the result if publishing fails, and the broker has no way to
        # know it should redeliver.
        await message.ack()

    async def _verify(self, payload: dict[str, Any]) -> Any:
        """Decode the images in a message and run the pipeline."""
        from hamqadam_ai.pipelines import VerificationImages

        def decode(field: str, role: str) -> Any:
            encoded = payload.get(field)
            if not encoded:
                return None
            return decode_image(base64.b64decode(encoded), role=role).pixels

        images = VerificationImages(
            live_selfie=decode("live_selfie", "live_selfie"),
            profile=decode("profile_image", "profile_image"),
            secondaries=[
                decode_image(
                    base64.b64decode(entry), role=f"secondary_image[{index}]"
                ).pixels
                for index, entry in enumerate(payload.get("secondary_images") or [])
            ],
            cnic=decode("cnic_image", "cnic_image"),
        )

        return await self._pipeline.verify(
            VerificationRequest(
                verification_id=str(payload.get("verification_id", "unknown")),
                user_reference=payload.get("user_reference"),
                enrol_on_success=bool(payload.get("enrol_on_success", False)),
            ),
            images,
        )

    async def _publish_failure(
        self, channel: Any, verification_id: str, reason: str
    ) -> None:
        """Tell the Backend a message will not be retried.

        A poison message that is simply dropped leaves the Backend waiting for
        a result that will never come, which is worse than a failure it can
        act on.
        """
        import aio_pika

        try:
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(
                        {
                            "verification_id": verification_id,
                            "status": "FAILED",
                            "reason": reason,
                        }
                    ).encode("utf-8"),
                    content_type="application/json",
                ),
                routing_key=self._queue.reply_queue,
            )
        except Exception as exc:  # noqa: BLE001 - already on the failure path
            log.error("worker.failure_publish_failed", reason=str(exc))

    @property
    def running(self) -> bool:
        """Whether the consume loop is active."""
        return self._running


def build_consumer(settings: Settings | None = None) -> VerificationConsumer:
    """Wire up a consumer and the pipeline beneath it."""
    from hamqadam_ai.pipelines import build_pipeline

    settings = settings or get_settings()
    return VerificationConsumer(
        pipeline=build_pipeline(settings),
        queue=QueueSettings(),
        settings=settings,
    )


def main() -> None:
    """Run a worker: ``python -m hamqadam_ai.workers``."""
    from hamqadam_ai.logging.setup import configure_logging

    settings = get_settings()
    configure_logging(settings)
    consumer = build_consumer(settings)

    try:
        asyncio.run(consumer.run())
    except KeyboardInterrupt:
        log.info("worker.stopped")


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "QueueSettings",
    "VerificationConsumer",
    "build_consumer",
    "main",
]
