"""Run the API with uvicorn: ``python -m hamqadam_ai.api``."""

from __future__ import annotations


def main() -> None:
    """Start the server using the configured host, port and worker count."""
    import uvicorn

    from hamqadam_ai.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "hamqadam_ai.api.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        timeout_keep_alive=30,
        access_log=False,  # structlog already logs every request with its id.
    )


if __name__ == "__main__":
    main()
