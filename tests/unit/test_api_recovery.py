"""A startup that failed because a backend was down must heal itself.

The defect these cover: `build_pipeline` dials Qdrant once, at startup. If the
container is restarting at that moment the API stays up - deliberately, so
/ready can explain why - but nothing ever retries, so every later request
returns MODEL_NOT_LOADED even after Qdrant is healthy again. The only cure was
a restart, prompted by an error that names a model rather than a socket.

Recovery must not become a second way to run without a gallery, so the tests
below pin the failure path as tightly as the success path: when the rebuild
still fails, `recover_state` returns None and the caller's existing error is
raised unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from hamqadam_ai.api import app as app_module


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    """Reset the module globals around each test."""
    app_module._state.clear()
    app_module._recovery.clear()
    yield
    app_module._state.clear()
    app_module._recovery.clear()


def _failed_startup(error: str = "VectorStoreError: could not reach Qdrant") -> None:
    """Put the module into the state a Qdrant-less startup leaves behind."""
    app_module._state.update(
        {
            "settings": None,
            "pipeline": None,
            "limiters": None,
            "started_at": 0.0,
            "ready": False,
            "ready_error": error,
        }
    )


class TestRecoversWhenTheBackendReturns:
    """The point of the change."""

    def test_rebuilds_once_the_backend_is_reachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _failed_startup()
        monkeypatch.setattr(
            app_module,
            "_build_state",
            lambda _s: {"pipeline": "REBUILT", "ready": True, "ready_error": None},
        )

        assert app_module.recover_state() == "REBUILT"
        assert app_module._state["ready"] is True
        assert app_module._state["ready_error"] is None

    def test_healthy_state_is_returned_without_rebuilding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A working process must not pay for this on every request."""
        app_module._state.update({"pipeline": "ORIGINAL", "ready": True})

        def _explode(_s: Any) -> Any:
            raise AssertionError("must not rebuild a working pipeline")

        monkeypatch.setattr(app_module, "_build_state", _explode)

        assert app_module.recover_state() == "ORIGINAL"


class TestStillBrokenStaysBroken:
    """Recovery must not paper over a backend that is genuinely down."""

    def test_returns_none_and_records_the_fresh_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _failed_startup(error="stale reason from startup")

        def _still_down(_s: Any) -> Any:
            raise RuntimeError("could not reach Qdrant at http://localhost:6333")

        monkeypatch.setattr(app_module, "_build_state", _still_down)

        assert app_module.recover_state() is None
        assert app_module._state["pipeline"] is None
        assert app_module._state["ready"] is False
        # The *current* failure, not the one from startup - otherwise a
        # changed cause is reported with the original message forever.
        assert "RuntimeError" in app_module._state["ready_error"]
        assert "stale reason" not in app_module._state["ready_error"]

    def test_no_memory_fallback_is_introduced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failure yields None, never a partially built substitute."""
        _failed_startup()
        monkeypatch.setattr(
            app_module,
            "_build_state",
            lambda _s: (_ for _ in ()).throw(RuntimeError("down")),
        )

        assert app_module.recover_state() is None


class TestCooldown:
    """A down backend costs one reconnect per interval, not one per request."""

    def test_second_attempt_inside_the_window_does_not_rebuild(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _failed_startup()
        calls: list[int] = []

        def _count(_s: Any) -> Any:
            calls.append(1)
            raise RuntimeError("down")

        monkeypatch.setattr(app_module, "_build_state", _count)

        assert app_module.recover_state() is None
        assert app_module.recover_state() is None
        assert app_module.recover_state() is None
        assert len(calls) == 1, "cooldown did not throttle the retries"

    def test_attempt_is_allowed_once_the_window_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _failed_startup()
        calls: list[int] = []

        def _count(_s: Any) -> Any:
            calls.append(1)
            raise RuntimeError("down")

        monkeypatch.setattr(app_module, "_build_state", _count)
        assert app_module.recover_state() is None

        # Age the last attempt past the cooldown rather than sleeping.
        app_module._recovery["attempted_at"] -= app_module.RECOVERY_COOLDOWN_SECONDS + 1
        assert app_module.recover_state() is None
        assert len(calls) == 2

    def test_first_ever_attempt_is_never_throttled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ledger must not read as 'attempted just now'."""
        _failed_startup()
        calls: list[int] = []
        monkeypatch.setattr(
            app_module,
            "_build_state",
            lambda _s: (calls.append(1), {"pipeline": "OK", "ready": True})[1],
        )

        assert app_module.recover_state() == "OK"
        assert len(calls) == 1
