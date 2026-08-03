"""The process-wide registry's lifecycle.

`ModelRegistry.close()` marks an instance dead permanently and shuts down its
thread pool. It cannot clear the module-level global that points at it - it has
no reference to it - so anything closing the registry directly rather than
through `reset_registry` used to leave every later caller holding a corpse.

The symptom was maximally misleading: the next model load failed with "could
not be loaded ... run scripts/download_models.py", which points at a missing
weights file rather than at a lifecycle bug. It was found when an in-process
`TestClient` shutdown silently disabled every integration test that ran after
it, and it is reachable in production too - a second `create_app` in one
process would find nothing loadable.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.models.registry import (
    ModelRegistry,
    get_registry,
    reset_registry,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Never leak a registry between these tests or into the rest of the suite."""
    reset_registry()


class TestRegistrySingleton:
    """`get_registry` must always return something usable."""

    def test_repeated_calls_share_one_registry(self) -> None:
        """The point of the singleton: models load once per process."""
        assert get_registry() is get_registry()

    def test_a_closed_registry_is_replaced_not_returned(self) -> None:
        """The defect. Closing directly must not poison the accessor."""
        first = get_registry()
        first.close()
        assert first.closed is True

        second = get_registry()
        assert second is not first
        assert second.closed is False

    def test_reset_then_get_yields_a_fresh_registry(self) -> None:
        first = get_registry()
        reset_registry()
        assert first.closed is True
        assert get_registry() is not first

    def test_close_is_idempotent(self) -> None:
        """Shutdown runs from more than one path; the second must be harmless."""
        registry = get_registry()
        registry.close()
        registry.close()
        assert registry.closed is True

    def test_reset_is_safe_with_no_registry(self) -> None:
        reset_registry()
        reset_registry()

    def test_closed_is_false_on_a_new_registry(self) -> None:
        assert ModelRegistry().closed is False
