"""The admin routes must be off by default and impossible in production.

They enumerate the duplicate gallery, which is a register of who has been
verified. Useful while proving the store works; not something to leave
reachable once it does.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hamqadam_ai.core.config import AdminConfig, Settings
from hamqadam_ai.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestAdminGate:
    def test_disabled_by_default(self) -> None:
        assert AdminConfig().enabled is False

    def test_production_refuses_to_start_with_them_on(self) -> None:
        """A hard refusal, not a warning.

        Leaving these reachable in production would expose the reference of
        every enrolled person to anyone holding an API key - including keys
        issued for verification only.
        """
        with pytest.raises(ConfigurationError):
            Settings(
                app={"environment": "production", "debug": False},
                security={"api_keys": ["k"], "require_api_key": True},
                logging={"renderer": "json"},
                server={"cors": {"allow_origins": []}},
                admin={"enabled": True},
            )

    def test_production_accepts_them_off(self) -> None:
        settings = Settings(
            app={"environment": "production", "debug": False},
            security={"api_keys": ["k"], "require_api_key": True},
            logging={"renderer": "json"},
            server={"cors": {"allow_origins": []}},
            admin={"enabled": False},
        )
        assert settings.admin.enabled is False

    def test_the_list_limit_is_bounded(self) -> None:
        """A gallery can hold hundreds of thousands of records.

        An unbounded scan is both a memory problem and a bulk-disclosure one.
        """
        assert AdminConfig().max_list_limit <= 1000
        with pytest.raises(ValidationError):
            AdminConfig(max_list_limit=100_000)


class TestInspectionNeverReturnsVectors:
    """A 512-float template is biometric data.

    An inspection route that returns it is an exfiltration route with a
    debugging excuse.
    """

    def test_memory_store_omits_the_vector(self) -> None:
        import datetime as dt

        import numpy as np

        from hamqadam_ai.duplicate_detection.base import VectorRecord
        from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore

        store = InMemoryVectorStore(max_records=10)
        vector = np.ones(512, dtype=np.float32) / np.sqrt(512)
        store.enrol(
            VectorRecord(
                reference="acct-A",
                vector=vector,
                model_version="m1",
                enrolled_at=dt.datetime(2026, 8, 5, tzinfo=dt.UTC),
            )
        )

        for row in (store.list_references(), [store.get("acct-A")]):
            for record in row:
                assert record is not None
                assert "vector" not in record
                assert record["reference"] == "acct-A"

    def test_an_absent_reference_returns_none(self) -> None:
        from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore

        assert InMemoryVectorStore(max_records=10).get("nobody") is None
