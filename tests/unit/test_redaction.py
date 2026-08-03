"""PII redaction — the guarantee that identity data cannot reach a log sink.

This is a security control, not a formatting nicety. Section 10 of the
development agreement makes CNIC images, names and verification data
confidential; section 21 requires access logging. Those two only coexist if the
log pipeline is physically incapable of emitting the confidential parts.

The processor runs last, after every other processor has finished adding
fields, so nothing can slip in behind it. These tests pin that behaviour
against the specific shapes a developer would plausibly log by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.core.config import RedactionConfig, RedactionPattern, get_settings
from hamqadam_ai.core.context import (
    RequestContext,
    fingerprint_api_key,
    pseudonymise,
)
from hamqadam_ai.logging.processors import (
    RedactionProcessor,
    add_context_fields,
    add_service_metadata,
    drop_color_message_key,
    rename_event_key,
)


@pytest.fixture
def processor() -> RedactionProcessor:
    """A processor built from the real production redaction configuration."""
    return RedactionProcessor(get_settings().logging.redaction)


def redact(processor: RedactionProcessor, **fields: object) -> dict:
    """Run one event through the processor and return the result."""
    return dict(processor(None, "info", dict(fields)))


# --------------------------------------------------------------------------- #
# Key dropping — bulk and wholly-sensitive values
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "image",
        "images",
        "image_bytes",
        "cnic_image",
        "profile_image",
        "live_selfie",
        "secondary_images",
        "embedding",
        "embeddings",
        "password",
        "api_key",
        "authorization",
        "secret",
        "token",
    ],
)
def test_configured_keys_are_dropped_entirely(
    processor: RedactionProcessor, key: str
) -> None:
    event = redact(processor, **{key: "the-actual-secret-value-12345"})
    assert "the-actual-secret-value" not in str(event[key])


@pytest.mark.unit
def test_key_matching_is_case_insensitive(processor: RedactionProcessor) -> None:
    event = redact(processor, Authorization="Bearer abc", API_KEY="k-123")
    assert "abc" not in str(event["Authorization"])
    assert "k-123" not in str(event["API_KEY"])


@pytest.mark.unit
def test_biometric_embedding_is_never_serialised(
    processor: RedactionProcessor,
) -> None:
    """A face embedding is biometric data. It must not appear in any form."""
    vector = np.random.default_rng(1).random(512).astype(np.float32)
    event = redact(processor, embedding=vector)
    rendered = str(event)
    assert "0." not in str(event["embedding"])
    assert str(float(vector[0]))[:8] not in rendered


@pytest.mark.unit
def test_an_unlisted_numpy_array_reports_shape_not_contents(
    processor: RedactionProcessor,
) -> None:
    """Even under a key nobody thought to list, array contents must not leak."""
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    event = redact(processor, some_unlisted_tensor=array)
    rendered = str(event["some_unlisted_tensor"])
    assert "ARRAY" in rendered
    assert "(2, 3, 4)" in rendered
    assert "17.0" not in rendered


@pytest.mark.unit
def test_raw_bytes_report_length_not_content(processor: RedactionProcessor) -> None:
    event = redact(processor, blob=b"\x89PNG\r\n\x1a\nsensitive-image-data")
    assert event["blob"] == "[BYTES:28]"


# --------------------------------------------------------------------------- #
# Key masking — correlatable but not disclosed
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cnic_number_is_masked_but_stays_correlatable(
    processor: RedactionProcessor,
) -> None:
    event = redact(processor, cnic_number="35202-1234567-1")
    masked = event["cnic_number"]
    assert "1234567" not in masked
    assert masked.startswith("35")
    # Two identical values must mask identically, so lines can be correlated.
    assert redact(processor, cnic_number="35202-1234567-1")["cnic_number"] == masked


@pytest.mark.unit
def test_two_different_cnics_mask_differently(
    processor: RedactionProcessor,
) -> None:
    first = redact(processor, cnic_number="35202-1234567-1")["cnic_number"]
    second = redact(processor, cnic_number="42101-7654321-9")["cnic_number"]
    assert first != second


@pytest.mark.unit
def test_short_values_are_fully_masked(processor: RedactionProcessor) -> None:
    """Keeping 2+2 characters of a 5-character name discloses most of it."""
    assert redact(processor, name="Ali")["name"] == "[REDACTED]"


@pytest.mark.unit
def test_full_name_is_masked(processor: RedactionProcessor) -> None:
    masked = redact(processor, full_name="Ubaid Malik")["full_name"]
    assert "baid Mali" not in masked


# --------------------------------------------------------------------------- #
# Pattern scrubbing — the net that catches what key rules cannot
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cnic_inside_free_text_is_scrubbed(processor: RedactionProcessor) -> None:
    """The OCR stage produces free text. Key-based rules cannot see inside it."""
    event = redact(
        processor,
        ocr_text="Name: Ubaid Malik  Number 35202-1234567-1  DOB 01-01-1990",
    )
    assert "35202-1234567-1" not in event["ocr_text"]
    assert "[CNIC]" in event["ocr_text"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw", ["35202-1234567-1", "3520212345671", "35202-12345671"]
)
def test_cnic_pattern_covers_hyphenated_and_bare_forms(
    processor: RedactionProcessor, raw: str
) -> None:
    event = redact(processor, note=f"the number is {raw} exactly")
    assert raw not in event["note"]


@pytest.mark.unit
def test_email_in_free_text_is_scrubbed(processor: RedactionProcessor) -> None:
    event = redact(processor, note="reach me at ubaid.malik@example.com please")
    assert "@example.com" not in event["note"]
    assert "[EMAIL]" in event["note"]


@pytest.mark.unit
def test_bearer_token_in_free_text_is_scrubbed(
    processor: RedactionProcessor,
) -> None:
    event = redact(
        processor, detail="request failed with Bearer eyJhbGciOiJIUzI1NiJ9.a.b"
    )
    assert "eyJhbGciOiJIUzI1NiJ9" not in event["detail"]


@pytest.mark.unit
def test_exception_messages_are_scrubbed(processor: RedactionProcessor) -> None:
    """A raised exception frequently quotes the offending value verbatim."""
    event = redact(
        processor,
        exception="ValueError: could not parse CNIC '35202-1234567-1' from field",
    )
    assert "35202-1234567-1" not in event["exception"]


# --------------------------------------------------------------------------- #
# Nesting, recursion and hostile shapes
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_nested_dictionaries_are_walked(processor: RedactionProcessor) -> None:
    event = redact(
        processor,
        details={"ocr": {"fields": {"cnic_number": "35202-1234567-1"}}},
    )
    leaf = event["details"]["ocr"]["fields"]["cnic_number"]
    assert "1234567" not in leaf


@pytest.mark.unit
def test_values_inside_lists_are_walked(processor: RedactionProcessor) -> None:
    event = redact(
        processor, candidates=[{"cnic_number": "35202-1234567-1"}, "plain string"]
    )
    assert "1234567" not in str(event["candidates"])


@pytest.mark.unit
def test_deeply_nested_structure_is_truncated_not_stack_overflowed() -> None:
    """A pathological payload must not take the process down from inside the
    logging path."""
    config = RedactionConfig(enabled=True, drop_keys=["secret"], mask_keys=[])
    processor = RedactionProcessor(config)

    nested: dict = {"leaf": "value"}
    for _ in range(200):
        nested = {"child": nested}

    result = processor(None, "info", {"payload": nested})
    assert "TRUNCATED" in str(result)


@pytest.mark.unit
def test_very_long_strings_are_truncated() -> None:
    """Guards against a base64 image accidentally logged as text."""
    config = RedactionConfig(enabled=True)
    processor = RedactionProcessor(config)
    event = processor(None, "info", {"blob": "A" * 10_000})
    assert len(event["blob"]) < 3_000
    assert "chars]" in event["blob"]


@pytest.mark.unit
def test_long_sequences_are_summarised() -> None:
    config = RedactionConfig(enabled=True)
    processor = RedactionProcessor(config)
    event = processor(None, "info", {"scores": list(range(500))})
    assert "more]" in str(event["scores"][-1])
    assert len(event["scores"]) < 100


@pytest.mark.unit
def test_scalars_pass_through_untouched(processor: RedactionProcessor) -> None:
    """Redaction must not corrupt the operational fields."""
    event = redact(
        processor, face_count=2, confidence=0.9731, passed=True, duration_ms=284.7
    )
    assert event["face_count"] == 2
    assert event["confidence"] == 0.9731
    assert event["passed"] is True
    assert event["duration_ms"] == 284.7


@pytest.mark.unit
def test_disabled_redaction_is_a_passthrough() -> None:
    processor = RedactionProcessor(RedactionConfig(enabled=False))
    event = processor(None, "info", {"cnic_number": "35202-1234567-1"})
    assert event["cnic_number"] == "35202-1234567-1"


@pytest.mark.unit
def test_custom_pattern_is_applied() -> None:
    config = RedactionConfig(
        enabled=True,
        patterns=[
            RedactionPattern(name="account", regex=r"ACC-\d{6}", replacement="[ACC]")
        ],
    )
    processor = RedactionProcessor(config)
    event = processor(None, "info", {"note": "see ACC-123456 for detail"})
    assert event["note"] == "see [ACC] for detail"


@pytest.mark.unit
def test_the_event_dict_is_mutated_in_place(processor: RedactionProcessor) -> None:
    """structlog requires the same mapping object back, not a copy."""
    event = {"cnic_number": "35202-1234567-1"}
    returned = processor(None, "info", event)
    assert returned is event


# --------------------------------------------------------------------------- #
# Pseudonymisation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_user_id_is_pseudonymised_not_logged_raw() -> None:
    context = RequestContext.create(user_id="user-42", verification_id="VER-1")
    fields = context.as_log_fields()
    assert "user-42" not in str(fields)
    assert fields["user"].startswith("u_")


@pytest.mark.unit
def test_pseudonym_is_stable_for_the_same_user() -> None:
    assert pseudonymise("user-42") == pseudonymise("user-42")


@pytest.mark.unit
def test_pseudonym_differs_between_users() -> None:
    assert pseudonymise("user-42") != pseudonymise("user-43")


@pytest.mark.unit
def test_api_key_is_fingerprinted_not_logged_raw() -> None:
    key = "sk-live-a-very-secret-api-key"
    context = RequestContext.create(api_key=key)
    assert key not in str(context.as_log_fields())
    assert context.api_key_id == fingerprint_api_key(key)
    assert len(context.api_key_id or "") == 8


@pytest.mark.unit
def test_verification_id_is_not_masked() -> None:
    """It is the Backend's own correlation key and carries no personal data."""
    context = RequestContext.create(verification_id="VER-123456")
    assert context.as_log_fields()["verification_id"] == "VER-123456"


# --------------------------------------------------------------------------- #
# The other processors in the chain
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_service_metadata_is_stamped() -> None:
    stamp = add_service_metadata("svc", "1.0.0", "production", "pod-7")
    event = stamp(None, "info", {})
    assert event == {
        "service": "svc",
        "version": "1.0.0",
        "env": "production",
        "instance": "pod-7",
    }


@pytest.mark.unit
def test_explicit_fields_win_over_service_metadata() -> None:
    stamp = add_service_metadata("svc", "1.0.0", "production", "pod-7")
    event = stamp(None, "info", {"service": "override"})
    assert event["service"] == "override"


@pytest.mark.unit
def test_context_fields_are_merged() -> None:
    from hamqadam_ai.core.context import request_context

    with request_context(RequestContext.create(verification_id="VER-9")):
        event = add_context_fields(None, "info", {})
    assert event["verification_id"] == "VER-9"


@pytest.mark.unit
def test_explicit_fields_win_over_context() -> None:
    from hamqadam_ai.core.context import request_context

    with request_context(RequestContext.create(verification_id="VER-9")):
        event = add_context_fields(None, "info", {"verification_id": "VER-OTHER"})
    assert event["verification_id"] == "VER-OTHER"


@pytest.mark.unit
def test_event_key_is_renamed_to_message() -> None:
    """Loki, Elasticsearch and CloudWatch all expect `message`."""
    event = rename_event_key(None, "info", {"event": "detection.completed"})
    assert event == {"message": "detection.completed"}


@pytest.mark.unit
def test_uvicorn_colour_duplicate_is_dropped() -> None:
    event = drop_color_message_key(None, "info", {"color_message": "\x1b[32mok"})
    assert "color_message" not in event
