"""Configuration layering, validation and cross-section invariants.

Configuration is the mechanism by which every accept/reject threshold in the
service is controlled, so a silent mis-load is a silent change to the
verification decision. These tests pin the layering order, the normalisation
rules and the production-hardening gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hamqadam_ai.core.config import (
    DetectionConfig,
    MatchThresholds,
    OcclusionConfig,
    PoseConfig,
    SecurityConfig,
    Settings,
    VisibilityConfig,
    _deep_merge,
    resolve_config_dir,
)
from hamqadam_ai.core.exceptions import ConfigurationError

# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_deep_merge_recurses_into_mappings() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overlay = {"a": {"c": 99, "e": 4}}
    assert _deep_merge(base, overlay) == {"a": {"b": 1, "c": 99, "e": 4}, "d": 3}


@pytest.mark.unit
def test_deep_merge_replaces_lists_wholesale() -> None:
    """An operator overriding a chain means 'use exactly this chain'."""
    base = {"chain": ["scrfd", "yolo", "haar"]}
    overlay = {"chain": ["haar"]}
    assert _deep_merge(base, overlay)["chain"] == ["haar"]


@pytest.mark.unit
def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"a": {"b": 1}}
    overlay = {"a": {"b": 2}}
    _deep_merge(base, overlay)
    assert base == {"a": {"b": 1}}


@pytest.mark.unit
def test_real_config_directory_is_discoverable() -> None:
    config_dir = resolve_config_dir()
    assert config_dir.is_dir()
    assert (config_dir / "app.yaml").is_file()
    assert (config_dir / "thresholds.yaml").is_file()


@pytest.mark.unit
def test_settings_load_every_yaml_section(settings: Settings) -> None:
    """All four base files must be merged, not just the first one found."""
    assert settings.app.name == "hamqadam-ai-verification"  # app.yaml
    assert settings.logging.redaction.enabled is True  # logging.yaml
    assert "face_detector_scrfd" in settings.models  # models.yaml
    assert settings.detection.policy.max_faces_allowed == 1  # thresholds.yaml


@pytest.mark.unit
def test_digest_lock_overrides_the_null_in_models_yaml(settings: Settings) -> None:
    """The generated lock file must win over the placeholder declaration."""
    config_dir = resolve_config_dir()
    lock_path = config_dir / "model_digests.lock.yaml"
    if not lock_path.is_file():
        pytest.skip("digest lock not generated in this checkout")

    locked = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["models"]
    for key, entry in locked.items():
        if entry.get("sha256") and key in settings.models:
            assert settings.models[key].sha256 == entry["sha256"], (
                f"{key} did not pick up its pinned digest"
            )


@pytest.mark.unit
def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HQ_SERVER__PORT", "9911")
    monkeypatch.setenv("HQ_DETECTION__SCORE_THRESHOLD", "0.77")
    fresh = Settings()
    assert fresh.server.port == 9911
    assert fresh.detection.score_threshold == pytest.approx(0.77)


@pytest.mark.unit
def test_init_kwargs_outrank_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HQ_SERVER__PORT", "9911")
    fresh = Settings(server={"port": 8123})
    assert fresh.server.port == 8123


# --------------------------------------------------------------------------- #
# Coercion and normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_api_keys_accept_a_comma_separated_string() -> None:
    """`HQ_SECURITY__API_KEYS=a,b,c` must become a list, not a single key."""
    config = SecurityConfig(api_keys="alpha, beta ,gamma")  # type: ignore[arg-type]
    assert config.api_keys == ["alpha", "beta", "gamma"]


@pytest.mark.unit
def test_api_keys_ignore_empty_entries() -> None:
    config = SecurityConfig(api_keys="alpha,,  ,beta")  # type: ignore[arg-type]
    assert config.api_keys == ["alpha", "beta"]


@pytest.mark.unit
def test_visibility_weights_are_normalised_to_one() -> None:
    """An operator raising one weight must not have to rebalance the rest."""
    config = VisibilityConfig(
        weights={
            "detector_confidence": 2.0,
            "occlusion": 3.0,
            "pose": 2.0,
            "face_size": 2.0,
            "framing": 1.0,
        }
    )
    assert sum(config.weights.values()) == pytest.approx(1.0)
    # Relative ordering must survive normalisation.
    assert config.weights["occlusion"] > config.weights["detector_confidence"]


@pytest.mark.unit
def test_visibility_weights_reject_missing_components() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        VisibilityConfig(weights={"pose": 1.0})


@pytest.mark.unit
def test_occlusion_region_weights_are_normalised() -> None:
    config = OcclusionConfig(region_weights={"left_eye": 3.0, "right_eye": 1.0})
    assert sum(config.region_weights.values()) == pytest.approx(1.0)
    assert config.region_weights["left_eye"] == pytest.approx(0.75)


@pytest.mark.unit
def test_primary_detector_is_forced_into_the_chain() -> None:
    config = DetectionConfig(primary="haar", fallback_chain=["scrfd"])
    assert config.fallback_chain[0] == "haar"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_detection_input_size_must_tile_the_largest_stride() -> None:
    """SCRFD's coarsest FPN level has stride 32."""
    with pytest.raises(ValueError, match="multiple of 32"):
        DetectionConfig(detection_input_size=(640, 500))


@pytest.mark.unit
def test_soft_pose_limit_cannot_exceed_the_hard_limit() -> None:
    with pytest.raises(ValueError, match="max_yaw"):
        PoseConfig(max_yaw=60.0, hard_max_yaw=50.0)


@pytest.mark.unit
def test_match_review_threshold_must_be_below_strong_match() -> None:
    with pytest.raises(ValueError, match="review threshold"):
        MatchThresholds(strong_match=0.4, review=0.6)


@pytest.mark.unit
def test_area_ratio_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="min_face_area_ratio"):
        DetectionConfig(policy={"min_face_area_ratio": 0.9, "max_face_area_ratio": 0.5})


@pytest.mark.unit
def test_unknown_detector_in_chain_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown detectors"):
        Settings(detection={"primary": "scrfd", "fallback_chain": ["scrfd", "magic"]})


@pytest.mark.unit
def test_malformed_yaml_raises_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.yaml").write_text("app:\n  name: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("HQ_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        Settings()


@pytest.mark.unit
def test_non_mapping_yaml_raises_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    monkeypatch.setenv("HQ_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ConfigurationError, match="mapping at the top level"):
        Settings()


@pytest.mark.unit
def test_missing_config_dir_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HQ_CONFIG_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(ConfigurationError, match="not a directory"):
        Settings()


# --------------------------------------------------------------------------- #
# Production hardening
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_production_refuses_wildcard_cors() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(
            app={"environment": "production"},
            security={"api_keys": ["k"], "require_api_key": True},
            server={"cors": {"allow_origins": ["*"]}},
        )
    assert any("cors" in problem for problem in excinfo.value.details["problems"])


@pytest.mark.unit
def test_production_refuses_empty_api_keys() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(
            app={"environment": "production"},
            security={"api_keys": [], "require_api_key": True},
        )
    assert any("api_keys" in problem for problem in excinfo.value.details["problems"])


@pytest.mark.unit
def test_production_refuses_disabled_redaction() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(
            app={"environment": "production"},
            security={"api_keys": ["k"]},
            server={"cors": {"allow_origins": ["https://backend"]}},
            logging={"redaction": {"enabled": False}},
        )
    assert any("redaction" in problem for problem in excinfo.value.details["problems"])


@pytest.mark.unit
def test_production_refuses_unverified_model_checksums() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings(
            app={"environment": "production"},
            security={"api_keys": ["k"]},
            server={"cors": {"allow_origins": ["https://backend"]}},
            model_store={"verify_checksum": False},
        )
    assert any(
        "verify_checksum" in problem for problem in excinfo.value.details["problems"]
    )


@pytest.mark.unit
def test_hardened_production_config_is_accepted() -> None:
    config = Settings(
        app={"environment": "production", "debug": False},
        security={"api_keys": ["a-real-key"], "require_api_key": True},
        server={"cors": {"allow_origins": ["https://backend.hamqadam"]}},
        logging={"renderer": "json", "redaction": {"enabled": True}},
        model_store={"verify_checksum": True},
    )
    assert config.app.is_production


@pytest.mark.unit
def test_hardening_can_be_disabled_for_a_sandbox() -> None:
    config = Settings(
        app={"environment": "production", "debug": True},
        security={"api_keys": [], "enforce_production_hardening": False},
    )
    assert config.app.debug is True


@pytest.mark.unit
def test_development_is_not_subject_to_hardening() -> None:
    config = Settings(app={"environment": "development", "debug": True})
    assert config.app.debug is True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_spec_lookup_reports_available_keys(settings: Settings) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        settings.model_spec("no_such_model")
    assert "face_detector_scrfd" in excinfo.value.details["available"]


@pytest.mark.unit
def test_version_map_covers_enabled_models_only(settings: Settings) -> None:
    versions = settings.model_version_map()
    assert "face_detector_scrfd" in versions
    # The occlusion classifier is disabled by default and ships no weights.
    assert "face_occlusion_classifier" not in versions


@pytest.mark.unit
def test_providers_for_unknown_device_falls_back_to_cpu(settings: Settings) -> None:
    providers = settings.providers_for("quantum")
    assert [spec.name for spec in providers] == ["CPUExecutionProvider"]


@pytest.mark.unit
def test_model_paths_resolve_under_the_store(settings: Settings) -> None:
    store = settings.storage.resolved_model_dir.resolve()
    assert store.is_absolute()


@pytest.mark.unit
def test_hmac_secret_is_required_when_hmac_is_enabled() -> None:
    with pytest.raises(ValueError, match="hmac.secret"):
        SecurityConfig(hmac={"enabled": True, "secret": None})


# --------------------------------------------------------------------------- #
# MODULE 10 - API keys from the environment
# --------------------------------------------------------------------------- #


class TestApiKeysFromEnvironment:
    """`HQ_SECURITY__API_KEYS` must accept what an operator will actually type.

    The field is annotated `NoDecode` so pydantic-settings hands the raw string
    to the validator. Without it, `list[str]` is treated as a complex field and
    JSON-parsed inside the *source*, so a plainly-written key failed with
    `SettingsError: error parsing value for field "security"` - an error naming
    the whole section, not the field, and saying nothing about JSON.

    Environment variables are strings and an operator setting one key will write
    it plainly, which puts that trap squarely in the deployment path.
    """

    @staticmethod
    def _keys(raw: str) -> list[str]:
        return SecurityConfig(api_keys=raw).api_keys  # type: ignore[arg-type]

    def test_a_single_bare_key_is_accepted(self) -> None:
        assert self._keys("my-secret-key") == ["my-secret-key"]

    def test_a_comma_separated_list_is_split_and_trimmed(self) -> None:
        assert self._keys("key-a, key-b ,key-c") == ["key-a", "key-b", "key-c"]

    def test_a_json_array_still_works(self) -> None:
        """Anyone already passing JSON must not be broken by the above."""
        assert self._keys('["j1", "j2"]') == ["j1", "j2"]

    def test_malformed_json_is_refused_rather_than_silently_split(self) -> None:
        """`["broken` must not become the single key `["broken`.

        A truncated array is a quoting mistake, and turning it into a literal
        key would leave the deployment authenticating against a string nobody
        intended - working, and wrong.
        """
        with pytest.raises(ValidationError, match="not valid JSON"):
            self._keys('["broken')

    def test_empty_entries_are_dropped(self) -> None:
        """A trailing comma must not create an empty key.

        An empty string in the list would be compared against every request's
        header, and a caller sending an empty key would authenticate.
        """
        assert self._keys("key-a,,key-b,") == ["key-a", "key-b"]

    def test_an_actual_list_passes_through(self) -> None:
        """YAML and Python callers already supply a list."""
        assert SecurityConfig(api_keys=["a", "b"]).api_keys == ["a", "b"]
