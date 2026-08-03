"""Every `HQ_` variable the deployment sets must correspond to a real setting.

Why this test exists
--------------------
`Settings` is configured with `extra="ignore"`, so an unrecognised
`HQ_`-prefixed variable is discarded in silence. That is the right behaviour at
runtime - an unrelated `HQ_` variable in an operator's shell should not stop the
service booting - but it means a **typo in a deployment file is invisible**.

The deployment shipped with six such names, all wrong:

===============================  =====================================
`HQ_DUPLICATE__STORE__BACKEND`   `HQ_DUPLICATE__BACKEND`
`HQ_DUPLICATE__STORE__URL`       `HQ_DUPLICATE__QDRANT__URL`
`HQ_CACHE__BACKEND`              `HQ_EMBEDDING__CACHE__BACKEND`
`HQ_CACHE__URL`                  `HQ_REDIS__URL`
`HQ_QUEUE__URL`                  correct, but the worker never read it
`HQ_MODELS__CACHE_DIR`           `HQ_STORAGE__MODEL_DIR`
===============================  =====================================

Five passed every test and started cleanly: Qdrant and Redis containers running
and connected to by nothing, an in-memory gallery losing every enrolled template
on restart and shared by no replica, and a worker dialling `localhost`. Nothing
failed. It simply was not the system the compose file described.

The sixth was worse and is the reason this checks the Dockerfile too.
`HQ_MODELS__CACHE_DIR` was not ignored, because `models` *is* a real section - a
`dict[str, ModelSpec]` - so `cache_dir` was read as a model named `cache_dir`
and failed validation. Every container would have exited at startup. It sat in
the Dockerfile's ENV block, which no test read.

Reading the file and checking each name against the settings tree is the only
thing that catches that class of mistake, because the mistake is precisely that
nothing complains.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from hamqadam_ai.core.config import Settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "deploy" / "docker-compose.yml"
DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

#: Variables read directly from `os.environ` by an adapter rather than through
#: the settings tree. Each is asserted below to be genuinely referenced in the
#: source, so this list cannot rot into a way of silencing real typos.
DIRECT_ENV_VARS = {
    "HQ_REDIS__URL",
    "HQ_QUEUE__URL",
    "HQ_QUEUE__REQUEST_QUEUE",
    "HQ_QUEUE__REPLY_QUEUE",
    "HQ_CONFIG_DIR",
}


def _resolve(path: list[str], model: type[BaseModel]) -> bool:
    """Whether a `__`-delimited path names a real field on the settings tree."""
    head, *rest = path
    fields = model.model_fields
    match = next((name for name in fields if name.lower() == head.lower()), None)
    if match is None:
        return False
    if not rest:
        return True

    annotation: Any = fields[match].annotation

    # Dict-valued sections such as `models` accept arbitrary keys, so anything
    # below them resolves. Checked *before* unwrapping, because `dict[str,
    # ModelSpec]` exposes ModelSpec in `__args__` and descending into it would
    # test the caller's dictionary key against ModelSpec's field names.
    if getattr(annotation, "__origin__", None) is dict:
        return True

    # Unwrap Optional[...] / unions to find a model to descend into.
    for candidate in (annotation, *getattr(annotation, "__args__", ())):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return _resolve(rest, candidate)
    return False


def _compose_env_vars() -> dict[str, str]:
    """Every `HQ_`-prefixed variable the compose file sets."""
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    found: dict[str, str] = {}

    def harvest(block: Any) -> None:
        if isinstance(block, dict):
            for key, value in block.items():
                if isinstance(key, str) and key.startswith("HQ_"):
                    found[key] = str(value)
                else:
                    harvest(value)
        elif isinstance(block, list):
            for item in block:
                harvest(item)

    harvest(document)
    return found


def _dockerfile_env_vars() -> set[str]:
    """Every `HQ_` variable the Dockerfile sets in an ENV instruction."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    # ENV lines continue with a trailing backslash; join them before matching.
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    names: set[str] = set()
    for line in joined.splitlines():
        if line.strip().startswith("ENV "):
            names.update(re.findall(r"\b(HQ_[A-Z0-9_]+)=", line))
    return names


class TestComposeEnvironment:
    """The compose stack must configure the service it claims to."""

    def test_every_variable_names_a_real_setting(self) -> None:
        unknown = [
            name
            for name in _compose_env_vars()
            if name not in DIRECT_ENV_VARS
            and not _resolve(name[len("HQ_") :].split("__"), Settings)
        ]
        assert not unknown, (
            f"{len(unknown)} variable(s) in docker-compose.yml name nothing in "
            f"the settings tree and would be silently ignored: {unknown}"
        )

    def test_the_gallery_is_configured_for_a_shared_store(self) -> None:
        """The specific failure that hid behind the typo.

        An in-memory gallery in a multi-replica deployment loses every template
        on restart and is not shared between replicas, so duplicate detection
        quietly checks against almost nothing.
        """
        env = _compose_env_vars()
        assert env.get("HQ_DUPLICATE__BACKEND") == "qdrant"
        assert "qdrant" in env.get("HQ_DUPLICATE__QDRANT__URL", "")

    def test_the_worker_is_pointed_at_the_broker(self) -> None:
        """Not at localhost, which is where the default sends it."""
        url = _compose_env_vars().get("HQ_QUEUE__URL", "")
        assert url.startswith("amqp://")
        assert "localhost" not in url

    def test_production_hardening_is_not_switched_off(self) -> None:
        env = _compose_env_vars()
        assert env.get("HQ_APP__ENVIRONMENT") == "production"
        assert env.get("HQ_SECURITY__REQUIRE_API_KEY", "true").lower() == "true"


class TestDockerfileEnvironment:
    """The image's own defaults must also be real."""

    def test_every_variable_names_a_real_setting(self) -> None:
        unknown = [
            name
            for name in _dockerfile_env_vars()
            if name not in DIRECT_ENV_VARS
            and not _resolve(name[len("HQ_") :].split("__"), Settings)
        ]
        assert not unknown, f"unknown HQ_ variables in the Dockerfile: {unknown}"


class TestDirectEnvVars:
    """The allow-list must stay honest.

    Every exemption has to be a variable something genuinely reads from
    `os.environ`. Without this, the list becomes a place to hide typos.
    """

    def test_each_exemption_is_actually_read_somewhere(self) -> None:
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "src").rglob("*.py")
        )
        unused = [name for name in DIRECT_ENV_VARS if name not in sources]
        assert not unused, (
            f"exempted from the settings-tree check but read by nothing: "
            f"{unused}"
        )


class TestBuildContext:
    """Everything the Dockerfile copies has to exist.

    `COPY` fails the build on a missing path, and the image is not built by the
    test suite - so a file referenced but absent is discovered by whoever runs
    the deploy. `README.md` was exactly that: referenced twice, and not present.
    """

    def test_every_copied_path_exists(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        missing: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("COPY") or "--from=" in stripped:
                continue
            parts = stripped.split()[1:]
            for source in parts[:-1]:  # the last token is the destination
                if source.startswith("--"):
                    continue
                if not (REPO_ROOT / source.rstrip("/")).exists():
                    missing.append(source)
        assert not missing, f"Dockerfile copies paths that do not exist: {missing}"


class TestEnvExample:
    """The committed environment template must configure real settings.

    `.env.example` is what an operator copies to `.env` and edits, so a wrong
    name there propagates into every deployment that starts from it - silently,
    because an unrecognised HQ_ variable is discarded rather than rejected.

    Seven were wrong when this test was written: HQ_QDRANT__* (the real prefix
    is HQ_DUPLICATE__QDRANT__*), HQ_RABBITMQ__* (HQ_QUEUE__*), and
    HQ_REDIS__PASSWORD, which does not exist at all - Redis credentials belong
    inside the URL. Someone following the template would have configured a
    Qdrant gallery and a RabbitMQ worker that were never actually switched on.
    """

    @staticmethod
    def _declared() -> list[str]:
        names: list[str] = []
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HQ_") and "=" in line:
                names.append(line.split("=", 1)[0].strip())
        return names

    def test_every_variable_names_a_real_setting(self) -> None:
        unknown = [
            name
            for name in self._declared()
            if name not in DIRECT_ENV_VARS
            and not _resolve(name[len("HQ_") :].split("__"), Settings)
        ]
        assert not unknown, (
            f"{len(unknown)} variable(s) in .env.example name nothing in the "
            f"settings tree and would be silently ignored: {unknown}"
        )

    def test_it_contains_no_real_secret(self) -> None:
        """A committed template must hold placeholders, never a usable key."""
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "API_KEYS=" in line and not line.strip().startswith("#"):
                value = line.split("=", 1)[1].strip()
                assert not value or "replace" in value.lower(), (
                    f"a real-looking API key is committed in .env.example: {value}"
                )
