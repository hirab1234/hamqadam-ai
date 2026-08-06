"""Every engine the service is configured to use must be an installable dependency.

The bug this pins
-----------------
`ocr.engine_chain` starts with `onnx_ppocr`, which is provided by
`rapidocr-onnxruntime`. That package appeared in **no requirements file**. It
happened to be installed in the development virtualenv, so the host used it and
every OCR figure ever measured came from it - while the container, built from
`requirements/ml.txt`, silently fell through to the third engine in the chain.

The logs said so plainly and nobody was reading them::

    {"engine": "easyocr", "requested": "onnx_ppocr", "overridden": false}

Two different recognisers with different accuracy, one measured and one
deployed. Nothing failed, because the fallback chain did exactly what it was
designed to do - which is what made it invisible.

The chain is a resilience feature, not a substitute for declaring what you
depend on. Falling back is right when an engine breaks at runtime; falling back
on every request because the primary was never installed is a different thing
wearing the same clothes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hamqadam_ai.core.config import get_settings

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements"

#: Which distribution provides each OCR engine. Mirrors the `extra=` argument
#: each adapter passes to DependencyUnavailableError, which is the string a user
#: is told to `pip install`.
ENGINE_PACKAGES = {
    "onnx_ppocr": "rapidocr-onnxruntime",
    "paddleocr": "paddleocr",
    "easyocr": "easyocr",
}


def _declared_packages() -> set[str]:
    """Every distribution named across the requirements files."""
    names: set[str] = set()
    for path in REQUIREMENTS.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r", "--")):
                continue
            name = re.split(r"[=<>~!\[; ]", line, maxsplit=1)[0].strip().lower()
            if name:
                names.add(name)
    return names


class TestOcrEngineChainIsInstallable:
    """The configured chain cannot reference packages nobody installs."""

    def test_the_primary_engine_is_a_declared_dependency(self) -> None:
        """The one that was wrong.

        The primary is the engine whose accuracy the thresholds were tuned
        against. If it is missing at runtime the service still answers - just
        with a different recogniser - so this has to be caught here rather than
        by anything at startup.
        """
        chain = get_settings().ocr.engine_chain
        assert chain, "the OCR engine chain is empty"

        primary = chain[0]
        package = ENGINE_PACKAGES.get(primary)
        assert package is not None, (
            f"engine {primary!r} is configured but this test does not know "
            f"which package provides it; add it to ENGINE_PACKAGES"
        )
        assert package.lower() in _declared_packages(), (
            f"the primary OCR engine {primary!r} needs {package!r}, which is "
            f"in no requirements file. The container will silently fall through "
            f"to a different engine with different accuracy."
        )

    @pytest.mark.parametrize("engine", ["onnx_ppocr", "paddleocr", "easyocr"])
    def test_every_engine_in_the_chain_is_declared(self, engine: str) -> None:
        """A fallback nobody installed is not a fallback."""
        if engine not in get_settings().ocr.engine_chain:
            pytest.skip(f"{engine} is not in the configured chain")
        package = ENGINE_PACKAGES[engine].lower()
        assert package in _declared_packages(), (
            f"{engine!r} is in the fallback chain but {package!r} is not "
            f"declared anywhere, so that rung of the chain does not exist"
        )

    def test_the_mapping_covers_the_whole_chain(self) -> None:
        """Guards the guard.

        A new engine added to the chain without an entry here would make the
        tests above skip silently, which is how this class of bug returns.
        """
        unknown = [
            engine
            for engine in get_settings().ocr.engine_chain
            if engine not in ENGINE_PACKAGES
        ]
        assert not unknown, f"engines with no known package: {unknown}"
