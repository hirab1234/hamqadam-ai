"""Fetch, verify and pin every model artefact declared in ``configs/models.yaml``.

Usage
-----
::

    python scripts/download_models.py                  # fetch what is missing
    python scripts/download_models.py --force          # re-fetch everything
    python scripts/download_models.py --only face_detector_scrfd
    python scripts/download_models.py --verify-only    # check existing files
    python scripts/download_models.py --write-lock     # (re)write the digest lock

Integrity model
---------------
Upstream does not sign these releases, so the honest workflow is
trust-on-first-download followed by pinning:

1. This script downloads each artefact over HTTPS and computes its SHA-256.
2. Digests are written to ``configs/model_digests.lock.yaml``.
3. That file is committed and reviewed. Every subsequent load - by this script
   and by the service at start-up - is verified against it, and a mismatch is a
   hard refusal, not a warning.

Re-running after the lock file exists verifies rather than overwrites. Use
``--write-lock`` to deliberately re-pin after an intentional model upgrade.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

# Allow `python scripts/download_models.py` from a source checkout without an
# editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hamqadam_ai.core.config import (  # noqa: E402
    ModelSpec,
    get_settings,
    resolve_config_dir,
)
from hamqadam_ai.utils.hashing import sha256_file  # noqa: E402

LOCK_FILENAME = "model_digests.lock.yaml"

_CHUNK = 1024 * 512


@dataclass(slots=True)
class Outcome:
    """Result of processing one registry key."""

    key: str
    action: str  # downloaded | present | verified | failed | skipped | bundled
    path: Path | None = None
    digest: str | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        """Whether this key ended in a usable state."""
        return self.action not in {"failed"}


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def download(url: str, destination: Path, *, timeout: float, label: str) -> None:
    """Stream ``url`` to ``destination`` atomically, with a progress readout.

    Writes to a sibling ``.part`` file and renames on success, so an
    interrupted download can never leave a truncated artefact that would then
    fail an opaque ONNX parse error at start-up.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    with httpx.stream(
        "GET", url, timeout=timeout, follow_redirects=True
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK):
                handle.write(chunk)
                written += len(chunk)
                _progress(label, written, total)
    print()

    partial.replace(destination)


def _progress(label: str, written: int, total: int) -> None:
    """Render a single-line progress indicator to stderr.

    ``content-length`` is advisory: with transfer encoding or content encoding
    in play the declared size can be smaller than the decoded stream, so the
    fraction is clamped rather than allowed to run past 100%.
    """
    if total > 0:
        fraction = min(1.0, written / total)
        pct = fraction * 100.0
        bar_width = 28
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stderr.write(
            f"\r  {label:34s} [{bar}] {pct:5.1f}%  {written / 1e6:7.1f} MB"
        )
    else:
        sys.stderr.write(f"\r  {label:34s} {written / 1e6:7.1f} MB")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Source handlers
# --------------------------------------------------------------------------- #


def fetch_insightface_pack(
    spec: ModelSpec, store_root: Path, pack_url: str, *, timeout: float, force: bool
) -> Path:
    """Extract one member from an InsightFace model-zoo zip.

    The zip is downloaded once into the store and reused for every member, so
    fetching both ``det_10g.onnx`` and ``w600k_r50.onnx`` from ``buffalo_l``
    costs a single 300 MB transfer rather than two.
    """
    pack = spec.source.pack or "buffalo_l"
    member = spec.source.member
    if not member:
        raise ValueError(f"insightface_pack source for {spec.version!r} has no member")

    target = store_root / spec.path
    if target.is_file() and not force:
        return target

    archive = store_root / ".packs" / f"{pack}.zip"
    if not archive.is_file() or force:
        url = pack_url if pack in pack_url else pack_url.replace("buffalo_l", pack)
        download(url, archive, timeout=timeout, label=f"{pack}.zip")

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        candidates = [
            name
            for name in zf.namelist()
            if Path(name).name == member and not name.endswith("/")
        ]
        if not candidates:
            available = sorted({Path(n).name for n in zf.namelist() if n.endswith(".onnx")})
            raise FileNotFoundError(
                f"{member!r} not found in {pack}.zip. Available: {available}"
            )
        with zf.open(candidates[0]) as source, tempfile.NamedTemporaryFile(
            delete=False, dir=target.parent
        ) as sink:
            shutil.copyfileobj(source, sink)
            temp_path = Path(sink.name)
    temp_path.replace(target)
    return target


def fetch_url(spec: ModelSpec, store_root: Path, *, timeout: float, force: bool) -> Path:
    """Download a single-file artefact."""
    if not spec.source.url:
        raise ValueError("url source has no `url` field")
    target = store_root / spec.path
    if target.is_file() and not force:
        return target
    download(spec.source.url, target, timeout=timeout, label=Path(spec.path).name)
    return target


def fetch_multi_url(
    spec: ModelSpec, store_root: Path, *, timeout: float, force: bool
) -> Path:
    """Download several files (weights plus a config, typically Caffe)."""
    primary: Path | None = None
    for entry in spec.source.urls:
        destination = store_root / entry["dest"]
        if not destination.is_file() or force:
            download(entry["url"], destination, timeout=timeout, label=destination.name)
        if primary is None:
            primary = destination
    if primary is None:
        raise ValueError("multi_url source declares no urls")
    return store_root / spec.path


def locate_bundled(spec: ModelSpec) -> Path:
    """Resolve an artefact that ships inside the package itself.

    The Haar cascades are vendored at ``hamqadam_ai/detectors/cascades/``
    rather than taken from ``cv2.data``, because ``opencv-python-headless``
    ships an empty data directory. They need no download, which is precisely
    why they are the terminal fallback: the service can always detect
    *something*, even on a host with no network access at all.
    """
    from hamqadam_ai.detectors.cascades import cascade_path

    candidate = cascade_path(spec.path)
    if candidate is None:
        raise FileNotFoundError(
            f"Vendored cascade {spec.path!r} is missing from the package. "
            "A packaging step has stripped hamqadam_ai/detectors/cascades/*.xml."
        )
    return candidate


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def process(
    key: str,
    spec: ModelSpec,
    store_root: Path,
    pack_url: str,
    *,
    timeout: float,
    force: bool,
    verify_only: bool,
    pinned: dict[str, str],
) -> Outcome:
    """Fetch (or verify) one artefact and return the outcome."""
    source_type = spec.source.type

    if source_type == "manual":
        target = store_root / spec.path
        if target.is_file():
            return Outcome(key, "present", target, sha256_file(target))
        return Outcome(
            key,
            "skipped",
            None,
            None,
            "manual source: place the file yourself, or leave disabled",
        )

    if source_type == "bundled":
        try:
            path = locate_bundled(spec)
        except (FileNotFoundError, ImportError) as exc:
            return Outcome(key, "failed", None, None, str(exc))
        return Outcome(key, "bundled", path, sha256_file(path))

    target = store_root / spec.path

    if verify_only:
        if not target.is_file():
            return Outcome(key, "failed", target, None, "file missing")
        return _verify(key, target, pinned)

    try:
        if source_type == "insightface_pack":
            path = fetch_insightface_pack(
                spec, store_root, pack_url, timeout=timeout, force=force
            )
        elif source_type == "url":
            path = fetch_url(spec, store_root, timeout=timeout, force=force)
        elif source_type == "multi_url":
            path = fetch_multi_url(spec, store_root, timeout=timeout, force=force)
        else:
            return Outcome(key, "failed", None, None, f"unknown source type {source_type!r}")
    except (httpx.HTTPError, OSError, ValueError, FileNotFoundError, zipfile.BadZipFile) as exc:
        return Outcome(key, "failed", target, None, f"{type(exc).__name__}: {exc}")

    if not path.is_file():
        return Outcome(key, "failed", path, None, "download reported success but no file exists")

    return _verify(key, path, pinned, downloaded=True)


def _verify(
    key: str, path: Path, pinned: dict[str, str], *, downloaded: bool = False
) -> Outcome:
    """Compute the digest and compare it against the lock file if pinned."""
    digest = sha256_file(path)
    expected = pinned.get(key)
    if expected and digest.lower() != expected.lower():
        return Outcome(
            key,
            "failed",
            path,
            digest,
            f"DIGEST MISMATCH - locked {expected[:16]}..., got {digest[:16]}...",
        )
    action = "downloaded" if downloaded else ("verified" if expected else "present")
    return Outcome(key, action, path, digest)


def load_lock(config_dir: Path) -> dict[str, str]:
    """Read the pinned digests from the lock file, if it exists."""
    lock_path = config_dir / LOCK_FILENAME
    if not lock_path.is_file():
        return {}
    with lock_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    models = data.get("models") or {}
    return {
        key: entry["sha256"]
        for key, entry in models.items()
        if isinstance(entry, dict) and entry.get("sha256")
    }


def write_lock(config_dir: Path, digests: dict[str, str], versions: dict[str, str]) -> Path:
    """Write the digest lock file in a shape that deep-merges into ``models``."""
    lock_path = config_dir / LOCK_FILENAME
    lines = [
        "# ======================================================================= #",
        "# GENERATED by scripts/download_models.py - review before committing.",
        "#",
        "# Pinned SHA-256 digests of every model artefact. This file is merged over",
        "# models.yaml at start-up, so these values become the integrity contract",
        "# the runtime enforces. A mismatch at load time is a hard refusal.",
        "#",
        "# Re-generate deliberately with:  python scripts/download_models.py --write-lock",
        "# ======================================================================= #",
        "",
        "models:",
    ]
    for key in sorted(digests):
        lines.append(f"  {key}:")
        lines.append(f"    # version: {versions.get(key, 'unknown')}")
        lines.append(f'    sha256: "{digests[key]}"')
    lines.append("")
    lock_path.write_text("\n".join(lines), encoding="utf-8")
    return lock_path


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Download, verify and pin Hamqadam AI model artefacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", nargs="+", metavar="KEY", help="Limit to these keys.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify what is on disk; download nothing."
    )
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="Write configs/model_digests.lock.yaml from the computed digests.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also process models marked `enabled: false`.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    config_dir = resolve_config_dir()
    store_root = settings.storage.resolved_model_dir
    store_root.mkdir(parents=True, exist_ok=True)

    pinned = load_lock(config_dir)

    selected = {
        key: spec
        for key, spec in settings.models.items()
        if (args.only is None or key in args.only)
        and (spec.enabled or args.include_disabled)
    }
    if args.only:
        unknown = set(args.only) - set(settings.models)
        if unknown:
            parser.error(f"Unknown model keys: {sorted(unknown)}")

    print("=" * 78)
    print("HAMQADAM AI - MODEL ARTEFACT FETCH")
    print("=" * 78)
    print(f"  store       : {store_root}")
    print(f"  lock file   : {config_dir / LOCK_FILENAME} "
          f"({'present' if pinned else 'absent'})")
    print(f"  models      : {len(selected)} selected")
    print(f"  mode        : {'verify-only' if args.verify_only else 'fetch'}"
          f"{' (force)' if args.force else ''}")
    print()

    outcomes: list[Outcome] = []
    for key, spec in selected.items():
        print(f"- {key}  [{spec.version}]")
        outcome = process(
            key,
            spec,
            store_root,
            settings.model_store.insightface_pack_url,
            timeout=float(settings.model_store.download_timeout_seconds),
            force=args.force,
            verify_only=args.verify_only,
            pinned=pinned,
        )
        outcomes.append(outcome)
        symbol = {
            "downloaded": "OK   downloaded",
            "present": "OK   already present",
            "verified": "OK   digest verified",
            "bundled": "OK   bundled with opencv",
            "skipped": "--   skipped",
            "failed": "FAIL",
        }[outcome.action]
        detail = f" {outcome.message}" if outcome.message else ""
        digest = f" sha256={outcome.digest[:16]}..." if outcome.digest else ""
        print(f"    {symbol}{digest}{detail}")
        print()

    digests = {o.key: o.digest for o in outcomes if o.digest}
    if args.write_lock and digests:
        versions = {key: spec.version for key, spec in selected.items()}
        path = write_lock(config_dir, digests, versions)
        print(f"Wrote {len(digests)} digests to {path}")
        print("Review and commit this file - it is the integrity contract.")
        print()
    elif digests and not pinned:
        print("No lock file exists yet. Re-run with --write-lock to pin these digests:")
        for key in sorted(digests):
            print(f"    {key:32s} {digests[key]}")
        print()

    print("=" * 78)
    failures = [o for o in outcomes if not o.ok]
    required_failures = [
        o for o in failures if settings.models[o.key].required
    ]
    print(f"  succeeded : {len(outcomes) - len(failures)}/{len(outcomes)}")
    if failures:
        print(f"  failed    : {[o.key for o in failures]}")
    if required_failures:
        print()
        print("  REQUIRED models are missing. The service will not start.")
        for outcome in required_failures:
            print(f"    {outcome.key}: {outcome.message}")
        return 1
    if failures:
        print()
        print("  Only optional models failed. The service will run in a degraded")
        print("  configuration using its fallback chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
