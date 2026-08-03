"""Secure scratch storage with guaranteed cleanup.

Section 21 of the requirements document forbids permanent AI-side storage of
user images. This module is how that guarantee is implemented and, just as
importantly, how it survives the failure cases:

* :class:`ScratchSpace` deletes its directory in a ``finally`` block, so a
  raised exception cannot leak files.
* Deletion overwrites file contents before unlinking when
  ``storage.secure_delete`` is on, so the bytes are not trivially recoverable
  from the filesystem's free list.
* :class:`TempFileJanitor` sweeps orphaned directories left behind by a hard
  crash, ``SIGKILL`` or an OOM kill - the cases a ``finally`` block cannot
  cover.
* An ``atexit`` hook makes a final attempt on normal interpreter shutdown.

The pipeline itself works entirely in memory; scratch space exists only for the
third-party libraries (PaddleOCR in particular) that insist on a file path.
"""

from __future__ import annotations

import atexit
import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)

#: Prefix identifying directories this service owns. The janitor refuses to
#: delete anything without it, so a misconfigured temp_dir pointing at a shared
#: location cannot cause collateral damage.
SCRATCH_PREFIX = "hqai-"

#: Number of overwrite passes before unlinking. One pass of random bytes
#: defeats casual recovery on any journalling filesystem; more passes are
#: cargo-cult on modern storage and merely slow the request down.
_OVERWRITE_PASSES = 1

#: Files above this size are unlinked without overwriting. Shredding a 40 MB
#: image adds latency to the request for no benefit, since the payload never
#: touches disk in the first place unless a library demanded it.
_MAX_SHRED_BYTES = 8 * 1024 * 1024

#: Live scratch directories, for the atexit sweep.
_active_spaces: set[Path] = set()
_active_lock = threading.Lock()


class ScratchSpace:
    """A private temporary directory that deletes itself.

    Prefer the :func:`scratch_space` context manager; construct directly only
    when the lifetime genuinely cannot be expressed as a ``with`` block.

    Args:
        root: Parent directory. Created if absent.
        label: Short tag embedded in the directory name, for diagnosability.
        secure_delete: Overwrite file contents before unlinking.
    """

    __slots__ = ("_closed", "_label", "_path", "_secure_delete")

    def __init__(
        self,
        root: Path,
        *,
        label: str = "req",
        secure_delete: bool = True,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self._label = label
        self._secure_delete = secure_delete
        self._closed = False
        self._path = root / f"{SCRATCH_PREFIX}{label}-{uuid.uuid4().hex[:16]}"
        self._path.mkdir(mode=0o700, parents=False, exist_ok=False)
        _restrict_permissions(self._path)

        with _active_lock:
            _active_spaces.add(self._path)

    @property
    def path(self) -> Path:
        """The scratch directory.

        Raises:
            RuntimeError: if the space has already been closed. Handing out a
                path to a deleted directory is a bug worth failing loudly on.
        """
        if self._closed:
            raise RuntimeError("ScratchSpace has been closed and its path is no longer valid")
        return self._path

    @property
    def closed(self) -> bool:
        """Whether the directory has been removed."""
        return self._closed

    def file(self, name: str) -> Path:
        """Return a path inside the scratch directory.

        Args:
            name: Filename. Path separators and ``..`` are rejected, so a
                filename derived from user input cannot escape the directory.

        Raises:
            ValueError: if ``name`` is not a plain filename.
        """
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"Invalid scratch filename: {name!r}")
        return self.path / name

    def write_bytes(self, name: str, data: bytes) -> Path:
        """Write ``data`` to a private file inside the scratch directory."""
        target = self.file(name)
        # Open with 0600 from the outset rather than chmod-ing afterwards,
        # which would leave a window where the file is world-readable.
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
        except BaseException:
            with suppress(OSError):
                target.unlink()
            raise
        return target

    def close(self) -> None:
        """Delete the directory and everything in it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with _active_lock:
            _active_spaces.discard(self._path)
        _destroy_tree(self._path, secure=self._secure_delete)

    def __enter__(self) -> ScratchSpace:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        with suppress(Exception):
            self.close()


@contextmanager
def scratch_space(
    root: Path,
    *,
    label: str = "req",
    secure_delete: bool = True,
) -> Iterator[ScratchSpace]:
    """Create a scratch directory for the duration of the ``with`` block.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with scratch_space(Path(tempfile.gettempdir())) as space:
        ...     _ = space.write_bytes("probe.bin", b"data")
        ...     space.path.is_dir()
        True
    """
    space = ScratchSpace(root, label=label, secure_delete=secure_delete)
    try:
        yield space
    finally:
        space.close()


def _restrict_permissions(path: Path) -> None:
    """Make a directory owner-only where the platform supports it."""
    with suppress(OSError, NotImplementedError):
        path.chmod(0o700)


def _shred_file(path: Path) -> None:
    """Overwrite a file's contents with random bytes, then truncate it."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size == 0 or size > _MAX_SHRED_BYTES:
        return
    try:
        with path.open("r+b", buffering=0) as handle:
            for _ in range(_OVERWRITE_PASSES):
                handle.seek(0)
                handle.write(os.urandom(size))
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            handle.truncate(0)
    except OSError as exc:
        log.warning("scratch.shred_failed", reason=str(exc))


def _destroy_tree(path: Path, *, secure: bool) -> None:
    """Remove a directory tree, optionally shredding files first."""
    if not path.exists():
        return
    if secure:
        for entry in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if entry.is_file() and not entry.is_symlink():
                _shred_file(entry)
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        # On Windows a handle held by a third-party library blocks removal.
        # Retry once after a short pause, then give up to the janitor.
        time.sleep(0.05)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            log.warning("scratch.cleanup_failed", path=str(path), reason=str(exc))


class TempFileJanitor:
    """Background sweeper for scratch directories orphaned by a crash.

    A ``finally`` block cannot run after ``SIGKILL`` or an OOM kill, so without
    this the temp volume of a container that restarts under memory pressure
    fills with CNIC images. The janitor removes any directory carrying the
    service's prefix whose modification time is older than the TTL.

    Args:
        root: Directory to sweep.
        ttl_seconds: Age above which an orphan is removed.
        interval_seconds: Delay between sweeps.
        secure_delete: Shred file contents before unlinking.
    """

    __slots__ = (
        "_interval",
        "_root",
        "_secure_delete",
        "_stop",
        "_thread",
        "_ttl",
        "_sweeps",
        "_removed",
    )

    def __init__(
        self,
        root: Path,
        *,
        ttl_seconds: float = 300.0,
        interval_seconds: float = 60.0,
        secure_delete: bool = True,
    ) -> None:
        self._root = root
        self._ttl = ttl_seconds
        self._interval = interval_seconds
        self._secure_delete = secure_delete
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sweeps = 0
        self._removed = 0

    def start(self) -> None:
        """Start the sweeper thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="hqai-temp-janitor", daemon=True
        )
        self._thread.start()
        log.info(
            "janitor.started",
            root=str(self._root),
            ttl_seconds=self._ttl,
            interval_seconds=self._interval,
        )

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the sweeper to exit and wait briefly for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        log.info("janitor.stopped", sweeps=self._sweeps, removed=self._removed)

    def sweep_once(self) -> int:
        """Run a single sweep synchronously.

        Returns:
            The number of orphaned directories removed.
        """
        self._sweeps += 1
        if not self._root.is_dir():
            return 0

        cutoff = time.time() - self._ttl
        removed = 0

        try:
            entries = list(self._root.iterdir())
        except OSError as exc:
            log.warning("janitor.scan_failed", root=str(self._root), reason=str(exc))
            return 0

        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith(SCRATCH_PREFIX):
                continue
            with _active_lock:
                if entry in _active_spaces:
                    continue
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue

            _destroy_tree(entry, secure=self._secure_delete)
            removed += 1
            log.info("janitor.removed_orphan", path=entry.name)

        self._removed += removed
        return removed

    @property
    def stats(self) -> dict[str, Any]:
        """Counters for the health endpoint."""
        return {
            "sweeps": self._sweeps,
            "orphans_removed": self._removed,
            "running": self._thread is not None and self._thread.is_alive(),
            "root": str(self._root),
        }

    def _run(self) -> None:
        # Sweep immediately on start: the most likely orphans are the ones this
        # very process left behind when it was killed a moment ago.
        while not self._stop.is_set():
            try:
                self.sweep_once()
            except Exception as exc:  # noqa: BLE001 - the janitor must never die
                log.error("janitor.sweep_error", reason=str(exc), exc_info=True)
            self._stop.wait(self._interval)


@atexit.register
def _cleanup_on_exit() -> None:  # pragma: no cover - interpreter shutdown
    """Last-chance removal of live scratch directories at normal shutdown."""
    with _active_lock:
        remaining = list(_active_spaces)
        _active_spaces.clear()
    for path in remaining:
        with suppress(Exception):
            _destroy_tree(path, secure=False)


__all__ = ["SCRATCH_PREFIX", "ScratchSpace", "TempFileJanitor", "scratch_space"]
