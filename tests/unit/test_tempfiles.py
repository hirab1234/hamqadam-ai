"""Scratch storage — the "no permanent AI-side storage" guarantee.

Section 21 of the requirements document forbids the AI service from retaining
user images, and section 10 of the development agreement makes CNIC images and
selfies confidential. The pipeline works in memory, but some third-party
libraries (PaddleOCR in particular) insist on a file path, so scratch space
exists and must be provably self-destructing.

Four separate mechanisms are tested here, because each covers a failure the
others cannot:

* the ``with`` block, for the normal path,
* deletion inside ``finally``, for the exception path,
* content overwriting before unlink, for forensic recovery,
* the janitor, for the ``SIGKILL`` / OOM-kill path where no Python code runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hamqadam_ai.utils.tempfiles import (
    SCRATCH_PREFIX,
    ScratchSpace,
    TempFileJanitor,
    scratch_space,
)

SENSITIVE = b"CNIC-IMAGE-BYTES-35202-1234567-1-" * 64


# --------------------------------------------------------------------------- #
# Normal lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_context_manager_removes_the_directory(tmp_path: Path) -> None:
    with scratch_space(tmp_path, label="unit") as space:
        location = space.path
        space.write_bytes("payload.bin", SENSITIVE)
        assert location.is_dir()
    assert not location.exists()


@pytest.mark.unit
def test_directory_is_removed_even_when_the_block_raises(tmp_path: Path) -> None:
    """The exception path is the one that leaks in a naive implementation."""
    location: Path | None = None
    with (
        pytest.raises(RuntimeError, match="pipeline blew up"),
        scratch_space(tmp_path, label="unit") as space,
    ):
        location = space.path
        space.write_bytes("cnic.jpg", SENSITIVE)
        raise RuntimeError("pipeline blew up")

    assert location is not None
    assert not location.exists()


@pytest.mark.unit
def test_close_is_idempotent(tmp_path: Path) -> None:
    space = ScratchSpace(tmp_path, label="unit")
    space.close()
    space.close()
    assert space.closed is True


@pytest.mark.unit
def test_path_access_after_close_fails_loudly(tmp_path: Path) -> None:
    """Handing out a path to a deleted directory is a bug worth failing on."""
    space = ScratchSpace(tmp_path, label="unit")
    space.close()
    with pytest.raises(RuntimeError, match="no longer valid"):
        _ = space.path


@pytest.mark.unit
def test_each_space_is_unique(tmp_path: Path) -> None:
    with scratch_space(tmp_path) as first, scratch_space(tmp_path) as second:
        assert first.path != second.path


@pytest.mark.unit
def test_directory_carries_the_ownership_prefix(tmp_path: Path) -> None:
    """The janitor refuses to touch anything without it, so a misconfigured
    temp_dir pointing at a shared location cannot cause collateral damage."""
    with scratch_space(tmp_path, label="verify") as space:
        assert space.path.name.startswith(SCRATCH_PREFIX)
        assert "verify" in space.path.name


@pytest.mark.unit
def test_nested_content_is_removed(tmp_path: Path) -> None:
    with scratch_space(tmp_path) as space:
        location = space.path
        (location / "sub" / "deeper").mkdir(parents=True)
        (location / "sub" / "deeper" / "cnic.jpg").write_bytes(SENSITIVE)
    assert not location.exists()


# --------------------------------------------------------------------------- #
# Path-traversal containment
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["../escape.txt", "..", ".", "sub/file.txt", "sub\\file.txt", ""]
)
def test_filenames_cannot_escape_the_directory(tmp_path: Path, name: str) -> None:
    """A filename derived from user input must not reach the parent directory."""
    with (
        scratch_space(tmp_path) as space,
        pytest.raises(ValueError, match="Invalid scratch filename"),
    ):
        space.file(name)


@pytest.mark.unit
def test_a_plain_filename_is_accepted(tmp_path: Path) -> None:
    with scratch_space(tmp_path) as space:
        assert space.file("cnic.jpg").parent == space.path


# --------------------------------------------------------------------------- #
# Secure deletion
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_file_contents_are_overwritten_before_unlink(tmp_path: Path) -> None:
    """Read the raw bytes back off disk before rmtree runs.

    A pure ``unlink`` leaves the payload in the filesystem's free list. This
    asserts the shred step genuinely replaces the bytes.
    """
    space = ScratchSpace(tmp_path, label="shred", secure_delete=True)
    target = space.write_bytes("cnic.jpg", SENSITIVE)
    assert target.read_bytes() == SENSITIVE

    from hamqadam_ai.utils.tempfiles import _shred_file

    _shred_file(target)
    assert target.read_bytes() != SENSITIVE
    assert target.stat().st_size == 0
    space.close()


@pytest.mark.unit
def test_secure_delete_can_be_disabled(tmp_path: Path) -> None:
    """Shredding a 40 MB image costs latency; it is a configurable trade-off."""
    with scratch_space(tmp_path, secure_delete=False) as space:
        location = space.path
        space.write_bytes("payload.bin", SENSITIVE)
    assert not location.exists()


@pytest.mark.unit
def test_write_bytes_creates_an_owner_only_file(tmp_path: Path) -> None:
    """Opened 0600 from the outset, not chmod-ed afterwards — otherwise there
    is a window where the file is world-readable."""
    with scratch_space(tmp_path) as space:
        target = space.write_bytes("cnic.jpg", SENSITIVE)
        assert target.is_file()
        if os.name != "nt":  # POSIX permission bits are meaningless on Windows
            assert (target.stat().st_mode & 0o777) == 0o600


@pytest.mark.unit
def test_writing_the_same_name_twice_is_refused(tmp_path: Path) -> None:
    """O_EXCL: silently overwriting would mask a caller bug."""
    with scratch_space(tmp_path) as space:
        space.write_bytes("cnic.jpg", SENSITIVE)
        with pytest.raises(FileExistsError):
            space.write_bytes("cnic.jpg", SENSITIVE)


# --------------------------------------------------------------------------- #
# The janitor — the crash path a `finally` cannot cover
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_janitor_removes_an_aged_orphan(tmp_path: Path) -> None:
    """Simulates a pod killed by the OOM killer mid-request."""
    orphan = tmp_path / f"{SCRATCH_PREFIX}crashed-abcdef0123456789"
    orphan.mkdir()
    (orphan / "cnic.jpg").write_bytes(SENSITIVE)
    os.utime(orphan, (0, 0))  # far in the past

    janitor = TempFileJanitor(tmp_path, ttl_seconds=60.0)
    assert janitor.sweep_once() == 1
    assert not orphan.exists()


@pytest.mark.unit
def test_janitor_leaves_a_fresh_orphan_alone(tmp_path: Path) -> None:
    """A directory younger than the TTL may belong to an in-flight request."""
    fresh = tmp_path / f"{SCRATCH_PREFIX}inflight-abcdef0123456789"
    fresh.mkdir()

    janitor = TempFileJanitor(tmp_path, ttl_seconds=3600.0)
    assert janitor.sweep_once() == 0
    assert fresh.exists()


@pytest.mark.unit
def test_janitor_never_touches_foreign_directories(tmp_path: Path) -> None:
    """A misconfigured temp_dir pointing at /tmp must not delete other tenants."""
    foreign = tmp_path / "someone-elses-important-data"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("do not delete", encoding="utf-8")
    os.utime(foreign, (0, 0))

    janitor = TempFileJanitor(tmp_path, ttl_seconds=1.0)
    assert janitor.sweep_once() == 0
    assert (foreign / "keep.txt").is_file()


@pytest.mark.unit
def test_janitor_never_touches_loose_files(tmp_path: Path) -> None:
    stray = tmp_path / f"{SCRATCH_PREFIX}not-a-directory.txt"
    stray.write_text("x", encoding="utf-8")
    os.utime(stray, (0, 0))

    janitor = TempFileJanitor(tmp_path, ttl_seconds=1.0)
    assert janitor.sweep_once() == 0
    assert stray.exists()


@pytest.mark.unit
def test_janitor_skips_a_live_scratch_space(tmp_path: Path) -> None:
    """An in-flight request's directory must survive even past the TTL."""
    with scratch_space(tmp_path, label="live") as space:
        os.utime(space.path, (0, 0))  # pretend it is old
        janitor = TempFileJanitor(tmp_path, ttl_seconds=1.0)
        assert janitor.sweep_once() == 0
        assert space.path.is_dir()


@pytest.mark.unit
def test_janitor_sweeps_a_missing_root_without_raising(tmp_path: Path) -> None:
    janitor = TempFileJanitor(tmp_path / "never-created", ttl_seconds=1.0)
    assert janitor.sweep_once() == 0


@pytest.mark.unit
def test_janitor_reports_its_counters(tmp_path: Path) -> None:
    orphan = tmp_path / f"{SCRATCH_PREFIX}old-0123456789abcdef"
    orphan.mkdir()
    os.utime(orphan, (0, 0))

    janitor = TempFileJanitor(tmp_path, ttl_seconds=1.0)
    janitor.sweep_once()
    stats = janitor.stats

    assert stats["sweeps"] == 1
    assert stats["orphans_removed"] == 1
    assert stats["running"] is False


@pytest.mark.unit
def test_janitor_thread_starts_and_stops(tmp_path: Path) -> None:
    janitor = TempFileJanitor(tmp_path, ttl_seconds=60.0, interval_seconds=30.0)
    janitor.start()
    try:
        assert janitor.stats["running"] is True
    finally:
        janitor.stop(timeout=2.0)
    assert janitor.stats["running"] is False


@pytest.mark.unit
def test_janitor_start_is_idempotent(tmp_path: Path) -> None:
    janitor = TempFileJanitor(tmp_path, ttl_seconds=60.0, interval_seconds=30.0)
    janitor.start()
    janitor.start()
    try:
        assert janitor.stats["running"] is True
    finally:
        janitor.stop(timeout=2.0)
