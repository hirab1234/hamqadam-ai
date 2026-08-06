"""Show what is enrolled in the duplicate gallery.

    python scripts/show_gallery.py            # table of every enrolled reference
    python scripts/show_gallery.py --json     # machine-readable
    python scripts/show_gallery.py --limit 20

Why this exists
---------------
"Is anything actually being stored?" is the first question anyone asks while
testing duplicate detection, and the response to /v1/verify does not answer it:
``gallery_size`` is measured during the duplicate *search*, which happens before
the enrolment, so a first successful verification reports 0 and looks like a
failure. This reads the store directly and reports the state *after* the write.

It talks to Qdrant over HTTP using the same URL the application resolves, so it
tells you about the gallery the API is actually using rather than one you
assumed. It never enrols and never deletes.

No vector is printed. A 512-float template is biometric data; only the
reference, model version and enrolment timestamp are shown - the same fields
the admin routes expose, and for the same reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hamqadam_ai.core.config import get_settings  # noqa: E402


def _scroll(url: str, collection: str, limit: int) -> list[dict[str, Any]]:
    """Page through the collection, returning payloads only."""
    endpoint = f"{url.rstrip('/')}/collections/{collection}/points/scroll"
    records: list[dict[str, Any]] = []
    offset: Any = None

    while True:
        body: dict[str, Any] = {
            "limit": min(limit - len(records), 256),
            "with_payload": True,
            # Never ask for the vectors. Beyond the privacy point, a gallery of
            # any size would print megabytes of floats.
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        request = urllib.request.Request(  # noqa: S310 - fixed http(s) endpoint
            endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            result = json.loads(response.read())["result"]

        records.extend(point.get("payload") or {} for point in result["points"])
        offset = result.get("next_page_offset")
        if offset is None or len(records) >= limit:
            return records


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Show the duplicate gallery.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows.")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    url = settings.duplicate.qdrant.url
    collection = settings.duplicate.qdrant.collection

    if not url.startswith(("http://", "https://")):
        print(f"duplicate.qdrant.url is {url!r}, which is not an HTTP endpoint.")
        print("This script reads a Qdrant server; nothing to connect to.")
        return 1

    try:
        records = _scroll(url, collection, args.limit)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"Collection {collection!r} does not exist yet at {url}.")
            print("It is created on first use - run one verification.")
            return 1
        print(f"Qdrant returned HTTP {exc.code} from {url}: {exc.reason}")
        return 1
    except OSError as exc:
        # The common one on Windows is WinError 10061 when the container is not
        # publishing 6333 - the base compose file exposes it to the compose
        # network only.
        print(f"Could not reach Qdrant at {url}: {exc}")
        print()
        print("If the stack is up but the port is not published, use the dev")
        print("override, which publishes 6333 on 127.0.0.1:")
        print(
            "  docker compose -f deploy/docker-compose.yml "
            "-f deploy/docker-compose.dev.yml up -d"
        )
        return 1

    if args.as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0

    print(f"{url}  collection={collection}")
    print(f"{len(records)} enrolled reference(s)")
    if not records:
        print()
        print("Empty. A verification only enrols when it carries a")
        print("user_reference AND the outcome satisfies duplicate.enrol_policy")
        print(f"(currently {settings.duplicate.enrol_policy!r}).")
        return 0

    print()
    print(f"{'reference':<28} {'enrolled_at':<34} model_version")
    print(f"{'-' * 28} {'-' * 34} {'-' * 40}")
    for record in records:
        print(
            f"{str(record.get('reference', '?')):<28} "
            f"{str(record.get('enrolled_at', '?')):<34} "
            f"{record.get('model_version', '?')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
