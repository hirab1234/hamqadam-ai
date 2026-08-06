"""Generate `deploy/.env` so the compose stack starts locally.

    python scripts/make_dev_env.py            # create it if absent
    python scripts/make_dev_env.py --force    # regenerate, rotating secrets
    python scripts/make_dev_env.py --print    # show what it would write

Why this exists
---------------
`deploy/docker-compose.yml` guards its secrets with the `${VAR:?}` form, so a
run without them fails immediately and names the missing variable::

    RABBITMQ_PASSWORD  - required variable is not set

That is deliberate and stays. Its whole purpose is that a real deployment
cannot start with an unset password and quietly accept traffic, and nothing here
weakens it - this script satisfies the guard locally by writing values, rather
than removing the guard.

What it writes, and where
-------------------------
`deploy/.env`, because Compose reads `.env` from the **project directory**,
which defaults to the directory holding the compose file. A `.env` at the
repository root is not read by `docker compose -f deploy/docker-compose.yml`.

Not to be confused with the repository's own `.env.example`: that documents the
application's `HQ_`-prefixed settings, which the container reads. This file
holds the handful of variables Compose itself interpolates. Different consumers,
different files.

Safety
------
Secrets are random per generation, not fixed strings, so two developers never
share a key and nothing guessable reaches a machine that later gets exposed.
The file is gitignored.

An existing `.env` is never overwritten without `--force`: it may hold
credentials for a real environment somebody pointed at this checkout, and
silently replacing those would be an outage with no message attached.
"""

from __future__ import annotations

import argparse
import contextlib
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / "deploy" / ".env"

TEMPLATE = """\
# Local development secrets for docker compose. GENERATED - do not commit.
#
# Written by `python scripts/make_dev_env.py`. Regenerate with --force.
#
# These satisfy the `${{VAR:?}}` guards in docker-compose.yml, which exist so a
# real deployment cannot start with unset passwords. The guards are unchanged;
# this file simply provides values for a laptop.
#
# Every secret below is random and local. Do not reuse any of them anywhere
# that matters, and do not copy this file to a server - generate a fresh one
# there, or supply the variables from your secret manager.

# API key the Backend sends as `X-API-Key`. Comma-separate for rotation.
HQ_API_KEYS={api_key}

# Broker credentials. The username is fixed to `hamqadam` in the compose file.
RABBITMQ_PASSWORD={rabbitmq_password}

# Grafana admin login (user: admin), reachable on 127.0.0.1:3000 only.
GRAFANA_PASSWORD={grafana_password}

# ---------------------------------------------------------------------------
# Optional policy switches. Both have defaults in the compose file; these are
# spelled out so the behaviour of a local stack is visible rather than implied.
# ---------------------------------------------------------------------------

# on_approve | unless_rejected
#   on_approve      only an APPROVE enrols into the duplicate gallery
#   unless_rejected APPROVE or MANUAL_REVIEW enrol - closes the hole where a
#                   reviewed applicant is never enrolled, so their second
#                   account has nothing to collide with
HQ_ENROL_POLICY=on_approve

# manual_review | reject
HQ_ON_DUPLICATE=manual_review
"""


def render() -> str:
    """Build the file contents with freshly generated secrets."""
    return TEMPLATE.format(
        api_key=f"dev-{secrets.token_hex(16)}",
        rabbitmq_password=secrets.token_urlsafe(24),
        grafana_password=secrets.token_urlsafe(18),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate deploy/.env for local docker compose runs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file, rotating every secret.",
    )
    parser.add_argument(
        "--print",
        dest="show",
        action="store_true",
        help="Print what would be written and exit.",
    )
    args = parser.parse_args(argv)

    if args.show:
        print(render())
        return 0

    if ENV_PATH.exists() and not args.force:
        # Refusing is the safe answer. The existing file may point this checkout
        # at a real environment, and replacing those credentials silently would
        # break it with nothing in the output to say why.
        print(f"{ENV_PATH} already exists; leaving it alone.")
        print("Pass --force to regenerate, which rotates every secret.")
        return 0

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(render(), encoding="utf-8")

    # 0600 where the platform honours it. Windows ignores the mode; the file is
    # gitignored either way, which is the protection that actually applies here.
    with contextlib.suppress(OSError):
        ENV_PATH.chmod(0o600)

    key = next(
        line.split("=", 1)[1]
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("HQ_API_KEYS=")
    )
    print(f"wrote {ENV_PATH}")
    print(f"  X-API-Key: {key}")
    print()
    print("Start the stack with:")
    print("  docker compose -f deploy/docker-compose.yml up -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
