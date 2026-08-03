"""Export the OpenAPI schema for the Backend team.

    python scripts/export_openapi.py                      # to stdout
    python scripts/export_openapi.py -o docs/openapi.json
    python scripts/export_openapi.py --summary             # human-readable

Why a file rather than "just read /openapi.json"
------------------------------------------------
The live endpoint requires a running service, which requires the model weights,
which is several minutes and several gigabytes before the Backend team can see
the shape of a response. Exporting it needs neither: the schema comes from the
Pydantic models, and those import without any weights at all.

Committing the output also makes a contract change **visible in a diff**. A
field quietly renamed between releases is otherwise discovered by the Backend at
runtime, in production, on a real applicant's verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_schema() -> dict[str, Any]:
    """Produce the OpenAPI document without loading a single model.

    The app is constructed with authentication and metrics disabled. Neither
    affects the schema, and both would otherwise demand configuration that has
    nothing to do with the contract being exported.
    """
    from hamqadam_ai.core.config import get_settings
    from hamqadam_ai.logging.setup import configure_logging

    settings = get_settings().model_copy(deep=True)
    settings.security.require_api_key = False
    settings.observability.metrics.enabled = False

    # Logging must be silenced *before* anything logs, and it matters more here
    # than it looks: structlog's unconfigured default writes to stdout, so
    # `export_openapi.py > openapi.json` captured a route-registration line as
    # the first line of the file and produced invalid JSON. The schema is the
    # only thing allowed on stdout.
    settings.logging.level = "CRITICAL"
    configure_logging(settings)

    # Routes are registered by `create_app`; the lifespan - which is what loads
    # the weights - only runs when the app is actually served.
    from hamqadam_ai.api.app import create_app

    app = create_app(settings)
    schema: dict[str, Any] = app.openapi()
    return schema


def _render_type(spec: dict[str, Any]) -> str:
    """Describe one property's type the way a Backend engineer needs to read it.

    `anyOf` is unwrapped rather than reported as "object". An optional float
    arrives as ``anyOf: [number, null]``, and rendering that as "object" told the
    reader nothing - worse, it implied a nested body where a bare number is
    sent. `identity_confidence_score` is precisely such a field, and precisely
    the one whose nullability the Backend must handle: it is null when no face
    was usable, which is not the same as a score of zero.
    """
    if "anyOf" in spec or "oneOf" in spec:
        options = spec.get("anyOf") or spec.get("oneOf") or []
        rendered = [_render_type(option) for option in options]
        # Deduplicate while keeping order, so `[number, number]` does not appear.
        seen: list[str] = []
        for item in rendered:
            if item not in seen:
                seen.append(item)
        return "|".join(seen)
    if spec.get("type") == "null":
        return "null"
    if "$ref" in spec:
        return spec["$ref"].rsplit("/", 1)[-1]
    if "enum" in spec:
        return "enum"
    if spec.get("type") == "array":
        return f"{_render_type(spec.get('items', {}))}[]"
    return str(spec.get("type") or "object")


def summarise(schema: dict[str, Any]) -> str:
    """Render the parts a Backend engineer needs before reading any JSON."""
    lines: list[str] = []
    info = schema.get("info", {})
    lines.append(f"{info.get('title', 'API')}  v{info.get('version', '?')}")
    lines.append("=" * 74)

    lines.append("\nEndpoints")
    lines.append("-" * 74)
    for path, operations in sorted(schema.get("paths", {}).items()):
        for method, operation in sorted(operations.items()):
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            summary = operation.get("summary", "")
            lines.append(f"  {method.upper():<7} {path:<34} {summary}")

    result = schema.get("components", {}).get("schemas", {}).get(
        "VerificationResult"
    )
    if result:
        lines.append("\nVerificationResult fields")
        lines.append("-" * 74)
        required = set(result.get("required", []))
        for name, spec in result.get("properties", {}).items():
            flag = "required" if name in required else "optional"
            lines.append(f"  {name:<28} {_render_type(spec):<16} {flag}")

    lines.append(
        f"\n{len(schema.get('components', {}).get('schemas', {}))} schema "
        f"definitions in total."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Export the service's OpenAPI schema."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the JSON here instead of to stdout.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary instead of the JSON.",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation."
    )
    args = parser.parse_args(argv)

    try:
        schema = build_schema()
    except Exception as exc:  # noqa: BLE001 - a clear message beats a traceback
        print(f"could not build the schema: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        print(summarise(schema))
        return 0

    payload = json.dumps(schema, indent=args.indent, sort_keys=False)
    if args.output is None:
        print(payload)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({len(payload):,} bytes, "
        f"{len(schema.get('paths', {}))} paths, "
        f"{len(schema.get('components', {}).get('schemas', {}))} schemas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
