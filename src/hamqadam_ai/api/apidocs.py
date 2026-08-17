"""Serve `docs/api-reference.md` as a readable page.

Two routes, both unauthenticated:

    GET /api-docs      the reference, rendered
    GET /api-docs.md   the same file, raw

Why this exists alongside /docs
-------------------------------
Swagger describes the *shape* of the API - field names, types, status codes -
and it is generated, so it is never out of date. What it cannot express is the
part an integrator actually gets wrong: that MANUAL_REVIEW is not a soft
approval, that `user_reference` is what makes duplicate detection work at all,
that `gallery_size` is measured before enrolment, that a verification takes
fifteen seconds and must not block a user-facing request. Those are prose, and
they live in one Markdown file that is also readable in the repository.

/docs is additionally an interactive console with an Authorize box that persists
a key in browser storage, which is why the production overlay does not serve it.
This page holds no credential and can call nothing, so it stays on.

One source of truth
-------------------
The Markdown file is the document. This module renders it and adds no content of
its own, so the repository copy and the served page cannot disagree.

The stylesheet is inlined. A documentation page that fetches a stylesheet from a
CDN is a page that renders unstyled on an air-gapped host, and it hands a third
party a log of who read your API reference.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi.responses import HTMLResponse, PlainTextResponse

from hamqadam_ai.core.config import REPO_ROOT
from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)

#: The document. Copied into the image by the Dockerfile - see the note there
#: about why that COPY is load-bearing.
REFERENCE_PATH: Path = REPO_ROOT / "docs" / "api-reference.md"

#: Palette taken from hamqadam.com/api-docs so the two pages read as one set:
#: page #f5f6fa, ink #1a1a2e, content on white, system font stack.
_STYLE = """
:root {
  --page: #f5f6fa; --card: #ffffff; --fg: #1a1a2e; --muted: #5b6672;
  --line: #e1e5ee; --accent: #0b6bcb; --code-bg: #f5f7fa; --head-bg: #eef1f7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #101319; --card: #171b22; --fg: #e6eaef; --muted: #9aa6b2;
    --line: #2a323b; --accent: #6db3f2; --code-bg: #1b2027; --head-bg: #1f262e;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0; background: var(--page); color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
}
.layout {
  display: grid; grid-template-columns: 16rem minmax(0, 1fr); gap: 2rem;
  max-width: 84rem; margin: 0 auto; padding: 2rem 1.25rem 6rem;
}
nav.toc {
  position: sticky; top: 2rem; align-self: start; max-height: calc(100vh - 4rem);
  overflow-y: auto; font-size: .9rem;
}
nav.toc .label {
  text-transform: uppercase; letter-spacing: .06em; font-size: .72rem;
  font-weight: 700; color: var(--muted); margin-bottom: .6rem;
}
nav.toc a {
  display: block; padding: .3rem .6rem; border-radius: 6px; color: var(--fg);
  text-decoration: none; border-left: 2px solid transparent;
}
nav.toc a:hover { background: var(--card); border-left-color: var(--accent); }
main {
  background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 2.25rem 2.5rem 3rem; min-width: 0;
}
h1, h2, h3, h4 { line-height: 1.25; margin: 2.2rem 0 .8rem; font-weight: 650; }
h1 { font-size: 2rem; margin-top: 0; }
h2 {
  font-size: 1.4rem; padding-bottom: .35rem; margin-top: 2.75rem;
  border-bottom: 1px solid var(--line); scroll-margin-top: 1.5rem;
}
h3 { font-size: 1.12rem; scroll-margin-top: 1.5rem; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }
code {
  background: var(--code-bg); padding: .15em .4em; border-radius: 4px;
  font: .875em/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre {
  background: var(--code-bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem; overflow-x: auto;
}
pre code { background: none; padding: 0; font-size: .84rem; }
table {
  border-collapse: collapse; width: 100%; margin: 1rem 0; display: block;
  overflow-x: auto;
}
th, td {
  border: 1px solid var(--line); padding: .5rem .7rem; text-align: left;
  vertical-align: top; font-size: .93rem;
}
th { background: var(--head-bg); font-weight: 620; white-space: nowrap; }
blockquote {
  margin: 1rem 0; padding: .1rem 1rem; border-left: 3px solid var(--line);
  color: var(--muted);
}
@media (max-width: 60rem) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  nav.toc { position: static; max-height: none; }
  main { padding: 1.5rem 1.25rem 2rem; }
}
"""

_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>{title}</title>
<style>{style}</style>
</head><body><div class="layout">{nav}<main>{body}</main></div></body></html>"""


def read_reference() -> str:
    """The Markdown source.

    Raises:
        FileNotFoundError: if the document is not on disk. Deliberately not
            swallowed here - the caller turns it into a 503 naming the path,
            which is what makes a missing COPY in the Dockerfile diagnosable
            from the response instead of from a stack trace.
    """
    return REFERENCE_PATH.read_text(encoding="utf-8")


def _slug(text: str) -> str:
    """A URL fragment for a heading."""
    cleaned = re.sub(r"[`*_]", "", text).strip().lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", cleaned)).strip("-")


def _headings(source: str) -> list[tuple[int, str, str]]:
    """Every `##` and `###` heading, in document order.

    Read from the Markdown rather than the rendered HTML because the source is
    unambiguous: a `##` line is a heading, whereas matching `<h2>` in HTML would
    also match one written inside a fenced code block.
    """
    found: list[tuple[int, str, str]] = []
    fenced = False
    for line in source.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^(#{2,3})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            found.append((level, text, _slug(text)))
    return found


@lru_cache(maxsize=1)
def _render(source: str) -> tuple[str, str]:
    """Markdown to (body HTML, sidebar HTML).

    Cached on the source text, so the file is parsed once per distinct version
    rather than on every request, and an edit still takes effect without a
    restart because changed text is a different cache key.
    """
    from markdown_it import MarkdownIt

    md = (
        MarkdownIt("commonmark", {"html": False})
        .enable("table")
        .enable("strikethrough")
    )
    body: str = md.render(source)

    # markdown-it's commonmark preset emits no heading ids, so the sidebar would
    # have nothing to link to. Injected in document order, which is sound
    # because both lists come from the same source in the same order.
    headings = _headings(source)
    for level, text, slug in headings:
        body = body.replace(
            f"<h{level}>", f'<h{level} id="{html.escape(slug)}">', 1
        )
        del text

    links = "".join(
        f'<a href="#{html.escape(slug)}" '
        f'style="padding-left:{0.6 if level == 2 else 1.5}rem">'
        f"{html.escape(re.sub(r'[`]', '', text))}</a>"
        for level, text, slug in headings
        if level == 2
    )
    nav = f'<nav class="toc"><div class="label">On this page</div>{links}</nav>'
    return body, nav


def register_api_docs(app: Any, settings: Any) -> None:
    """Attach the documentation routes, if they are enabled."""
    if not settings.server.api_docs_enabled:
        log.info("apidocs.disabled", reason="server.api_docs_enabled is false")
        return

    @app.get(
        "/api-docs",
        summary="API reference for integrators",
        tags=["operations"],
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def api_docs() -> HTMLResponse:
        """The reference, rendered as a page."""
        try:
            source = read_reference()
        except OSError as exc:
            log.error("apidocs.unavailable", path=str(REFERENCE_PATH), reason=str(exc))
            return HTMLResponse(
                _PAGE.format(
                    title="API reference unavailable",
                    style=_STYLE,
                    nav="",
                    body=(
                        "<h1>API reference unavailable</h1><p>The document was "
                        f"not found at <code>{html.escape(str(REFERENCE_PATH))}"
                        "</code>. In a container this usually means "
                        "<code>docs/</code> was not copied into the image.</p>"
                    ),
                ),
                status_code=503,
            )

        # First heading wins, so the browser tab matches the document.
        title = next(
            (
                line.lstrip("# ").strip()
                for line in source.splitlines()
                if line.startswith("# ")
            ),
            "API reference",
        )
        body, nav = _render(source)
        return HTMLResponse(
            _PAGE.format(
                title=html.escape(title), style=_STYLE, nav=nav, body=body
            )
        )

    @app.get(
        "/api-docs.md",
        summary="API reference as Markdown",
        tags=["operations"],
        response_class=PlainTextResponse,
        include_in_schema=False,
    )
    async def api_docs_markdown() -> PlainTextResponse:
        """The same document, unrendered, for anything that wants the source."""
        try:
            source = read_reference()
        except OSError as exc:
            log.error("apidocs.unavailable", path=str(REFERENCE_PATH), reason=str(exc))
            return PlainTextResponse(
                f"The API reference was not found at {REFERENCE_PATH}.",
                status_code=503,
            )
        return PlainTextResponse(source, media_type="text/markdown; charset=utf-8")

    log.info("apidocs.registered", routes=["/api-docs", "/api-docs.md"])


__all__ = ["REFERENCE_PATH", "read_reference", "register_api_docs"]
