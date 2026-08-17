"""The /api-docs route, and the things that silently break it.

The route itself is four lines of plumbing. What is worth testing is everything
around it, because each has already gone wrong once in this repository:

* the document not being copied into the image (`scripts/` did this, and
  model-init died with "can't open file");
* depending on a package nobody declared (`rapidocr-onnxruntime` did this, and
  the container ran a different OCR engine from the host for weeks);
* documentation drifting out of step with the routes it describes.

The last one has no runtime symptom at all - the page keeps serving, it just
stops being true - so `TestTheDocumentIsCurrent` is the only thing that would
ever catch it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hamqadam_ai.api.apidocs import REFERENCE_PATH, read_reference
from hamqadam_ai.api.app import create_app
from hamqadam_ai.core.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client on the real application."""
    with TestClient(create_app()) as test_client:
        yield test_client


class TestTheRoutesServe:
    def test_the_rendered_page_is_html(self, client: TestClient) -> None:
        response = client.get("/api-docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_the_raw_route_is_markdown(self, client: TestClient) -> None:
        response = client.get("/api-docs.md")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert response.text == read_reference()

    def test_no_api_key_is_required(self, client: TestClient) -> None:
        """Documentation you need a credential to read is documentation the
        person integrating cannot get to."""
        assert client.get("/api-docs", headers={}).status_code == 200
        assert client.get("/api-docs.md", headers={}).status_code == 200

    def test_the_markdown_actually_rendered(self, client: TestClient) -> None:
        """Guards against serving the source as an HTML body, which looks fine
        in a diff and is unreadable in a browser."""
        body = client.get("/api-docs").text
        assert "<table>" in body
        assert "<h1>" in body
        assert "<pre>" in body
        assert "| Field | Type |" not in body, "a table was left unrendered"

    def test_nothing_is_fetched_from_the_internet(
        self, client: TestClient
    ) -> None:
        """An inlined stylesheet renders on an air-gapped host and does not hand
        a third party a log of who read the API reference."""
        head = client.get("/api-docs").text.split("<body>")[0]
        for pattern in ("<script", "cdn.", "googleapis", "unpkg", "jsdelivr"):
            assert pattern not in head


class TestTheSidebar:
    """The reference is long; without a contents list it is a single scroll."""

    def test_a_contents_list_is_rendered(self, client: TestClient) -> None:
        body = client.get("/api-docs").text
        assert '<nav class="toc"' in body
        assert "On this page" in body

    def test_headings_carry_anchors_for_it_to_link_to(
        self, client: TestClient
    ) -> None:
        """markdown-it's commonmark preset emits no heading ids, so without the
        injection step every sidebar link would go nowhere."""
        body = client.get("/api-docs").text
        assert body.count("<h2 id=") >= 8
        assert '<h2>' not in body, "a heading was left without an anchor"

    def test_every_sidebar_link_has_a_target(self, client: TestClient) -> None:
        body = client.get("/api-docs").text
        nav = body.split('<nav class="toc"')[1].split("</nav>")[0]
        targets = set(re.findall(r'<h2 id="([^"]+)"', body))
        links = set(re.findall(r'href="#([^"]+)"', nav))
        assert links, "the sidebar rendered no links"
        assert links <= targets, f"sidebar links with no heading: {links - targets}"


class TestItCanBeTurnedOff:
    def test_disabled_by_configuration(self) -> None:
        settings = get_settings().model_copy(deep=True)
        settings.server.api_docs_enabled = False
        with TestClient(create_app(settings)) as client:
            assert client.get("/api-docs").status_code == 404
            assert client.get("/api-docs.md").status_code == 404

    def test_on_by_default(self) -> None:
        """Off by default would mean the Backend team is told to read a page
        that is not there."""
        assert get_settings().server.api_docs_enabled is True


class TestTheDocumentIsCurrent:
    """Drift here is invisible at runtime: the page keeps serving, it just stops
    describing the service."""

    def test_every_public_route_is_documented(self) -> None:
        source = read_reference()
        undocumented = []
        for route in create_app().routes:
            path = getattr(route, "path", "")
            if not path.startswith(("/v1", "/admin", "/health", "/ready", "/metrics")):
                continue
            # Path parameters are written {reference} in both places.
            if path not in source:
                undocumented.append(path)
        assert not undocumented, f"routes missing from the reference: {undocumented}"

    def test_every_documented_method_and_path_exists(self) -> None:
        """The other direction: a reference describing an endpoint that was
        removed sends integrators to a 404."""
        source = read_reference()
        real = {
            (method, getattr(route, "path", ""))
            for route in create_app().routes
            for method in getattr(route, "methods", set()) or set()
        }
        # Rows are `| Feature | `METHOD` | `/endpoint` | Payload |` - the layout
        # the Backend team asked for, matching hamqadam.com/api-docs.
        documented = set(
            re.findall(
                r"^\|[^|]+\|\s*`(GET|POST|DELETE|PUT|PATCH)`\s*\|\s*`(/[^`]+)`",
                source,
                re.MULTILINE,
            )
        )
        # Query strings are illustrative in the table, not part of the path.
        documented = {(m, p.split("?")[0]) for m, p in documented}
        assert documented, "no method/endpoint rows were found to check"
        assert documented <= real, f"documented but not routed: {documented - real}"

    def test_the_three_recommendations_are_explained(self) -> None:
        source = read_reference()
        for value in ("APPROVE", "REJECT", "MANUAL_REVIEW"):
            assert value in source

    def test_the_error_envelope_is_shown(self) -> None:
        """Integrators need the shape of a failure, not only of a success."""
        source = read_reference()
        assert '"error"' in source
        assert '"retryable"' in source


class TestItSurvivesDeployment:
    """Each of these is a defect this repository has already shipped once."""

    def test_the_document_is_in_the_repository(self) -> None:
        assert REFERENCE_PATH.is_file(), f"{REFERENCE_PATH} is missing"

    def test_the_dockerfile_copies_the_docs_directory(self) -> None:
        """Without this the route works in development and 503s in the
        container - exactly what happened to scripts/ and model-init."""
        dockerfile = (REPO_ROOT / "deploy" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert re.search(r"^COPY .*docs/ docs/", dockerfile, re.MULTILINE), (
            "deploy/Dockerfile does not copy docs/ into the image"
        )

    def test_the_markdown_renderer_is_declared(self) -> None:
        """It arrives transitively via `rich` on a developer machine and is
        absent from the image, which is the quietest possible way for this
        route to fail."""
        declared = "".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "requirements").glob("*.txt")
        )
        assert "markdown-it-py==" in declared


class TestItLeaksNothing:
    """The page is unauthenticated, so its content is public to anyone who can
    reach the service."""

    def test_no_real_api_key_is_printed(self) -> None:
        source = read_reference()
        for key in get_settings().security.api_keys:
            assert key not in source, "a configured API key appears in the docs"

    def test_no_cnic_shaped_number_other_than_the_placeholder(self) -> None:
        """A real identity number must never reach a public page."""
        found = re.findall(r"\b\d{5}-\d{7}-\d\b", read_reference())
        assert set(found) <= {"00000-0000000-0"}, f"real-looking CNIC: {found}"
