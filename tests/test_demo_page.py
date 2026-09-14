"""Step 7: the demo page is served and wired to the real /explain endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_demo_page_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Understand your report" in response.text


def test_demo_page_posts_to_explain(client):
    assert 'fetch("/explain"' in client.get("/").text


def test_demo_page_references_report_text_field(client):
    body = client.get("/").text
    assert "report_text" in body
    assert "target_language" in body
    assert "include_audio" in body


def test_demo_page_offers_hindi_and_audio(client):
    body = client.get("/").text
    assert 'value="hi"' in body
    assert 'id="include-audio"' in body
    assert 'id="audio-el"' in body
