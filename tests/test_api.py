"""Tests for the read-only dashboard API.

WHAT: Requests both routes through FastAPI's in-process test client.
HOW:  Builds the app around a fresh Engine and Journal, as the runner does, and calls
      `/` and `/api/state` without starting a real server.
WHY:  The dashboard page is UTF-8 and was once read with the OS default code page,
      which crashed `/` on Windows machines with a non-UTF-8 locale (e.g. cp1255).
      This keeps both routes exercised on every run.
"""
from fastapi.testclient import TestClient

from openbell.api import create_app
from openbell.config import Config
from openbell.engine import Engine
from openbell.journal import Journal


def make_client(tmp_path):
    cfg = Config()
    j = Journal(str(tmp_path / "t.db"))
    return TestClient(create_app(Engine(cfg, j), j, cfg, {"agent_active": False}))


def test_dashboard_page_is_served(tmp_path):
    r = make_client(tmp_path).get("/")
    assert r.status_code == 200
    assert "✕" in r.text                # a non-ASCII character from the page decodes correctly


def test_state_endpoint_returns_snapshot(tmp_path):
    r = make_client(tmp_path).get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "idle" and body["equity"] == 100_000 and body["agent_active"] is False
