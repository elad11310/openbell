"""Read-only status API and dashboard. There are no endpoints that place orders."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import Config
from .engine import Engine
from .journal import Journal

PAGE = Path(__file__).with_name("dashboard.html")


def _thin(rows: list, n: int = 240) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n - 1)] + [rows[-1]]


def create_app(engine: Engine, journal: Journal, cfg: Config, info: dict) -> FastAPI:
    app = FastAPI(title="OpenBell", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.read_text()

    @app.get("/api/state")
    def state() -> dict:
        snap = engine.snapshot()
        day = journal.rows("SELECT * FROM days ORDER BY day DESC LIMIT 1")
        snap.update({
            "now": datetime.now(timezone.utc).isoformat(),
            "display_tz": cfg.api.display_tz,
            "benchmark": cfg.benchmark,
            "starting_equity": cfg.starting_equity,
            "next_session": info.get("next_session"),
            "agent_active": info.get("agent_active", False),
            "stats": journal.stats(),
            "trades": journal.trades(limit=25),
            "events": journal.rows("SELECT ts, kind, symbol, message FROM events "
                                   "ORDER BY id DESC LIMIT 40"),
            "equity_curve": _thin(journal.rows("SELECT ts, equity, benchmark FROM equity ORDER BY ts")),
            "scan": journal.scan_rows(),
            "scan_stats": journal.scan_stats(),
            "review": day[0]["review"] if day else None,
            "review_day": day[0]["day"] if day else None,
        })
        return snap

    return app
