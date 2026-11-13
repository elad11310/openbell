"""Read-only status API and dashboard. There are no endpoints that place orders.

WHAT: `create_app` builds a small FastAPI web app with two routes:
        GET /           serves `dashboard.html` (a single self-contained page)
        GET /api/state  returns one JSON document with everything the page shows:
                        engine snapshot, stats, trades, events, equity curve, today's
                        scan and the latest agent review
HOW:  The dashboard page fetches /api/state every 3 seconds and re-renders. The
      endpoint reads the live Engine object in memory (positions, prices, status) and
      queries the Journal for history. The app is created by the runner (live) or by the
      `sim --serve` command, which pass in their engine and journal.
WHY:  * Read-only by design: there is no route that changes anything, so even if
        someone reached the page they could not trade. It still has no login, which is
        why config binds it to 127.0.0.1 (this machine only) by default.
      * One endpoint with everything keeps the page simple: one request per refresh.
      * The API holds references to the same Engine/Journal objects the trading loop
        uses (closures below), so it shows live state with no copying or messaging.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import Config
from .engine import Engine
from .journal import Journal

# dashboard.html lives next to this file, wherever the package is installed.
PAGE = Path(__file__).with_name("dashboard.html")


def _thin(rows: list, n: int = 240) -> list:
    """Down-sample a long list to about `n` evenly spaced rows, always keeping the last.

    The equity curve gains a point every minute (~390 a day, thousands over months).
    The chart cannot show more than a few hundred points anyway, so sending all of them
    would only slow the page. Picks indexes 0, step, 2*step, ... (n-1 of them), then
    appends the final row so the chart always ends at the current equity."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n - 1)] + [rows[-1]]


def create_app(engine: Engine, journal: Journal, cfg: Config, info: dict) -> FastAPI:
    """Build the app. `info` is a dict the runner keeps updating (next session, agent on/off)."""
    # Auto-generated API docs pages are switched off; nothing here needs them.
    app = FastAPI(title="OpenBell", docs_url=None, redoc_url=None)

    # The route functions are defined inside create_app, so they "close over" (keep
    # references to) engine, journal, cfg and info. No global variables are needed.
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Re-read on every request, so editing the HTML needs no server restart.
        return PAGE.read_text()

    @app.get("/api/state")
    def state() -> dict:
        snap = engine.snapshot()             # live, in-memory state
        day = journal.rows("SELECT * FROM days ORDER BY day DESC LIMIT 1")   # latest day row
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
        return snap                          # FastAPI serialises the dict to JSON

    return app
