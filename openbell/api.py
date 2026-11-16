"""Read-only status API and dashboard. There are no endpoints that place orders.

WHAT: `create_app` builds a small FastAPI web app with three routes:
        GET /           serves `dashboard.html` (a single self-contained page)
        GET /api/state  returns one JSON document with everything the page shows:
                        engine snapshot, stats, trades, events, equity curve, today's
                        scan and the latest agent review
        GET /api/chart/{symbol}?range=1D|5D|1M|6M|1Y
                        candles, volume, EMA20/50, VWAP, trade markers and price levels
                        for one symbol (see charts.py); 503 when no chart data source
                        was given, 400 for a bad symbol or range, 502 if the data
                        provider fails
HOW:  The dashboard page fetches /api/state every 3 seconds and re-renders. The
      endpoint reads the live Engine object in memory (positions, prices, status) and
      queries the Journal for history. The app is created by the runner (live) or by the
      `sim --serve` command, which pass in their engine and journal, plus a
      `ChartService` that fetches bars when a symbol's chart is opened.
WHY:  * Read-only by design: there is no route that changes anything, so even if
        someone reached the page they could not trade. It still has no login, which is
        why config binds it to 127.0.0.1 (this machine only) by default.
      * One endpoint with everything keeps the page simple: one request per refresh.
      * The API holds references to the same Engine/Journal objects the trading loop
        uses (closures below), so it shows live state with no copying or messaging.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .charts import RANGES, SYMBOL_RE, ChartService, overlays
from .config import Config
from .engine import Engine
from .journal import Journal

log = logging.getLogger("openbell.api")
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


def create_app(engine: Engine, journal: Journal, cfg: Config, info: dict,
               charts: Optional[ChartService] = None) -> FastAPI:
    """Build the app. `info` is a dict the runner keeps updating (next session, agent on/off).
    `charts` serves the price charts; without it the page shows no chart links."""
    # Auto-generated API docs pages are switched off; nothing here needs them.
    app = FastAPI(title="OpenBell", docs_url=None, redoc_url=None)

    # The route functions are defined inside create_app, so they "close over" (keep
    # references to) engine, journal, cfg and info. No global variables are needed.
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Re-read on every request, so editing the HTML needs no server restart.
        # The file is UTF-8 (it contains symbols like "✕"). Without an explicit encoding
        # Python uses the OS code page (e.g. cp1255 on a Hebrew Windows), which fails.
        return PAGE.read_text(encoding="utf-8")

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
            # Tells the page whether symbols can open a chart, and which ranges exist.
            "chart_ranges": list(RANGES) if charts else [],
        })
        return snap                          # FastAPI serialises the dict to JSON

    # `async def` (unlike the routes above): it awaits the network while fetching bars.
    # Live, this runs on the trading event loop, so it must never block; the Alpaca
    # client is async, and the indicator arithmetic over a few hundred bars is instant.
    # `alias="range"`: the query parameter is called range, which is a Python builtin.
    @app.get("/api/chart/{symbol}")
    async def chart(symbol: str, rng: str = Query("1D", alias="range")) -> dict:
        if charts is None:
            raise HTTPException(503, "charts are not available in this mode")
        symbol = symbol.upper()
        if not SYMBOL_RE.match(symbol):
            raise HTTPException(400, "not a ticker symbol")
        if rng not in RANGES:
            raise HTTPException(400, f"range must be one of {', '.join(RANGES)}")
        try:
            data = await charts.chart(symbol, rng)
        except Exception as e:
            # Network trouble, plan limits, an unknown symbol rejected by the provider:
            # the page shows the message; trading is unaffected.
            log.warning("chart %s %s failed: %s: %s", symbol, rng, type(e).__name__, e)
            raise HTTPException(502, f"market data unavailable ({type(e).__name__})")
        return overlays(data, engine, journal)

    return app
