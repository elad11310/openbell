"""A local stand-in for Alpaca and Finnhub, speaking the protocol as the app understands it.
It proves the client code is self-consistent; only `openbell doctor` proves it against the real thing.

WHAT: A FastAPI app that imitates the external endpoints the app calls:
        GET  /v2/clock, /v2/assets, /v2/stocks/bars   Alpaca REST
        GET  /v1beta1/news                             Alpaca news
        WS   /v2/iex, /v2/test                         Alpaca websocket stream
        GET  /api/v1/calendar/earnings, /stock/metric  Finnhub
      `start()` runs it on a free localhost port in a background thread.
HOW:  Each endpoint returns small, predictable data with deliberate traps that the
      clients must handle: pagination (bars and news are split over two pages),
      refused recent SIP data (HTTP 403), nanosecond timestamps, a news "round-up"
      tagged with 7 symbols, a trade with a condition code to skip ("W"), a bad API key,
      and a 31-symbol subscription that must be refused.
      test_live_protocol.py points the clients' URL constants at this server.
WHY:  The real services need keys and a network connection, and were not reachable when
      the app was built. This fake lets the HTTP and websocket client code (pagination,
      parsing, reconnects, error classes) run in CI without either.
"""
import asyncio
import json
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import uvicorn
from fastapi import FastAPI, Request, WebSocket

app = FastAPI()
UTC = timezone.utc


def _bars(sym, tf, start, end):
    """Generate bars from `start` to `end` in Alpaca's JSON shape: daily bars at
    05:00 UTC on weekdays, or every minute. Prices climb 0.1 per bar from a
    symbol-dependent start; capped at 400 bars."""
    step = timedelta(days=1) if tf == "1Day" else timedelta(minutes=1)
    t = start.replace(second=0, microsecond=0)
    if tf == "1Day":
        t = t.replace(hour=5, minute=0)          # like Alpaca: daily bars stamped at midnight New York
    out, px = [], 100.0 + len(sym)
    while t <= end and len(out) < 400:
        if tf != "1Day" or t.weekday() < 5:      # no weekend daily bars
            out.append({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": px, "h": px + 1, "l": px - 1,
                        "c": px + 0.5, "v": 1000000 if tf == "1Day" else 5000, "n": 10, "vw": px})
            px += 0.1
        t += step
    return out


@app.get("/v2/clock")
def clock(request: Request):
    # Only the key "good" is accepted. Note the fake answers 200 either way; with a bad
    # key the reply has no "next_open", doctor's check crashes on it (KeyError) and is
    # reported as "[FAIL] alpaca", which is what test_doctor_reports_bad_keys expects.
    if request.headers.get("apca-api-key-id") != "good":
        return {"message": "forbidden."}
    # next_open is set by the test fixture to the app's own calendar answer, so the
    # doctor's "calendars agree" check passes.
    return {"is_open": False, "next_open": app.state.next_open, "next_close": app.state.next_open}


@app.get("/v2/assets")
def assets():
    # 1200 fake NASDAQ stocks (SAA, SAB, ...) plus SPY on ARCA, which the stock filter
    # must drop (fund name, wrong exchange). 1201 listings in total.
    names = [f"S{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(1200)]
    out = [{"symbol": s, "name": s + " Inc", "exchange": "NASDAQ", "tradable": True, "status": "active"}
           for s in names]
    return out + [{"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA",
                   "tradable": True, "status": "active"}]


@app.get("/v2/stocks/bars")
def bars(symbols: str, timeframe: str, start: str, end: str, feed: str = "iex", page_token: str = ""):
    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    # Free-plan rule: full-market (SIP) data newer than 15 minutes is refused with 403.
    if feed == "sip" and e > datetime.now(UTC) - timedelta(minutes=15):
        from fastapi.responses import JSONResponse
        return JSONResponse({"message": "subscription does not permit querying recent SIP data"}, 403)
    syms = symbols.split(",")
    half = max(len(syms) // 2, 1)
    # First request: return only the first half of the symbols plus a page token; the
    # request with page_token="p2" returns the rest.
    if not page_token and len(syms) > 1:          # force one page break to exercise pagination
        return {"bars": {x: _bars(x, timeframe, s, e) for x in syms[:half]}, "next_page_token": "p2"}
    rest = syms[half:] if page_token else syms
    return {"bars": {x: _bars(x, timeframe, s, e) for x in rest}, "next_page_token": None}


@app.get("/v1beta1/news")
def news(page_token: str = ""):
    now = datetime.now(UTC)
    # mk(i, symbols, headline): an article published i hours ago, with a 9-digit
    # (nanosecond) timestamp like the real feed.
    mk = lambda i, syms, h: {"id": i, "headline": h, "symbols": syms, "source": "benzinga", "summary": "",
                             "author": "a", "url": "", "images": [], "content": "",
                             "created_at": (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S.123456789Z"),
                             "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if page_token:
        return {"news": [mk(30, ["NVDA"], "NVDA older item")], "next_page_token": None}
    # Page 1 traps: messy whitespace in a headline, a two-ticker story, a 7-ticker market
    # wrap (must be ignored) and an untagged macro note (must be ignored).
    return {"news": [mk(1, ["NVDA"], "NVDA   raises\n guidance"), mk(2, ["AAPL", "MSFT"], "AAPL and MSFT sign deal"),
                     mk(3, list("ABCDEFG"), "Market wrap"), mk(4, [], "Macro note")], "next_page_token": "n2"}


async def _stream(ws: WebSocket, fake: bool):
    """Alpaca's websocket handshake: connected -> auth -> authenticated -> subscribe ->
    subscription confirmation -> data. `fake=True` imitates the /v2/test feed, which
    always trades the made-up symbol FAKEPACA."""
    await ws.accept()
    await ws.send_text(json.dumps([{"T": "success", "msg": "connected"}]))
    authed = False
    while True:
        m = json.loads(await ws.receive_text())
        if m.get("action") == "auth":
            if m.get("key") != "good":
                # 402 = auth failed; the app's client treats it as fatal.
                await ws.send_text(json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]))
                continue
            authed = True
            await ws.send_text(json.dumps([{"T": "success", "msg": "authenticated"}]))
        elif m.get("action") == "subscribe" and authed:
            syms = m.get("trades") or []
            if len(syms) > 30:                    # free-plan symbol limit
                await ws.send_text(json.dumps([{"T": "error", "code": 405, "msg": "symbol limit exceeded"}]))
                continue
            await ws.send_text(json.dumps([{"T": "subscription", "trades": syms, "quotes": [],
                                            "bars": m.get("bars") or []}]))
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000111222Z")
            sym = "FAKEPACA" if fake else syms[0]
            # One normal trade, one trade with condition "W" (must be skipped), one bar.
            await ws.send_text(json.dumps([
                {"T": "t", "S": sym, "i": 1, "x": "V", "p": 101.25, "s": 100, "c": ["@"], "z": "C", "t": ts},
                {"T": "t", "S": sym, "i": 2, "x": "V", "p": 55.0, "s": 5, "c": ["@", "W"], "z": "C", "t": ts},
                {"T": "b", "S": sym, "o": 101, "h": 102, "l": 100.5, "c": 101.5, "v": 4200, "t": ts}]))


@app.websocket("/v2/iex")
async def ws_iex(ws: WebSocket):
    try:
        await _stream(ws, False)
    except Exception:
        pass                                      # client disconnected: normal end


@app.websocket("/v2/test")
async def ws_test(ws: WebSocket):
    try:
        await _stream(ws, True)
    except Exception:
        pass


@app.get("/api/v1/calendar/earnings")
def earnings(token: str = ""):
    # NVDA reports in two days; "BAD" has no date and must be skipped by the client.
    d = (datetime.now(UTC).date() + timedelta(days=2)).isoformat()
    return {"earningsCalendar": [{"symbol": "NVDA", "date": d, "epsEstimate": 1.0, "hour": "amc"},
                                 {"symbol": "BAD", "date": None}]}


@app.get("/api/v1/stock/metric")
def metric(symbol: str):
    # Every company is profitable except one called "LOSS".
    return {"metric": {"netProfitMarginTTM": -3.1 if symbol == "LOSS" else 24.5, "epsTTM": 6.1}, "symbol": symbol}


def start() -> str:
    """Start the fake server in a daemon thread and return "127.0.0.1:<port>"."""
    # Ask the OS for a free port: bind to port 0, read which port was assigned, release it.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    # daemon=True: the thread dies with the test process, so no shutdown code is needed.
    threading.Thread(target=server.run, daemon=True).start()
    # Poll for up to ~5 s (100 x 0.05 s) until the server accepts connections.
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return f"127.0.0.1:{port}"
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("fake server did not start")
