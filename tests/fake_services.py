"""A local stand-in for Alpaca and Finnhub, speaking the protocol as the app understands it.
It proves the client code is self-consistent; only `openbell doctor` proves it against the real thing."""
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
    step = timedelta(days=1) if tf == "1Day" else timedelta(minutes=1)
    t = start.replace(second=0, microsecond=0)
    if tf == "1Day":
        t = t.replace(hour=5, minute=0)
    out, px = [], 100.0 + len(sym)
    while t <= end and len(out) < 400:
        if tf != "1Day" or t.weekday() < 5:
            out.append({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": px, "h": px + 1, "l": px - 1,
                        "c": px + 0.5, "v": 1000000 if tf == "1Day" else 5000, "n": 10, "vw": px})
            px += 0.1
        t += step
    return out


@app.get("/v2/clock")
def clock(request: Request):
    if request.headers.get("apca-api-key-id") != "good":
        return {"message": "forbidden."}
    return {"is_open": False, "next_open": app.state.next_open, "next_close": app.state.next_open}


@app.get("/v2/assets")
def assets():
    names = [f"S{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(1200)]
    out = [{"symbol": s, "name": s + " Inc", "exchange": "NASDAQ", "tradable": True, "status": "active"}
           for s in names]
    return out + [{"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA",
                   "tradable": True, "status": "active"}]


@app.get("/v2/stocks/bars")
def bars(symbols: str, timeframe: str, start: str, end: str, feed: str = "iex", page_token: str = ""):
    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if feed == "sip" and e > datetime.now(UTC) - timedelta(minutes=15):
        from fastapi.responses import JSONResponse
        return JSONResponse({"message": "subscription does not permit querying recent SIP data"}, 403)
    syms = symbols.split(",")
    half = max(len(syms) // 2, 1)
    if not page_token and len(syms) > 1:          # force one page break to exercise pagination
        return {"bars": {x: _bars(x, timeframe, s, e) for x in syms[:half]}, "next_page_token": "p2"}
    rest = syms[half:] if page_token else syms
    return {"bars": {x: _bars(x, timeframe, s, e) for x in rest}, "next_page_token": None}


@app.get("/v1beta1/news")
def news(page_token: str = ""):
    now = datetime.now(UTC)
    mk = lambda i, syms, h: {"id": i, "headline": h, "symbols": syms, "source": "benzinga", "summary": "",
                             "author": "a", "url": "", "images": [], "content": "",
                             "created_at": (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S.123456789Z"),
                             "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if page_token:
        return {"news": [mk(30, ["NVDA"], "NVDA older item")], "next_page_token": None}
    return {"news": [mk(1, ["NVDA"], "NVDA   raises\n guidance"), mk(2, ["AAPL", "MSFT"], "AAPL and MSFT sign deal"),
                     mk(3, list("ABCDEFG"), "Market wrap"), mk(4, [], "Macro note")], "next_page_token": "n2"}


async def _stream(ws: WebSocket, fake: bool):
    await ws.accept()
    await ws.send_text(json.dumps([{"T": "success", "msg": "connected"}]))
    authed = False
    while True:
        m = json.loads(await ws.receive_text())
        if m.get("action") == "auth":
            if m.get("key") != "good":
                await ws.send_text(json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]))
                continue
            authed = True
            await ws.send_text(json.dumps([{"T": "success", "msg": "authenticated"}]))
        elif m.get("action") == "subscribe" and authed:
            syms = m.get("trades") or []
            if len(syms) > 30:
                await ws.send_text(json.dumps([{"T": "error", "code": 405, "msg": "symbol limit exceeded"}]))
                continue
            await ws.send_text(json.dumps([{"T": "subscription", "trades": syms, "quotes": [],
                                            "bars": m.get("bars") or []}]))
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000111222Z")
            sym = "FAKEPACA" if fake else syms[0]
            await ws.send_text(json.dumps([
                {"T": "t", "S": sym, "i": 1, "x": "V", "p": 101.25, "s": 100, "c": ["@"], "z": "C", "t": ts},
                {"T": "t", "S": sym, "i": 2, "x": "V", "p": 55.0, "s": 5, "c": ["@", "W"], "z": "C", "t": ts},
                {"T": "b", "S": sym, "o": 101, "h": 102, "l": 100.5, "c": 101.5, "v": 4200, "t": ts}]))


@app.websocket("/v2/iex")
async def ws_iex(ws: WebSocket):
    try:
        await _stream(ws, False)
    except Exception:
        pass


@app.websocket("/v2/test")
async def ws_test(ws: WebSocket):
    try:
        await _stream(ws, True)
    except Exception:
        pass


@app.get("/api/v1/calendar/earnings")
def earnings(token: str = ""):
    d = (datetime.now(UTC).date() + timedelta(days=2)).isoformat()
    return {"earningsCalendar": [{"symbol": "NVDA", "date": d, "epsEstimate": 1.0, "hour": "amc"},
                                 {"symbol": "BAD", "date": None}]}


@app.get("/api/v1/stock/metric")
def metric(symbol: str):
    return {"metric": {"netProfitMarginTTM": -3.1 if symbol == "LOSS" else 24.5, "epsTTM": 6.1}, "symbol": symbol}


def start() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return f"127.0.0.1:{port}"
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("fake server did not start")
