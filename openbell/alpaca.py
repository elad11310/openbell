"""Alpaca market data: REST for history, websocket for the live stream.

Only market data is used. Orders are filled by the engine's own paper broker
against these live prices, so no order ever leaves this process."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

import httpx
import websockets

from .models import Bar, Tick

log = logging.getLogger("openbell.alpaca")
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
# Trade conditions that do not represent a regular, in-sequence last sale
# (average-price, out-of-sequence, extended-hours, derivatively priced, ...).
SKIP_CONDITIONS = {"C", "N", "R", "W", "Z", "T", "U", "4", "H", "V", "7"}


class FatalFeedError(RuntimeError):
    pass


def parse_ts(s: str) -> datetime:
    """RFC 3339 with up to nanosecond precision -> aware UTC datetime."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        frac, tz = rest[:-6], rest[-6:]
        s = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class AlpacaData:
    def __init__(self, key: str, secret: str, feed: str = "iex") -> None:
        if not key or not secret:
            raise FatalFeedError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.key, self.secret, self.feed = key, secret, feed

    async def bars(self, symbols: list, timeframe: str, start: datetime, end: datetime,
                   feed: Optional[str] = None) -> dict:
        """Historical bars per symbol, oldest first, all pages."""
        out: dict = {s: [] for s in symbols}
        params = {"symbols": ",".join(symbols), "timeframe": timeframe,
                  "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": 10000, "adjustment": "all", "sort": "asc",
                  "feed": feed or self.feed}
        async with httpx.AsyncClient(timeout=30, headers=self.headers) as c:
            while True:
                r = await c.get(DATA_URL, params=params)
                if r.status_code in (401, 403):
                    raise FatalFeedError(f"Alpaca rejected the request ({r.status_code}): {r.text[:200]}")
                if r.status_code == 429:
                    await asyncio.sleep(3)
                    continue
                r.raise_for_status()
                body = r.json()
                for sym, rows in (body.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(
                        Bar(sym, parse_ts(x["t"]), x["o"], x["h"], x["l"], x["c"], x["v"])
                        for x in rows)
                token = body.get("next_page_token")
                if not token:
                    return out
                params["page_token"] = token

    async def assets(self) -> list:
        """All active US equity listings (dicts with symbol, name, exchange, tradable, status)."""
        async with httpx.AsyncClient(timeout=60, headers=self.headers) as c:
            r = await c.get(ASSETS_URL, params={"status": "active", "asset_class": "us_equity"})
            if r.status_code in (401, 403):
                raise FatalFeedError(f"Alpaca rejected the request ({r.status_code}): {r.text[:200]}")
            r.raise_for_status()
            return r.json()

    async def news_all(self, wanted: set, start: datetime, end: datetime,
                       max_pages: int = 40) -> Optional[dict]:
        """symbol -> [(timestamp, headline)] newest first, for every wanted symbol mentioned
        in the window. One bulk pull of the whole feed. None if news is unavailable."""
        out: dict = {}
        params = {"limit": 50, "sort": "desc",
                  "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        try:
            async with httpx.AsyncClient(timeout=30, headers=self.headers) as c:
                for _ in range(max_pages):
                    r = await c.get(NEWS_URL, params=params)
                    r.raise_for_status()
                    body = r.json()
                    for n in body.get("news") or []:
                        tagged = n.get("symbols") or []
                        if not tagged or len(tagged) > 5:   # market round-ups say little about one name
                            continue
                        ts = parse_ts(n["created_at"])
                        head = " ".join(str(n.get("headline", "")).split())[:180]
                        for s in wanted.intersection(tagged):
                            out.setdefault(s, []).append((ts, head))
                    token = body.get("next_page_token")
                    if not token:
                        break
                    params["page_token"] = token
            return out
        except Exception as e:
            log.warning("news unavailable (%s); scanning without headlines", type(e).__name__)
            return None

    async def premarket(self, symbols: list, start: datetime, now: datetime) -> Optional[dict]:
        """symbol -> {"price", "dollar_vol"} from today's pre-market minute bars on the
        full-market feed, which the free plan serves once it is 15 minutes old.
        None if that data is unavailable."""
        end = now - timedelta(minutes=16)
        if end <= start:
            return None
        out: dict = {}
        try:
            for i in range(0, len(symbols), 200):
                bars = await self.bars(symbols[i:i + 200], "1Min", start, end, feed="sip")
                for sym, rows in bars.items():
                    if rows:
                        out[sym] = {"price": rows[-1].c, "dollar_vol": sum(b.c * b.v for b in rows)}
        except Exception as e:
            log.warning("pre-market data unavailable (%s)", type(e).__name__)
            return None
        return out or None

    async def stream(self, symbols: list, until: datetime) -> AsyncIterator:
        """Yield Tick and Bar objects until `until`. Reconnects with backoff."""
        url = STREAM_URL.format(feed=self.feed)
        delay = 1.0
        while datetime.now(timezone.utc) < until:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    while datetime.now(timezone.utc) < until:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        for m in json.loads(raw):
                            kind = m.get("T")
                            if kind == "t":
                                if SKIP_CONDITIONS.intersection(m.get("c") or []):
                                    continue
                                yield Tick(m["S"], parse_ts(m["t"]), float(m["p"]), float(m.get("s", 0)))
                            elif kind == "b":
                                yield Bar(m["S"], parse_ts(m["t"]), m["o"], m["h"], m["l"], m["c"], m["v"])
                            elif kind == "error":
                                code = m.get("code")
                                if code in (401, 402, 404, 409):
                                    raise FatalFeedError(f"stream error {code}: {m.get('msg')}")
                                log.warning("stream error %s: %s", code, m.get("msg"))
                                if code in (405, 406):      # symbol / connection limit
                                    raise ConnectionError(m.get("msg"))
                            elif kind == "success" and m.get("msg") == "authenticated":
                                delay = 1.0
                                await ws.send(json.dumps({"action": "subscribe",
                                                          "trades": symbols, "bars": symbols}))
                                log.info("stream authenticated (%s feed)", self.feed)
            except FatalFeedError:
                raise
            except Exception as e:  # network drop, server close, bad frame
                log.warning("stream disconnected (%s); reconnecting in %.0fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
