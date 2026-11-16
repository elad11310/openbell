"""Alpaca market data: REST for history, websocket for the live stream.

Only market data is used. Orders are filled by the engine's own paper broker
against these live prices, so no order ever leaves this process.

WHAT: `AlpacaData` provides:
        bars()       historical OHLCV bars (daily or 1-minute), all pages
        assets()     the list of every US-listed security (for the scanner universe)
        news_all()   recent headlines, grouped by ticker
        premarket()  today's pre-market price and dollar volume per symbol
        latest_trades()  each symbol's most recent trade, for names polled instead of streamed
        stream()     an async generator of live Tick and Bar objects, reconnecting on drops;
                     its symbol list can change while it runs (subscribe / unsubscribe)
      `FatalFeedError` signals problems that retrying cannot fix (bad keys, plan
      limits); `parse_ts` converts Alpaca's timestamps.
HOW:  `httpx.AsyncClient` for REST and the `websockets` library for streaming, both
      async so the trading loop never blocks while waiting on the network. Endpoint URLs
      are module-level constants so the tests can point them at a local fake server.
WHY:  * Data feeds: "iex" is free and real-time but covers only trades on the IEX
        exchange (a small share of volume); "sip" is the whole market, real-time only on
        a paid plan, but free once the data is 15+ minutes old. That is why history and
        pre-market requests pass feed="sip" and keep their end time 15+ minutes in the past.
      * Error classes: a 401/403 means the keys are wrong and retrying will not help, so
        it raises FatalFeedError and the app stops with a clear message. Network drops
        are temporary, so the stream reconnects with exponential backoff instead.
"""
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
LATEST_TRADES_URL = "https://data.alpaca.markets/v2/stocks/trades/latest"
ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"   # {feed} -> "iex" or "sip"
# Trade conditions that do not represent a regular, in-sequence last sale
# (average-price, out-of-sequence, extended-hours, derivatively priced, ...).
# Such prints can be far from the real market price (e.g. a late report of a trade made
# an hour ago) and would trigger stops falsely, so the stream drops them.
SKIP_CONDITIONS = {"C", "N", "R", "W", "Z", "T", "U", "4", "H", "V", "7"}


class FatalFeedError(RuntimeError):
    """An error retrying cannot fix (bad keys, refused subscription). Stops the app."""
    pass


def parse_ts(s: str) -> datetime:
    """RFC 3339 with up to nanosecond precision -> aware UTC datetime.

    Alpaca sends timestamps like "2026-11-10T14:30:00.123456789Z". Python's
    `fromisoformat` handles at most 6 fractional digits (microseconds) and, in older
    versions, no "Z". So:
      1. replace "Z" with "+00:00";
      2. split off the fraction, cut it to 6 digits (or pad it with zeros), and
         reattach the 6-character time-zone suffix;
      3. parse and convert to UTC."""
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, rest = s.split(".", 1)
        frac, tz = rest[:-6], rest[-6:]          # "+00:00" is always 6 characters
        s = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class AlpacaData:
    def __init__(self, key: str, secret: str, feed: str = "iex") -> None:
        # Fail at construction, before any trading starts, if the keys are missing.
        if not key or not secret:
            raise FatalFeedError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.key, self.secret, self.feed = key, secret, feed

    async def bars(self, symbols: list, timeframe: str, start: datetime, end: datetime,
                   feed: Optional[str] = None) -> dict:
        """Historical bars per symbol, oldest first, all pages."""
        # Pre-fill every requested symbol with [] so callers can index any of them, even
        # ones with no data.
        out: dict = {s: [] for s in symbols}
        params = {"symbols": ",".join(symbols), "timeframe": timeframe,
                  "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  # adjustment="all": prices adjusted for splits and dividends, so a 2:1
                  # split does not look like a 50% crash to the indicators.
                  "limit": 10000, "adjustment": "all", "sort": "asc",
                  "feed": feed or self.feed}
        async with httpx.AsyncClient(timeout=30, headers=self.headers) as c:
            # Pagination: Alpaca returns up to `limit` bars per response plus a
            # next_page_token while more remain. Keep requesting with the token until
            # it is absent. One symbol's bars can span several pages, hence `extend`.
            while True:
                r = await c.get(DATA_URL, params=params)
                if r.status_code in (401, 403):
                    raise FatalFeedError(f"Alpaca rejected the request ({r.status_code}): {r.text[:200]}")
                if r.status_code == 429:             # rate limited: wait, then retry the same page
                    await asyncio.sleep(3)
                    continue
                r.raise_for_status()                 # any other HTTP error -> exception
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
        in the window. One bulk pull of the whole feed. None if news is unavailable.

        Why one bulk pull: asking for news symbol by symbol would take 400 requests;
        reading the whole feed for the window (up to max_pages x 50 articles) and
        filtering locally takes a few."""
        out: dict = {}
        params = {"limit": 50, "sort": "desc",
                  "start": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        try:
            async with httpx.AsyncClient(timeout=30, headers=self.headers) as c:
                for _ in range(max_pages):           # hard cap on requests
                    r = await c.get(NEWS_URL, params=params)
                    r.raise_for_status()
                    body = r.json()
                    for n in body.get("news") or []:
                        tagged = n.get("symbols") or []
                        if not tagged or len(tagged) > 5:   # market round-ups say little about one name
                            continue
                        ts = parse_ts(n["created_at"])
                        # Collapse whitespace/newlines and cap the length for the prompt.
                        head = " ".join(str(n.get("headline", "")).split())[:180]
                        # Attach the headline to every wanted symbol it is tagged with.
                        for s in wanted.intersection(tagged):
                            out.setdefault(s, []).append((ts, head))
                    token = body.get("next_page_token")
                    if not token:
                        break
                    params["page_token"] = token
            return out
        except Exception as e:
            # News is optional: the scan continues without it and logs that it did.
            log.warning("news unavailable (%s); scanning without headlines", type(e).__name__)
            return None

    async def premarket(self, symbols: list, start: datetime, now: datetime) -> Optional[dict]:
        """symbol -> {"price", "dollar_vol"} from today's pre-market minute bars on the
        full-market feed, which the free plan serves once it is 15 minutes old.
        None if that data is unavailable."""
        # 16 minutes, not 15, as a safety margin on the free plan's 15-minute delay.
        end = now - timedelta(minutes=16)
        if end <= start:                     # pre-market has not been running long enough yet
            return None
        out: dict = {}
        try:
            for i in range(0, len(symbols), 200):
                bars = await self.bars(symbols[i:i + 200], "1Min", start, end, feed="sip")
                for sym, rows in bars.items():
                    if rows:
                        # Latest pre-market price, and total dollar volume traded so far.
                        out[sym] = {"price": rows[-1].c, "dollar_vol": sum(b.c * b.v for b in rows)}
        except Exception as e:
            log.warning("pre-market data unavailable (%s)", type(e).__name__)
            return None
        return out or None                   # an empty result also counts as "unavailable"

    async def latest_trades(self, symbols: list) -> dict:
        """symbol -> Tick of its most recent trade on the configured feed.

        One request covers up to 200 symbols, which is what makes polling affordable: the
        names that do not fit on the websocket cost one request per poll, all together.
        A trade with a condition in SKIP_CONDITIONS is left out (the symbol is then
        missing from the result), exactly as the stream drops it."""
        out: dict = {}
        async with httpx.AsyncClient(timeout=10, headers=self.headers) as c:
            for i in range(0, len(symbols), 200):
                r = await c.get(LATEST_TRADES_URL, params={"symbols": ",".join(symbols[i:i + 200]),
                                                           "feed": self.feed})
                if r.status_code in (401, 403):
                    raise FatalFeedError(f"Alpaca rejected the request ({r.status_code}): {r.text[:200]}")
                # Anything else, a 429 (rate limit) included, raises an ordinary error; the
                # poller waits longer and tries again.
                r.raise_for_status()
                for sym, m in (r.json().get("trades") or {}).items():
                    if SKIP_CONDITIONS.intersection(m.get("c") or []):
                        continue
                    out[sym] = Tick(sym, parse_ts(m["t"]), float(m["p"]), float(m.get("s", 0)))
        return out

    async def stream(self, symbols: list, until: datetime) -> AsyncIterator:
        """Yield Tick and Bar objects until `until`. Reconnects with backoff.

        This is an async generator: callers write `async for ev in data.stream(...)` and
        each `yield` hands them one event while the connection stays open.

        `symbols` is read again before every message, so the caller may change the list
        while the stream runs. The difference is sent as "unsubscribe" / "subscribe"
        actions on the open connection; the runner uses this to move names between the
        stream and REST polling during the session (see feeds.py).

        Alpaca protocol, in order:
          1. connect; the server sends [{"T":"success","msg":"connected"}]
          2. we send {"action":"auth", key, secret}
          3. server: [{"T":"success","msg":"authenticated"}] -> we send "subscribe"
          4. server sends batches (JSON arrays) of messages:
               T="t" trade  -> Tick     T="b" minute bar -> Bar     T="error" -> see below
        Outer loop = reconnect loop; inner loop = read messages from one connection."""
        url = STREAM_URL.format(feed=self.feed)
        delay = 1.0                          # reconnect backoff, doubles up to 30 s
        while datetime.now(timezone.utc) < until:
            # What this connection is subscribed to. A new connection starts with nothing.
            subscribed: set = set()
            authed = False
            try:
                # ping_interval/ping_timeout: the library sends keep-alive pings and drops
                # the connection if the server stops answering, so a dead link is noticed.
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    while datetime.now(timezone.utc) < until:
                        if authed:
                            await self._resubscribe(ws, symbols, subscribed)
                        try:
                            # Wait at most 5 s so the `until` check above runs regularly
                            # even when no messages arrive (quiet market).
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue
                        for m in json.loads(raw):
                            kind = m.get("T")
                            if kind == "t":
                                # "c" is the list of trade condition codes.
                                if SKIP_CONDITIONS.intersection(m.get("c") or []):
                                    continue
                                yield Tick(m["S"], parse_ts(m["t"]), float(m["p"]), float(m.get("s", 0)))
                            elif kind == "b":
                                yield Bar(m["S"], parse_ts(m["t"]), m["o"], m["h"], m["l"], m["c"], m["v"])
                            elif kind == "error":
                                code = m.get("code")
                                # Not fixable by reconnecting: bad auth (401/402),
                                # plan forbids it (404/409) -> stop the app.
                                if code in (401, 402, 404, 409):
                                    raise FatalFeedError(f"stream error {code}: {m.get('msg')}")
                                log.warning("stream error %s: %s", code, m.get("msg"))
                                if code in (405, 406):      # symbol / connection limit
                                    # Raised as an ordinary error so the outer handler
                                    # backs off and reconnects.
                                    raise ConnectionError(m.get("msg"))
                            elif kind == "success" and m.get("msg") == "authenticated":
                                delay = 1.0          # a good connection resets the backoff
                                authed = True
                                await self._resubscribe(ws, symbols, subscribed)
                                log.info("stream authenticated (%s feed)", self.feed)
            except FatalFeedError:
                raise
            except Exception as e:  # network drop, server close, bad frame
                # Exponential backoff (1, 2, 4, ... 30 s): avoids hammering a struggling
                # server while still recovering quickly from a short blip.
                log.warning("stream disconnected (%s); reconnecting in %.0fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    @staticmethod
    async def _resubscribe(ws, symbols: list, subscribed: set) -> None:
        """Bring the connection's subscription in line with `symbols`, updating
        `subscribed` in place. Unsubscribes first, so the plan's symbol cap is never
        exceeded for a moment while names are swapped. Sends nothing if nothing changed."""
        want = set(symbols)
        # `new` keeps the caller's order (benchmark and positions first); dict.fromkeys
        # drops duplicates.
        gone, new = sorted(subscribed - want), [s for s in dict.fromkeys(symbols) if s not in subscribed]
        if gone:
            await ws.send(json.dumps({"action": "unsubscribe", "trades": gone, "bars": gone}))
        if new:
            await ws.send(json.dumps({"action": "subscribe", "trades": new, "bars": new}))
        if subscribed and (gone or new):
            log.info("stream symbols changed: +%s -%s", new, gone)
        subscribed.clear()
        subscribed.update(want)
