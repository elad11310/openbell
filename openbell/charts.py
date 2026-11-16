"""Price charts for the dashboard: candles, volume and indicators for one symbol.

WHAT: `ChartService.chart(symbol, range_key)` returns everything the dashboard's chart
      panel draws for one symbol and one range ("1D", "5D", "1M", "6M", "1Y"), as plain
      JSON types:
        candles     OHLCV candles: 5-minute (1D), 15-minute (5D) or daily (1M/6M/1Y)
        ema_fast    EMA of the closes, same period as the swing strategy (20)
        ema_slow    EMA of the closes, same period as the swing strategy (50)
        vwap        session VWAP, intraday ranges only: one line per session, since
                    VWAP restarts at every open
        volume_ma   20-candle average volume
        base        the close just before the range (the reference for the change figure;
                    on 1D this is the previous session's close)
      `overlays(payload, engine, journal)` adds what the app itself knows about the
      symbol: trade markers (every entry and exit) and price levels (open position's
      entry/stop/target, today's swing trigger and opening range).
HOW:  * Intraday ranges fetch 1-minute bars for the last sessions, keep only the regular
        session (09:30-16:00 New York; Alpaca also returns pre- and after-market bars)
        and merge them into 5- or 15-minute candles aligned to the open. VWAP is computed
        on the 1-minute bars, as the engine does, and sampled at the end of each candle.
        One extra session is fetched in front so the EMAs are already warm at the first
        visible candle.
      * Daily ranges fetch ~120 extra calendar days before the range for the same reason
        (EMA50 needs 50 sessions before it has a value).
      * The data source is anything with AlpacaData's `bars()` signature: `AlpacaData`
        when live, `SyntheticMarket` under `sim --serve`.
      * Results are cached for 60 s (intraday) or 10 min (daily), so a chart left open
        (it refreshes every minute) does not turn into a stream of API requests.
WHY:  * Seeing the price action behind a scan row or a watchlist name is the quickest way
        to sanity-check what the scanner and the agent picked, and why a trade happened.
      * Live charts use the full-market (SIP) feed and therefore end 16 minutes ago: SIP
        is free only once it is 15 minutes old. The free IEX feed is real-time but carries
        only a few percent of the volume, which would make the volume bars misleading.
        The payload states the delay and the feed so the page can say so.
      * Display only: the engine never reads any of this, so charts cannot affect trading.
      * Time axis: the chart library draws timestamps as UTC. Intraday candle times are
        therefore sent as New York wall-clock time written as if it were UTC, so the axis
        reads 09:30-16:00 wherever the browser is. Daily candles use plain dates.
"""
from __future__ import annotations

import bisect
import re
import time as clock      # renamed: monotonic clock for the cache; datetime is used below
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .alpaca import FatalFeedError
from .config import Config
from .indicators import ema_series, sma_series, vwap_series
from .market_calendar import MarketCalendar, NY
from .models import Bar

# The free plan serves full-market (SIP) data once it is 15 minutes old; one spare minute.
SIP_DELAY = timedelta(minutes=16)
# What the API accepts as a ticker: capital letter first, then letters, digits, "." or
# "-" (BRK.B, BF-B). Anything else is refused before it reaches Alpaca.
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
CACHE_MAX = 200           # cached charts kept before the cache is emptied


@dataclass(frozen=True)
class Range:
    """One button on the chart panel."""
    bar_minutes: int      # candle size in minutes; 0 = daily candles
    sessions: int         # trading sessions shown
    ttl: int              # seconds a fetched chart is reused before refetching


RANGES = {
    "1D": Range(5, 1, 60),        # 78 candles on a full day
    "5D": Range(15, 5, 60),       # 26 a day, 130 in all
    "1M": Range(0, 21, 600),      # ~21 sessions in a month
    "6M": Range(0, 126, 600),
    "1Y": Range(0, 252, 600),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def chart_time(ts: datetime) -> int:
    """Intraday candle time as the chart expects it: the New York wall-clock time,
    written as if it were UTC, in epoch seconds. 14:30 UTC in winter becomes the epoch
    value of "09:30 UTC", which the library then displays as 09:30."""
    return int(ts.astimezone(NY).replace(tzinfo=timezone.utc).timestamp())


def chart_day(ts: datetime) -> str:
    """Daily candle time: the session date in New York, as "YYYY-MM-DD". Alpaca stamps
    daily bars at midnight New York, which is 04:00/05:00 UTC, so the UTC date would
    also work, but converting first is safe for any source."""
    return ts.astimezone(NY).date().isoformat()


class ChartService:
    """Fetches bars for one symbol and range, and computes the indicator lines."""

    def __init__(self, source, cfg: Config, cal: MarketCalendar,
                 now: Callable[[], datetime] = _utcnow, delay: timedelta = SIP_DELAY,
                 feed: Optional[str] = "sip") -> None:
        # source: object with `async bars(symbols, timeframe, start, end, feed=None)`.
        # now:    the "current" time; the simulation passes its last session's close.
        # delay:  how far behind `now` the data must end (0 for the simulation).
        # feed:   feed requested first; None = whatever the source uses by default.
        self.source, self.cfg, self.cal = source, cfg, cal
        self.now, self.delay, self.feed = now, delay, feed
        self._cache: dict = {}    # (symbol, range) -> (expiry on the monotonic clock, payload)

    async def chart(self, symbol: str, key: str) -> dict:
        """Candles + indicator lines for `symbol` over range `key` (a RANGES key; the
        API validates both). The returned dict is shared with the cache: do not modify
        it, copy it (as `overlays` does)."""
        rng = RANGES[key]
        hit = self._cache.get((symbol, key))
        if hit and hit[0] > clock.monotonic():
            return hit[1]
        asof = self.now() - self.delay        # the newest moment the data may cover
        if rng.bar_minutes:
            payload = await self._intraday(symbol, rng, asof)
        else:
            payload = await self._daily(symbol, rng, asof)
        payload.update(symbol=symbol, range=key, asof=asof.isoformat(),
                       delay_minutes=int(self.delay.total_seconds() // 60))
        if len(self._cache) >= CACHE_MAX:     # crude bound: start over rather than track age
            self._cache.clear()
        self._cache[(symbol, key)] = (clock.monotonic() + rng.ttl, payload)
        return payload

    async def _fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> tuple:
        """(bars, feed used). Asks for `self.feed` first; if the account may not read it
        (FatalFeedError = 401/403), falls back to the source's default feed, as the
        runner does for daily bars."""
        try:
            got = await self.source.bars([symbol], timeframe, start, end, feed=self.feed)
            used = self.feed
        except FatalFeedError:
            if self.feed is None:             # nothing to fall back to
                raise
            got = await self.source.bars([symbol], timeframe, start, end)
            used = self.cfg.feed.feed
        return got.get(symbol, []), used or "synthetic"

    async def _intraday(self, symbol: str, rng: Range, asof: datetime) -> dict:
        """5- or 15-minute candles for the last `rng.sessions` sessions that have data."""
        # Sessions that had opened by `asof`, newest last. Two more than shown: one warms
        # up the EMAs, and one covers a latest session without bars yet (its first 16
        # minutes, when the delayed feed has nothing for it).
        recent = self.cal.sessions_between(asof.date() - timedelta(days=14), asof.date())
        sessions = [s for s in recent if s.open <= asof][-(rng.sessions + 2):]
        if not sessions:
            return self._payload([], 0, [], None, "none", rng)
        minutes, used = await self._fetch(symbol, "1Min", sessions[0].open,
                                          min(asof, sessions[-1].close))
        by_day = {s.day: s for s in sessions}
        day_of = lambda b: b.ts.astimezone(NY).date()      # the bar's New York date

        # Keep the regular session only. Pre- and after-market bars are thin and would
        # distort VWAP, the candles and the volume scale.
        regular = []
        for b in minutes:
            s = by_day.get(day_of(b))
            if s and s.open <= b.ts < s.close:
                regular.append(b)

        # VWAP on the 1-minute bars (as the engine computes it), restarting each session.
        vw = vwap_series(regular, key=day_of)

        # Merge 1-minute bars into candles of `bar_minutes`. The candle a bar belongs to
        # starts at open + whole multiples of the candle size: (b.ts - open) // size is
        # the number of complete candles before the bar (timedelta // timedelta -> int).
        # Bars arrive oldest first, so the dict keeps candles in time order.
        size = timedelta(minutes=rng.bar_minutes)
        candles: dict = {}                    # start -> [o, h, l, c, v, vwap at candle end]
        for b, w in zip(regular, vw):
            s = by_day[day_of(b)]
            start = s.open + (b.ts - s.open) // size * size
            c = candles.get(start)
            if c is None:
                candles[start] = [b.o, b.h, b.l, b.c, b.v, w]
            else:                             # extend the candle with this minute
                c[1], c[2] = max(c[1], b.h), min(c[2], b.l)
                c[3], c[5] = b.c, w           # the latest close and VWAP
                c[4] += b.v
        bars = [Bar(symbol, t, *c[:5]) for t, c in candles.items()]
        vwaps = [c[5] for c in candles.values()]

        # Show the last `rng.sessions` session dates that have candles; the candles
        # before them were fetched only to warm up the EMAs.
        days = sorted({day_of(b) for b in bars})[-rng.sessions:]
        first = next((i for i, b in enumerate(bars) if day_of(b) >= days[0]), len(bars)) if days else 0
        # VWAP restarts every session, so it is sent as one line per session (the chart
        # would otherwise join yesterday's last value to today's first). `starts` holds
        # the index of each later visible session's first candle.
        starts = {i for i in range(first + 1, len(bars)) if day_of(bars[i]) != day_of(bars[i - 1])}
        return self._payload(bars, first, [chart_time(b.ts) for b in bars], vwaps, used, rng, starts)

    async def _daily(self, symbol: str, rng: Range, asof: datetime) -> dict:
        """Daily candles for the last `rng.sessions` sessions, plus warm-up history."""
        # sessions * 7/5 turns trading days into calendar days (5 trading days a week);
        # the extra 120 calendar days (~80 sessions) warm up EMA50 and the volume average.
        start = asof - timedelta(days=rng.sessions * 7 // 5 + 120)
        bars, used = await self._fetch(symbol, "1Day", start, asof)
        first = max(len(bars) - rng.sessions, 0)
        return self._payload(bars, first, [chart_day(b.ts) for b in bars], None, used, rng)

    def _payload(self, bars: list, first: int, times: list, vwaps: Optional[list],
                 feed: str, rng: Range, starts: frozenset = frozenset()) -> dict:
        """Indicators over ALL fetched bars (so they are warm), then only the visible
        part, bars[first:], is sent. `times` holds each bar's chart time; `starts` the
        indexes where a new session begins (VWAP is split there)."""
        sw = self.cfg.swing
        closes = [b.c for b in bars]
        ema_f, ema_s = ema_series(closes, sw.ema_fast), ema_series(closes, sw.ema_slow)
        vol_ma = sma_series([b.v for b in bars], 20)

        def line(values: list) -> list:
            # Visible points only, skipping the None warm-up gaps (a line series needs a
            # number at every point it draws; the line simply starts later).
            return [{"time": times[i], "value": round(values[i], 4)}
                    for i in range(first, len(bars)) if values[i] is not None]

        def per_session(values: list) -> list:
            # The same points as line(), cut into one list per session at `starts`.
            out: list = []
            for i in range(first, len(bars)):
                if i == first or i in starts:
                    out.append([])
                if values[i] is not None:
                    out[-1].append({"time": times[i], "value": round(values[i], 4)})
            return out

        shown = range(first, len(bars))
        return {
            "intraday": vwaps is not None, "bar_minutes": rng.bar_minutes, "feed": feed,
            "ema_periods": [sw.ema_fast, sw.ema_slow],
            "candles": [{"time": times[i], "open": round(bars[i].o, 4), "high": round(bars[i].h, 4),
                         "low": round(bars[i].l, 4), "close": round(bars[i].c, 4),
                         "volume": int(bars[i].v)} for i in shown],
            "ema_fast": line(ema_f), "ema_slow": line(ema_s), "volume_ma": line(vol_ma),
            # A list of lines, one per session (empty on daily ranges).
            "vwap": per_session(vwaps) if vwaps is not None else [],
            # The close before the first visible candle; without one, the first open.
            "base": round(bars[first - 1].c, 4) if 0 < first <= len(bars)
                    else (round(bars[0].o, 4) if bars else None),
        }


def overlays(payload: dict, engine, journal) -> dict:
    """A copy of `payload` with the app's own annotations for this symbol:
         markers  an arrow per trade entry ("buy") and exit ("sell"), placed on the
                  candle that contains the fill
         levels   horizontal price lines: the open position's entry, stop and target;
                  the swing trigger; today's opening range (intraday ranges)
    Computed on every request (cheap) because positions and stops change live, while
    the candles underneath come from the cache."""
    out = dict(payload, markers=[], levels=[])   # copy: the cached payload stays clean
    sym, candles = payload["symbol"], payload["candles"]
    if not candles:
        return out
    times = [c["time"] for c in candles]
    intraday, span = payload["intraday"], payload["bar_minutes"] * 60

    def place(ts_iso: str):
        """The time of the candle holding `ts_iso`, or None if it is off the chart.
        bisect_right finds the first candle starting AFTER the fill; the one before it
        is the candle the fill happened in."""
        ts = datetime.fromisoformat(ts_iso)
        k = chart_time(ts) if intraday else chart_day(ts)
        i = bisect.bisect_right(times, k) - 1
        if i < 0:
            return None                       # before the first candle
        if intraday and k >= times[i] + span:
            return None                       # after the last candle (e.g. delayed data)
        if not intraday and k != times[i]:
            return None                       # a day the chart has no candle for
        return times[i]

    def mark(ts_iso: Optional[str], side: str, text: str) -> None:
        t = place(ts_iso) if ts_iso else None
        if t is not None:
            out["markers"].append({"time": t, "side": side, "text": text})

    for t in journal.rows("SELECT strategy, qty, entry_ts, exit_ts, r_multiple FROM trades "
                          "WHERE symbol=? ORDER BY id", (sym,)):
        mark(t["entry_ts"], "buy", f"{t['strategy']} {t['qty']}")
        mark(t["exit_ts"], "sell", f"{t['r_multiple']:+.1f}R")
    p = engine.positions.get(sym)
    if p:
        mark(p.entry_ts.isoformat(), "buy", f"{p.strategy} {p.qty}")
        out["levels"] += [{"price": p.entry_px, "kind": "entry", "label": "entry"},
                          {"price": p.stop, "kind": "stop", "label": "stop"}]
        if p.target:
            out["levels"].append({"price": p.target, "kind": "target", "label": "target"})
    c = engine.ctx.get(sym)
    # A trigger matters only for a tradable name that is not already held. engine.ctx
    # also covers the benchmark (SPY), which is never traded, hence the watchlist check.
    if c and c.swing_trigger and not p and sym in engine.symbols:
        out["levels"].append({"price": c.swing_trigger, "kind": "trigger", "label": "swing trigger"})
    st = engine.state.get(sym)
    if intraday and st and st.or_bars:        # opening range exists once its first bar arrived
        out["levels"] += [{"price": st.or_high, "kind": "range", "label": "range high"},
                          {"price": st.or_low, "kind": "range", "label": "range low"}]
    # The chart library requires markers in time order.
    out["markers"].sort(key=lambda m: m["time"])
    return out
