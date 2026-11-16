"""Live prices for more names than the websocket allows: stream some, poll the rest.

WHAT: * `plan_feeds` splits today's names into the websocket set (the benchmark plus up to
        `feed.stream_limit` names) and the polled set.
      * `Poller` fetches the polled names' latest trades every `feed.poll_seconds` and
        their finished 1-minute bars once a minute, over REST, and yields them as Tick and
        Bar objects, just like `AlpacaData.stream`.
      * `completed_until` gives the end time that takes every finished minute bar and
        leaves out the one still forming.
HOW:  The runner keeps two lists, `streamed` and `polled`. The stream and the poller
      re-read them all the time, so moving a name from one list to the other moves its
      data source. Every second the runner calls `plan_feeds` again and applies any change:
        tier 2  an open position: its stop and target need every tick, so it streams;
        tier 1  a name that can still enter today (ORB attempt unused before the cutoff,
                or an untried swing trigger);
        tier 0  everything else: only shown on the dashboard.
      Within a tier, names already streaming keep their place (so names do not flip back
      and forth), then the scanner's rank decides.
WHY:  * The free Alpaca websocket takes 30 symbols. The REST API has no symbol cap: one
        "latest trades" request covers up to 200 names, and the free plan allows about
        200 requests a minute. At the default 5 s that is 12 requests a minute for trades
        plus up to 4 for bars, whatever the number of polled names.
      * The cost of polling is delay: a polled price is up to `poll_seconds` old, and a
        polled minute bar arrives a few seconds after the minute ends. The strategies
        decide on 1-minute bars (ORB) or a price crossing a level set the night before
        (SWING), so a few seconds rarely matter for an entry. They do matter for a stop,
        which is why a name moves to the stream as soon as it becomes a position.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Callable, Optional

from .alpaca import FatalFeedError

log = logging.getLogger("openbell.feeds")
EPOCH = datetime.min.replace(tzinfo=timezone.utc)   # "never" for the last-seen dictionaries
# Each bar request re-reads the last 5 minutes, so a bar Alpaca publishes late, or one
# missed by a failed poll, is still picked up. The runner drops bars it already has.
BAR_LOOKBACK = timedelta(minutes=5)
# Alpaca publishes a minute bar a few seconds after the minute ends. During the first 20 s
# of each minute the poller asks again on every round until the bar is there; after that,
# once a minute is enough.
BAR_RETRY = timedelta(seconds=20)


def completed_until(now: datetime) -> datetime:
    """End time for a minute-bar request that includes every finished bar and excludes
    the one still forming: one second before the current minute began. (Bars are stamped
    with their start time and a request's end is inclusive, so ending at hh:mm:00 could
    return the half-built hh:mm bar.)"""
    return now.replace(second=0, microsecond=0) - timedelta(seconds=1)


def plan_feeds(engine, names: list, benchmark: str, limit: int, current=()) -> tuple:
    """Split `names` (today's watchlist in rank order) into (streamed, polled).

    `streamed` is the benchmark plus at most `limit` names; `polled` is everything else.
    Open positions always come first, even if they are not on the watchlist. Both lists
    keep the rank order of `names`. `current` is what streams now: within a tier, those
    names keep their place, so a swap happens only when a polled name needs real-time
    prices more than a streamed one (see the module docstring for the tiers)."""
    names = [s for s in dict.fromkeys(list(engine.positions) + list(names)) if s != benchmark]
    rank = {s: i for i, s in enumerate(names)}
    cur = set(current)

    def tier(s: str) -> int:
        return 2 if s in engine.positions else 1 if engine.can_still_enter(s) else 0

    # Sort keys: higher tier first, then names already streaming, then scanner rank.
    best = set(sorted(names, key=lambda s: (-tier(s), s not in cur, rank[s]))[:limit])
    return [benchmark] + [s for s in names if s in best], [s for s in names if s not in best]


class Poller:
    """REST polling for the names that do not fit on the websocket.

    `symbols` is the runner's `polled` list, read again on every round. `last_ok` is the
    monotonic time of the last successful round (or of an idle round with nothing to
    poll); the runner blocks entries when it is older than `feed.stale_seconds`, as it
    does for a silent stream."""

    def __init__(self, data, symbols: list, every: float,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self.data, self.symbols, self.every = data, symbols, every
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.last_ok = time.monotonic()
        self._trade_ts: dict = {}            # symbol -> time of the last trade yielded
        self._bars_for: Optional[datetime] = None   # minute whose bars were last fetched

    async def run(self, until: datetime) -> AsyncIterator:
        """Yield Tick and Bar objects until `until`, one round every `every` seconds.
        Errors are logged and the wait doubles (up to 60 s) until a round succeeds;
        a FatalFeedError (bad keys, refused feed) stops the app, as in the stream."""
        wait = self.every
        while self.now() < until:
            names = list(self.symbols)
            events: list = []
            if names:
                try:
                    events = await self.poll(names)
                    wait = self.every
                except FatalFeedError:
                    raise
                except Exception as e:
                    wait = min(wait * 2, 60)
                    log.warning("polling failed (%s: %s); next try in %.0fs", type(e).__name__, e, wait)
                    events = None
            if events is not None:
                self.last_ok = time.monotonic()
            # Yield outside the try, so an error in the consumer is never taken for a
            # polling failure.
            for ev in events or ():
                yield ev
            await asyncio.sleep(wait)

    async def poll(self, names: list) -> list:
        """One round: finished minute bars when due, then the latest trades, oldest first.

        Only trades newer than the last one yielded for that symbol come back, so a quiet
        name does not repeat the same print every round. Bars are not filtered here: the
        runner drops a bar it already has, whichever source sent it first."""
        trades = await self.data.latest_trades(names)
        now = self.now()
        minute = now.replace(second=0, microsecond=0)
        bars: dict = {}
        if minute != self._bars_for or now - minute < BAR_RETRY:
            bars = await self.data.bars(names, "1Min", minute - BAR_LOOKBACK, completed_until(now))
            self._bars_for = minute
        out = sorted((b for rows in bars.values() for b in rows), key=lambda b: (b.ts, b.symbol))
        fresh = [t for s, t in trades.items() if t.ts > self._trade_ts.get(s, EPOCH)]
        for t in fresh:
            self._trade_ts[t.symbol] = t.ts
        return out + sorted(fresh, key=lambda t: (t.ts, t.symbol))
