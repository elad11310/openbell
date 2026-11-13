"""Live loop: wait for the next NYSE session, prepare, trade it, review it, repeat.

WHAT: `LiveRunner` is the conductor of `python -m openbell run`. It owns one of each
      component (Journal, Engine, MarketCalendar, AlpacaData, Agent, Finnhub) and walks
      them through every trading day:
        1. sleep until 25 minutes before the open           (run)
        2. scan for today's watchlist                        (build_watchlist)
        3. build daily contexts, ask the agent for a plan,
           start the engine's day                            (prepare)
        4. stream live ticks and bars into the engine until
           the close; watch for a silent feed                (trade)
        5. close the day's books and ask for a review        (after_close)
      The read-only dashboard server runs alongside, on the same event loop.
HOW:  Everything is asyncio: one thread, many cooperating tasks. While one task waits
      on the network (`await`), the others run. The blocking Claude API calls are pushed
      to a worker thread with `asyncio.to_thread` so they cannot freeze the price loop.
WHY:  * Separation of duties: the runner does all the I/O and timing; the engine stays a
        pure event processor that the replay and the simulation can drive identically.
      * Resilience: a failure inside one day (an API outage, a bug) is logged and retried
        after 30 s instead of killing the process. Only a FatalFeedError (bad keys),
        which a retry cannot fix, stops the app.
      * Restart safety: if the process starts mid-session, `prepare` rebuilds VWAP and
        the opening range from today's bars, and the engine restores positions and
        counters from the journal, so trading resumes as if nothing happened.
"""
from __future__ import annotations

import asyncio
import logging
import time as clock      # renamed: `time` below is datetime.time (a time of day)
from datetime import datetime, time, timedelta, timezone

import uvicorn

from .agent import Agent
from .alpaca import AlpacaData, FatalFeedError
from .api import create_app
from .config import Config
from .engine import Engine
from .journal import Journal
from .market_calendar import MarketCalendar, NY
from .models import Tick, Session
from . import scanner
from .finnhub import Finnhub
from .replay import contexts_for

log = logging.getLogger("openbell.runner")


def utcnow() -> datetime:
    """Current time in UTC. A module-level function so tests can replace it with a fixed
    time (monkeypatch) and run the pre-market flow as if it were any moment."""
    return datetime.now(timezone.utc)


async def sleep_until(when: datetime) -> None:
    """Sleep until `when`, in steps of at most 30 s.

    Why not one long sleep: a laptop can suspend, or the clock can be adjusted, during
    an hours-long sleep. Waking every 30 s and rechecking the real time keeps the wake-up
    accurate. The walrus operator (:=) computes the time left and tests it in one step."""
    while (left := (when - utcnow()).total_seconds()) > 0:
        await asyncio.sleep(min(left, 30))


class LiveRunner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.journal = Journal(cfg.db_path)
        self.engine = Engine(cfg, self.journal)      # reloads cash + positions from the journal
        self.cal = MarketCalendar()
        self.data = AlpacaData(cfg.alpaca_key, cfg.alpaca_secret, cfg.feed.feed)  # raises if keys missing
        self.agent = Agent(cfg)                      # inactive without ANTHROPIC_API_KEY
        self.finnhub = Finnhub(cfg.finnhub_key)      # inactive without FINNHUB_API_KEY
        # Symbols to stream: today's watchlist + benchmark (replaced every morning).
        self.universe = list(dict.fromkeys(cfg.symbols + [cfg.benchmark]))
        # Extra facts for the dashboard. The dict is shared with the API app, so updating
        # it here shows up on the next page refresh.
        self.info = {"next_session": None, "agent_active": self.agent.active}

    async def run(self) -> None:
        """Main loop. Never returns normally; stops on Ctrl+C or a FatalFeedError."""
        app = create_app(self.engine, self.journal, self.cfg, self.info)
        # Run the web server as a task on this same event loop (not a separate process),
        # so it can read the engine's in-memory state directly.
        server = uvicorn.Server(uvicorn.Config(app, host=self.cfg.api.host, port=self.cfg.api.port,
                                               log_level="warning"))
        api = asyncio.create_task(server.serve())
        log.info("dashboard on http://%s:%s", self.cfg.api.host, self.cfg.api.port)
        try:
            while True:
                # The session in progress (if started mid-day) or the next one.
                sess = self.cal.current_or_next(utcnow())
                self.info["next_session"] = {"open": sess.open.isoformat(), "close": sess.close.isoformat(),
                                             "early_close": sess.early_close}
                try:
                    lead = timedelta(minutes=self.cfg.feed.premarket_lead_minutes)
                    if utcnow() < sess.open - lead:
                        self.engine.status = "waiting for next session"
                        log.info("next session %s opens %s", sess.day, sess.open.isoformat())
                        await sleep_until(sess.open - lead)
                    await self.prepare(sess)         # scan + plan (about 25 min before the open)
                    await sleep_until(sess.open)     # returns at once if already open
                    await self.trade(sess)           # until the close
                    await self.after_close(sess)     # books + review; then loop to the next day
                except FatalFeedError:
                    raise
                except Exception as e:
                    # Anything else: record it, wait, and start the loop again. The next
                    # pass re-reads the calendar, so a mid-session retry resumes trading
                    # (prepare handles the mid-session rebuild).
                    log.exception("session loop error; retrying in 30s")
                    self.journal.event(utcnow(), "ERROR", "", f"{type(e).__name__}: {e}")
                    await asyncio.sleep(30)
        finally:
            # Shut the web server down cleanly whatever ended the loop.
            server.should_exit = True
            await api

    async def _daily(self, symbols: list, sess: Session) -> dict:
        """Daily bars for `symbols`: ~420 calendar days (enough for EMA50 and the scanner's
        3-month strength), ending 12 hours before the open so only completed bars are used."""
        start, end = sess.open - timedelta(days=420), sess.open - timedelta(hours=12)
        out: dict = {}
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i + 200]
            try:   # consolidated history is free once it is older than 15 minutes
                out.update(await self.data.bars(chunk, "1Day", start, end, feed="sip"))
            except FatalFeedError:
                # If this account cannot read SIP at all, fall back to the configured feed
                # (IEX). Its volumes are smaller, but prices are fine for indicators.
                out.update(await self.data.bars(chunk, "1Day", start, end))
        return out

    async def build_watchlist(self, sess: Session) -> tuple:
        """Returns (watchlist, daily bars, earnings map). Falls back to config symbols on failure.

        The full live pipeline: universe -> daily bars -> pre-market + news -> numeric
        scan -> Finnhub enrichment -> agent gate -> picks. Held positions always stay on
        the list, so only the remaining slots are filled with new names."""
        sc, bench = self.cfg.scanner, self.cfg.benchmark
        held = list(self.engine.positions)
        slots = max(sc.max_watchlist - len(held), 0)
        day = sess.day.isoformat()
        if sc.enabled:
            try:
                universe = await scanner.ensure_universe(self.data, self.journal, self.cfg, utcnow())
                daily = await self._daily(list(dict.fromkeys(universe + held + [bench])), sess)
                # Use the fresh bars to grade earlier scans (what did past picks do next?).
                self.journal.update_scan_outcomes(daily)
                now = utcnow()
                # US pre-market trading starts at 04:00 New York time; convert to UTC.
                early = datetime.combine(sess.day, time(4, 0), tzinfo=NY).astimezone(timezone.utc)
                pre = await self.data.premarket(universe, early, now)
                news = await self.data.news_all(set(universe),
                                                now - timedelta(hours=sc.news_lookback_hours), now)
                # "Fresh" news = published after the previous session's close. The newest
                # benchmark daily bar is the previous session; its timestamp is midnight
                # New York, so take that date and set 20:00 UTC (16:00 New York in summer).
                # In winter the close is 21:00 UTC, so up to one extra hour of headlines
                # counts as fresh: a deliberate, harmless simplification.
                prev_close = max((b.ts for b in daily.get(bench, [])), default=now - timedelta(hours=18))
                fresh_since = prev_close.replace(hour=20, minute=0)   # 20:00 UTC: the summer-time close
                cands = scanner.scan(self.cfg, {s: daily.get(s, []) for s in universe},
                                     daily.get(bench, []), pre, news, fresh_since)
                names = [c.symbol for c in cands]
                earn = await self.finnhub.earnings(sess.day)
                profit = await self.finnhub.profitable(names)
                for c in cands:
                    c.profitable = profit.get(c.symbol)
                    c.earnings_in_days = self._sessions_until(sess, earn.get(c.symbol))
                # The model call is blocking; run it in a worker thread.
                decisions = await asyncio.to_thread(self.agent.select, cands)
                picks = scanner.apply_decisions(cands, decisions, slots, set(held))
                self.journal.save_scan(day, cands, "agent" if decisions is not None else "rank")
                rejected = sum(c.decision == "reject" for c in cands)
                # One summary line in the activity log, including which inputs were missing.
                self.journal.event(utcnow(), "SCAN", "", f"{len(universe)} scanned, {len(cands)} "
                                   f"candidates, {rejected} rejected, {len(picks)} added"
                                   f"{'' if decisions is not None else ' (ranked, agent not used)'}; "
                                   f"pre-market data {'on' if pre else 'off'}, "
                                   f"news {'on' if news is not None else 'off'}")
                return held + picks, daily, earn
            except FatalFeedError:
                raise
            except Exception as e:
                # The scanner is an optimisation, not a necessity: on any failure, trade
                # the fixed fallback list from config instead of skipping the day.
                log.exception("scanner failed; using the fallback symbol list")
                self.journal.event(utcnow(), "WARN", "", f"scanner failed ({type(e).__name__}: {e}); "
                                   "using fallback symbols")
        watch = list(dict.fromkeys(held + self.cfg.symbols))[:sc.max_watchlist]
        daily = await self._daily(list(dict.fromkeys(watch + [bench])), sess)
        return watch, daily, await self.finnhub.earnings(sess.day)

    def _sessions_until(self, sess: Session, when) -> "int | None":
        """Trading sessions from today until date `when` (0 = today), or None if unknown.
        Counts trading days, not calendar days, so a weekend does not inflate the gap."""
        if when is None:
            return None
        return max(len(self.cal.sessions_between(sess.day, when)) - 1, 0)

    async def prepare(self, sess: Session) -> None:
        """Pre-market: watchlist -> daily contexts -> agent plan -> engine.start_day.
        Also handles a start in the middle of the session (see the last block)."""
        self.engine.status = "scanning"
        watch, daily, earn = await self.build_watchlist(sess)
        self.universe = list(dict.fromkeys(watch + [self.cfg.benchmark]))
        ctx = contexts_for(self.cfg, {s: daily.get(s, []) for s in self.universe}, sess)
        for s, c in ctx.items():
            c.earnings_in_days = self._sessions_until(sess, earn.get(s))
        missing = [s for s in watch if s not in ctx]
        if missing:
            self.journal.event(utcnow(), "WARN", "", "no daily history for: " + ", ".join(missing))
        recent = self.journal.rows("SELECT day, pnl, trades FROM days WHERE end_equity IS NOT NULL "
                                   "ORDER BY day DESC LIMIT 5")
        plan = await asyncio.to_thread(self.agent.make_plan, sess, ctx, self.engine.snapshot(),
                                       self.journal.stats(), recent)
        self.engine.start_day(sess, ctx, plan, watch)
        if utcnow() > sess.open + timedelta(minutes=1):
            # started mid-session: rebuild VWAP / opening range from today's bars, no orders
            #
            # Fetch today's minute bars so far and feed them through the engine in time
            # order with warmup=True: the engine updates prices, VWAP and the opening
            # range (and marks ORB attempts already "used"), but on_tick/on_bar place no
            # orders and on_clock does nothing. Then warm-up ends and live trading resumes
            # with the correct intraday state.
            bars = await self.data.bars(self.universe, "1Min", sess.open, utcnow())
            self.engine.warmup = True
            for b in sorted((b for v in bars.values() for b in v), key=lambda x: x.ts):
                self.engine.on_tick(Tick(b.symbol, b.ts, b.c))
                self.engine.on_bar(b)
            self.engine.warmup = False
            self.journal.event(utcnow(), "INFO", "", "started mid-session; state rebuilt from today's bars")

    async def trade(self, sess: Session) -> None:
        """Stream market data into the engine until the close, ticking its clock each second.

        Two concurrent activities:
          * `pump` (a background task) reads the websocket and calls on_tick / on_bar;
          * the loop below wakes every second to: re-raise pump errors, check that data
            is still arriving, and call engine.on_clock (flatten, loss limit, equity).
        Why separate: on_clock must keep running even when no prices arrive (a quiet
        market or a dead connection); otherwise the end-of-day flatten could be missed."""
        last_msg = clock.monotonic()   # monotonic: unaffected by system clock changes

        async def pump() -> None:
            nonlocal last_msg          # update the variable of the enclosing function
            async for ev in self.data.stream(self.universe, sess.close):
                last_msg = clock.monotonic()
                if isinstance(ev, Tick):
                    self.engine.on_tick(ev)
                else:
                    self.engine.on_bar(ev)

        task = asyncio.create_task(pump())
        try:
            while utcnow() < sess.close:
                # If the stream task died (e.g. FatalFeedError), surface its exception here.
                if task.done() and task.exception():
                    raise task.exception()
                # Stale-feed guard: trading on old prices is dangerous, so if nothing has
                # arrived for stale_seconds the engine blocks new entries (stops still
                # work on whatever prices do arrive). Logged only when the state changes.
                ok = clock.monotonic() - last_msg < self.cfg.feed.stale_seconds
                if ok != self.engine.feed_ok:
                    self.engine.feed_ok = ok
                    self.journal.event(utcnow(), "FEED", "", "data feed live" if ok
                                       else "data feed silent; entries blocked")
                self.engine.on_clock(utcnow())
                await asyncio.sleep(1)
        finally:
            # Always stop the stream task, and wait for it so it is not left dangling.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def after_close(self, sess: Session) -> None:
        """Close the books for the day, then ask the agent for a written review."""
        day = sess.day.isoformat()
        self.engine.end_day()
        row = self.journal.rows("SELECT * FROM days WHERE day=?", (day,))[0]
        events = self.journal.rows("SELECT ts, kind, symbol, message FROM events WHERE ts>=? ORDER BY id",
                                   (sess.open.isoformat(),))
        review = await asyncio.to_thread(self.agent.review, day, self.engine.plan.to_dict(),
                                         self.journal.trades(day), events,
                                         {k: row[k] for k in ("start_equity", "end_equity", "pnl", "trades")})
        if review:
            self.journal.upsert_day(day, review=review)
