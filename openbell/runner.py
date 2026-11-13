"""Live loop: wait for the next NYSE session, prepare, trade it, review it, repeat."""
from __future__ import annotations

import asyncio
import logging
import time as clock
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
    return datetime.now(timezone.utc)


async def sleep_until(when: datetime) -> None:
    while (left := (when - utcnow()).total_seconds()) > 0:
        await asyncio.sleep(min(left, 30))


class LiveRunner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.journal = Journal(cfg.db_path)
        self.engine = Engine(cfg, self.journal)
        self.cal = MarketCalendar()
        self.data = AlpacaData(cfg.alpaca_key, cfg.alpaca_secret, cfg.feed.feed)
        self.agent = Agent(cfg)
        self.finnhub = Finnhub(cfg.finnhub_key)
        self.universe = list(dict.fromkeys(cfg.symbols + [cfg.benchmark]))
        self.info = {"next_session": None, "agent_active": self.agent.active}

    async def run(self) -> None:
        app = create_app(self.engine, self.journal, self.cfg, self.info)
        server = uvicorn.Server(uvicorn.Config(app, host=self.cfg.api.host, port=self.cfg.api.port,
                                               log_level="warning"))
        api = asyncio.create_task(server.serve())
        log.info("dashboard on http://%s:%s", self.cfg.api.host, self.cfg.api.port)
        try:
            while True:
                sess = self.cal.current_or_next(utcnow())
                self.info["next_session"] = {"open": sess.open.isoformat(), "close": sess.close.isoformat(),
                                             "early_close": sess.early_close}
                try:
                    lead = timedelta(minutes=self.cfg.feed.premarket_lead_minutes)
                    if utcnow() < sess.open - lead:
                        self.engine.status = "waiting for next session"
                        log.info("next session %s opens %s", sess.day, sess.open.isoformat())
                        await sleep_until(sess.open - lead)
                    await self.prepare(sess)
                    await sleep_until(sess.open)
                    await self.trade(sess)
                    await self.after_close(sess)
                except FatalFeedError:
                    raise
                except Exception as e:
                    log.exception("session loop error; retrying in 30s")
                    self.journal.event(utcnow(), "ERROR", "", f"{type(e).__name__}: {e}")
                    await asyncio.sleep(30)
        finally:
            server.should_exit = True
            await api

    async def _daily(self, symbols: list, sess: Session) -> dict:
        start, end = sess.open - timedelta(days=420), sess.open - timedelta(hours=12)
        out: dict = {}
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i + 200]
            try:   # consolidated history is free once it is older than 15 minutes
                out.update(await self.data.bars(chunk, "1Day", start, end, feed="sip"))
            except FatalFeedError:
                out.update(await self.data.bars(chunk, "1Day", start, end))
        return out

    async def build_watchlist(self, sess: Session) -> tuple:
        """Returns (watchlist, daily bars, earnings map). Falls back to config symbols on failure."""
        sc, bench = self.cfg.scanner, self.cfg.benchmark
        held = list(self.engine.positions)
        slots = max(sc.max_watchlist - len(held), 0)
        day = sess.day.isoformat()
        if sc.enabled:
            try:
                universe = await scanner.ensure_universe(self.data, self.journal, self.cfg, utcnow())
                daily = await self._daily(list(dict.fromkeys(universe + held + [bench])), sess)
                self.journal.update_scan_outcomes(daily)
                now = utcnow()
                early = datetime.combine(sess.day, time(4, 0), tzinfo=NY).astimezone(timezone.utc)
                pre = await self.data.premarket(universe, early, now)
                news = await self.data.news_all(set(universe),
                                                now - timedelta(hours=sc.news_lookback_hours), now)
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
                decisions = await asyncio.to_thread(self.agent.select, cands)
                picks = scanner.apply_decisions(cands, decisions, slots, set(held))
                self.journal.save_scan(day, cands, "agent" if decisions is not None else "rank")
                rejected = sum(c.decision == "reject" for c in cands)
                self.journal.event(utcnow(), "SCAN", "", f"{len(universe)} scanned, {len(cands)} "
                                   f"candidates, {rejected} rejected, {len(picks)} added"
                                   f"{'' if decisions is not None else ' (ranked, agent not used)'}; "
                                   f"pre-market data {'on' if pre else 'off'}, "
                                   f"news {'on' if news is not None else 'off'}")
                return held + picks, daily, earn
            except FatalFeedError:
                raise
            except Exception as e:
                log.exception("scanner failed; using the fallback symbol list")
                self.journal.event(utcnow(), "WARN", "", f"scanner failed ({type(e).__name__}: {e}); "
                                   "using fallback symbols")
        watch = list(dict.fromkeys(held + self.cfg.symbols))[:sc.max_watchlist]
        daily = await self._daily(list(dict.fromkeys(watch + [bench])), sess)
        return watch, daily, await self.finnhub.earnings(sess.day)

    def _sessions_until(self, sess: Session, when) -> "int | None":
        if when is None:
            return None
        return max(len(self.cal.sessions_between(sess.day, when)) - 1, 0)

    async def prepare(self, sess: Session) -> None:
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
            bars = await self.data.bars(self.universe, "1Min", sess.open, utcnow())
            self.engine.warmup = True
            for b in sorted((b for v in bars.values() for b in v), key=lambda x: x.ts):
                self.engine.on_tick(Tick(b.symbol, b.ts, b.c))
                self.engine.on_bar(b)
            self.engine.warmup = False
            self.journal.event(utcnow(), "INFO", "", "started mid-session; state rebuilt from today's bars")

    async def trade(self, sess: Session) -> None:
        last_msg = clock.monotonic()

        async def pump() -> None:
            nonlocal last_msg
            async for ev in self.data.stream(self.universe, sess.close):
                last_msg = clock.monotonic()
                if isinstance(ev, Tick):
                    self.engine.on_tick(ev)
                else:
                    self.engine.on_bar(ev)

        task = asyncio.create_task(pump())
        try:
            while utcnow() < sess.close:
                if task.done() and task.exception():
                    raise task.exception()
                ok = clock.monotonic() - last_msg < self.cfg.feed.stale_seconds
                if ok != self.engine.feed_ok:
                    self.engine.feed_ok = ok
                    self.journal.event(utcnow(), "FEED", "", "data feed live" if ok
                                       else "data feed silent; entries blocked")
                self.engine.on_clock(utcnow())
                await asyncio.sleep(1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def after_close(self, sess: Session) -> None:
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
