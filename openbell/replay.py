"""Drive the engine from minute bars: used by the synthetic simulation and by
historical replays. Each bar is expanded into four ticks so stops and targets are
checked inside the bar (low first on up-bars, high first on down-bars).

WHAT: * `contexts_for`   builds each symbol's DailyContext using only data from BEFORE
                         the session (also used by the live runner);
      * `replay_session` plays one session's minute bars through the Engine;
      * `pick_watchlist` is the numeric-only scanner path (no news, no agent);
      * `SyntheticMarket` is a seeded random-walk price generator; its `bars()` mimics
                         AlpacaData.bars so the dashboard can chart simulated prices;
      * `run_simulation` ties these together for `python -m openbell sim` and returns
                         the market, which `sim --serve` hands to the chart service.
HOW:  Historical data contains only bars, but the engine expects a live-like stream of
      ticks, bars and clock events. `replay_session` recreates that stream: for every
      bar (in time order) it sends four synthetic ticks (O, then L/H, then H/L, then C),
      then the bar itself, then a clock event at the bar's end.
WHY:  Testing a strategy on past data is only meaningful if it is the SAME code that
      trades live. Replaying through the real Engine (not a separate backtester)
      guarantees that. The simulation adds a no-keys-needed smoke test of the whole
      pipeline (scanner -> engine -> journal -> dashboard); its P&L means nothing.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from .config import Config
from .engine import Engine
from .indicators import build_daily_context
from .models import Bar, Tick, Session, DayPlan

# Minute bars kept per symbol for the dashboard's intraday charts: 7 full sessions,
# enough for the 5-day chart plus the warm-up session in front of it.
MINUTES_KEPT = 7 * 390


def bar_ticks(b: Bar) -> list:
    """Approximate the price path inside a bar with four ticks.

    A bar only tells us open, high, low and close, not the order in which high and low
    happened. The usual assumption: on an up-bar (close >= open) the price dipped to the
    low first and then rallied to the high; on a down-bar the high came first.
    This matters when a bar touches both a stop and a target: the order decides which one
    fills, and this choice resolves it the cautious way for the common case.
    Ticks are spaced 15 s apart (at +1 s, 16 s, 31 s, 46 s) so they stay inside the minute,
    and each carries a quarter of the bar's volume."""
    path = [b.o, b.l, b.h, b.c] if b.c >= b.o else [b.o, b.h, b.l, b.c]
    return [Tick(b.symbol, b.ts + timedelta(seconds=1 + 15 * i), px, b.v / 4)
            for i, px in enumerate(path)]


def contexts_for(cfg: Config, daily: dict, session: Session) -> dict:
    """Daily contexts using only bars from sessions strictly before `session`."""
    out = {}
    for sym, bars in daily.items():
        # Filtering by date prevents look-ahead bias: the engine must never know today's
        # daily bar (its close, high, ...) at the open. Without this, a replay would look
        # far better than live trading ever could.
        past = [b for b in bars if b.ts.date() < session.day]
        ctx = build_daily_context(sym, past, cfg.swing)
        if ctx:                         # None = not enough history; that symbol is skipped
            out[sym] = ctx
    return out


def replay_session(engine: Engine, session: Session, contexts: dict, plan: DayPlan,
                   minute_bars: Iterable[Bar], watchlist: list = None) -> None:
    """Play one whole session through the engine, as if it were live."""
    engine.start_day(session, contexts, plan, watchlist)
    # Sort by (time, symbol): all symbols' bars for 09:30 first, then 09:31, ... exactly
    # as they would interleave in a live stream. The symbol tie-break makes it deterministic.
    for b in sorted(minute_bars, key=lambda x: (x.ts, x.symbol)):
        for t in bar_ticks(b):
            engine.on_tick(t)                       # stops, targets, swing entries
        engine.on_bar(b)                            # VWAP, opening range, ORB entries
        engine.on_clock(b.ts + timedelta(minutes=1))   # flatten, loss limit, equity curve
    engine.on_clock(session.close)                  # make sure close-time rules run
    engine.end_day()


# ---------------------------------------------------------------- synthetic market
class SyntheticMarket:
    """Seeded random-walk market. It exists to exercise the plumbing end to end;
    its P&L says nothing about how the strategies do on real prices.

    "Seeded" means `random.Random(seed)`: the same seed always produces the same prices,
    so a simulation is exactly repeatable (a test relies on this).

    Prices follow geometric Brownian motion: each step multiplies the price by
    exp(normal noise), the standard simple model of stock prices (prices stay positive,
    moves are proportional to the price level)."""

    def __init__(self, symbols: list, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.symbols = symbols
        # Each symbol gets its own random starting price, daily drift (trend) and daily
        # volatility, so the scanner has a varied market to choose from.
        self.px = {s: self.rng.uniform(40, 400) for s in symbols}
        self.drift = {s: self.rng.uniform(-0.0002, 0.0012) for s in symbols}
        self.vol = {s: self.rng.uniform(0.012, 0.035) for s in symbols}
        self.daily: dict = {s: [] for s in symbols}   # generated daily bars per symbol
        # The latest minute bars per symbol, only for charts; the simulation never reads them.
        self.minutes: dict = {}

    def seed_history(self, sessions: list) -> None:
        """Generate one daily bar per symbol per session (the warm-up history that the
        indicators need before the simulated trading starts)."""
        for sess in sessions:
            for s in self.symbols:
                o = self.px[s] * math.exp(self.rng.gauss(0, self.vol[s] * 0.3))   # overnight gap
                c = o * math.exp(self.rng.gauss(self.drift[s], self.vol[s]))      # the day's move
                # High/low extend beyond the open/close body by a random amount.
                h = max(o, c) * (1 + abs(self.rng.gauss(0, self.vol[s] * 0.4)))
                l = min(o, c) * (1 - abs(self.rng.gauss(0, self.vol[s] * 0.4)))
                self.px[s] = c
                self.daily[s].append(Bar(s, _day_ts(sess), o, h, l, c, self.rng.uniform(5e6, 5e7)))

    def daily_step(self, session: Session, symbols: list) -> None:
        """Advance symbols that are not being watched today by one daily bar."""
        # Temporarily narrow self.symbols so seed_history only touches these names, then
        # restore it. Watched names get their daily bar from minute_bars instead.
        keep, self.symbols = self.symbols, symbols
        self.seed_history([session])
        self.symbols = keep

    def minute_bars(self, session: Session, symbols: list = None) -> list:
        """Generate a full session of 1-minute bars for `symbols`, and append the matching
        daily bar so tomorrow's indicators include today."""
        out = []
        n = int((session.close - session.open).total_seconds() // 60)   # 390, or 210 on a half-day
        for s in (symbols if symbols is not None else self.symbols):
            first = len(out)                  # where this symbol's bars start in `out`
            px = self.px[s] * math.exp(self.rng.gauss(0, self.vol[s] * 0.3))   # opening gap
            # Scale daily volatility to one minute: variance adds up over time, so the
            # per-minute standard deviation is daily_vol / sqrt(minutes per day).
            sig = self.vol[s] / math.sqrt(390)
            # Per-minute drift: the symbol's trend plus a random "today's mood" component.
            mu = self.drift[s] / 390 + self.rng.gauss(0, self.vol[s] * 0.6) / 390
            day_o, day_h, day_l, day_v = px, px, px, 0.0
            for i in range(n):
                # U-shaped activity: real markets are busiest at the open and the close.
                # u is 1.0 at midday and rises to 2.8 at both ends; it scales volatility and volume.
                u = 1.0 + 1.8 * (abs(i - n / 2) / (n / 2)) ** 2      # U-shaped activity
                o = px
                c = o * math.exp(self.rng.gauss(mu, sig * math.sqrt(u)))
                h = max(o, c) * (1 + abs(self.rng.gauss(0, sig * 0.5)))
                l = min(o, c) * (1 - abs(self.rng.gauss(0, sig * 0.5)))
                v = max(100.0, self.rng.lognormvariate(9.5, 0.6) * u)   # log-normal: always positive, skewed
                out.append(Bar(s, session.open + timedelta(minutes=i), o, h, l, c, v))
                px = c
                day_h, day_l, day_v = max(day_h, h), min(day_l, l), day_v + v
            self.px[s] = px
            self.daily[s].append(Bar(s, _day_ts(session), day_o, day_h, day_l, px, day_v))
            # Remember today's minutes for the charts, dropping the oldest beyond the cap.
            # (Uses no random numbers, so simulations stay identical with or without it.)
            kept = self.minutes.setdefault(s, [])
            kept.extend(out[first:])
            del kept[:-MINUTES_KEPT]
        return out

    async def bars(self, symbols: list, timeframe: str, start: datetime, end: datetime,
                   feed: str = None) -> dict:
        """Same call as `AlpacaData.bars`, served from memory, so `ChartService` can
        chart a simulation exactly as it charts live data. Only "1Day" and "1Min" exist
        (`feed` is ignored). Minute bars exist only for days a symbol was watched, so
        a name that was merely scanned can have an empty intraday chart."""
        src = {"1Day": self.daily, "1Min": self.minutes}[timeframe]
        return {s: [b for b in src.get(s, []) if start <= b.ts < end] for s in symbols}


def _day_ts(session: Session) -> datetime:
    """Timestamp for a synthetic daily bar: noon UTC on the session date, safely inside
    the date whatever the time zone, so `b.ts.date()` returns the session day."""
    return datetime(session.day.year, session.day.month, session.day.day, 12, tzinfo=timezone.utc)


def pick_watchlist(cfg: Config, engine: Engine, daily: dict, session: Session) -> list:
    """Numeric scan only (no news, no agent): what a replay or simulation can reproduce."""
    from . import scanner               # imported here, where it is used, not at module load
    held = list(engine.positions)
    if not cfg.scanner.enabled:
        return list(dict.fromkeys(held + cfg.symbols))
    # Same look-ahead guard as contexts_for: the scan sees only bars before today.
    past = {s: [b for b in bars if b.ts.date() < session.day] for s, bars in daily.items()}
    engine.j.update_scan_outcomes(past)
    cands = scanner.scan(cfg, past, past.get(cfg.benchmark, []))
    picks = scanner.apply_decisions(cands, None, max(cfg.scanner.max_watchlist - len(held), 0), set(held))
    engine.j.save_scan(session.day.isoformat(), cands, "rank")
    return held + picks


def run_simulation(cfg: Config, engine: Engine, sessions: list, history: list,
                   planner: Callable = None, seed: int = 7, universe: int = 60) -> SyntheticMarket:
    """Simulate `sessions` on a synthetic market warmed up over `history`.

    Each day: scan -> build contexts -> plan (default unless a planner is given) ->
    generate minute bars for watched names -> advance unwatched names by a daily bar ->
    replay the session through the engine. Returns the market, whose bars the
    dashboard's charts read under `sim --serve`."""
    # Universe = the config symbols plus made-up names SIMAA, SIMAB, ... up to `universe`
    # in total. chr(65) is "A"; i // 26 and i % 26 give two letters, like base-26 digits.
    names = list(dict.fromkeys(cfg.symbols + [f"SIM{chr(65 + i // 26)}{chr(65 + i % 26)}"
                                              for i in range(max(universe - len(cfg.symbols), 0))]))
    mkt = SyntheticMarket(names + [cfg.benchmark], seed)
    mkt.seed_history(history)
    for sess in sessions:
        watch = pick_watchlist(cfg, engine, mkt.daily, sess)
        live = list(dict.fromkeys(watch + [cfg.benchmark]))   # benchmark is always "streamed"
        ctx = contexts_for(cfg, {s: mkt.daily[s] for s in live}, sess)
        plan = planner(sess, ctx) if planner else DayPlan()
        bars = mkt.minute_bars(sess, live)
        # Unwatched names still need a daily bar for today, or their history would have a
        # hole and tomorrow's scan would be wrong.
        mkt.daily_step(sess, [s for s in mkt.symbols if s not in live])
        replay_session(engine, sess, ctx, plan, bars, watch)
    return mkt
