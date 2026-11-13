"""Drive the engine from minute bars: used by the synthetic simulation and by
historical replays. Each bar is expanded into four ticks so stops and targets are
checked inside the bar (low first on up-bars, high first on down-bars)."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from .config import Config
from .engine import Engine
from .indicators import build_daily_context
from .models import Bar, Tick, Session, DayPlan


def bar_ticks(b: Bar) -> list:
    path = [b.o, b.l, b.h, b.c] if b.c >= b.o else [b.o, b.h, b.l, b.c]
    return [Tick(b.symbol, b.ts + timedelta(seconds=1 + 15 * i), px, b.v / 4)
            for i, px in enumerate(path)]


def contexts_for(cfg: Config, daily: dict, session: Session) -> dict:
    """Daily contexts using only bars from sessions strictly before `session`."""
    out = {}
    for sym, bars in daily.items():
        past = [b for b in bars if b.ts.date() < session.day]
        ctx = build_daily_context(sym, past, cfg.swing)
        if ctx:
            out[sym] = ctx
    return out


def replay_session(engine: Engine, session: Session, contexts: dict, plan: DayPlan,
                   minute_bars: Iterable[Bar], watchlist: list = None) -> None:
    engine.start_day(session, contexts, plan, watchlist)
    for b in sorted(minute_bars, key=lambda x: (x.ts, x.symbol)):
        for t in bar_ticks(b):
            engine.on_tick(t)
        engine.on_bar(b)
        engine.on_clock(b.ts + timedelta(minutes=1))
    engine.on_clock(session.close)
    engine.end_day()


# ---------------------------------------------------------------- synthetic market
class SyntheticMarket:
    """Seeded random-walk market. It exists to exercise the plumbing end to end;
    its P&L says nothing about how the strategies do on real prices."""

    def __init__(self, symbols: list, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.symbols = symbols
        self.px = {s: self.rng.uniform(40, 400) for s in symbols}
        self.drift = {s: self.rng.uniform(-0.0002, 0.0012) for s in symbols}
        self.vol = {s: self.rng.uniform(0.012, 0.035) for s in symbols}
        self.daily: dict = {s: [] for s in symbols}

    def seed_history(self, sessions: list) -> None:
        for sess in sessions:
            for s in self.symbols:
                o = self.px[s] * math.exp(self.rng.gauss(0, self.vol[s] * 0.3))
                c = o * math.exp(self.rng.gauss(self.drift[s], self.vol[s]))
                h = max(o, c) * (1 + abs(self.rng.gauss(0, self.vol[s] * 0.4)))
                l = min(o, c) * (1 - abs(self.rng.gauss(0, self.vol[s] * 0.4)))
                self.px[s] = c
                self.daily[s].append(Bar(s, _day_ts(sess), o, h, l, c, self.rng.uniform(5e6, 5e7)))

    def daily_step(self, session: Session, symbols: list) -> None:
        """Advance symbols that are not being watched today by one daily bar."""
        keep, self.symbols = self.symbols, symbols
        self.seed_history([session])
        self.symbols = keep

    def minute_bars(self, session: Session, symbols: list = None) -> list:
        out = []
        n = int((session.close - session.open).total_seconds() // 60)
        for s in (symbols if symbols is not None else self.symbols):
            px = self.px[s] * math.exp(self.rng.gauss(0, self.vol[s] * 0.3))
            sig = self.vol[s] / math.sqrt(390)
            mu = self.drift[s] / 390 + self.rng.gauss(0, self.vol[s] * 0.6) / 390
            day_o, day_h, day_l, day_v = px, px, px, 0.0
            for i in range(n):
                u = 1.0 + 1.8 * (abs(i - n / 2) / (n / 2)) ** 2      # U-shaped activity
                o = px
                c = o * math.exp(self.rng.gauss(mu, sig * math.sqrt(u)))
                h = max(o, c) * (1 + abs(self.rng.gauss(0, sig * 0.5)))
                l = min(o, c) * (1 - abs(self.rng.gauss(0, sig * 0.5)))
                v = max(100.0, self.rng.lognormvariate(9.5, 0.6) * u)
                out.append(Bar(s, session.open + timedelta(minutes=i), o, h, l, c, v))
                px = c
                day_h, day_l, day_v = max(day_h, h), min(day_l, l), day_v + v
            self.px[s] = px
            self.daily[s].append(Bar(s, _day_ts(session), day_o, day_h, day_l, px, day_v))
        return out


def _day_ts(session: Session) -> datetime:
    return datetime(session.day.year, session.day.month, session.day.day, 12, tzinfo=timezone.utc)


def pick_watchlist(cfg: Config, engine: Engine, daily: dict, session: Session) -> list:
    """Numeric scan only (no news, no agent): what a replay or simulation can reproduce."""
    from . import scanner
    held = list(engine.positions)
    if not cfg.scanner.enabled:
        return list(dict.fromkeys(held + cfg.symbols))
    past = {s: [b for b in bars if b.ts.date() < session.day] for s, bars in daily.items()}
    engine.j.update_scan_outcomes(past)
    cands = scanner.scan(cfg, past, past.get(cfg.benchmark, []))
    picks = scanner.apply_decisions(cands, None, max(cfg.scanner.max_watchlist - len(held), 0), set(held))
    engine.j.save_scan(session.day.isoformat(), cands, "rank")
    return held + picks


def run_simulation(cfg: Config, engine: Engine, sessions: list, history: list,
                   planner: Callable = None, seed: int = 7, universe: int = 60) -> None:
    names = list(dict.fromkeys(cfg.symbols + [f"SIM{chr(65 + i // 26)}{chr(65 + i % 26)}"
                                              for i in range(max(universe - len(cfg.symbols), 0))]))
    mkt = SyntheticMarket(names + [cfg.benchmark], seed)
    mkt.seed_history(history)
    for sess in sessions:
        watch = pick_watchlist(cfg, engine, mkt.daily, sess)
        live = list(dict.fromkeys(watch + [cfg.benchmark]))
        ctx = contexts_for(cfg, {s: mkt.daily[s] for s in live}, sess)
        plan = planner(sess, ctx) if planner else DayPlan()
        bars = mkt.minute_bars(sess, live)
        mkt.daily_step(sess, [s for s in mkt.symbols if s not in live])
        replay_session(engine, sess, ctx, plan, bars, watch)
