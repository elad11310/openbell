from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .config import Config
from .journal import Journal
from .models import Bar, Tick, Session, Signal, Position, DailyContext, DayPlan


@dataclass
class IntradayState:
    bars: int = 0
    vols: list = field(default_factory=list)
    cum_pv: float = 0.0
    cum_v: float = 0.0
    or_high: float = -math.inf
    or_low: float = math.inf
    or_bars: int = 0
    or_done: bool = False
    orb_taken: bool = False

    @property
    def vwap(self) -> Optional[float]:
        return self.cum_pv / self.cum_v if self.cum_v > 0 else None


class Engine:
    """Deterministic trading core: strategies, risk limits and a paper broker.

    It is driven only by the events it is handed (ticks, minute bars, clock), so the
    same code runs live, in the synthetic simulation and in a historical replay.
    The LLM never calls into this class; it only supplies a DayPlan before the open.
    """

    def __init__(self, cfg: Config, journal: Journal) -> None:
        self.cfg = cfg
        self.j = journal
        self.cash: float = journal.get_account("cash", cfg.starting_equity)
        self.positions: dict = journal.load_positions()
        self.last: dict = {}
        self.session: Optional[Session] = None
        self.ctx: dict = {}
        self.plan = DayPlan()
        self.state: dict = {}
        self.symbols: list = list(cfg.symbols)
        self.day_start_equity = self.equity()
        self.trades_today = 0
        self.halted = False
        self.flattened = False
        self.feed_ok = True
        self.warmup = False
        self.status = "idle"
        self._attempted: set = set()
        self._last_equity_minute: Optional[datetime] = None
        self.now: Optional[datetime] = None

    # ------------------------------------------------------------ accounting
    def equity(self) -> float:
        return self.cash + sum(p.qty * self.last.get(p.symbol, p.entry_px)
                               for p in self.positions.values())

    def exposure(self) -> float:
        return sum(p.qty * self.last.get(p.symbol, p.entry_px) for p in self.positions.values())

    def day_pnl(self) -> float:
        return self.equity() - self.day_start_equity

    # ------------------------------------------------------------ day lifecycle
    def start_day(self, session: Session, contexts: dict, plan: DayPlan,
                  watchlist: Optional[list] = None) -> None:
        """`watchlist` is today's tradable names; held positions are always managed."""
        self.session = session
        self.symbols = list(dict.fromkeys(watchlist if watchlist is not None else self.cfg.symbols))
        self.ctx = contexts
        self.plan = plan
        self.state = {s: IntradayState() for s in self.symbols}
        self._attempted = set()
        self.trades_today = 0
        self.halted = False
        self.flattened = False
        self._last_equity_minute = None
        for s, c in contexts.items():
            self.last[s] = c.prev_close
        self.day_start_equity = self.equity()
        # Restart in the middle of a session: restore today's counters from the journal
        # so the day's limits and one-attempt-per-symbol rule still hold.
        day = session.day.isoformat()
        prior = self.j.rows("SELECT start_equity FROM days WHERE day=?", (day,))
        if prior and prior[0]["start_equity"]:
            self.day_start_equity = prior[0]["start_equity"]
        done = [(t["symbol"], t["strategy"]) for t in self.j.trades(day)]
        done += [(p.symbol, p.strategy) for p in self.positions.values()
                 if p.entry_ts >= session.open]
        self._attempted.update(done)
        self.trades_today = len(done)
        for sym, strat in done:
            if strat == "ORB" and sym in self.state:
                self.state[sym].orb_taken = True
        sw = self.cfg.swing
        for p in self.positions.values():
            c = contexts.get(p.symbol)
            if p.strategy != "SWING":
                if p.entry_ts < session.open:
                    p.exit_flag = "intraday position carried overnight"
            elif c:
                p.high_close = max(p.high_close, c.prev_close)
                p.stop = round(max(p.stop, p.high_close - sw.trail_atr * c.atr), 2)
                if c.prev_close < c.ema_slow:
                    p.exit_flag = f"close below EMA{sw.ema_slow}"
                elif (session.day - p.entry_ts.date()).days >= sw.max_hold_days:
                    p.exit_flag = "max holding period"
            self.j.save_position(p)
        self.status = "ready"
        self.j.upsert_day(session.day.isoformat(), start_equity=self.day_start_equity,
                          plan=plan.to_dict())
        self.j.event(session.open, "PLAN", "", f"[{plan.source}] {plan.regime}, "
                     f"risk x{plan.risk_multiplier:.2f}: {plan.summary}")

    def end_day(self) -> None:
        if not self.session:
            return
        s = self.session
        self._record_equity(s.close, force=True)
        self.j.upsert_day(s.day.isoformat(), end_equity=self.equity(), pnl=self.day_pnl(),
                          trades=self.trades_today, benchmark_close=self.last.get(self.cfg.benchmark))
        self.j.event(s.close, "CLOSE", "", f"day P&L {self.day_pnl():+.2f}, "
                     f"equity {self.equity():.2f}, trades {self.trades_today}")
        self.session = None
        self.status = "closed"

    # ------------------------------------------------------------ event handlers
    def on_tick(self, t: Tick) -> None:
        s = self.session
        if not s or not (s.open <= t.ts < s.close) or t.price <= 0:
            return
        self.now = t.ts
        self.last[t.symbol] = t.price
        if self.warmup:
            return
        p = self.positions.get(t.symbol)
        if p:
            if p.exit_flag:
                self._close(p, t.price, t.ts, p.exit_flag)
            elif t.price <= p.stop:
                self._close(p, t.price, t.ts, "stop")
            elif p.target and t.price >= p.target:
                self._close(p, p.target, t.ts, "target", slip=False)
            return
        self._check_swing(t)

    def on_bar(self, b: Bar) -> None:
        s = self.session
        st = self.state.get(b.symbol)
        if not s or st is None or not (s.open <= b.ts < s.close):
            return
        ic = self.cfg.intraday
        end = b.ts + timedelta(minutes=1)
        st.bars += 1
        st.cum_pv += (b.h + b.l + b.c) / 3 * b.v
        st.cum_v += b.v
        avg_vol = sum(st.vols) / len(st.vols) if st.vols else 0.0
        st.vols.append(b.v)
        if b.ts < s.open + timedelta(minutes=ic.opening_range_minutes):
            st.or_high = max(st.or_high, b.h)
            st.or_low = min(st.or_low, b.l)
            st.or_bars += 1
            return
        st.or_done = True
        if st.orb_taken or st.or_bars < ic.min_range_bars or b.c <= st.or_high:
            return
        # first close above the opening range: one attempt per symbol per day
        st.orb_taken = True
        if self.warmup or not ic.enabled:
            return
        if b.symbol in self.positions or not self.plan.for_symbol(b.symbol).allow_intraday:
            return
        c = self.ctx.get(b.symbol)
        vwap = st.vwap
        if end > s.close - timedelta(minutes=ic.no_entry_minutes_before_close):
            return self._reject(end, b.symbol, "ORB", "past entry cutoff")
        if vwap is None or b.c <= vwap:
            return self._reject(end, b.symbol, "ORB", "below VWAP")
        if avg_vol <= 0 or b.v < ic.rvol_min * avg_vol:
            return self._reject(end, b.symbol, "ORB", "breakout volume too light")
        stop = st.or_low
        if c:
            stop = max(stop, b.c - ic.max_stop_atr * c.atr)
        risk = b.c - stop
        if risk <= 0 or risk / b.c < 0.001:
            return self._reject(end, b.symbol, "ORB", "stop too tight")
        sig = Signal(b.symbol, "ORB", b.c, round(stop, 2), round(b.c + ic.target_r * risk, 2),
                     f"close {b.c:.2f} > opening range high {st.or_high:.2f}, "
                     f"above VWAP {vwap:.2f}, volume {b.v / avg_vol:.1f}x")
        self._enter(sig, end)

    def on_clock(self, now: datetime) -> None:
        s = self.session
        if not s or self.warmup:
            return
        self.now = now
        if now >= s.open:
            self.status = "trading"
        ic = self.cfg.intraday
        if not self.flattened and now >= s.close - timedelta(minutes=ic.flatten_minutes_before_close):
            self.flattened = True
            self._flatten_intraday(now, "end-of-day flatten")
        limit = self.cfg.risk.daily_loss_limit_pct / 100 * self.day_start_equity
        if not self.halted and self.day_pnl() <= -limit:
            self.halted = True
            self.j.event(now, "HALT", "", f"daily loss limit hit ({self.day_pnl():+.2f}); "
                         "no new entries today")
            self._flatten_intraday(now, "daily loss limit")
        self._record_equity(now)

    # ------------------------------------------------------------ strategies
    def _check_swing(self, t: Tick) -> None:
        sw = self.cfg.swing
        c: Optional[DailyContext] = self.ctx.get(t.symbol)
        if not sw.enabled or not c or c.swing_trigger is None or t.symbol not in self.state:
            return
        if (t.symbol, "SWING") in self._attempted or t.price < c.swing_trigger:
            return
        if t.ts < self.session.open + timedelta(minutes=sw.wait_minutes_after_open):
            return
        if not self.plan.for_symbol(t.symbol).allow_swing:
            self._attempted.add((t.symbol, "SWING"))
            return
        blackout = self.cfg.scanner.earnings_blackout_days
        if c.earnings_in_days is not None and 0 <= c.earnings_in_days <= blackout:
            return self._reject(t.ts, t.symbol, "SWING",
                                f"earnings in {c.earnings_in_days} session(s)")
        if t.price > c.swing_trigger + sw.max_chase_atr * c.atr:
            return self._reject(t.ts, t.symbol, "SWING", "gapped too far past trigger")
        stop = round(t.price - sw.stop_atr * c.atr, 2)
        self._enter(Signal(t.symbol, "SWING", t.price, stop, None, c.swing_reason), t.ts)

    # ------------------------------------------------------------ orders
    def _reject(self, ts: datetime, symbol: str, strategy: str, why: str) -> None:
        if (symbol, strategy) not in self._attempted:
            self._attempted.add((symbol, strategy))
            self.j.event(ts, "SKIP", symbol, f"{strategy}: {why}")

    def _enter(self, sig: Signal, ts: datetime) -> None:
        r = self.cfg.risk
        eq = self.equity()
        why = None
        if self.halted:
            why = "trading halted for the day"
        elif not self.feed_ok:
            why = "data feed stale"
        elif len(self.positions) >= r.max_positions:
            why = "max open positions"
        elif self.trades_today >= r.max_trades_per_day:
            why = "max trades per day"
        elif self.plan.risk_multiplier <= 0:
            why = "plan set risk to zero"
        if why:
            return self._reject(ts, sig.symbol, sig.strategy, why)
        px = round(sig.entry * (1 + r.slippage_bps / 1e4), 4)
        risk_ps = px - sig.stop
        budget = eq * r.risk_per_trade_pct / 100 * min(max(self.plan.risk_multiplier, 0.0), 1.0)
        qty = math.floor(budget / risk_ps) if risk_ps > 0 else 0
        qty = min(qty,
                  math.floor(eq * r.max_position_pct / 100 / px),
                  math.floor(max(eq * r.max_gross_exposure_pct / 100 - self.exposure(), 0) / px),
                  math.floor(self.cash / px))
        if qty < 1:
            return self._reject(ts, sig.symbol, sig.strategy, "size rounds to zero under risk limits")
        self._attempted.add((sig.symbol, sig.strategy))
        p = Position(sig.symbol, sig.strategy, qty, px, ts, sig.stop, sig.stop, sig.target,
                     high_close=px)
        self.positions[sig.symbol] = p
        self.cash -= qty * px
        self.trades_today += 1
        self.last[sig.symbol] = sig.entry
        self.j.save_position(p)
        self.j.set_account("cash", self.cash)
        tgt = f", target {sig.target:.2f}" if sig.target else ""
        self.j.event(ts, "BUY", sig.symbol, f"{sig.strategy} {qty} @ {px:.2f}, stop {sig.stop:.2f}"
                     f"{tgt}. {sig.reason}")

    def _close(self, p: Position, price: float, ts: datetime, reason: str, slip: bool = True) -> None:
        px = round(price * (1 - self.cfg.risk.slippage_bps / 1e4), 4) if slip else price
        pnl = (px - p.entry_px) * p.qty
        self.cash += p.qty * px
        del self.positions[p.symbol]
        self.j.delete_position(p.symbol)
        self.j.set_account("cash", self.cash)
        self.j.add_trade(day=self.session.day.isoformat(), symbol=p.symbol, strategy=p.strategy,
                         qty=p.qty, entry_ts=p.entry_ts.isoformat(), entry_px=p.entry_px,
                         exit_ts=ts.isoformat(), exit_px=px, initial_stop=p.initial_stop,
                         pnl=round(pnl, 2),
                         r_multiple=round((px - p.entry_px) / p.risk_per_share, 2),
                         exit_reason=reason)
        self.j.event(ts, "SELL", p.symbol, f"{p.strategy} {p.qty} @ {px:.2f} ({reason}), "
                     f"P&L {pnl:+.2f}")

    def _flatten_intraday(self, ts: datetime, reason: str) -> None:
        for p in [p for p in self.positions.values() if p.strategy != "SWING"]:
            self._close(p, self.last.get(p.symbol, p.entry_px), ts, reason)

    def _record_equity(self, ts: datetime, force: bool = False) -> None:
        minute = ts.replace(second=0, microsecond=0)
        if force or minute != self._last_equity_minute:
            self._last_equity_minute = minute
            self.j.add_equity(ts if force else minute, self.equity(), self.cash,
                              self.last.get(self.cfg.benchmark))

    # ------------------------------------------------------------ view
    def snapshot(self) -> dict:
        s = self.session
        pos = []
        for p in self.positions.values():
            last = self.last.get(p.symbol, p.entry_px)
            pos.append({"symbol": p.symbol, "strategy": p.strategy, "qty": p.qty,
                        "entry": p.entry_px, "last": last, "stop": p.stop, "target": p.target,
                        "pnl": round((last - p.entry_px) * p.qty, 2),
                        "r": round((last - p.entry_px) / p.risk_per_share, 2)})
        watch = []
        for sym in self.symbols:
            c, st = self.ctx.get(sym), self.state.get(sym)
            last = self.last.get(sym)
            watch.append({
                "symbol": sym, "last": last,
                "chg_pct": round((last / c.prev_close - 1) * 100, 2) if c and last else None,
                "vwap": round(st.vwap, 2) if st and st.vwap else None,
                "or_high": st.or_high if st and st.or_bars else None,
                "swing_trigger": c.swing_trigger if c else None,
                "earnings_in_days": c.earnings_in_days if c else None,
                "note": self.plan.for_symbol(sym).note})
        eq = self.equity()
        return {
            "status": self.status, "halted": self.halted, "feed_ok": self.feed_ok,
            "session": {"day": s.day.isoformat(), "open": s.open.isoformat(),
                        "close": s.close.isoformat(), "early_close": s.early_close} if s else None,
            "equity": round(eq, 2), "cash": round(self.cash, 2),
            "day_pnl": round(self.day_pnl(), 2),
            "day_pnl_pct": round(self.day_pnl() / self.day_start_equity * 100, 2),
            "total_pnl_pct": round((eq / self.cfg.starting_equity - 1) * 100, 2),
            "exposure_pct": round(self.exposure() / eq * 100, 1) if eq else 0,
            "trades_today": self.trades_today,
            "positions": pos, "watchlist": watch, "plan": self.plan.to_dict(),
        }
