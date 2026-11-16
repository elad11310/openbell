"""The trading engine: strategies, risk limits and a simulated (paper) broker.

WHAT: `Engine` decides when to buy and sell and keeps the books. It runs two long-only
      strategies:
        * ORB (opening-range breakout, intraday): buy the first 1-minute close above the
          high of the first 15 minutes, if above VWAP and on strong volume; stop at the
          range low (capped at 0.5 ATR), target 2R; everything closed before the bell.
        * SWING (trend pullback, multi-day): buy when price trades through the trigger
          computed pre-market by `indicators.build_daily_context`; 2-ATR stop that then
          trails 3 ATR below the highest close; exits on trend break or after 30 days.
      Every entry passes the hard risk limits from config (risk per trade, position size,
      exposure, open positions, trades per day, daily loss limit, stale feed).
      Fills are simulated against the last price plus a fixed slippage.

HOW:  The engine is a pure event processor. It never reads the clock, the network or a
      file by itself; it only reacts to what it is handed:
        start_day(session, contexts, plan, watchlist)   before the open
        on_tick(tick)      every trade print   -> stops, targets, swing entries
        on_bar(bar)        every 1-minute bar  -> VWAP, opening range, ORB entries
        on_clock(now)      about once a second -> end-of-day flatten, loss limit, equity curve
        end_day()          after the close
      Every state change is written to the Journal straight away.

WHY:  * Determinism: because the engine only reacts to events, the exact same code runs
        live (runner.py), in historical replays and in the synthetic simulation
        (replay.py). A test can feed it a handful of bars and check the outcome exactly.
      * Safety: the LLM never calls into this class. Its only influence is the DayPlan,
        which can lower the risk multiplier (clamped to [0, 1]) or forbid a strategy on a
        symbol. It can never raise size or bypass a limit.
      * Speed: a stop must act on the very next tick, so the decisions here are cheap
        arithmetic, not model calls.

Key ideas used below:
  R          the dollar risk per share at entry (entry - initial stop). "Target 2R" means
             take profit at twice the distance to the stop.
  ATR        typical daily range, so stops scale with each stock's volatility.
  VWAP       volume-weighted average price of today's session; buyers above it are, on
             average, in profit, a common intraday strength filter.
  RVOL       this bar's volume relative to the session's average bar so far.
"""
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
    """Per-symbol running state for the current session, rebuilt every morning.

    Accumulated bar by bar in `Engine.on_bar`; after a mid-session restart it is rebuilt
    by replaying today's bars with the engine in warm-up mode (see runner.prepare)."""
    bars: int = 0                 # minute bars seen today
    vols: list = field(default_factory=list)   # volume of each bar, for the average (RVOL)
    cum_pv: float = 0.0           # running sum of typical_price * volume (VWAP numerator)
    cum_v: float = 0.0            # running sum of volume (VWAP denominator)
    or_high: float = -math.inf    # opening range high; -inf so the first bar always replaces it
    or_low: float = math.inf      # opening range low;  +inf for the same reason
    or_bars: int = 0              # bars that fell inside the opening range window
    or_done: bool = False         # the opening-range window has finished
    orb_taken: bool = False       # the one ORB attempt for today has been used

    @property
    def vwap(self) -> Optional[float]:
        """Volume-weighted average price so far, or None before any volume traded."""
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
        # Durable state comes back from the journal, so a restart keeps cash and positions.
        # On the very first run the account table is empty and cash = starting_equity.
        self.cash: float = journal.get_account("cash", cfg.starting_equity)
        self.positions: dict = journal.load_positions()   # symbol -> Position
        self.last: dict = {}            # symbol -> last known price (marks positions to market)
        self.session: Optional[Session] = None   # None between sessions
        self.ctx: dict = {}             # symbol -> DailyContext for today
        self.plan = DayPlan()           # today's agent plan (default until start_day)
        self.state: dict = {}           # symbol -> IntradayState for today's watchlist
        self.symbols: list = list(cfg.symbols)   # today's tradable watchlist
        self.day_start_equity = self.equity()
        self.trades_today = 0
        self.halted = False             # daily loss limit hit: no new entries today
        self.flattened = False          # end-of-day flatten already done
        self.feed_ok = True             # set by the runner; False blocks new entries
        self.warmup = False             # True while replaying today's bars after a restart
        self.status = "idle"            # shown on the dashboard
        # (symbol, strategy) pairs already tried today: each strategy gets one attempt per
        # symbol per day, whether it was filled or rejected.
        self._attempted: set = set()
        self._last_equity_minute: Optional[datetime] = None
        self.now: Optional[datetime] = None

    # ------------------------------------------------------------ accounting
    def equity(self) -> float:
        """Cash plus the market value of every position, marked at the last known price
        (or at the entry price if no price has arrived yet)."""
        return self.cash + sum(p.qty * self.last.get(p.symbol, p.entry_px)
                               for p in self.positions.values())

    def exposure(self) -> float:
        """Market value of all positions (gross exposure)."""
        return sum(p.qty * self.last.get(p.symbol, p.entry_px) for p in self.positions.values())

    def day_pnl(self) -> float:
        """Profit or loss since the start of the session, realised and unrealised."""
        return self.equity() - self.day_start_equity

    # ------------------------------------------------------------ day lifecycle
    def start_day(self, session: Session, contexts: dict, plan: DayPlan,
                  watchlist: Optional[list] = None) -> None:
        """`watchlist` is today's tradable names; held positions are always managed."""
        # --- 1. reset today's state ---
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
        # Mark everything at yesterday's close until today's first trade arrives.
        for s, c in contexts.items():
            self.last[s] = c.prev_close
        self.day_start_equity = self.equity()
        # --- 2. restart recovery ---
        # Restart in the middle of a session: restore today's counters from the journal
        # so the day's limits and one-attempt-per-symbol rule still hold.
        day = session.day.isoformat()
        prior = self.j.rows("SELECT start_equity FROM days WHERE day=?", (day,))
        if prior and prior[0]["start_equity"]:
            # Keep the morning's figure; recomputing it now would include today's P&L and
            # quietly move the daily loss limit.
            self.day_start_equity = prior[0]["start_equity"]
        # Today's attempts = trades already closed today + positions opened today and
        # still open. Both count toward max_trades_per_day.
        done = [(t["symbol"], t["strategy"]) for t in self.j.trades(day)]
        done += [(p.symbol, p.strategy) for p in self.positions.values()
                 if p.entry_ts >= session.open]
        self._attempted.update(done)
        self.trades_today = len(done)
        for sym, strat in done:
            if strat == "ORB" and sym in self.state:
                self.state[sym].orb_taken = True
        # --- 3. pre-market pass over open positions ---
        sw = self.cfg.swing
        for p in self.positions.values():
            c = contexts.get(p.symbol)
            if p.strategy != "SWING":
                # An intraday position from a previous day should not exist (they are
                # flattened before the close), but if a crash left one, sell it at the open.
                if p.entry_ts < session.open:
                    p.exit_flag = "intraday position carried overnight"
            elif c:
                # Swing trailing stop: track the highest daily close since entry and keep
                # the stop trail_atr ATRs below it. max(p.stop, ...) means the stop only
                # ever moves up, never down.
                p.high_close = max(p.high_close, c.prev_close)
                p.stop = round(max(p.stop, p.high_close - sw.trail_atr * c.atr), 2)
                # Exit rules checked on yesterday's close; exit_flag makes on_tick sell at
                # the first trade of the session.
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
        """Write the closing equity point and the day's summary row, then go idle."""
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
        """Handle one trade print: manage an open position, or check for a swing entry."""
        s = self.session
        # Ignore prints outside regular hours (pre/after-market) and bad prices.
        if not s or not (s.open <= t.ts < s.close) or t.price <= 0:
            return
        self.now = t.ts
        self.last[t.symbol] = t.price
        if self.warmup:
            # Rebuilding state after a restart: update prices only, place no orders.
            return
        p = self.positions.get(t.symbol)
        if p:
            # Exit priority: a pre-market exit flag first, then the stop, then the target.
            if p.exit_flag:
                self._close(p, t.price, t.ts, p.exit_flag)
            elif t.price <= p.stop:
                # Filled at the tick price, not the stop price: if the stock gaps through
                # the stop you get the worse price, as you would with a real stop order.
                self._close(p, t.price, t.ts, "stop")
            elif p.target and t.price >= p.target:
                # A take-profit is a limit order, which fills at its own price with no
                # slippage, hence slip=False.
                self._close(p, p.target, t.ts, "target", slip=False)
            return
        # No position in this symbol: maybe a swing entry triggers on this price.
        self._check_swing(t)

    def on_bar(self, b: Bar) -> None:
        """Handle one completed 1-minute bar: update VWAP and the opening range, and look
        for the ORB entry.

        Flow for each bar:
          1. Update VWAP and the volume history (always, for every bar).
          2. If the bar is inside the opening-range window (first 15 minutes): widen the
             range high/low and stop there.
          3. After the window: the first bar that CLOSES above the range high is the
             breakout. That single bar is the one ORB attempt for the day (orb_taken),
             whether it then passes the filters or not; later breakouts are ignored.
          4. Filters: entry cutoff time, close above VWAP, volume >= rvol_min x average.
          5. Stop = range low, but never more than max_stop_atr x ATR below the entry.
             Target = entry + target_r x risk.
        """
        s = self.session
        st = self.state.get(b.symbol)
        # Only symbols on today's watchlist have state; bars outside the session are ignored.
        if not s or st is None or not (s.open <= b.ts < s.close):
            return
        ic = self.cfg.intraday
        end = b.ts + timedelta(minutes=1)     # bar timestamps mark the START; it is complete at `end`
        st.bars += 1
        # Typical price (h+l+c)/3 is the usual stand-in for the bar's average traded price.
        st.cum_pv += (b.h + b.l + b.c) / 3 * b.v
        st.cum_v += b.v
        # Average volume of the PREVIOUS bars, taken before this bar is appended, so the
        # breakout bar is compared against the bars before it rather than diluting itself.
        avg_vol = sum(st.vols) / len(st.vols) if st.vols else 0.0
        st.vols.append(b.v)
        if b.ts < s.open + timedelta(minutes=ic.opening_range_minutes):
            st.or_high = max(st.or_high, b.h)
            st.or_low = min(st.or_low, b.l)
            st.or_bars += 1
            return
        st.or_done = True
        # Not a breakout (yet): already used today, too few bars to trust the range
        # (sparse IEX data in a quiet name), or no close above the range high.
        if st.orb_taken or st.or_bars < ic.min_range_bars or b.c <= st.or_high:
            return
        # first close above the opening range: one attempt per symbol per day
        st.orb_taken = True
        if self.warmup or not ic.enabled:
            return
        # Already holding it (e.g. a swing position), or the agent disallowed intraday here.
        if b.symbol in self.positions or not self.plan.for_symbol(b.symbol).allow_intraday:
            return
        c = self.ctx.get(b.symbol)
        vwap = st.vwap
        # Each `return self._reject(...)` logs a SKIP event saying why, then stops.
        if end > s.close - timedelta(minutes=ic.no_entry_minutes_before_close):
            return self._reject(end, b.symbol, "ORB", "past entry cutoff")
        if vwap is None or b.c <= vwap:
            return self._reject(end, b.symbol, "ORB", "below VWAP")
        if avg_vol <= 0 or b.v < ic.rvol_min * avg_vol:
            return self._reject(end, b.symbol, "ORB", "breakout volume too light")
        stop = st.or_low
        if c:
            # A very wide opening range would mean a far-away stop and a tiny position;
            # cap the stop distance at max_stop_atr x daily ATR. max() picks the HIGHER,
            # i.e. closer, of the two stops.
            stop = max(stop, b.c - ic.max_stop_atr * c.atr)
        risk = b.c - stop
        # Reject stops closer than 0.1% of the price: normal noise would hit them, and
        # the tiny risk would make the share count huge.
        if risk <= 0 or risk / b.c < 0.001:
            return self._reject(end, b.symbol, "ORB", "stop too tight")
        sig = Signal(b.symbol, "ORB", b.c, round(stop, 2), round(b.c + ic.target_r * risk, 2),
                     f"close {b.c:.2f} > opening range high {st.or_high:.2f}, "
                     f"above VWAP {vwap:.2f}, volume {b.v / avg_vol:.1f}x")
        self._enter(sig, end)

    def on_clock(self, now: datetime) -> None:
        """Time-driven rules, called about once a second live and once per bar in replays."""
        s = self.session
        if not s or self.warmup:
            return
        self.now = now
        if now >= s.open:
            self.status = "trading"
        ic = self.cfg.intraday
        # Close all intraday positions shortly before the bell (once per day). The session
        # close already reflects half-days, so this works on 13:00 closes too.
        if not self.flattened and now >= s.close - timedelta(minutes=ic.flatten_minutes_before_close):
            self.flattened = True
            self._flatten_intraday(now, "end-of-day flatten")
        # Daily loss limit: measured on total (realised + unrealised) P&L since the open.
        limit = self.cfg.risk.daily_loss_limit_pct / 100 * self.day_start_equity
        if not self.halted and self.day_pnl() <= -limit:
            self.halted = True
            self.j.event(now, "HALT", "", f"daily loss limit hit ({self.day_pnl():+.2f}); "
                         "no new entries today")
            # Swing positions keep their own stops; only intraday trades are cut.
            self._flatten_intraday(now, "daily loss limit")
        self._record_equity(now)

    # ------------------------------------------------------------ strategies
    def _check_swing(self, t: Tick) -> None:
        """Swing entry: buy when price trades at or above the pre-computed buy-stop trigger.

        Order of checks: enabled + setup exists + on watchlist -> not tried today and
        price reached the trigger -> waited past the open -> agent allows it -> not near
        earnings -> not chasing a gap. Then stop = price - stop_atr x ATR, no fixed target
        (winners are managed by the trailing stop in start_day)."""
        sw = self.cfg.swing
        c: Optional[DailyContext] = self.ctx.get(t.symbol)
        # `t.symbol not in self.state` = not on today's watchlist. Held names are still
        # managed in on_tick, but new entries only come from today's list.
        if not sw.enabled or not c or c.swing_trigger is None or t.symbol not in self.state:
            return
        if (t.symbol, "SWING") in self._attempted or t.price < c.swing_trigger:
            return
        # Skip the first minutes: opening prints are noisy and can briefly spike through.
        if t.ts < self.session.open + timedelta(minutes=sw.wait_minutes_after_open):
            return
        if not self.plan.for_symbol(t.symbol).allow_swing:
            # Marked as attempted without a SKIP event, so the log is not flooded on every tick.
            self._attempted.add((t.symbol, "SWING"))
            return
        # Earnings can move a stock far beyond any stop overnight; do not open a
        # multi-day position into a report.
        blackout = self.cfg.scanner.earnings_blackout_days
        if c.earnings_in_days is not None and 0 <= c.earnings_in_days <= blackout:
            return self._reject(t.ts, t.symbol, "SWING",
                                f"earnings in {c.earnings_in_days} session(s)")
        # If the stock opened far above the trigger, buying now means a worse price and a
        # wider real risk than the setup assumed, so skip it.
        if t.price > c.swing_trigger + sw.max_chase_atr * c.atr:
            return self._reject(t.ts, t.symbol, "SWING", "gapped too far past trigger")
        stop = round(t.price - sw.stop_atr * c.atr, 2)
        self._enter(Signal(t.symbol, "SWING", t.price, stop, None, c.swing_reason), t.ts)

    # ------------------------------------------------------------ orders
    def _reject(self, ts: datetime, symbol: str, strategy: str, why: str) -> None:
        """Mark the (symbol, strategy) as tried today and log why it was skipped, once."""
        if (symbol, strategy) not in self._attempted:
            self._attempted.add((symbol, strategy))
            self.j.event(ts, "SKIP", symbol, f"{strategy}: {why}")

    def _enter(self, sig: Signal, ts: datetime) -> None:
        """The paper broker's buy: apply account-level limits, size the trade, fill it."""
        r = self.cfg.risk
        eq = self.equity()
        # --- account-level gates (the first that fails wins) ---
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
        # --- simulated fill price: buys pay the slippage (bps / 10_000 = fraction) ---
        px = round(sig.entry * (1 + r.slippage_bps / 1e4), 4)
        risk_ps = px - sig.stop
        # --- position sizing ---
        # Risk budget = equity x risk_per_trade_pct, scaled by the agent's multiplier. The
        # multiplier is clamped again here as a second line of defence: even a plan that
        # skipped parse_plan cannot raise size above 1x.
        # Example: 100k equity, 0.5% -> $500 at risk; stop $1.00 away -> 500 shares.
        budget = eq * r.risk_per_trade_pct / 100 * min(max(self.plan.risk_multiplier, 0.0), 1.0)
        qty = math.floor(budget / risk_ps) if risk_ps > 0 else 0
        # Then cap by each limit and take the smallest:
        #   max position size, remaining gross exposure room, and available cash
        #   (no margin, so you can never spend more cash than you have).
        qty = min(qty,
                  math.floor(eq * r.max_position_pct / 100 / px),
                  math.floor(max(eq * r.max_gross_exposure_pct / 100 - self.exposure(), 0) / px),
                  math.floor(self.cash / px))
        if qty < 1:
            return self._reject(ts, sig.symbol, sig.strategy, "size rounds to zero under risk limits")
        # --- book the trade ---
        self._attempted.add((sig.symbol, sig.strategy))
        p = Position(sig.symbol, sig.strategy, qty, px, ts, sig.stop, sig.stop, sig.target,
                     high_close=px)
        self.positions[sig.symbol] = p
        self.cash -= qty * px
        self.trades_today += 1
        self.last[sig.symbol] = sig.entry
        # Persist immediately: if the process dies now, the position (and its stop) survive.
        self.j.save_position(p)
        self.j.set_account("cash", self.cash)
        tgt = f", target {sig.target:.2f}" if sig.target else ""
        self.j.event(ts, "BUY", sig.symbol, f"{sig.strategy} {qty} @ {px:.2f}, stop {sig.stop:.2f}"
                     f"{tgt}. {sig.reason}")

    def _close(self, p: Position, price: float, ts: datetime, reason: str, slip: bool = True) -> None:
        """The paper broker's sell: fill, book the P&L, record the trade, drop the position."""
        # Sells receive slightly less than the price (slippage), except limit-style exits.
        px = round(price * (1 - self.cfg.risk.slippage_bps / 1e4), 4) if slip else price
        pnl = (px - p.entry_px) * p.qty
        self.cash += p.qty * px
        del self.positions[p.symbol]
        self.j.delete_position(p.symbol)
        self.j.set_account("cash", self.cash)
        # r_multiple: the result in units of initial risk. +2.0 = won twice what was
        # risked, -1.0 = a normal stop-out, below -1 = slipped or gapped past the stop.
        self.j.add_trade(day=self.session.day.isoformat(), symbol=p.symbol, strategy=p.strategy,
                         qty=p.qty, entry_ts=p.entry_ts.isoformat(), entry_px=p.entry_px,
                         exit_ts=ts.isoformat(), exit_px=px, initial_stop=p.initial_stop,
                         pnl=round(pnl, 2),
                         r_multiple=round((px - p.entry_px) / p.risk_per_share, 2),
                         exit_reason=reason)
        self.j.event(ts, "SELL", p.symbol, f"{p.strategy} {p.qty} @ {px:.2f} ({reason}), "
                     f"P&L {pnl:+.2f}")

    def _flatten_intraday(self, ts: datetime, reason: str) -> None:
        """Sell every non-swing position at its last price."""
        # Build the list first: _close deletes from self.positions, and deleting from a
        # dict while iterating over it raises an error.
        for p in [p for p in self.positions.values() if p.strategy != "SWING"]:
            self._close(p, self.last.get(p.symbol, p.entry_px), ts, reason)

    def _record_equity(self, ts: datetime, force: bool = False) -> None:
        """Add a point to the equity curve at most once per minute (on_clock fires every
        second, which would bloat the table). `force` writes the exact closing point."""
        minute = ts.replace(second=0, microsecond=0)
        if force or minute != self._last_equity_minute:
            self._last_equity_minute = minute
            self.j.add_equity(ts if force else minute, self.equity(), self.cash,
                              self.last.get(self.cfg.benchmark))

    # ------------------------------------------------------------ view
    def can_still_enter(self, symbol: str) -> bool:
        """Could `symbol` still produce a new entry today? Read-only.

        True while its ORB attempt is unused and the entry cutoff has not passed, or while
        its swing trigger is untried, as far as the plan and the day's limits allow.
        The runner uses this to decide which names deserve the real-time stream (see
        feeds.plan_feeds). It is not an entry check: on_tick and on_bar still apply
        every rule."""
        s, st = self.session, self.state.get(symbol)
        if not s or st is None or self.halted or self.trades_today >= self.cfg.risk.max_trades_per_day:
            return False
        ic, sw, allow = self.cfg.intraday, self.cfg.swing, self.plan.for_symbol(symbol)
        now = self.now or s.open
        orb = (ic.enabled and allow.allow_intraday and not st.orb_taken
               and now < s.close - timedelta(minutes=ic.no_entry_minutes_before_close))
        c = self.ctx.get(symbol)
        swing = (sw.enabled and allow.allow_swing and c is not None and c.swing_trigger is not None
                 and (symbol, "SWING") not in self._attempted)
        return bool(orb or swing)

    def snapshot(self) -> dict:
        """Everything the dashboard (and the agent's plan prompt) needs, as plain JSON
        types. Read-only: building it never changes engine state."""
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
            # The conditional expressions avoid errors for symbols with missing data.
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
