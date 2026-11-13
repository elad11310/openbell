"""Shared data types: the "nouns" every other module passes around.

WHAT: Plain dataclasses for market data (Bar, Tick), the trading calendar (Session),
      trading decisions (Signal, Position), per-symbol pre-market facts (DailyContext)
      and the agent's instructions to the engine (SymbolPlan, DayPlan).
HOW:  Each type is a `@dataclass`, so Python writes `__init__`, `__repr__` and `__eq__`
      for us. Market data and sessions are `frozen=True` (immutable): once a price
      arrives it must never be edited by mistake. Positions and plans are mutable
      because the engine updates stops and flags on them.
WHY:  Keeping the data shapes in one module, with no logic and no I/O, means the
      engine, the data client, the scanner, the journal and the tests all agree on one
      vocabulary. Nothing in here imports other OpenBell modules, so it can never cause
      a circular import.

Conventions used everywhere:
  * every datetime is timezone-aware UTC (conversion to New York or local time happens
    only for display or calendar maths);
  * prices are floats in dollars; quantities (`qty`) are whole shares.
"""
from __future__ import annotations   # lets type hints name classes before they are defined

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle: open/high/low/close prices and volume over a time window.

    Used with two window sizes: 1 minute (intraday stream and replays) and 1 day
    (history for indicators and the scanner)."""
    symbol: str
    ts: datetime          # bar START, timezone-aware UTC
    o: float              # first trade price in the window
    h: float              # highest trade price
    l: float              # lowest trade price
    c: float              # last trade price
    v: float              # shares traded


@dataclass(frozen=True)
class Tick:
    """A single trade print from the live stream (or synthesised from a bar in replays).
    Ticks drive stop/target checks and swing entries, because those must react to the
    very next price, not wait for the minute bar to finish."""
    symbol: str
    ts: datetime          # timezone-aware UTC
    price: float
    size: float = 0.0     # shares in the trade; informational only


@dataclass(frozen=True)
class Session:
    """One NYSE trading day. Built by `MarketCalendar`, which already accounts for
    holidays, 13:00 half-days and daylight-saving changes, so no other module has to."""
    day: date
    open: datetime        # UTC
    close: datetime       # UTC
    early_close: bool = False   # True on half-days (close before 16:00 New York)


@dataclass
class Signal:
    """A strategy's request to buy, before risk checks and sizing.
    The engine turns it into a Position only if every limit allows it."""
    symbol: str
    strategy: str         # "ORB" (intraday) or "SWING"
    entry: float          # price the strategy wants to buy at (before slippage)
    stop: float           # price at which the trade is proven wrong and sold
    target: Optional[float]   # take-profit price; None for swing (it trails instead)
    reason: str           # human-readable explanation written to the event log


@dataclass
class Position:
    """An open (simulated) holding. Mutable: the swing trail raises `stop` over time and
    the pre-market pass may set `exit_flag`. Persisted in the `positions` table so it
    survives a restart."""
    symbol: str
    strategy: str
    qty: int
    entry_px: float
    entry_ts: datetime
    stop: float           # current stop (moves up for swing trades)
    initial_stop: float   # stop at entry; fixed, used to measure risk ("R")
    target: Optional[float] = None
    high_close: float = 0.0       # highest daily close since entry (swing trail)
    exit_flag: Optional[str] = None  # set pre-market: exit at the open, with reason

    @property
    def risk_per_share(self) -> float:
        """Dollars lost per share if the original stop is hit ("1R").

        R-multiples (profit / risk_per_share) make trades of different sizes and prices
        comparable. The `max(..., 1e-9)` guards against a division by zero if a
        position were ever stored with its stop at or above the entry."""
        return max(self.entry_px - self.initial_stop, 1e-9)

    def to_row(self) -> dict:
        """Flat dict with the timestamp as an ISO string, ready for JSON or SQLite."""
        d = asdict(self)
        d["entry_ts"] = self.entry_ts.isoformat()
        return d


@dataclass
class DailyContext:
    """Per-symbol facts computed once, before the open, from completed daily bars.

    Built by `indicators.build_daily_context`. The engine reads it all day (ATR for stop
    distances, the swing trigger level); the agent sees a summary of it in its prompt."""
    symbol: str
    prev_close: float
    prev_high: float
    prev_low: float
    atr: float            # Average True Range: the stock's typical daily move in dollars
    ema_fast: float       # 20-day exponential moving average (short-term trend)
    ema_slow: float       # 50-day exponential moving average (medium-term trend)
    ret_5d: float         # % return over the last 5 sessions
    ret_20d: float        # % return over the last 20 sessions
    dist_from_high_pct: float     # % below the highest close in the lookback
    swing_trigger: Optional[float] = None   # buy-stop level if a setup exists
    swing_reason: str = ""
    earnings_in_days: Optional[int] = None  # trading sessions until the next report, if known


@dataclass
class SymbolPlan:
    """The agent's per-symbol permissions. Defaults allow everything, so a symbol the
    agent did not mention trades normally."""
    allow_intraday: bool = True
    allow_swing: bool = True
    note: str = ""        # short reason, shown on the dashboard


@dataclass
class DayPlan:
    """What the high-level agent hands the engine. It can only restrict risk.

    The default instance (`DayPlan()`) is the "agent not used" plan: neutral regime,
    full risk, no symbol restrictions. It is what the engine runs on when there is no
    API key, when the model call fails, and in every simulation and replay."""
    regime: str = "neutral"               # risk_on | neutral | risk_off
    risk_multiplier: float = 1.0          # clamped to [0, 1]
    symbols: dict = field(default_factory=dict)   # symbol -> SymbolPlan
    summary: str = "Default plan (agent not used)."
    source: str = "default"               # "default" or "agent": who wrote this plan

    def for_symbol(self, symbol: str) -> SymbolPlan:
        """Permissions for one symbol; unmentioned symbols get the permissive default."""
        return self.symbols.get(symbol, SymbolPlan())

    def to_dict(self) -> dict:
        """JSON-friendly form, stored in the `days` table and shown on the dashboard."""
        return {
            "regime": self.regime,
            "risk_multiplier": self.risk_multiplier,
            "summary": self.summary,
            "source": self.source,
            "symbols": {k: asdict(v) for k, v in self.symbols.items()},
        }
