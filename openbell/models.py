from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: datetime          # bar START, timezone-aware UTC
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass(frozen=True)
class Tick:
    symbol: str
    ts: datetime          # timezone-aware UTC
    price: float
    size: float = 0.0


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime        # UTC
    close: datetime       # UTC
    early_close: bool = False


@dataclass
class Signal:
    symbol: str
    strategy: str         # "ORB" (intraday) or "SWING"
    entry: float
    stop: float
    target: Optional[float]
    reason: str


@dataclass
class Position:
    symbol: str
    strategy: str
    qty: int
    entry_px: float
    entry_ts: datetime
    stop: float
    initial_stop: float
    target: Optional[float] = None
    high_close: float = 0.0       # highest daily close since entry (swing trail)
    exit_flag: Optional[str] = None  # set pre-market: exit at the open, with reason

    @property
    def risk_per_share(self) -> float:
        return max(self.entry_px - self.initial_stop, 1e-9)

    def to_row(self) -> dict:
        d = asdict(self)
        d["entry_ts"] = self.entry_ts.isoformat()
        return d


@dataclass
class DailyContext:
    """Per-symbol facts computed once, before the open, from completed daily bars."""
    symbol: str
    prev_close: float
    prev_high: float
    prev_low: float
    atr: float
    ema_fast: float
    ema_slow: float
    ret_5d: float
    ret_20d: float
    dist_from_high_pct: float     # % below the highest close in the lookback
    swing_trigger: Optional[float] = None   # buy-stop level if a setup exists
    swing_reason: str = ""
    earnings_in_days: Optional[int] = None  # trading sessions until the next report, if known


@dataclass
class SymbolPlan:
    allow_intraday: bool = True
    allow_swing: bool = True
    note: str = ""


@dataclass
class DayPlan:
    """What the high-level agent hands the engine. It can only restrict risk."""
    regime: str = "neutral"               # risk_on | neutral | risk_off
    risk_multiplier: float = 1.0          # clamped to [0, 1]
    symbols: dict = field(default_factory=dict)   # symbol -> SymbolPlan
    summary: str = "Default plan (agent not used)."
    source: str = "default"

    def for_symbol(self, symbol: str) -> SymbolPlan:
        return self.symbols.get(symbol, SymbolPlan())

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "risk_multiplier": self.risk_multiplier,
            "summary": self.summary,
            "source": self.source,
            "symbols": {k: asdict(v) for k, v in self.symbols.items()},
        }
