from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class RiskCfg:
    risk_per_trade_pct: float = 0.5      # % of equity lost if the stop is hit
    max_position_pct: float = 20.0       # % of equity in one name
    max_gross_exposure_pct: float = 100.0
    max_positions: int = 6
    max_trades_per_day: int = 8
    daily_loss_limit_pct: float = 2.0    # halt new entries + flatten intraday
    slippage_bps: float = 2.0


@dataclass
class IntradayCfg:
    enabled: bool = True
    opening_range_minutes: int = 15
    min_range_bars: int = 3              # bars needed for a valid opening range
    rvol_min: float = 1.2                # breakout bar volume vs session average
    target_r: float = 2.0
    max_stop_atr: float = 0.5            # stop no further than this x daily ATR
    no_entry_minutes_before_close: int = 60
    flatten_minutes_before_close: int = 10


@dataclass
class SwingCfg:
    enabled: bool = True
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    pullback_atr: float = 1.0            # prior low within this x ATR of fast EMA
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    max_chase_atr: float = 0.5           # skip if price is this far past trigger
    max_hold_days: int = 30              # calendar days
    wait_minutes_after_open: int = 5


@dataclass
class AgentCfg:
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    max_tokens: int = 2000


@dataclass
class FeedCfg:
    feed: str = "iex"                    # "iex" (free) or "sip" (paid)
    stale_seconds: int = 45              # block entries if the stream is silent
    premarket_lead_minutes: int = 25


@dataclass
class ScannerCfg:
    enabled: bool = True
    universe_size: int = 400             # most liquid US stocks kept for scanning
    universe_refresh_days: int = 7
    min_price: float = 10.0
    min_dollar_volume: float = 50e6      # 20-day average daily dollar volume
    min_rel_strength_pct: float = 0.0    # swing only: 3-month return minus the benchmark's
    min_gap_pct: float = 1.0             # intraday: pre-market gap up that counts as 'in play'
    min_premarket_share: float = 0.005   # ... with pre-market dollar volume >= this share of a normal day
    candidates: int = 45                 # shortlist shown to the agent
    max_watchlist: int = 29              # live names per day (free stream: 30 incl. benchmark)
    news_lookback_hours: int = 48
    earnings_blackout_days: int = 3      # no new swing entry this many sessions before a report


@dataclass
class ApiCfg:
    host: str = "127.0.0.1"
    port: int = 8000
    display_tz: str = "Asia/Jerusalem"


@dataclass
class Config:
    # Used only when the scanner is disabled or fails.
    symbols: list = field(default_factory=lambda: [
        "NVDA", "AVGO", "MRVL", "INTC", "AMD", "CAT", "MU",
        "GOOGL", "IBM", "PLTR", "CRWD", "IONQ", "BE"])
    benchmark: str = "SPY"
    starting_equity: float = 100_000.0
    db_path: str = "openbell.db"
    risk: RiskCfg = field(default_factory=RiskCfg)
    intraday: IntradayCfg = field(default_factory=IntradayCfg)
    swing: SwingCfg = field(default_factory=SwingCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    feed: FeedCfg = field(default_factory=FeedCfg)
    scanner: ScannerCfg = field(default_factory=ScannerCfg)
    api: ApiCfg = field(default_factory=ApiCfg)

    # secrets come from the environment only, never from the YAML file
    @property
    def alpaca_key(self) -> str:
        return os.environ.get("ALPACA_API_KEY", "")

    @property
    def alpaca_secret(self) -> str:
        return os.environ.get("ALPACA_API_SECRET", "")

    @property
    def finnhub_key(self) -> str:
        return os.environ.get("FINNHUB_API_KEY", "")

    @property
    def anthropic_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("config: symbols is empty")
        self.symbols = [s.upper() for s in self.symbols]
        self.symbols = list(dict.fromkeys(self.symbols))
        if self.feed.feed == "iex" and self.scanner.max_watchlist > 29:
            raise ValueError("config: the free IEX stream allows 30 symbols (29 + benchmark)")
        if len(self.symbols) > self.scanner.max_watchlist:
            raise ValueError("config: symbols is longer than scanner.max_watchlist")
        if not 0 < self.risk.risk_per_trade_pct <= 2:
            raise ValueError("config: risk_per_trade_pct must be in (0, 2]")


def _merge(obj, data: dict):
    names = {f.name for f in fields(obj)}
    for k, v in (data or {}).items():
        if k not in names:
            raise ValueError(f"config: unknown key '{k}'")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | None = "config.yaml") -> Config:
    load_dotenv()
    cfg = Config()
    if path and Path(path).exists():
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    cfg.validate()
    return cfg
