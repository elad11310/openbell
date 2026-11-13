"""Configuration: every tunable number in one typed object, plus secret loading.

WHAT: Defines `Config`, a tree of dataclasses (risk, intraday, swing, agent, feed,
      scanner, api), and `load_config()`, which builds it from defaults + config.yaml +
      environment variables.
HOW:  1. `load_dotenv()` copies KEY=VALUE lines from `.env` into the process
         environment (without overwriting variables that are already set).
      2. `Config()` is created with the defaults written in the dataclasses below.
      3. `_merge()` walks the YAML dict and overwrites matching fields, recursing into
         nested sections. An unknown key raises an error instead of being ignored.
      4. `validate()` normalises symbols and rejects settings that cannot work.
      Secrets are never stored on the object: they are read from `os.environ` on demand
      through properties (`cfg.alpaca_key`, ...).
WHY:  * Typed defaults mean the app runs even with an empty config.yaml, and every
        module reads settings as attributes (`cfg.risk.max_positions`) instead of
        fragile dictionary lookups.
      * Failing on unknown keys catches typos: `max_postions: 3` would otherwise be
        silently ignored and the app would trade with the default 6.
      * Keeping secrets out of YAML means config.yaml can be committed and shared.
      * Some settings (e.g. `min_range_bars`, `pullback_atr`, `max_chase_atr`,
        `premarket_lead_minutes`) exist only here as defaults; they can still be set in
        config.yaml because the merge accepts any field defined below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class RiskCfg:
    """Hard limits enforced by the engine on every entry, whatever the agent says."""
    risk_per_trade_pct: float = 0.5      # % of equity lost if the stop is hit
    max_position_pct: float = 20.0       # % of equity in one name
    max_gross_exposure_pct: float = 100.0  # total market value of positions / equity (100 = no leverage)
    max_positions: int = 6
    max_trades_per_day: int = 8
    daily_loss_limit_pct: float = 2.0    # halt new entries + flatten intraday
    slippage_bps: float = 2.0            # 1 bp = 0.01%; added to buys, taken off sells


@dataclass
class IntradayCfg:
    """Opening-range breakout (ORB) settings. All positions are closed before the bell."""
    enabled: bool = True
    opening_range_minutes: int = 15
    min_range_bars: int = 3              # bars needed for a valid opening range
    rvol_min: float = 1.2                # breakout bar volume vs session average
    target_r: float = 2.0                # take profit at 2x the risk taken
    max_stop_atr: float = 0.5            # stop no further than this x daily ATR
    no_entry_minutes_before_close: int = 60
    flatten_minutes_before_close: int = 10


@dataclass
class SwingCfg:
    """Trend-pullback settings. Positions are held for days with a trailing stop."""
    enabled: bool = True
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    pullback_atr: float = 1.0            # prior low within this x ATR of fast EMA
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    max_chase_atr: float = 0.5           # skip if price is this far past trigger
    max_hold_days: int = 30              # calendar days
    wait_minutes_after_open: int = 5     # let the opening auction noise settle first


@dataclass
class AgentCfg:
    """The LLM that writes the day plan, screens scanner candidates and reviews the day."""
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    max_tokens: int = 2000


@dataclass
class FeedCfg:
    """Live market-data stream settings."""
    feed: str = "iex"                    # "iex" (free) or "sip" (paid)
    stale_seconds: int = 45              # block entries if the stream is silent
    premarket_lead_minutes: int = 25     # how long before the open to wake up and scan


@dataclass
class ScannerCfg:
    """Pre-market scanner settings: which stocks get watched today."""
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
    """Read-only dashboard server."""
    host: str = "127.0.0.1"              # localhost only: the dashboard has no login
    port: int = 8000
    display_tz: str = "Asia/Jerusalem"   # the browser shows times in this zone


@dataclass
class Config:
    """Root of the settings tree. `field(default_factory=...)` gives every Config its
    own fresh sub-objects; a plain default would share one instance between all
    Configs, so changing `cfg.risk` in one test would leak into the next."""
    # Used only when the scanner is disabled or fails.
    symbols: list = field(default_factory=lambda: [
        "NVDA", "AVGO", "MRVL", "INTC", "AMD", "CAT", "MU",
        "GOOGL", "IBM", "PLTR", "CRWD", "IONQ", "BE"])
    benchmark: str = "SPY"               # market proxy: always streamed, never traded
    starting_equity: float = 100_000.0   # paper cash on the very first run
    db_path: str = "openbell.db"         # SQLite journal file
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
        """Normalise and sanity-check settings. Raises ValueError so the app refuses to
        start rather than trade with a broken configuration."""
        if not self.symbols:
            raise ValueError("config: symbols is empty")
        # Upper-case, then de-duplicate while keeping order (dict keys keep insertion order;
        # a set would scramble it).
        self.symbols = [s.upper() for s in self.symbols]
        self.symbols = list(dict.fromkeys(self.symbols))
        # The free IEX websocket accepts at most 30 symbols; one slot is the benchmark.
        if self.feed.feed == "iex" and self.scanner.max_watchlist > 29:
            raise ValueError("config: the free IEX stream allows 30 symbols (29 + benchmark)")
        if len(self.symbols) > self.scanner.max_watchlist:
            raise ValueError("config: symbols is longer than scanner.max_watchlist")
        # Above 2% per trade, a few losses in a row would do serious damage.
        if not 0 < self.risk.risk_per_trade_pct <= 2:
            raise ValueError("config: risk_per_trade_pct must be in (0, 2]")


def _merge(obj, data: dict):
    """Copy values from the YAML dict onto the dataclass `obj`, recursively.

    For each YAML key: if the current attribute is itself a dataclass (a section such as
    `risk:`), recurse into it with the nested dict; otherwise overwrite the value.
    Unknown keys raise, which turns typos into a clear startup error."""
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
    """Minimal .env reader (avoids a python-dotenv dependency).

    Skips blank lines and comments, splits on the first '=', strips optional quotes.
    `setdefault` means a variable already set in the real environment wins, so a
    deployment can override the file."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | None = "config.yaml") -> Config:
    """Entry point used by every CLI command: .env -> defaults -> YAML -> validate."""
    load_dotenv()
    cfg = Config()
    if path and Path(path).exists():
        # `or {}` handles an empty YAML file, which safe_load returns as None.
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    cfg.validate()
    return cfg
