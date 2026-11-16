# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenBell is an autonomous **paper-trading** agent for the US cash session (Python, asyncio). No real orders are ever sent; fills are simulated inside the app. See README.md for strategy rules, scanner details and known limits.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # full suite, no keys or network needed
python -m pytest tests/test_engine.py::test_name -q   # single test
python -m openbell sim --days 20 --serve              # synthetic smoke run + dashboard (with charts), no keys
python -m openbell replay --start 2026-09-01 --end 2026-10-30   # real historical minute bars (Alpaca keys)
python -m openbell doctor                             # checks every external service with real keys
python -m openbell run                                # live loop, dashboard on 127.0.0.1:8000
```

`--config` (default `config.yaml`) goes before the subcommand. `sim` and `replay` delete and recreate their own DB (`sim.db` / `replay.db`); `run` uses `db_path` from config (`openbell.db`). Secrets come from `.env` / environment only (`ALPACA_API_KEY`, `ALPACA_API_SECRET`, optional `FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`), never from YAML.

## Architecture

**One deterministic engine, three drivers.** `Engine` (`engine.py`) holds strategies (ORB intraday, SWING trend-pullback), position sizing, risk limits and the paper broker. It is driven purely by `on_tick`, `on_bar`, `on_clock` plus `start_day`/`end_day`, so identical code runs:
- live: `runner.LiveRunner` feeds Alpaca websocket ticks/bars and a 1s clock;
- `sim`: `replay.run_simulation` over a seeded random-walk `SyntheticMarket`;
- `replay`: `replay.replay_session` over real Alpaca minute bars.

`replay.bar_ticks` expands each minute bar into four ticks (O-L-H-C on up bars, O-H-L-C on down bars) so stops/targets trigger inside the bar. Keep new engine logic driven by these events, never by wall-clock or I/O, or the three paths diverge.

**The LLM is never in the price loop.** `agent.py` runs three blocking calls, each wrapped in `asyncio.to_thread` by the runner: `select` (approve/reject scanner candidates), `make_plan` (pre-market `DayPlan`: regime, `risk_multiplier` clamped to [0,1], per-symbol allow flags), and `review` (post-close text). The plan can only *restrict* the engine; hard limits in `config.yaml` are enforced regardless. `select` and `make_plan` request structured output (`output_config.format` with `SELECT_SCHEMA`/`PLAN_SCHEMA`; per-ticker answers are lists with a `ticker` field because schemas can't have arbitrary keys), and `agent.build_request` is shared with `doctor.py` so the health check sends the live request. Keep `agent.max_tokens` large: thinking is on by default and counts against it (2000 truncated the 45-candidate reply). Every model reply goes through `parse_plan`/`parse_selection`, which clamp and validate; any failure falls back to `DayPlan()` / numeric ranking (`select` returning `None` means "rank by score"). Preserve this fail-safe property when touching the agent.

**Daily cycle** (`runner.py`): sleep until `premarket_lead_minutes` before the open → `prepare` (scanner builds watchlist → daily contexts via `replay.contexts_for` → agent plan → `engine.start_day`) → `trade` (stream + poll until close, stale-feed detection blocks entries) → `after_close` (`end_day`, agent review). `FatalFeedError` propagates and stops the process; other exceptions are logged as `ERROR` events and the loop retries.

**Scanner pipeline** (`scanner.py`): weekly liquid universe (stored in `universe` table) → daily numeric scan producing `Candidate`s of kind SWING / INTRADAY / BOTH → enrichment (Alpaca news, Finnhub earnings/profitability) → agent gate → `apply_decisions` (held positions always first). Replay/sim use the numeric-only path `replay.pick_watchlist`. The watchlist is `scanner.max_watchlist` (45) names; its rank order matters, because it decides which names get the websocket.

**Live feeds** (`feeds.py`): the free IEX websocket allows 30 symbols, so only the benchmark + `feed.stream_limit` (29, enforced by `Config.validate` on `iex`) stream; the rest of the watchlist is polled by `Poller` (`AlpacaData.latest_trades` every `feed.poll_seconds`, plus finished 1-minute bars via `bars()` a few times a minute, never the bar still forming: `completed_until`). `plan_feeds` re-splits every second by tier (open position > `Engine.can_still_enter` > the rest; ties keep names already streaming, then rank), and `LiveRunner.refresh_feeds` edits `runner.streamed` / `runner.polled` **in place**: `AlpacaData.stream` and the poller re-read those lists, and the stream sends `unsubscribe`/`subscribe` on the open connection. All events go through `LiveRunner.deliver`, which drops a minute bar already delivered (`last_bar`), since the two sources overlap around a move. Sim and replay do not use any of this.

**Journal is durable state, not just a log.** `journal.py` (SQLite) stores cash, positions, trades, events, equity curve, days (plan/review) and scan rows. `Engine.__init__` loads cash and positions from it, and `start_day` restores the day's start equity, trade count and attempted (symbol, strategy) pairs, so a mid-session restart keeps limits and one-attempt-per-symbol intact. The runner additionally rebuilds VWAP/opening range from today's bars with `engine.warmup = True` (no orders during warmup). Any new per-day engine state must be recoverable the same way.

**Config** (`config.py`): nested dataclasses with defaults; `config.yaml` is merged over them and **unknown keys raise**, so a new setting needs a dataclass field first. Some tunables (e.g. `min_range_bars`, `pullback_atr`, `max_chase_atr`, `premarket_lead_minutes`) exist only as dataclass defaults.

**Dashboard**: `api.py` (FastAPI, read-only, no auth) serves `dashboard.html` and JSON from `engine.snapshot()` and the journal. Keep it bound to localhost.

**Charts** (`charts.py`): `GET /api/chart/{symbol}?range=1D|5D|1M|6M|1Y` returns candles plus EMA (the swing periods), VWAP and volume average, and `overlays()` adds trade markers and position/setup levels from the engine and journal. `ChartService` takes any source with `AlpacaData.bars()`'s signature: the runner passes its `AlpacaData` (SIP feed, data ending 16 min ago, fallback to the configured feed on `FatalFeedError`), and `sim --serve` passes the `SyntheticMarket` that `run_simulation` returns. Intraday ranges fetch 1-minute bars, keep 09:30-16:00 only and merge them into candles aligned to the open. Intraday times are sent as New York wall-clock time encoded as UTC epoch seconds, because the chart library displays UTC. The page draws with Lightweight Charts 4.2.0 from unpkg, and series colours are CSS tokens (`--ema-fast`, `--ema-slow`, `--vwap`). The route is `async` and runs on the trading event loop when live, so it must never block. Charts are display only; never feed them back into the engine.

## Testing notes

- All times are timezone-aware UTC internally; `market_calendar.py` (exchange_calendars, NYSE) produces `Session(day, open, close, early_close)`. Display conversion uses `api.display_tz`.
- `tests/fake_services.py` is a local FastAPI/websocket stand-in for Alpaca and Finnhub. `tests/test_live_protocol.py` points the clients at it by monkeypatching module-level URL constants (`alpaca.DATA_URL`, `NEWS_URL`, `ASSETS_URL`, `STREAM_URL`, `LATEST_TRADES_URL`, `finnhub.BASE`, and `doctor.*`). New endpoints should be added as module constants the same way so they can be redirected in tests.
- The real services have not been exercised by the test suite; `openbell doctor` is the only check against them.
- Always pass `encoding="utf-8"` when reading or writing text files. The dev machine is Windows with a cp1255 default code page, and `dashboard.html` contains non-ASCII characters; a bare `read_text()` crashed the dashboard (`tests/test_api.py` guards this).
