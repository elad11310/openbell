# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenBell is an autonomous **paper-trading** agent for the US cash session (Python, asyncio). No real orders are ever sent; fills are simulated inside the app. See README.md for strategy rules, scanner details and known limits.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # full suite, no keys or network needed
python -m pytest tests/test_engine.py::test_name -q   # single test
python -m openbell sim --days 20 --serve              # synthetic smoke run + dashboard, no keys
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

**Daily cycle** (`runner.py`): sleep until `premarket_lead_minutes` before the open → `prepare` (scanner builds watchlist → daily contexts via `replay.contexts_for` → agent plan → `engine.start_day`) → `trade` (stream until close, stale-feed detection blocks entries) → `after_close` (`end_day`, agent review). `FatalFeedError` propagates and stops the process; other exceptions are logged as `ERROR` events and the loop retries.

**Scanner pipeline** (`scanner.py`): weekly liquid universe (stored in `universe` table) → daily numeric scan producing `Candidate`s of kind SWING / INTRADAY / BOTH → enrichment (Alpaca news, Finnhub earnings/profitability) → agent gate → `apply_decisions` (held positions always first). Replay/sim use the numeric-only path `replay.pick_watchlist`. Watchlist is capped at 29 + benchmark because the free IEX stream allows 30 symbols (`Config.validate` enforces this).

**Journal is durable state, not just a log.** `journal.py` (SQLite) stores cash, positions, trades, events, equity curve, days (plan/review) and scan rows. `Engine.__init__` loads cash and positions from it, and `start_day` restores the day's start equity, trade count and attempted (symbol, strategy) pairs, so a mid-session restart keeps limits and one-attempt-per-symbol intact. The runner additionally rebuilds VWAP/opening range from today's bars with `engine.warmup = True` (no orders during warmup). Any new per-day engine state must be recoverable the same way.

**Config** (`config.py`): nested dataclasses with defaults; `config.yaml` is merged over them and **unknown keys raise**, so a new setting needs a dataclass field first. Some tunables (e.g. `min_range_bars`, `pullback_atr`, `max_chase_atr`, `premarket_lead_minutes`) exist only as dataclass defaults.

**Dashboard**: `api.py` (FastAPI, read-only, no auth) serves `dashboard.html` and JSON from `engine.snapshot()` and the journal. Keep it bound to localhost.

## Testing notes

- All times are timezone-aware UTC internally; `market_calendar.py` (exchange_calendars, NYSE) produces `Session(day, open, close, early_close)`. Display conversion uses `api.display_tz`.
- `tests/fake_services.py` is a local FastAPI/websocket stand-in for Alpaca and Finnhub. `tests/test_live_protocol.py` points the clients at it by monkeypatching module-level URL constants (`alpaca.DATA_URL`, `NEWS_URL`, `ASSETS_URL`, `STREAM_URL`, `finnhub.BASE`, and `doctor.*`). New endpoints should be added as module constants the same way so they can be redirected in tests.
- The real services have not been exercised by the test suite; `openbell doctor` is the only check against them.
- Always pass `encoding="utf-8"` when reading or writing text files. The dev machine is Windows with a cp1255 default code page, and `dashboard.html` contains non-ASCII characters; a bare `read_text()` crashed the dashboard (`tests/test_api.py` guards this).
