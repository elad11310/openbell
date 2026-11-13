"""Command-line entry point: `python -m openbell <command>`. START READING HERE.

WHAT: Parses the command line and dispatches to one of four commands:
        run      trade live NYSE sessions on paper, forever    (needs Alpaca keys)
        sim      synthetic market through the real engine       (no keys needed)
        replay   real historical minute bars through the engine (needs Alpaca keys)
        doctor   check every external service with your keys
      Global option: `--config PATH` (default config.yaml), placed before the command.
HOW:  `python -m openbell` makes Python run this file as `__main__`. `argparse` builds
      one sub-parser per command and attaches the function to call with
      `set_defaults(fn=...)`; `main()` then simply calls `args.fn(args)`.
      Each command imports its modules inside the function (lazy imports), so a
      command only loads what it uses (e.g. `doctor` never imports the runner or the
      web app).
WHY:  One entry point keeps start-up uniform: every command loads the same config
      (`load_config`: .env + config.yaml + validation) and builds the same Engine and
      Journal, so the simulation and the replay exercise exactly the code that runs live.
      Each command returns an integer exit code (0 = success) for scripts and systemd.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from .config import load_config


def cmd_run(args) -> int:
    """Live paper trading. Hands control to LiveRunner.run(), which loops forever."""
    from .alpaca import FatalFeedError
    from .runner import LiveRunner
    cfg = load_config(args.config)
    try:
        # asyncio.run starts the event loop that the runner, the stream and the dashboard share.
        asyncio.run(LiveRunner(cfg).run())
    except FatalFeedError as e:
        # Unfixable data problem (missing/bad keys, plan limits): explain and exit with code 2.
        print(f"Cannot start: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass                                # Ctrl+C is a normal way to stop
    return 0


def cmd_sim(args) -> int:
    """Synthetic sessions through the real engine. Tests the plumbing, not the strategy."""
    from .engine import Engine
    from .journal import Journal
    from .market_calendar import MarketCalendar
    from .replay import run_simulation
    cfg = load_config(args.config)
    cfg.db_path = args.db
    # Start from an empty journal each time, so results never mix with a previous run.
    if os.path.exists(args.db):
        os.remove(args.db)
    cal = MarketCalendar()
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    # Real NYSE session dates (holidays and half-days included), ~700 calendar days back.
    sessions = cal.sessions_between(end - timedelta(days=700), end)
    journal = Journal(args.db)
    engine = Engine(cfg, journal)
    # The last `days` sessions are traded; the 300 sessions before them only generate
    # price history so the indicators (EMA50, 3-month strength, ...) have data to work on.
    run_simulation(cfg, engine, sessions[-args.days:], sessions[-args.days - 300:-args.days], seed=args.seed)
    print(json.dumps({"days": args.days, "equity": round(engine.equity(), 2),
                      "open_positions": len(engine.positions), "stats": journal.stats()}, indent=2))
    if args.serve:
        # Optionally open the dashboard on the finished simulation (blocks until Ctrl+C).
        import uvicorn
        from .api import create_app
        print(f"dashboard on http://{cfg.api.host}:{cfg.api.port} (simulated data)")
        uvicorn.run(create_app(engine, journal, cfg, {}), host=cfg.api.host, port=cfg.api.port,
                    log_level="warning")
    return 0


def cmd_replay(args) -> int:
    """Run the engine over real historical minute bars from Alpaca."""
    from .alpaca import AlpacaData
    from .engine import Engine
    from .journal import Journal
    from .market_calendar import MarketCalendar
    from .models import DayPlan
    from . import scanner
    from .replay import contexts_for, pick_watchlist, replay_session
    cfg = load_config(args.config)
    cfg.db_path = args.db
    if os.path.exists(args.db):
        os.remove(args.db)
    data = AlpacaData(cfg.alpaca_key, cfg.alpaca_secret, cfg.feed.feed)
    sessions = MarketCalendar().sessions_between(args.start, args.end)
    journal = Journal(args.db)
    engine = Engine(cfg, journal)

    async def go() -> None:
        # The scan here is numeric only: past headlines and agent decisions are not replayed,
        # and the universe is today's most liquid stocks (survivorship bias).
        names = list(cfg.symbols)
        if cfg.scanner.enabled:
            names = await scanner.ensure_universe(data, journal, cfg, datetime.now(timezone.utc))
        names = list(dict.fromkeys(names + [cfg.benchmark]))
        # Download all daily history once, up front (420 days before the first session
        # gives the indicators their warm-up), in chunks of 200 symbols per request.
        daily: dict = {}
        for i in range(0, len(names), 200):
            daily.update(await data.bars(names[i:i + 200], "1Day", sessions[0].open - timedelta(days=420),
                                         sessions[-1].close, feed="sip"))
        # Then, session by session: pick the watchlist from data before that day, fetch
        # only that day's minute bars for the watched names, replay them through the engine.
        for s in sessions:
            watch = pick_watchlist(cfg, engine, daily, s)
            live = list(dict.fromkeys(watch + [cfg.benchmark]))
            minute = await data.bars(live, "1Min", s.open, s.close, feed="sip")
            # Flatten {symbol: [bars]} into one list; replay_session sorts it by time.
            replay_session(engine, s, contexts_for(cfg, {x: daily.get(x, []) for x in live}, s),
                           DayPlan(), [b for v in minute.values() for b in v], watch)
            print(f"{s.day}  equity {engine.equity():,.2f}  watching {len(watch)}")

    asyncio.run(go())
    print(json.dumps(journal.stats(), indent=2))
    return 0


def cmd_doctor(args) -> int:
    """Check Alpaca, Finnhub and Anthropic with the configured keys; exit 1 on any failure."""
    from .doctor import run
    return asyncio.run(run(load_config(args.config)))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="openbell")
    ap.add_argument("--config", default="config.yaml")
    # dest="cmd", required=True: running with no command prints usage instead of doing nothing.
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="trade live sessions on paper (needs Alpaca keys)").set_defaults(fn=cmd_run)
    sub.add_parser("doctor", help="check Alpaca, Finnhub and Anthropic with your keys").set_defaults(fn=cmd_doctor)
    s = sub.add_parser("sim", help="synthetic smoke run, no keys needed")
    s.add_argument("--days", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)       # same seed -> identical run
    s.add_argument("--db", default="sim.db")
    s.add_argument("--serve", action="store_true", help="open the dashboard on the result")
    s.set_defaults(fn=cmd_sim)
    r = sub.add_parser("replay", help="replay real history (needs Alpaca keys)")
    r.add_argument("--start", required=True)            # e.g. 2026-09-01
    r.add_argument("--end", required=True)
    r.add_argument("--db", default="replay.db")
    r.set_defaults(fn=cmd_replay)
    args = ap.parse_args()
    return args.fn(args)                                # dispatch to the chosen cmd_* function


if __name__ == "__main__":
    # sys.exit passes the command's return value to the shell as the exit code.
    sys.exit(main())
