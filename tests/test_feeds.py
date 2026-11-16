"""Tests for streaming some names and polling the rest (feeds.py and the runner's use of it).

WHAT: The split between the websocket and polling (tiers, the stream limit, stability),
      the engine's `can_still_enter`, the poller (bar timing, the forming minute left
      out, repeated trades dropped, failures retried, fatal errors raised), the runner's
      duplicate-bar guard, moves during the day, a short live session end to end, and the
      new config checks.
HOW:  A real Engine and Journal on a temp database, a stub data source with the same
      `latest_trades` / `bars` / `stream` calls as AlpacaData, and a fixed clock where the
      timing matters. The end-to-end test runs `LiveRunner.trade` on a session that
      closes three seconds from now.
WHY:  Polling is how the app watches more than 29 names on the free plan. A name stuck on
      the slow path while it holds a position, a bar counted twice (VWAP, ORB volume) or
      a half-built bar would quietly change what the engine trades.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from openbell.alpaca import FatalFeedError
from openbell.config import Config
from openbell.engine import Engine
from openbell.feeds import Poller, completed_until, plan_feeds
from openbell.journal import Journal
from openbell.market_calendar import MarketCalendar
from openbell.models import Bar, DailyContext, DayPlan, Position, Session, SymbolPlan, Tick

UTC = timezone.utc
SESS = MarketCalendar().sessions_between("2026-11-10", "2026-11-10")[0]   # opens 14:30 UTC
NAMES = ["A", "B", "C", "D", "E"]                     # today's watchlist, in rank order


def day_engine(tmp_path, cfg=None):
    """An engine in the middle of SESS with NAMES on the watchlist and no daily contexts,
    so every name can still try its ORB and none has a swing setup."""
    cfg = cfg or Config()
    e = Engine(cfg, Journal(str(tmp_path / "f.db")))
    e.start_day(SESS, {}, DayPlan(), NAMES)
    e.now = SESS.open + timedelta(minutes=20)
    return e


def hold(e, sym):
    e.positions[sym] = Position(sym, "ORB", 10, 50.0, SESS.open + timedelta(minutes=20), 49.0, 49.0, 52.0)


# ---------------------------------------------------------------- which names stream
def test_can_still_enter(tmp_path):
    e = day_engine(tmp_path)
    assert e.can_still_enter("A") and not e.can_still_enter("ZZZ")      # ZZZ: not on the watchlist
    e.state["A"].orb_taken = True
    assert not e.can_still_enter("A")                                   # ORB used, no swing setup
    e.ctx["A"] = DailyContext("A", 100, 101, 99, 2.0, 99, 95, 1, 3, 2, swing_trigger=101.01)
    assert e.can_still_enter("A")                                       # the swing trigger is still open
    e._attempted.add(("A", "SWING"))
    assert not e.can_still_enter("A")
    e.plan = DayPlan(symbols={"B": SymbolPlan(allow_intraday=False)})
    assert not e.can_still_enter("B") and e.can_still_enter("C")        # the plan forbids B's ORB
    e.now = SESS.close - timedelta(minutes=30)                          # past the entry cutoff
    assert not e.can_still_enter("C")
    e.now = SESS.open + timedelta(minutes=20)
    e.trades_today = e.cfg.risk.max_trades_per_day                      # no entries left today
    assert not e.can_still_enter("C")


def test_plan_feeds_puts_positions_then_live_setups_on_the_stream(tmp_path):
    e = day_engine(tmp_path)
    universe = NAMES + ["SPY"]                                          # as the runner keeps it
    assert plan_feeds(e, universe, "SPY", 2) == (["SPY", "A", "B"], ["C", "D", "E"])   # by rank
    # A's only chance is gone: C, which can still enter, takes its place.
    e.state["A"].orb_taken = True
    stream, poll = plan_feeds(e, universe, "SPY", 2, ["SPY", "A", "B"])
    assert (stream, poll) == (["SPY", "B", "C"], ["A", "D", "E"])
    # E becomes a position: it streams at once, and the lower-ranked of B and C gives way.
    hold(e, "E")
    stream, poll = plan_feeds(e, universe, "SPY", 2, stream)
    assert (stream, poll) == (["SPY", "E", "B"], ["A", "C", "D"])


def test_plan_feeds_keeps_streamed_names_when_nothing_changed(tmp_path):
    e = day_engine(tmp_path)
    # D and E stream (say, from earlier moves). A and B rank higher but need prices no
    # more urgently, so nobody moves: no churn on the websocket.
    assert plan_feeds(e, NAMES, "SPY", 2, ["SPY", "D", "E"]) == (["SPY", "D", "E"], ["A", "B", "C"])
    # A held name off the watchlist still gets a place.
    hold(e, "X")
    assert plan_feeds(e, NAMES, "SPY", 2, ["SPY", "D", "E"])[0] == ["SPY", "X", "D"]


# ---------------------------------------------------------------- the poller
T0 = datetime(2026, 11, 10, 15, 7, tzinfo=UTC)          # 10:07 New York
MINUTES = [Bar(s, T0 - timedelta(minutes=k), 10, 11, 9, 10.5, 100) for s in ("C", "D") for k in range(8, -1, -1)]


class PollData:
    """Same `latest_trades` and `bars` calls as AlpacaData. Serves minute bars from 09:59
    to 10:07 (10:07 being the minute in progress) and whatever trades a test sets."""

    def __init__(self):
        self.trades, self.calls, self.fail, self.fatal = {}, [], 0, False

    async def latest_trades(self, names):
        if self.fatal:
            raise FatalFeedError("bad keys")
        if self.fail:
            self.fail -= 1
            raise ConnectionError("timeout")
        return {s: t for s, t in self.trades.items() if s in names}

    async def bars(self, names, timeframe, start, end, feed=None):
        self.calls.append((start, end))
        return {s: [b for b in MINUTES if b.symbol == s and start <= b.ts <= end] for s in names}


def test_poller_round_gets_finished_bars_then_new_trades():
    data, clock = PollData(), [T0 + timedelta(seconds=4)]
    p = Poller(data, ["C", "D"], 5, now=lambda: clock[0])
    data.trades = {"C": Tick("C", T0 + timedelta(seconds=2), 10.6)}
    ev = asyncio.run(p.poll(["C", "D"]))
    # Bars from the last 5 minutes, oldest first, without the 10:07 bar still forming;
    # then the trade.
    assert data.calls == [(T0 - timedelta(minutes=5), completed_until(clock[0]))]
    bars = [x for x in ev if isinstance(x, Bar)]
    assert [b.ts for b in bars[:2]] == [T0 - timedelta(minutes=5)] * 2 and max(b.ts for b in bars) < T0
    assert ev[-1] == data.trades["C"] and len(ev) == 11
    # 5 s later: still early in the minute, so bars are asked again (Alpaca may be late);
    # the same trade is not repeated.
    clock[0] += timedelta(seconds=5)
    assert all(isinstance(x, Bar) for x in asyncio.run(p.poll(["C", "D"])))
    # 25 s into the minute: bars are not asked again until the next minute.
    clock[0] = T0 + timedelta(seconds=25)
    data.trades["C"] = Tick("C", T0 + timedelta(seconds=24), 10.7)
    assert asyncio.run(p.poll(["C", "D"])) == [data.trades["C"]] and len(data.calls) == 2
    clock[0] = T0 + timedelta(minutes=1, seconds=2)
    asyncio.run(p.poll(["C", "D"]))
    assert len(data.calls) == 3


def test_poller_retries_failures_and_stops_on_fatal_errors():
    data = PollData()
    data.fail = 2
    data.trades = {"C": Tick("C", T0, 10.0)}
    p = Poller(data, ["C"], 0.01)
    p.last_ok = 0.0

    async def collect(seconds):
        out = []
        async for ev in p.run(datetime.now(UTC) + timedelta(seconds=seconds)):
            out.append(ev)
        return out
    ticks = [x for x in asyncio.run(collect(0.5)) if isinstance(x, Tick)]
    assert ticks == [data.trades["C"]] and p.last_ok > 0     # recovered; the trade came once
    data.fatal = True
    with pytest.raises(FatalFeedError):
        asyncio.run(collect(1))


# ---------------------------------------------------------------- the runner
def make_runner(tmp_path, monkeypatch, limit=2):
    from openbell import runner as R
    monkeypatch.setenv("ALPACA_API_KEY", "k")          # AlpacaData refuses to construct without keys
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    cfg = Config()
    cfg.db_path = str(tmp_path / "live.db")
    cfg.feed.stream_limit = limit
    r = R.LiveRunner(cfg)
    r.universe = NAMES + ["SPY"]
    return r


def feed_events(r):
    return [x["message"] for x in r.journal.rows("SELECT message FROM events WHERE kind='FEED' ORDER BY id")]


def test_runner_moves_a_new_position_onto_the_stream(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)
    r.engine.start_day(SESS, {}, DayPlan(), NAMES)
    streamed, polled = r.streamed, r.polled
    r.refresh_feeds()
    assert (r.streamed, r.polled) == (["SPY", "A", "B"], ["C", "D", "E"])
    assert r.info["feeds"] == {"stream": ["SPY", "A", "B"], "poll": ["C", "D", "E"], "poll_seconds": 5.0}
    hold(r.engine, "D")                                # a polled name was just bought
    r.refresh_feeds()
    r.refresh_feeds()                                  # nothing new: no second message
    assert (r.streamed, r.polled) == (["SPY", "D", "A"], ["B", "C", "E"])
    assert r.streamed is streamed and r.polled is polled   # changed in place: the sources see it
    assert feed_events(r) == ["3 names streamed in real time, 3 polled every 5s",
                              "D now streamed in real time; B now polled"]


def test_runner_counts_each_minute_bar_once(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)
    r.engine.start_day(SESS, {}, DayPlan(), NAMES)
    bar = Bar("A", SESS.open, 10, 11, 9, 10.5, 100)
    r.deliver(bar)
    r.deliver(bar)                                     # the same bar from the other source
    r.deliver(Bar("A", SESS.open - timedelta(minutes=1), 1, 1, 1, 1, 1))   # older: dropped
    assert r.engine.state["A"].bars == 1
    r.deliver(Tick("A", SESS.open + timedelta(seconds=70), 10.8))
    assert r.engine.last["A"] == 10.8                  # ticks always pass


class LiveData:
    """Stand-in for AlpacaData during a session: the stream sends one trade for its first
    symbol, polled names trade at 20.0."""

    def __init__(self):
        self.stream_symbols = None

    async def stream(self, symbols, until):
        self.stream_symbols = symbols
        yield Tick(symbols[1], datetime.now(UTC), 10.0)
        while datetime.now(UTC) < until:
            await asyncio.sleep(0.05)

    async def latest_trades(self, names):
        return {s: Tick(s, datetime.now(UTC), 20.0) for s in names}

    async def bars(self, names, timeframe, start, end, feed=None):
        return {s: [] for s in names}


def test_live_session_streams_some_names_and_polls_the_rest(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)
    r.cfg.feed.poll_seconds = 0.2
    r.data = LiveData()
    now = datetime.now(UTC)
    sess = Session(now.date(), now - timedelta(minutes=1), now + timedelta(seconds=3))
    r.engine.start_day(sess, {}, DayPlan(), NAMES)
    asyncio.run(r.trade(sess))
    assert r.data.stream_symbols is r.streamed         # the stream follows the runner's list
    assert r.engine.last["A"] == 10.0                  # streamed
    assert all(r.engine.last[s] == 20.0 for s in ("C", "D", "E"))   # polled
    assert r.engine.feed_ok and r.info["feeds"] is None   # hidden again after the close
    assert feed_events(r)[0] == "3 names streamed in real time, 3 polled every 0.2s"


# ---------------------------------------------------------------- config
def test_config_limits_the_stream_not_the_watchlist():
    cfg = Config()
    cfg.validate()                                     # 45 names on the free feed is fine
    assert cfg.scanner.max_watchlist == 45 and cfg.feed.stream_limit == 29
    cfg.feed.stream_limit = 30
    with pytest.raises(ValueError, match="30 symbols"):
        cfg.validate()
    cfg.feed.feed = "sip"                              # the paid feed has no symbol cap
    cfg.validate()
    cfg.feed.poll_seconds = 0.5
    with pytest.raises(ValueError, match="poll_seconds"):
        cfg.validate()
