"""Tests for the dashboard's price charts (charts.py, the /api/chart route, the
indicator series and the simulation's bar source).

WHAT: Candle merging, regular-session filtering, VWAP reset, EMA warm-up, the range
      windows, the feed fallback, the cache, trade markers and price levels, the API's
      error codes, and charting a real simulation end to end.
HOW:  A fake bar source with the same `bars()` call as AlpacaData serves hand-made
      minute and daily bars for Mon 2026-11-09 and Tue 2026-11-10 (winter time: the
      session is 14:30-21:00 UTC). "Now" is fixed after Tuesday's close and the delay is
      zero, so every expected value can be worked out by hand.
WHY:  Charts are display only, but a wrong candle, a misplaced marker or an indicator
      line that disagrees with the engine's own numbers would mislead whoever reads the
      dashboard. No network or keys are needed.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from openbell.alpaca import FatalFeedError
from openbell.api import create_app
from openbell.charts import ChartService, chart_time, overlays
from openbell.config import Config
from openbell.engine import Engine
from openbell.indicators import ema, ema_series, sma_series, vwap_series
from openbell.journal import Journal
from openbell.market_calendar import MarketCalendar, NY
from openbell.models import Bar, DailyContext, Position
from openbell.replay import run_simulation

UTC = timezone.utc
MON = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)          # 09:30 New York
TUE = datetime(2026, 11, 10, 14, 30, tzinfo=UTC)
AFTER = datetime(2026, 11, 10, 22, 0, tzinfo=UTC)         # "now": after Tuesday's close
CAL = MarketCalendar()                                    # loading it is slow; share it


def session_minutes(open_, start_px):
    """390 one-minute bars rising 1 cent a minute, volume 1000 each."""
    return [Bar("AAA", open_ + timedelta(minutes=i), start_px + i * 0.01, start_px + i * 0.01 + 0.05,
                start_px + i * 0.01 - 0.05, start_px + i * 0.01 + 0.02, 1000) for i in range(390)]


MINUTES = (session_minutes(MON, 100.0)
           # A pre-market bar (08:00 New York) with a huge price and volume: must be dropped.
           + [Bar("AAA", TUE - timedelta(minutes=90), 500, 500, 500, 500, 10**9)]
           + session_minutes(TUE, 105.0))
# One daily bar per session for ~1.5 years, stamped at midnight New York like Alpaca's.
DAILY = [Bar("AAA", datetime(s.day.year, s.day.month, s.day.day, tzinfo=NY).astimezone(UTC),
             50 + i, 51 + i, 49 + i, 50.5 + i, 1e6 + i)
         for i, s in enumerate(CAL.sessions_between("2025-06-02", "2026-11-10"))]


class FakeSource:
    """Same `bars()` call as AlpacaData. Optionally refuses the SIP feed, like an
    account without access to it."""

    def __init__(self, refuse_sip=False):
        self.refuse_sip, self.calls = refuse_sip, []

    async def bars(self, symbols, timeframe, start, end, feed=None):
        self.calls.append((timeframe, feed))
        if self.refuse_sip and feed == "sip":
            raise FatalFeedError("subscription does not permit SIP")
        src = MINUTES if timeframe == "1Min" else DAILY
        return {s: [b for b in src if b.symbol == s and start <= b.ts < end] for s in symbols}


def service(source=None, **kw):
    return ChartService(source or FakeSource(), Config(), CAL, now=lambda: AFTER,
                        delay=timedelta(0), **kw)


def get(svc, key):
    return asyncio.run(svc.chart("AAA", key))


# ---------------------------------------------------------------- indicator series
def test_series_match_the_engine_indicators():
    vals = [float(v) for v in range(1, 80)]
    s = ema_series(vals, 20)
    assert s[:19] == [None] * 19 and s[-1] == pytest.approx(ema(vals, 20))
    assert sma_series([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]
    bars = [Bar("A", MON, 0, 12, 6, 9, 10), Bar("A", MON, 0, 30, 30, 30, 10),
            Bar("A", TUE, 0, 3, 3, 3, 5)]                 # typical prices 9, 30 | 3
    assert vwap_series(bars, key=lambda b: b.ts.date()) == [9, 19.5, 3]   # restarts on Tuesday


# ---------------------------------------------------------------- intraday ranges
def test_1d_merges_regular_minutes_into_5_minute_candles():
    d = get(service(), "1D")
    c = d["candles"]
    assert d["intraday"] and len(c) == 78                 # 390 minutes / 5, Tuesday only
    assert c[0]["time"] == chart_time(TUE) == int(datetime(2026, 11, 10, 9, 30, tzinfo=UTC).timestamp())
    # First candle = Tuesday's first five minutes; the 08:00 pre-market bar is ignored.
    assert (c[0]["open"], c[0]["high"], c[0]["low"], c[0]["close"], c[0]["volume"]) == \
        pytest.approx((105.0, 105.09, 104.95, 105.06, 5000))
    # VWAP restarts at Tuesday's open: after five minutes it is the average typical price.
    typical = [(b.h + b.l + b.c) / 3 for b in MINUTES[391:396]]
    assert d["vwap"][0][0]["value"] == pytest.approx(sum(typical) / 5, abs=1e-4)
    # Monday's candles were fetched only to warm up the EMAs: both lines are complete.
    assert len(d["ema_fast"]) == len(d["ema_slow"]) == 78
    assert d["base"] == pytest.approx(MINUTES[389].c)     # previous session's last close


def test_5d_shows_every_session_with_data_in_15_minute_candles():
    d = get(service(), "5D")
    assert len(d["candles"]) == 2 * 26                    # only Monday and Tuesday have bars
    assert d["candles"][26]["time"] == chart_time(TUE)
    # VWAP restarts each session, so it comes as one line per session, never joined.
    assert [len(day) for day in d["vwap"]] == [26, 26]
    assert d["vwap"][1][0]["time"] == chart_time(TUE)


# ---------------------------------------------------------------- daily ranges
def test_daily_ranges_show_their_sessions_with_warm_indicators():
    for key, n in (("1M", 21), ("6M", 126), ("1Y", 252)):
        d = get(service(), key)
        assert not d["intraday"] and len(d["candles"]) == n and d["vwap"] == []
        assert d["candles"][-1]["time"] == "2026-11-10"
        assert len(d["ema_slow"]) == n                    # EMA50 warmed up before the range
        assert d["base"] == pytest.approx(DAILY[-n - 1].c)


# ---------------------------------------------------------------- feed, cache
def test_falls_back_to_configured_feed_and_caches():
    src = FakeSource(refuse_sip=True)
    svc = service(src)
    assert get(svc, "1M")["feed"] == "iex"                # Config().feed.feed
    assert src.calls == [("1Day", "sip"), ("1Day", None)]
    get(svc, "1M")
    assert len(src.calls) == 2                            # served from the cache


# ---------------------------------------------------------------- overlays
def test_markers_land_on_their_candles_and_levels_follow_the_position(tmp_path):
    cfg = Config(symbols=["AAA"])
    j = Journal(str(tmp_path / "t.db"))
    e = Engine(cfg, j)
    j.add_trade(day="2026-11-10", symbol="AAA", strategy="ORB", qty=10,
                entry_ts=(TUE + timedelta(minutes=17, seconds=30)).isoformat(), entry_px=105.2,
                exit_ts=(TUE + timedelta(minutes=64)).isoformat(), exit_px=106.0,
                initial_stop=104.8, pnl=8.0, r_multiple=2.0, exit_reason="target")
    e.positions["AAA"] = Position("AAA", "SWING", 5, 108.0, TUE + timedelta(hours=6), 104.0, 104.0)
    d = overlays(get(service(), "1D"), e, j)
    assert d["markers"] == [
        {"time": chart_time(TUE + timedelta(minutes=15)), "side": "buy", "text": "ORB 10"},   # 09:47 -> 09:45
        {"time": chart_time(TUE + timedelta(minutes=60)), "side": "sell", "text": "+2.0R"},   # 10:34 -> 10:30
        {"time": chart_time(TUE + timedelta(hours=6)), "side": "buy", "text": "SWING 5"}]
    assert {(l["kind"], l["price"]) for l in d["levels"]} == {("entry", 108.0), ("stop", 104.0)}
    # Daily range: the same fills sit on Tuesday's daily candle.
    days = overlays(get(service(), "1M"), e, j)
    assert {m["time"] for m in days["markers"]} == {"2026-11-10"}


def test_swing_trigger_level_only_for_unheld_watchlist_names(tmp_path):
    cfg = Config(symbols=["AAA"])
    j = Journal(str(tmp_path / "t.db"))
    e = Engine(cfg, j)
    setup = DailyContext("AAA", 100, 101, 99, 2.0, 99, 95, 1, 3, 2, swing_trigger=101.01)
    e.ctx = {"AAA": setup, "BBB": setup}       # BBB stands in for the benchmark: not on the watchlist
    d = overlays(get(service(), "1M"), e, j)
    assert d["levels"] == [{"price": 101.01, "kind": "trigger", "label": "swing trigger"}]
    assert overlays(dict(d, symbol="BBB"), e, j)["levels"] == []


# ---------------------------------------------------------------- API
def client(tmp_path, charts):
    cfg = Config()
    j = Journal(str(tmp_path / "t.db"))
    return TestClient(create_app(Engine(cfg, j), j, cfg, {}, charts))


def test_chart_route_validates_and_serves(tmp_path):
    off = client(tmp_path, None)
    assert off.get("/api/chart/AAA").status_code == 503
    assert off.get("/api/state").json()["chart_ranges"] == []
    on = client(tmp_path, service())
    assert on.get("/api/state").json()["chart_ranges"] == ["1D", "5D", "1M", "6M", "1Y"]
    assert on.get("/api/chart/AAA?range=2W").status_code == 400
    assert on.get("/api/chart/A$B").status_code == 400
    body = on.get("/api/chart/aaa?range=5D").json()       # lower case is accepted
    assert body["symbol"] == "AAA" and len(body["candles"]) == 52 and body["markers"] == []


def test_chart_route_reports_provider_failure(tmp_path):
    class Down:
        async def bars(self, *a, **kw):
            raise ConnectionError("offline")
    assert client(tmp_path, service(Down())).get("/api/chart/AAA").status_code == 502


# ---------------------------------------------------------------- simulation
def test_simulation_market_serves_charts(tmp_path):
    cfg = Config()
    j = Journal(str(tmp_path / "s.db"))
    sessions = CAL.sessions_between("2026-01-01", "2026-06-30")
    mkt = run_simulation(cfg, Engine(cfg, j), sessions[-3:], sessions[-63:-3], seed=3)
    svc = ChartService(mkt, cfg, CAL, now=lambda: sessions[-1].close, delay=timedelta(0), feed=None)
    name = cfg.benchmark                                  # streamed every simulated day
    one_day = asyncio.run(svc.chart(name, "1D"))
    assert one_day["feed"] == "synthetic" and len(one_day["candles"]) == 78
    assert len(asyncio.run(svc.chart(name, "1M"))["candles"]) == 21
