import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from openbell import scanner
from openbell.agent import parse_selection
from openbell.config import Config
from openbell.engine import Engine
from openbell.journal import Journal
from openbell.market_calendar import MarketCalendar
from openbell.models import Bar, Tick, Session, DailyContext, DayPlan
from openbell.replay import run_simulation
from openbell.scanner import Candidate, apply_decisions, eligible_assets, rank_universe, scan

UTC = timezone.utc
D0 = datetime(2026, 6, 1, 12, tzinfo=UTC)


def series(sym, n=90, start=100.0, step=0.5, vol=2e6, last=None):
    """Steady trend; `last` overrides the final bar as (o, h, l, c, v)."""
    bars = [Bar(sym, D0 + timedelta(days=i), start + step * i, start + step * i + 1,
                start + step * i - 1, start + step * i + 0.5, vol) for i in range(n)]
    if last:
        bars[-1] = Bar(sym, bars[-1].ts, *last)
    return bars


def test_eligible_assets_drops_funds_and_odd_classes():
    a = lambda **k: {"tradable": True, "status": "active", "exchange": "NASDAQ", "name": "Acme Inc", **k}
    got = eligible_assets([a(symbol="ACME"), a(symbol="TQQQ", name="ProShares UltraPro QQQ"),
                           a(symbol="BRK.B"), a(symbol="OTCX", exchange="OTC"),
                           a(symbol="HALT", tradable=False), a(symbol="SPY", name="SPDR S&P 500 ETF Trust")])
    assert got == ["ACME"]


def test_rank_universe_applies_price_and_liquidity_floor():
    cfg = Config()
    cfg.scanner.universe_size = 2
    daily = {"BIG": series("BIG", 30, vol=5e6), "MID": series("MID", 30, vol=1e6),
             "SMALL": series("SMALL", 30, vol=6e5), "THIN": series("THIN", 30, vol=1e4),
             "PENNY": series("PENNY", 30, start=2, step=0.01, vol=9e7)}
    assert [r[0] for r in rank_universe(daily, cfg)] == ["BIG", "MID"]


def test_scan_finds_swing_and_intraday_candidates():
    cfg = Config()
    bench = series("SPY", step=0.1)
    daily = {
        # uptrend, yesterday dipped to the fast EMA and held: swing setup
        "PULL": series("PULL", last=(142, 143, 136, 138, 2e6)),
        # yesterday 3x volume, wide range, closed at the high: in play
        "HOT": series("HOT", last=(145, 152, 144.5, 151.5, 6e6)),
        # extended uptrend on normal volume: neither
        "DULL": series("DULL"),
        # weak for three months but heavy volume and a strong close yesterday:
        # still an intraday candidate (no strength filter there), never a swing one
        "TURN": series("TURN", start=200, step=-0.5, last=(155, 158, 154, 157.5, 9e6)),
        "SPY": bench,
    }
    got = {c.symbol: c for c in scan(cfg, daily, bench)}
    assert set(got) == {"PULL", "HOT", "TURN"}
    assert got["PULL"].kind == "SWING" and got["HOT"].kind == "INTRADAY" and got["TURN"].kind == "INTRADAY"
    assert got["TURN"].metrics["rs_3m_pct"] < 0 and got["HOT"].metrics["rvol"] == 3.0
    assert got["HOT"].metrics["gap_pct"] is None and got["HOT"].metrics["fresh_news"] is False


def test_scan_gap_and_news_routes():
    cfg = Config()
    bench = series("SPY", step=0.1)
    names = ["GAPUP", "NEWS", "OLDNEWS", "GAPDN", "THINGAP", "QUIET"]
    daily = {n: series(n) for n in names}                    # all dull yesterday, close 145.0
    daily["SPY"] = bench
    close, dv = 145.0, 145.0 * 2e6
    pre = {"GAPUP": {"price": close * 1.04, "dollar_vol": dv * 0.05},     # +4% on 5% of a day's volume
           "GAPDN": {"price": close * 0.95, "dollar_vol": dv * 0.05},
           "THINGAP": {"price": close * 1.04, "dollar_vol": dv * 0.0001}, # gap on almost no volume
           "NEWS": {"price": close * 1.002, "dollar_vol": dv * 0.001},
           "QUIET": {"price": close, "dollar_vol": 0.0}}
    since = D0 + timedelta(days=89, hours=8)
    news = {"NEWS": [(since + timedelta(hours=3), "NEWS wins contract")],
            "OLDNEWS": [(since - timedelta(hours=30), "OLDNEWS old story")],
            "GAPDN": [(since + timedelta(hours=1), "GAPDN cuts guidance")]}
    got = {c.symbol: c for c in scan(cfg, daily, bench, pre, news, since) if c.kind != "SWING"}
    assert set(got) == {"GAPUP", "NEWS"}                     # gap down, thin gap, stale news: out
    assert got["GAPUP"].rank < got["NEWS"].rank and got["GAPUP"].metrics["gap_pct"] == 4.0
    assert got["NEWS"].metrics["fresh_news"] and got["NEWS"].headlines == ["NEWS wins contract"]
    assert 5.0 <= got["GAPUP"].metrics["premarket_vol_pct_of_day"] < 5.5


def test_swing_keeps_the_strength_floor():
    cfg = Config()
    weak = series("WEAK", step=0.05, last=(104.5, 105.2, 103, 104.9, 2e6))   # uptrend, but lags the benchmark
    bench = series("SPY", step=0.5)
    assert [c for c in scan(cfg, {"WEAK": weak}, bench) if c.kind in ("SWING", "BOTH")] == []
    cfg.scanner.min_rel_strength_pct = -100
    assert [c.kind for c in scan(cfg, {"WEAK": weak}, bench)] == ["SWING"]


def test_scan_respects_candidate_cap_and_is_deterministic():
    cfg = Config()
    cfg.scanner.candidates = 5
    bench = series("SPY", step=0.1)
    daily = {f"H{i:02d}": series(f"H{i:02d}", step=0.3 + i * 0.02,
                                 last=(130, 140 + i, 129.5, 139.5 + i, 6e6)) for i in range(12)}
    a, b = scan(cfg, daily, bench), scan(cfg, daily, bench)
    assert len(a) == 5 and [c.symbol for c in a] == [c.symbol for c in b]
    assert [c.rank for c in a] == [1, 2, 3, 4, 5]


def cands(n):
    return [Candidate(f"C{i}", "SWING", 10 - i, rank=i + 1) for i in range(n)]


def test_gate_fallback_takes_top_ranked():
    cs = cands(5)
    assert apply_decisions(cs, None, 3, set()) == ["C0", "C1", "C2"]
    assert all(c.decision == "approve" for c in cs) and [c.picked for c in cs] == [True] * 3 + [False] * 2


def test_gate_agent_rejections_and_held_positions():
    cs = cands(5)
    picks = apply_decisions(cs, {"C0": (False, "share offering"), "C1": (True, "clean")}, 2, {"C2"})
    assert picks == ["C1", "C3"]                      # C0 rejected, C2 already held, C3 undecided -> kept
    assert cs[0].decision == "reject" and cs[0].reason == "share offering" and not cs[0].picked
    assert cs[2].picked and "no decision" in cs[3].reason
    assert apply_decisions(cands(3), {}, 0, set()) == []


def test_parse_selection_is_strict():
    txt = 'ok {"decisions":{"AAA":{"approve":false,"reason":"offering"},"ZZZ":{"approve":true},' \
          '"BBB":{"approve":"yes"}}}'
    assert parse_selection(txt, ["AAA", "BBB"]) == {"AAA": (False, "offering")}
    with pytest.raises(ValueError):
        parse_selection('{"picks": ["AAA"]}', ["AAA"])


def test_earnings_blackout_blocks_swing_only(tmp_path):
    cfg = Config()
    cfg.risk.slippage_bps = 0
    j = Journal(str(tmp_path / "t.db"))
    e = Engine(cfg, j)
    o = datetime(2026, 11, 10, 14, 30, tzinfo=UTC)
    sess = Session(o.date(), o, o + timedelta(hours=6, minutes=30))
    mk = lambda s, d: DailyContext(s, 100, 101, 99, 2.0, 99, 95, 1, 3, 2, swing_trigger=101.0,
                                   swing_reason="t", earnings_in_days=d)
    e.start_day(sess, {"SOON": mk("SOON", 2), "LATER": mk("LATER", 6), "NONE": mk("NONE", None)},
                DayPlan(), ["SOON", "LATER", "NONE"])
    for s in ("SOON", "LATER", "NONE"):
        e.on_tick(Tick(s, o + timedelta(minutes=6), 101.2))
    assert set(e.positions) == {"LATER", "NONE"}
    assert "earnings in 2" in j.rows("SELECT message FROM events WHERE kind='SKIP'")[0]["message"]


def test_engine_trades_only_todays_watchlist_but_manages_held(tmp_path):
    cfg = Config()
    cfg.risk.slippage_bps = 0
    e = Engine(cfg, Journal(str(tmp_path / "t.db")))
    o = datetime(2026, 11, 10, 14, 30, tzinfo=UTC)
    sess = Session(o.date(), o, o + timedelta(hours=6, minutes=30))
    mk = lambda s: DailyContext(s, 100, 101, 99, 2.0, 99, 95, 1, 3, 2, swing_trigger=101.0, swing_reason="t")
    e.start_day(sess, {"IN": mk("IN"), "OUT": mk("OUT")}, DayPlan(), ["IN"])
    e.on_tick(Tick("IN", o + timedelta(minutes=6), 101.2))
    e.on_tick(Tick("OUT", o + timedelta(minutes=6), 101.2))
    assert set(e.positions) == {"IN"}
    e.end_day()
    # next day IN is no longer picked, but its stop must still work
    d2 = Session(sess.day + timedelta(days=1), o + timedelta(days=1), sess.close + timedelta(days=1))
    e.start_day(d2, {"IN": mk("IN"), "OUT": mk("OUT")}, DayPlan(), ["OUT"])
    e.on_tick(Tick("IN", d2.open + timedelta(minutes=1), 90.0))
    assert "IN" not in e.positions


def test_scan_outcomes_and_stats(tmp_path):
    j = Journal(str(tmp_path / "t.db"))
    bars = series("AAA", 12)
    day = bars[5].ts.date().isoformat()
    a, r = Candidate("AAA", "SWING", 1, rank=1), Candidate("BBB", "SWING", 1, rank=2, decision="reject")
    j.save_scan(day, [a, r], "agent")
    j.update_scan_outcomes({"AAA": bars, "BBB": series("BBB", 12, step=-0.5)})
    row = {x["symbol"]: x for x in j.scan_rows(day)}
    assert row["AAA"]["ret_1d"] == pytest.approx((bars[5].c / bars[4].c - 1) * 100)
    assert row["AAA"]["ret_5d"] == pytest.approx((bars[9].c / bars[4].c - 1) * 100)
    st = j.scan_stats()
    assert st["approve"]["n"] == 1 and st["approve"]["avg_1d"] > 0 > st["reject"]["avg_1d"]


# ---------------------------------------------------------------- live pre-market path with stubs
class StubData:
    def __init__(self, daily, fail_news=False):
        self.daily, self.fail_news, self.asset_calls, self.pre = daily, fail_news, 0, None

    async def assets(self):
        self.asset_calls += 1
        return [{"symbol": s, "name": s + " Corp", "exchange": "NYSE", "tradable": True, "status": "active"}
                for s in self.daily if s != "SPY"]

    async def bars(self, symbols, timeframe, start, end, feed=None):
        return {s: [b for b in self.daily.get(s, []) if start <= b.ts <= end] for s in symbols}

    async def news_all(self, wanted, start, end, max_pages=40):
        if self.fail_news:
            return None
        return {s: [(end - timedelta(hours=2), f"{s} raises guidance")] for s in wanted if s.startswith("HOT")}

    async def premarket(self, symbols, start, now):
        return self.pre


class StubFinnhub:
    async def earnings(self, start, days=14):
        return {"HOTA": start + timedelta(days=1)}

    async def profitable(self, symbols):
        return {s: True for s in symbols}


class StubAgent:
    active = True

    def __init__(self, reject=()):
        self.reject, self.seen = set(reject), None

    def select(self, candidates):
        self.seen = candidates
        return {c.symbol: (c.symbol not in self.reject, "stub") for c in candidates}

    def make_plan(self, *a):
        return DayPlan()


def make_runner(tmp_path, monkeypatch, daily, **kw):
    from openbell import runner as R
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    cfg = Config()
    cfg.db_path = str(tmp_path / "live.db")
    cfg.scanner.max_watchlist = kw.pop("slots", 4)
    r = R.LiveRunner(cfg)
    r.data, r.finnhub, r.agent = StubData(daily, **kw), StubFinnhub(), StubAgent()
    return r


def market(sess):
    """Daily history ending the session before `sess`: six "in play" names, two in downtrends."""
    n = 90
    first = datetime(sess.day.year, sess.day.month, sess.day.day, 5, tzinfo=UTC) - timedelta(days=n)
    def mk(sym, step, last=None):
        bars = [Bar(sym, first + timedelta(days=i), 100 + step * i, 101 + step * i, 99 + step * i,
                    100.5 + step * i, 2e6) for i in range(n)]
        if last:
            c = bars[-1].c
            bars[-1] = Bar(sym, bars[-1].ts, c - 5, c + 2, c - 5.5, c + 1.5, last)
        return bars
    d = {"HOT" + "ABCDEF"[i]: mk("HOT" + "ABCDEF"[i], 0.4 + 0.05 * i, last=(4 + i) * 2e6) for i in range(6)}
    d.update({"DULLA": mk("DULLA", -0.2), "DULLB": mk("DULLB", -0.3), "SPY": mk("SPY", 0.1)})
    return d


def test_live_premarket_scan_end_to_end(tmp_path, monkeypatch):
    sess = MarketCalendar().current_or_next(datetime(2026, 11, 10, 5, tzinfo=UTC))
    r = make_runner(tmp_path, monkeypatch, market(sess))
    r.agent = StubAgent(reject={"HOTF"})
    monkeypatch.setattr("openbell.runner.utcnow", lambda: sess.open - timedelta(minutes=30))
    asyncio.run(r.prepare(sess))
    rows = {x["symbol"]: x for x in r.journal.scan_rows()}
    assert set(rows) == {"HOT" + x for x in "ABCDEF"}               # downtrends never proposed
    assert rows["HOTF"]["decision"] == "reject" and not rows["HOTF"]["picked"]
    assert rows["HOTF"]["rank"] == 1                                 # best score, still vetoed
    assert r.engine.symbols == ["HOTE", "HOTD", "HOTC", "HOTB"]      # next four by rank
    assert rows["HOTA"]["decision"] == "approve" and not rows["HOTA"]["picked"]   # no slot left
    assert r.universe == r.engine.symbols + ["SPY"] and len(r.universe) <= 30
    assert rows["HOTE"]["headlines"] == ["HOTE raises guidance"] and rows["HOTE"]["profitable"] == 1
    assert rows["HOTE"]["metrics"]["fresh_news"] and rows["HOTE"]["metrics"]["gap_pct"] is None
    assert rows["HOTA"]["earnings_in_days"] == 1
    assert r.agent.seen[0].headlines                                 # agent saw the enrichment
    assert r.engine.status == "ready" and r.journal.rows("SELECT * FROM events WHERE kind='SCAN'")
    # the weekly universe is cached: a second morning does not rebuild it
    asyncio.run(r.prepare(sess))
    assert r.data.asset_calls == 1


def test_live_scan_survives_missing_agent_news_and_scanner_failure(tmp_path, monkeypatch):
    sess = MarketCalendar().current_or_next(datetime(2026, 11, 10, 5, tzinfo=UTC))
    monkeypatch.setattr("openbell.runner.utcnow", lambda: sess.open - timedelta(minutes=30))
    r = make_runner(tmp_path, monkeypatch, market(sess), fail_news=True, slots=3)
    r.agent.select = lambda c: None                                  # agent unavailable
    asyncio.run(r.prepare(sess))
    assert r.engine.symbols == ["HOTF", "HOTE", "HOTD"]              # top-ranked
    assert r.journal.scan_rows()[0]["source"] == "rank"

    async def boom():
        raise RuntimeError("assets endpoint down")
    r2 = make_runner(tmp_path / "..", monkeypatch, market(sess), slots=29)
    r2.cfg.db_path = str(tmp_path / "b.db")
    r2.journal = Journal(r2.cfg.db_path)
    r2.engine = Engine(r2.cfg, r2.journal)
    r2.data.assets = boom
    asyncio.run(r2.prepare(sess))
    assert r2.engine.symbols == r2.cfg.symbols                       # fallback list from config
    assert "scanner failed" in r2.journal.rows("SELECT message FROM events WHERE kind='WARN'")[0]["message"]


def test_simulation_uses_a_changing_watchlist(tmp_path):
    cfg = Config()
    j = Journal(str(tmp_path / "s.db"))
    e = Engine(cfg, j)
    sessions = MarketCalendar().sessions_between("2025-01-01", "2026-06-30")
    run_simulation(cfg, e, sessions[-8:], sessions[-308:-8], seed=5)
    days = j.rows("SELECT day, COUNT(*) n, SUM(picked) p FROM scan GROUP BY day")
    assert len(days) == 8 and all(d["p"] <= cfg.scanner.max_watchlist for d in days)
    lists = [frozenset(r["symbol"] for r in j.scan_rows(d["day"]) if r["picked"]) for d in days]
    assert len(set(lists)) > 1                                       # not the same names every day
    traded = {t["symbol"] for t in j.trades(limit=1000)}
    assert traded and any(s.startswith("SIM") for s in traded)       # names from outside the config list
    assert j.scan_stats()["approve"]["n"] > 0


def test_live_scan_uses_premarket_gap(tmp_path, monkeypatch):
    sess = MarketCalendar().current_or_next(datetime(2026, 11, 10, 5, tzinfo=UTC))
    monkeypatch.setattr("openbell.runner.utcnow", lambda: sess.open - timedelta(minutes=20))
    daily = market(sess)
    r = make_runner(tmp_path, monkeypatch, daily, slots=2)
    a, f = daily["HOTA"][-1].c, daily["HOTF"][-1].c
    r.data.pre = {"HOTA": {"price": a * 1.06, "dollar_vol": a * 2e6 * 0.1},   # weakest yesterday, gapping up
                  "HOTF": {"price": f * 0.90, "dollar_vol": f * 2e6 * 0.1}}   # best yesterday, gapping down
    asyncio.run(r.prepare(sess))
    rows = {x["symbol"]: x for x in r.journal.scan_rows()}
    assert "HOTA" in r.engine.symbols and rows["HOTA"]["rank"] <= 2      # top of the intraday list
    assert rows["HOTA"]["metrics"]["gap_pct"] == 6.0
    assert "HOTF" not in r.engine.symbols
    assert "pre-market data on, news on" in r.journal.rows(
        "SELECT message FROM events WHERE kind='SCAN'")[0]["message"]
