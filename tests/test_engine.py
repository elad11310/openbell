from datetime import datetime, timedelta, timezone

import pytest

from openbell.agent import parse_plan
from openbell.alpaca import parse_ts
from openbell.config import Config
from openbell.engine import Engine
from openbell.indicators import atr, ema, build_daily_context
from openbell.journal import Journal
from openbell.market_calendar import MarketCalendar
from openbell.models import Bar, Tick, Session, DailyContext, DayPlan, SymbolPlan
from openbell.replay import run_simulation

UTC = timezone.utc
OPEN = datetime(2026, 11, 10, 14, 30, tzinfo=UTC)
SESS = Session(OPEN.date(), OPEN, OPEN + timedelta(hours=6, minutes=30))


def make(tmp_path, **risk):
    cfg = Config(symbols=["AAA", "BBB"])
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    cfg.risk.slippage_bps = 0
    j = Journal(str(tmp_path / "t.db"))
    return cfg, j, Engine(cfg, j)


def ctx(sym="AAA", close=100.0, atr_=2.0, trigger=None):
    return DailyContext(sym, close, close + 1, close - 1, atr_, close - 1, close - 5, 1, 3, 2,
                        swing_trigger=trigger, swing_reason="test")


def bar(i, o, h, l, c, v=1000, sym="AAA"):
    return Bar(sym, OPEN + timedelta(minutes=i), o, h, l, c, v)


def feed_opening_range(e, sym="AAA"):
    for i in range(15):
        e.on_bar(bar(i, 100, 101, 99.5, 100.5, sym=sym))


# ---------------------------------------------------------------- calendar
def test_calendar_regular_holiday_halfday_weekend():
    cal = MarketCalendar()
    s = cal.current_or_next(datetime(2026, 11, 10, 5, 17, tzinfo=UTC))
    assert (s.open, s.close, s.early_close) == (OPEN, SESS.close, False)
    assert cal.is_open(datetime(2026, 11, 10, 15, 0, tzinfo=UTC))
    assert not cal.is_open(SESS.close)
    # Thanksgiving is closed; the Friday after is a 13:00 ET half day
    s = cal.current_or_next(datetime(2026, 11, 26, 15, 0, tzinfo=UTC))
    assert s.day.isoformat() == "2026-11-27" and s.early_close
    assert s.close == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    # Saturday rolls to Monday
    assert cal.current_or_next(datetime(2026, 11, 14, 12, tzinfo=UTC)).day.isoformat() == "2026-11-16"
    # summer time: the open is 13:30 UTC
    assert cal.current_or_next(datetime(2026, 7, 1, 0, tzinfo=UTC)).open.hour == 13


# ---------------------------------------------------------------- indicators
def test_indicators():
    assert ema([1, 2, 3, 4, 5], 5) == 3
    bars = [Bar("A", OPEN + timedelta(days=i), 10, 11, 9, 10, 1) for i in range(20)]
    assert atr(bars, 14) == pytest.approx(2.0)
    assert build_daily_context("A", bars[:10], Config().swing) is None


def test_swing_setup_detected_only_in_uptrend():
    cfg = Config().swing
    up = [Bar("A", OPEN + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 1) for i in range(80)]
    down = [Bar("A", OPEN + timedelta(days=i), 200 - i, 201 - i, 199 - i, 199.5 - i, 1) for i in range(80)]
    assert build_daily_context("A", up, cfg).swing_trigger is None        # extended, no pullback
    dip = up + [Bar("A", OPEN + timedelta(days=80), 179, 180, 172, 178, 1)]
    assert build_daily_context("A", dip, cfg).swing_trigger == pytest.approx(180.01)
    assert build_daily_context("A", down, cfg).swing_trigger is None


# ---------------------------------------------------------------- ORB
def test_orb_entry_sizing_target_and_stop(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx()}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000))
    p = e.positions["AAA"]
    # stop = max(range low 99.5, 101.5 - 0.5*ATR) = 100.5 -> risk 1.00/share, 0.5% of 100k = 500
    assert (p.stop, p.target, p.qty) == (100.5, 103.5, 197)   # capped by 20% position limit
    e.on_tick(Tick("AAA", OPEN + timedelta(minutes=20), 103.6))
    assert "AAA" not in e.positions
    t = j.trades()[0]
    assert t["exit_reason"] == "target" and t["exit_px"] == 103.5 and t["r_multiple"] == 2.0
    assert e.cash == pytest.approx(100_000 + 197 * 2.0)


def test_orb_stop_fills_at_tick_through_stop(tmp_path):
    cfg, j, e = make(tmp_path, max_position_pct=100)
    e.start_day(SESS, {"AAA": ctx()}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000))
    assert e.positions["AAA"].qty == 500
    e.on_tick(Tick("AAA", OPEN + timedelta(minutes=20), 100.2))   # gaps through 100.5
    t = j.trades()[0]
    assert t["exit_reason"] == "stop" and t["exit_px"] == 100.2 and t["r_multiple"] == -1.3


@pytest.mark.parametrize("vol,close,why", [(1000, 101.5, "volume"), (2000, 100.9, None)])
def test_orb_filters(tmp_path, vol, close, why):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx()}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.0, 100.4, close, v=vol))
    assert not e.positions
    skips = [x for x in j.rows("SELECT * FROM events WHERE kind='SKIP'")]
    assert (len(skips) == 1 and why in skips[0]["message"]) if why else not skips


def test_one_orb_attempt_per_day_and_cutoff(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx()}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=100))       # rejected: light volume
    e.on_bar(bar(16, 101.5, 102.6, 101.4, 102.5, v=5000))      # no second attempt
    assert not e.positions
    feed_opening_range(e, "BBB")
    e.on_bar(bar(340, 100.5, 101.6, 100.4, 101.5, v=5000, sym="BBB"))   # inside last hour
    assert not e.positions


def test_end_of_day_flatten_keeps_swing(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx(), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000))
    e.on_tick(Tick("BBB", OPEN + timedelta(minutes=10), 101.2))
    assert set(e.positions) == {"AAA", "BBB"}
    e.on_clock(SESS.close - timedelta(minutes=10))
    assert set(e.positions) == {"BBB"}
    assert j.trades()[0]["exit_reason"] == "end-of-day flatten"


# ---------------------------------------------------------------- swing
def test_swing_trigger_wait_chase_and_trailing(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx(trigger=101.0), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    e.on_tick(Tick("AAA", OPEN + timedelta(minutes=2), 101.2))     # too early
    assert not e.positions
    e.on_tick(Tick("AAA", OPEN + timedelta(minutes=6), 101.2))
    p = e.positions["AAA"]
    assert p.strategy == "SWING" and p.stop == 97.2 and p.target is None   # 2 x ATR
    e.on_tick(Tick("BBB", OPEN + timedelta(minutes=6), 103.0))     # > trigger + 0.5 ATR
    assert "BBB" not in e.positions
    e.on_clock(SESS.close)
    e.end_day()
    # next day: close 110 -> trail = 110 - 3*2 = 104; then a close under the slow EMA exits at the open
    d2 = Session(SESS.day + timedelta(days=1), OPEN + timedelta(days=1), SESS.close + timedelta(days=1))
    e.start_day(d2, {"AAA": ctx(close=110.0)}, DayPlan())
    assert e.positions["AAA"].stop == 104.0
    e.end_day()
    d3 = Session(SESS.day + timedelta(days=2), OPEN + timedelta(days=2), SESS.close + timedelta(days=2))
    weak = ctx(close=108.0)
    weak.ema_slow = 109.0
    e.start_day(d3, {"AAA": weak}, DayPlan())
    e.on_tick(Tick("AAA", d3.open + timedelta(seconds=5), 107.5))
    assert not e.positions and "EMA" in j.trades()[0]["exit_reason"]


# ---------------------------------------------------------------- risk
def test_daily_loss_limit_halts_and_flattens(tmp_path):
    cfg, j, e = make(tmp_path, max_position_pct=100, daily_loss_limit_pct=0.3)
    e.start_day(SESS, {"AAA": ctx(), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000))
    e.last["AAA"] = 100.8                                          # -350 unrealised on 500 shares
    e.on_clock(OPEN + timedelta(minutes=20))
    assert e.halted and not e.positions
    e.on_tick(Tick("BBB", OPEN + timedelta(minutes=30), 101.2))
    assert not e.positions


def test_plan_restricts_but_never_raises_risk(tmp_path):
    cfg, j, e = make(tmp_path, max_position_pct=100)
    plan = DayPlan(risk_multiplier=5.0, symbols={"BBB": SymbolPlan(allow_intraday=False)})
    e.start_day(SESS, {"AAA": ctx(), "BBB": ctx("BBB")}, plan)
    for s in ("AAA", "BBB"):
        feed_opening_range(e, s)
        e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000, sym=s))
    assert set(e.positions) == {"AAA"} and e.positions["AAA"].qty == 500   # not 2500
    p = parse_plan('noise {"regime":"panic","risk_multiplier":7,"symbols":{"ZZZ":{},"AAA":{"allow_swing":false}}}',
                   ["AAA"])
    assert (p.regime, p.risk_multiplier, list(p.symbols)) == ("neutral", 1.0, ["AAA"])
    assert parse_plan('{"risk_multiplier": -3}', []).risk_multiplier == 0.0
    with pytest.raises(ValueError):
        parse_plan("I think today looks fine", [])


def test_max_positions_and_stale_feed(tmp_path):
    cfg, j, e = make(tmp_path, max_positions=1)
    e.start_day(SESS, {"AAA": ctx(trigger=101.0), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    e.feed_ok = False
    e.on_tick(Tick("AAA", OPEN + timedelta(minutes=6), 101.2))
    assert not e.positions
    e.feed_ok = True
    e.on_tick(Tick("BBB", OPEN + timedelta(minutes=6), 101.2))
    assert set(e.positions) == {"BBB"}


# ---------------------------------------------------------------- durability
def test_restart_restores_state_and_day_counters(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx(), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    feed_opening_range(e)
    e.on_bar(bar(15, 100.5, 101.6, 100.4, 101.5, v=2000))
    e.on_tick(Tick("BBB", OPEN + timedelta(minutes=10), 101.2))
    cash = e.cash
    e2 = Engine(cfg, Journal(str(tmp_path / "t.db")))               # process restart
    assert e2.cash == cash and set(e2.positions) == {"AAA", "BBB"}
    e2.start_day(SESS, {"AAA": ctx(), "BBB": ctx("BBB", trigger=101.0)}, DayPlan())
    assert e2.trades_today == 2 and e2.day_start_equity == 100_000
    assert e2.positions["AAA"].exit_flag is None and e2.state["AAA"].orb_taken


def test_ticks_outside_session_are_ignored(tmp_path):
    cfg, j, e = make(tmp_path)
    e.start_day(SESS, {"AAA": ctx(trigger=101.0)}, DayPlan())
    e.on_tick(Tick("AAA", OPEN - timedelta(minutes=30), 105.0))
    e.on_tick(Tick("AAA", SESS.close + timedelta(minutes=1), 105.0))
    assert not e.positions and e.last["AAA"] == 100.0


def test_parse_ts_nanoseconds():
    assert parse_ts("2026-11-10T14:30:00.123456789Z") == datetime(2026, 11, 10, 14, 30, 0, 123456, tzinfo=UTC)
    assert parse_ts("2026-11-10T05:00:00Z").hour == 5


# ---------------------------------------------------------------- whole system
def test_simulation_is_deterministic_and_books_balance(tmp_path):
    results = []
    for n in range(2):
        cfg = Config()
        j = Journal(str(tmp_path / f"s{n}.db"))
        e = Engine(cfg, j)
        sessions = MarketCalendar().sessions_between("2025-01-01", "2026-06-30")
        run_simulation(cfg, e, sessions[-10:], sessions[-310:-10], seed=3)
        realised = sum(t["pnl"] for t in j.trades(limit=10_000))
        open_pnl = sum((e.last[p.symbol] - p.entry_px) * p.qty for p in e.positions.values())
        assert e.equity() == pytest.approx(cfg.starting_equity + realised + open_pnl, abs=0.5)
        assert all(p.strategy == "SWING" for p in e.positions.values())      # nothing intraday overnight
        assert len(e.positions) <= cfg.risk.max_positions
        days = j.rows("SELECT * FROM days")
        assert len(days) == 10 and all(d["trades"] <= cfg.risk.max_trades_per_day for d in days)
        worst = min(d["pnl"] / d["start_equity"] for d in days)
        assert worst > -0.035       # loss limit 2% plus gap slack
        results.append((round(e.equity(), 2), len(j.trades(limit=10_000))))
    assert results[0] == results[1] and results[0][1] > 0
