import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from openbell import alpaca, doctor, finnhub
from openbell.alpaca import AlpacaData, FatalFeedError
from openbell.config import Config
from openbell.finnhub import Finnhub
from openbell.market_calendar import MarketCalendar
from openbell.models import Bar, Tick
from tests import fake_services

UTC = timezone.utc


@pytest.fixture(scope="module")
def host():
    return fake_services.start()


@pytest.fixture
def wired(host, monkeypatch):
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    urls = {"DATA_URL": f"http://{host}/v2/stocks/bars", "NEWS_URL": f"http://{host}/v1beta1/news",
            "ASSETS_URL": f"http://{host}/v2/assets", "STREAM_URL": f"ws://{host}/v2/{{feed}}"}
    for k, v in urls.items():
        monkeypatch.setattr(alpaca, k, v)
        monkeypatch.setattr(doctor, k, v)
    monkeypatch.setattr(doctor, "CLOCK_URL", f"http://{host}/v2/clock")
    monkeypatch.setattr(doctor, "TEST_STREAM", f"ws://{host}/v2/test")
    monkeypatch.setattr(doctor, "FINNHUB", f"http://{host}/api/v1")
    monkeypatch.setattr(finnhub, "BASE", f"http://{host}/api/v1")
    now = datetime.now(UTC)
    cal = MarketCalendar()
    nxt = cal.current_or_next(now)
    nxt_open = nxt.open if now < nxt.open else cal.current_or_next(nxt.close + timedelta(minutes=1)).open
    fake_services.app.state.next_open = nxt_open.isoformat()
    return AlpacaData("good", "secret")


def test_rest_clients_parse_and_paginate(wired):
    now = datetime.now(UTC)

    async def go():
        bars = await wired.bars(["SPY", "NVDA", "AMD"], "1Day", now - timedelta(days=30),
                                now - timedelta(hours=12), feed="sip")
        assert set(bars) == {"SPY", "NVDA", "AMD"} and all(len(v) >= 18 for v in bars.values())
        assert isinstance(bars["AMD"][0], Bar) and bars["AMD"][0].ts.tzinfo is not None
        with pytest.raises(FatalFeedError):                       # recent full-market data: refused
            await wired.bars(["SPY"], "1Min", now - timedelta(minutes=5), now, feed="sip")
        assert len(await wired.assets()) == 1201
        news = await wired.news_all({"NVDA", "AAPL", "A"}, now - timedelta(hours=48), now)
        assert [h for _, h in news["NVDA"]] == ["NVDA raises guidance", "NVDA older item"]   # 2 pages
        assert news["NVDA"][0][0] > news["NVDA"][1][0] and "A" not in news      # round-up ignored
        assert news["AAPL"][0][1] == "AAPL and MSFT sign deal"
        pre = await wired.premarket(["NVDA", "AMD"], now - timedelta(hours=3), now)
        assert pre["NVDA"]["price"] > 0 and pre["NVDA"]["dollar_vol"] > 0
        assert await wired.premarket(["NVDA"], now - timedelta(minutes=5), now) is None   # window too young
        fh = Finnhub("k")
        assert list(await fh.earnings(now.date())) == ["NVDA"]
        return fh
    asyncio.run(go())


def test_stream_yields_ticks_and_bars_and_stops(wired):
    async def go():
        got = []
        async for ev in wired.stream(["NVDA", "AMD"], datetime.now(UTC) + timedelta(seconds=3)):
            got.append(ev)
        return got
    got = asyncio.run(go())
    ticks = [e for e in got if isinstance(e, Tick)]
    assert len(ticks) == 1 and ticks[0].price == 101.25 and ticks[0].symbol == "NVDA"   # 'W' trade dropped
    assert [type(e) for e in got].count(Bar) == 1


def test_stream_bad_key_is_fatal(wired):
    bad = AlpacaData("wrong", "secret")

    async def go():
        async for _ in bad.stream(["NVDA"], datetime.now(UTC) + timedelta(seconds=5)):
            pass
    with pytest.raises(FatalFeedError):
        asyncio.run(go())


def test_doctor_full_run(wired, monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "good")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret")
    monkeypatch.setenv("FINNHUB_API_KEY", "fh-secret-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(doctor.MarketCalendar, "is_open", lambda self, now: False)
    code = asyncio.run(doctor.run(Config()))
    out = capsys.readouterr().out
    assert code == 0, out
    for name in ("alpaca.clock", "alpaca.assets", "alpaca.daily_sip", "alpaca.parse_bars", "alpaca.plan",
                 "alpaca.premarket", "alpaca.minute_iex", "alpaca.news", "alpaca.parse_news",
                 "alpaca.stream_30", "alpaca.stream_31", "alpaca.stream_test", "finnhub.earnings",
                 "finnhub.metric"):
        assert f"[PASS] {name}" in out, (name, out)
    assert "free plan, as expected" in out and "31st symbol refused" in out
    assert "[SKIP] anthropic" in out and "0 failed" in out
    assert "fh-secret-key" not in out and "secret" not in out.replace("fh-secret-key", "")


def test_doctor_reports_bad_keys(wired, monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "wrong")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert asyncio.run(doctor.run(Config())) == 1
    assert "[FAIL] alpaca" in capsys.readouterr().out
