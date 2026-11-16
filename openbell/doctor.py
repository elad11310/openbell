"""`python -m openbell doctor` - check every outside service with your own keys.

Prints a report you can paste to whoever supports the app. Keys are never printed.
Nothing here places orders or changes anything on any account.

WHAT: Runs a list of live checks and prints one PASS / WARN / FAIL / SKIP line each:
        Alpaca    keys + market clock (and whether our calendar agrees), asset list,
                  full-market daily history, which data plan the key is on, pre-market
                  bars, regular-session minute bars, latest trades (how names past the
                  stream limit are polled), news, the websocket (30-symbol
                  subscription, the 31st refused) and Alpaca's always-on test stream;
        Finnhub   earnings calendar and company metrics (if a key is set);
        Anthropic one real `select` call through the app's own prompt and parser.
      Exit code 1 if anything FAILed, else 0.
HOW:  Mostly raw HTTP/websocket calls (to see exactly what each service answers), plus
      a few calls through the app's own client code (`data.bars`, `data.news_all`,
      `parse_selection`) to prove the app parses real responses. Error details are
      trimmed and any key that might appear in an error message is replaced with "***".
WHY:  The code was built against published API documentation and a local fake server
      (tests/fake_services.py), never against the real services. This command is the
      real-world check: run it once with your keys before trusting a live session, and
      again whenever something looks wrong.
"""
from __future__ import annotations

import asyncio
import json
import platform
from datetime import datetime, time, timedelta, timezone

import httpx
import websockets

from . import __version__, scanner
from .alpaca import AlpacaData, ASSETS_URL, DATA_URL, NEWS_URL, STREAM_URL
from .config import Config
from .finnhub import BASE as FINNHUB
from .market_calendar import MarketCalendar, NY

CLOCK_URL = "https://paper-api.alpaca.markets/v2/clock"
TEST_STREAM = "wss://stream.data.alpaca.markets/v2/test"   # sends fake trades 24/7
# Liquid large caps used to pad the stream test past 30 symbols, to confirm the plan's
# 30-symbol cap is enforced.
FILLER = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "JPM", "V", "XOM", "UNH", "HD", "PG", "MA", "COST",
          "ABBV", "KO", "PEP", "MRK", "WMT", "BAC", "CVX", "ORCL", "NFLX", "ADBE", "CSCO", "CRM", "TMO",
          "ACN", "LIN", "MCD", "DIS", "WFC", "ABT", "QCOM", "TXN", "DHR", "INTU", "AMGN", "PM", "NEE"]
# Shown when a stream check fails: the most common cause is the app itself.
ONE_CONNECTION = "free plan = ONE stream connection; is `openbell run` running? stop it, rerun"


class Report:
    """Collects check results and prints each one as soon as it is added."""

    def __init__(self) -> None:
        self.rows: list = []

    def add(self, status: str, name: str, detail: str) -> None:
        # Collapse whitespace/newlines and cap the length so each check is one tidy line.
        self.rows.append((status, name, " ".join(str(detail).split())[:230]))
        # flush=True: show progress immediately; some checks take several seconds.
        print(f"[{status:4}] {name}: {self.rows[-1][2]}", flush=True)

    def counts(self) -> dict:
        return {s: sum(r[0] == s for r in self.rows) for s in ("PASS", "WARN", "FAIL", "SKIP")}


def _iso(dt: datetime) -> str:
    """Datetime -> the "YYYY-MM-DDTHH:MM:SSZ" format Alpaca expects."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _err(r: httpx.Response) -> str:
    return f"HTTP {r.status_code}: {r.text[:160]}"


async def check_alpaca(cfg: Config, rep: Report) -> None:
    if not cfg.alpaca_key or not cfg.alpaca_secret:
        rep.add("FAIL", "alpaca.keys", "ALPACA_API_KEY / ALPACA_API_SECRET missing from .env")
        return
    data = AlpacaData(cfg.alpaca_key, cfg.alpaca_secret, cfg.feed.feed)
    cal, now = MarketCalendar(), datetime.now(timezone.utc)
    nxt = cal.current_or_next(now)
    # the most recent session that has fully finished
    # (sessions in the last 10 days, keep those already closed, take the last one;
    # 10 days is enough to span any holiday cluster)
    done = [s for s in cal.sessions_between(now.date() - timedelta(days=10), now.date()) if s.close < now][-1]
    async with httpx.AsyncClient(timeout=60, headers=data.headers) as c:
        # 1. keys + clock
        r = await c.get(CLOCK_URL)
        if r.status_code != 200:
            # Without valid keys every other check would fail too; stop here.
            rep.add("FAIL", "alpaca.auth", _err(r) + " | are these PAPER keys?")
            return
        ck = r.json()
        theirs = datetime.fromisoformat(ck["next_open"]).astimezone(timezone.utc)
        # Alpaca's "next_open" is the NEXT open even while the market is open now, so if
        # a session is in progress compare against the session after it.
        ours = nxt.open if now < nxt.open else cal.current_or_next(nxt.close + timedelta(minutes=1)).open
        same = abs((theirs - ours).total_seconds()) < 60
        rep.add("PASS" if same else "WARN", "alpaca.clock",
                f"keys accepted; market {'open' if ck['is_open'] else 'closed'}; next open Alpaca "
                f"{_iso(theirs)} vs app {_iso(ours)}" + ("" if same else "  <-- calendars disagree"))

        # 2. asset list
        r = await c.get(ASSETS_URL, params={"status": "active", "asset_class": "us_equity"})
        if r.status_code == 200:
            assets = r.json()
            ok = scanner.eligible_assets(assets)
            ex = sorted({a.get("exchange") for a in assets})
            # Thousands of US stocks should pass; far fewer suggests a filter/field mismatch.
            rep.add("PASS" if len(ok) > 1000 else "WARN", "alpaca.assets",
                    f"{len(assets)} listings, {len(ok)} pass the stock filter; exchanges {ex}")
        else:
            rep.add("FAIL", "alpaca.assets", _err(r))

        # 3. full-market daily history (the scanner and swing rules depend on it)
        async def bars(feed, tf, start, end, syms="SPY,NVDA"):
            """Raw bars request (helper used by checks 3-6)."""
            return await c.get(DATA_URL, params={"symbols": syms, "timeframe": tf, "start": _iso(start),
                                                 "end": _iso(end), "limit": 10000, "adjustment": "all",
                                                 "feed": feed, "sort": "asc"})
        r = await bars("sip", "1Day", now - timedelta(days=30), now - timedelta(hours=12))
        if r.status_code == 200:
            n = {k: len(v) for k, v in (r.json().get("bars") or {}).items()}
            rep.add("PASS" if n.get("SPY", 0) >= 10 else "WARN", "alpaca.daily_sip",
                    f"full-market daily bars over 30 days: {n}")
        else:
            rep.add("FAIL", "alpaca.daily_sip", _err(r))
        # Same data again, through the app's own client, to prove parsing works.
        try:
            got = await data.bars(["SPY"], "1Day", now - timedelta(days=30), now - timedelta(hours=12), "sip")
            rep.add("PASS", "alpaca.parse_bars", f"app parsed {len(got['SPY'])} bars; last {got['SPY'][-1]}")
        except Exception as e:
            rep.add("FAIL", "alpaca.parse_bars", f"{type(e).__name__}: {e}")

        # 4. which plan is this? recent full-market data is refused on the free plan
        r = await bars("sip", "1Min", now - timedelta(minutes=10), now)
        rep.add("PASS", "alpaca.plan", "recent full-market data ALLOWED: paid plan" if r.status_code == 200
                else f"recent full-market data refused ({r.status_code}): free plan, as expected")

        # 5. pre-market bars in the delayed full-market feed (the gap scan depends on it)
        # Use today's pre-market if at least 30 minutes of it are already 16+ minutes old
        # (the free-plan delay); otherwise use the last completed session's pre-market.
        day = nxt.day if (now - timedelta(minutes=16)) > datetime.combine(
            nxt.day, time(4, 30), tzinfo=NY).astimezone(timezone.utc) else done.day
        pm_start = datetime.combine(day, time(4, 0), tzinfo=NY).astimezone(timezone.utc)
        pm_end = min(datetime.combine(day, time(9, 29), tzinfo=NY).astimezone(timezone.utc),
                     now - timedelta(minutes=16))
        r = await bars("sip", "1Min", pm_start, pm_end, "SPY,NVDA,AAPL")
        if r.status_code == 200:
            b = r.json().get("bars") or {}
            n = {k: len(v) for k, v in b.items()}
            first = {k: v[0]["t"] for k, v in b.items() if v}
            rep.add("PASS" if sum(n.values()) > 0 else "FAIL", "alpaca.premarket",
                    f"pre-market minute bars for {day}: {n}; first bar {first}")
        else:
            rep.add("FAIL", "alpaca.premarket", _err(r))

        # 6. regular-session minute bars on the configured live feed (used after a restart)
        r = await bars(cfg.feed.feed, "1Min", done.open, done.open + timedelta(minutes=30))
        if r.status_code == 200:
            n = {k: len(v) for k, v in (r.json().get("bars") or {}).items()}
            rep.add("PASS" if n else "WARN", "alpaca.minute_" + cfg.feed.feed,
                    f"first 30 min of {done.day} on the {cfg.feed.feed} feed: {n} bars (30 = every minute)")
        else:
            rep.add("FAIL", "alpaca.minute_" + cfg.feed.feed, _err(r))

        # 7. latest trades through the app's client: how watchlist names that do not fit
        # on the websocket are priced during the session (feeds.Poller)
        try:
            got = await data.latest_trades(["SPY", "NVDA", "AAPL"])
            # Ages from the current time, never below zero (the local clock can be a little
            # behind Alpaca's).
            at = datetime.now(timezone.utc)
            age = {s: f"{max((at - t.ts).total_seconds(), 0) / 60:.0f} min ago" for s, t in got.items()}
            rep.add("PASS" if got else "WARN", "alpaca.latest_trades",
                    f"{len(got)} of 3 symbols on the {cfg.feed.feed} feed; last trade {age}")
        except Exception as e:
            rep.add("FAIL", "alpaca.latest_trades", f"{type(e).__name__}: {e}")

        # 8. news
        r = await c.get(NEWS_URL, params={"limit": 50, "sort": "desc", "start": _iso(now - timedelta(hours=48)),
                                          "end": _iso(now)})
        if r.status_code == 200:
            items = r.json().get("news") or []
            keys = sorted(items[0].keys()) if items else []
            rep.add("PASS" if items else "WARN", "alpaca.news",
                    f"{len(items)} articles on first page of last 48h; more pages: "
                    f"{bool(r.json().get('next_page_token'))}; fields {keys}")
            # Through the app's own parser, capped at 6 pages to keep the check quick.
            got = await data.news_all({"NVDA", "AAPL", "TSLA", "AMD", "MSFT"}, now - timedelta(hours=48), now, 6)
            rep.add("PASS" if got is not None else "FAIL", "alpaca.parse_news",
                    "app parsed headlines for " + str({k: len(v) for k, v in (got or {}).items()}))
        else:
            rep.add("FAIL", "alpaca.news", _err(r))

    await check_stream(cfg, rep, data)


async def _ws_session(url: str, key: str, secret: str, sub: dict, listen: float) -> dict:
    """Connect, log in, subscribe; return what the server said.

    Same protocol as AlpacaData.stream, but it only counts and records messages for
    `listen` seconds instead of yielding them, so the report can describe what happened.
    If the server drops the connection, what was received so far is still returned, with
    the reason in "closed": the server's error message usually explains the drop."""
    out = {"auth": None, "subscription": None, "errors": [], "trades": 0, "bars": 0, "sample": None,
           "closed": None}
    try:
        await _ws_listen(url, key, secret, sub, listen, out)
    except websockets.ConnectionClosed as e:
        out["closed"] = f"{type(e).__name__}: {e}"
    return out


async def _ws_listen(url: str, key: str, secret: str, sub: dict, listen: float, out: dict) -> None:
    """The body of _ws_session; fills `out` as messages arrive."""
    async with websockets.connect(url, open_timeout=15) as ws:
        await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
        loop = asyncio.get_running_loop()
        end = loop.time() + listen
        sent = False
        while loop.time() < end:
            try:
                # Wait only as long as the listening window has left (at least 0.1 s).
                raw = await asyncio.wait_for(ws.recv(), timeout=max(end - loop.time(), 0.1))
            except asyncio.TimeoutError:
                break
            for m in json.loads(raw):
                t = m.get("T")
                if t == "success" and m.get("msg") == "authenticated":
                    out["auth"] = True
                    if not sent:
                        sent = True
                        await ws.send(json.dumps({"action": "subscribe", **sub}))
                elif t == "subscription":
                    # Record how many symbols the server confirmed per channel.
                    out["subscription"] = {k: len(v) for k, v in m.items() if isinstance(v, list) and v}
                elif t == "error":
                    out["errors"].append(f"{m.get('code')} {m.get('msg')}")
                elif t == "t":
                    out["trades"] += 1
                    out["sample"] = out["sample"] or m      # keep the first trade as a sample
                elif t == "b":
                    out["bars"] += 1


async def check_stream(cfg: Config, rep: Report, data: AlpacaData) -> None:
    """Three websocket checks: 30 symbols accepted, a 31st refused, the test feed flows."""
    url = STREAM_URL.format(feed=cfg.feed.feed)
    syms = list(dict.fromkeys(cfg.symbols + [cfg.benchmark] + FILLER))
    is_open = MarketCalendar().is_open(datetime.now(timezone.utc))
    try:
        # Listen longer while the market is open, when real trades should arrive.
        r = await _ws_session(url, data.key, data.secret, {"trades": syms[:30], "bars": syms[:30]},
                              75 if is_open else 6)
        ok = r["auth"] and r["subscription"] and not r["errors"] and not r["closed"]
        live = (f"; in 75s: {r['trades']} trades, {r['bars']} bars; sample {r['sample']}" if is_open
                else "; market closed, so no prices expected (rerun during the session to see flow)")
        closed = f"; connection dropped ({r['closed']})" if r["closed"] else ""
        # On failure the hint replaces the flow summary; report lines are cut at 230 characters.
        rep.add("PASS" if ok else "FAIL", "alpaca.stream_30",
                f"login {r['auth']}, subscribed {r['subscription']}, errors {r['errors']}{closed}"
                + (live if ok else f" | {ONE_CONNECTION}"))
        if is_open and ok and r["trades"] == 0:
            rep.add("WARN", "alpaca.stream_flow", "market is open but no trades arrived in 75s")
    except Exception as e:
        rep.add("FAIL", "alpaca.stream_30", f"{type(e).__name__}: {e}")
        return
    await asyncio.sleep(2)            # the free plan allows one connection at a time
    try:
        # 31 symbols: the free plan should refuse; if it accepts, the cap is higher.
        # Either answer PASSes; the point is to learn which plan this key is on.
        r = await _ws_session(url, data.key, data.secret, {"trades": syms[:31]}, 6)
        if r["errors"]:
            rep.add("PASS", "alpaca.stream_31", f"31st symbol refused as expected: {r['errors']}")
        else:
            rep.add("PASS", "alpaca.stream_31", f"31 symbols ACCEPTED ({r['subscription']}): no 30 cap on this key")
    except Exception as e:
        rep.add("WARN", "alpaca.stream_31", f"{type(e).__name__}: {e}")
    await asyncio.sleep(2)
    try:                              # always-on test feed: proves message parsing even on a weekend
        r = await _ws_session(TEST_STREAM, data.key, data.secret, {"trades": ["FAKEPACA"]}, 8)
        rep.add("PASS" if r["trades"] else "WARN", "alpaca.stream_test",
                f"test feed: {r['trades']} trades; errors {r['errors']}; sample {r['sample']}")
    except Exception as e:
        rep.add("WARN", "alpaca.stream_test", f"{type(e).__name__}: {e}")


async def check_finnhub(cfg: Config, rep: Report) -> None:
    if not cfg.finnhub_key:
        rep.add("SKIP", "finnhub", "FINNHUB_API_KEY not set (earnings dates and profitability stay blank)")
        return
    today = datetime.now(timezone.utc).date()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{FINNHUB}/calendar/earnings", params={
            "from": today.isoformat(), "to": (today + timedelta(days=14)).isoformat(), "token": cfg.finnhub_key})
        if r.status_code == 200:
            rows = r.json().get("earningsCalendar") or []
            rep.add("PASS" if rows else "WARN", "finnhub.earnings",
                    f"{len(rows)} reports in the next 14 days; sample {rows[0] if rows else None}")
        else:
            # Finnhub takes the key as a URL parameter, so it can be echoed in errors: mask it.
            rep.add("FAIL", "finnhub.earnings", _err(r).replace(cfg.finnhub_key, "***"))
        r = await c.get(f"{FINNHUB}/stock/metric", params={"symbol": "AAPL", "metric": "all",
                                                           "token": cfg.finnhub_key})
        if r.status_code == 200:
            m = r.json().get("metric") or {}
            v = {k: m.get(k) for k in ("netProfitMarginTTM", "epsTTM")}
            rep.add("PASS" if any(x is not None for x in v.values()) else "WARN", "finnhub.metric",
                    f"AAPL: {v} ({len(m)} metrics returned)")
        else:
            rep.add("FAIL", "finnhub.metric", _err(r).replace(cfg.finnhub_key, "***"))


def check_anthropic(cfg: Config, rep: Report) -> None:
    """One real model call with the app's own gate prompt and parser, on a tiny sample
    designed to have an obvious answer (an offering headline should be rejected)."""
    if not cfg.anthropic_key:
        rep.add("SKIP", "anthropic", "ANTHROPIC_API_KEY not set (default plan, no reviews, ranked picks)")
        return
    try:
        import anthropic
        from .agent import SELECT_SCHEMA, SELECT_SYSTEM, build_request, parse_selection, reply_text
        client = anthropic.Anthropic(api_key=cfg.anthropic_key)
        sample = [{"ticker": "AAA", "rank": 1, "kind": "INTRADAY", "gap_pct": 4.2,
                   "headlines": ["AAA prices $500M secondary share offering"]},
                  {"ticker": "BBB", "rank": 2, "kind": "SWING", "rs_3m_pct": 12.0, "headlines": []}]
        # Same request the live gate sends: model, max_tokens, effort and the JSON schema.
        r = client.messages.create(**build_request(cfg, SELECT_SYSTEM, json.dumps(sample), SELECT_SCHEMA))
        got = parse_selection(reply_text(r), ["AAA", "BBB"])
        rep.add("PASS" if len(got) == 2 else "WARN", "anthropic.select",
                f"model {cfg.agent.model}, effort {cfg.agent.effort}; decisions {got}; "
                f"stop {r.stop_reason}; tokens in/out {r.usage.input_tokens}/{r.usage.output_tokens}")
    except Exception as e:
        rep.add("FAIL", "anthropic.select", f"{type(e).__name__}: {str(e).replace(cfg.anthropic_key, '***')}")


async def run(cfg: Config) -> int:
    """Run all checks in order, print a summary, return the process exit code."""
    rep = Report()
    print(f"OpenBell doctor {__version__} | python {platform.python_version()} | "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | feed={cfg.feed.feed}\n")
    # Each service check is isolated: a crash in one is reported as a FAIL and the
    # remaining checks still run.
    for name, coro in (("alpaca", check_alpaca(cfg, rep)), ("finnhub", check_finnhub(cfg, rep))):
        try:
            await coro
        except Exception as e:
            rep.add("FAIL", name, f"check crashed: {type(e).__name__}: {e}")
    # The Anthropic client is synchronous; run it in a thread like the live app does.
    await asyncio.to_thread(check_anthropic, cfg, rep)
    c = rep.counts()
    print(f"\nSummary: {c['PASS']} passed, {c['WARN']} warnings, {c['FAIL']} failed, {c['SKIP']} skipped.")
    print("Paste everything above when asking for help. It contains no keys.")
    return 1 if c["FAIL"] else 0
