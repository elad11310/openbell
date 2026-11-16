# OpenBell

An autonomous **paper-trading** agent for the US cash session. It wakes up for every
NYSE trading day, streams real-time prices while the market is open, trades two
rule-based strategies under hard risk limits, and has an LLM agent that plans the
day before the open and reviews it after the close.

No real orders are ever sent. Fills are simulated inside the app against live prices.

## How it is put together

```
            before the open                 9:30-16:00 New York                after the close
 daily bars ──► Agent (LLM) ──► DayPlan ──► Engine (rules + risk + paper broker) ──► Agent review
                                              ▲
                          Alpaca websocket ───┘  ticks + 1-minute bars
```

| Layer | File | What it does |
|---|---|---|
| Schedule | `market_calendar.py` | NYSE sessions: weekends, holidays, half-days, daylight saving |
| Data | `alpaca.py` | Daily/minute history over REST, live trades and bars over websocket, auto-reconnect |
| Scanner | `scanner.py`, `finnhub.py` | Picks the daily watchlist from the 400 most liquid stocks |
| Engine | `engine.py` | Strategies, position sizing, limits, stops/targets, simulated fills. Deterministic |
| Agent | `agent.py` | Pre-market plan and post-close review. Can only *reduce* what the engine may do |
| Journal | `journal.py` | SQLite: positions, cash, trades, events, equity curve. Survives restarts |
| Runner | `runner.py` | The loop: wait for the session, prepare, trade, review, repeat |
| Dashboard | `api.py`, `dashboard.html` | Read-only status page, works on a phone |
| Charts | `charts.py` | Candles, volume, EMA 20/50 and VWAP for any ticker on the dashboard |

**Why the LLM is not in the price loop.** A model call takes seconds and is not
repeatable; a stop has to act on the next tick, every time. So rules own execution,
and the agent owns judgement: market regime, a risk multiplier between 0 and 1, and
which symbols may trade. Its reply is parsed and clamped, and if the call fails the
engine uses the default plan. The limits in `config.yaml` hold whatever the model says.

## The scanner

Each morning, about 25 minutes before the open, the app chooses that day's watchlist:

1. **Universe (weekly).** Every tradable NYSE/Nasdaq common stock from Alpaca, cut to the
   400 most liquid (price at least $10, average daily dollar volume at least $50M).
2. **Numeric scan (daily).** Two lists, interleaved by rank:
   - `SWING`: the engine's own trend-pullback setup is present and the stock has beaten SPY
     over three months; ranked by that relative strength.
   - `INTRADAY`: in play today, by any of: gapping up at least 1% before the open on real
     pre-market volume; a company headline since the previous close; or unusual volume or
     range yesterday. Ranked by gap size (in ATRs), pre-market volume, fresh news, then
     yesterday's activity. No three-month strength test here.

   Stocks gapping down more than half an ATR are dropped from both lists (long only).
   Pre-market prices come from Alpaca's full-market bars, which the free plan serves once
   they are 15 minutes old. If that data or the news feed is unavailable, the scan uses
   what is left and says so in the activity log.
3. **Enrichment.** Up to four headlines from the last 48 hours (one bulk pull of Alpaca
   news), plus the next earnings date and a profitability flag (Finnhub, optional).
4. **Agent gate.** The agent approves or discards each of the 45 candidates with a reason.
   It is told to reject on concrete problems (offering, legal trouble, guidance cut, thin
   hype) and that no news is not a reason to reject. If the agent is off or fails, the
   top-ranked candidates are used.
5. **Watchlist.** Symbols with open positions first, then approved candidates by rank, up
   to 29 names plus SPY.

Being on the watchlist triggers nothing. Entries still need the strategy rules below.
Swing entries are also blocked within 3 sessions of an earnings report.

Every candidate is stored in the `scan` table with its metrics, headlines, decision and
reason. Later runs fill in its next-day and 5-day move, and the dashboard shows the
average for approved versus rejected names, so you can check whether the gate adds value.

## Charts

Click any ticker on the dashboard (positions, watchlist, scan, trades, activity) to open
its price chart. You can also type a symbol in the chart's box, or open
`http://127.0.0.1:8000/#chart=NVDA`. Use the 1D, 5D, 1M, 6M and 1Y buttons to change the range.

| Range | Candles | Lines |
|---|---|---|
| 1D | 5-minute, today's (or the last) session | EMA 20, EMA 50, session VWAP |
| 5D | 15-minute, last 5 sessions | EMA 20, EMA 50, session VWAP |
| 1M, 6M, 1Y | daily | EMA 20, EMA 50 |

- Volume bars sit under the candles with a 20-bar average line.
- Green arrows mark this app's paper entries and red arrows its exits.
- Dotted lines show the open position's entry, stop and target, today's swing trigger,
  and the opening range on intraday charts.
- The legend shows open, high, low, close, volume and the indicator values under the cursor.
- The EMA periods are the swing strategy's, so the lines are the ones the engine trades on.

Charts are drawn with TradingView's open-source
[Lightweight Charts](https://github.com/tradingview/lightweight-charts) library, loaded
from a CDN. Live charts use Alpaca's full-market (SIP) bars, which the free plan serves
once they are 15 minutes old, so **live charts end 16 minutes ago** and say so. If your
account cannot read SIP, charts fall back to the configured feed (IEX), whose volume is
only one exchange's share.

Under `sim --serve`, charts show the synthetic prices. Intraday bars exist only for
days a symbol was on the watchlist.

Each chart costs one Alpaca request. It is cached for 60 s (intraday) or 10 min (daily),
and an open chart refreshes once a minute. Charts are for display only: the engine never
reads them.

## Strategies (long only)

- **ORB, intraday.** First 1-minute close above the 15-minute opening range, above
  session VWAP, on heavier-than-average volume. Stop at the range low (capped at
  0.5 daily ATR), target 2R, one attempt per symbol per day, no entries in the last
  hour, flat 10 minutes before the close (half-days included).
- **SWING, multi-day.** Uptrend (close above EMA50, EMA20 above EMA50) with a
  pullback toward EMA20. Buy-stop above yesterday's high, skipped if price gaps too
  far past it. Stop 2 ATR, then a 3 ATR trail from the highest close. Exits at the
  open after a close below EMA50 or after 30 days.

These are sensible, well-known rule sets, not a proven edge. Treat the first
months of paper results as the test.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # Alpaca keys (free paper account); Anthropic and Finnhub keys are optional
python -m openbell doctor   # checks every outside service with your keys; paste the report if anything fails
python -m pytest -q         # no keys or network needed
python -m openbell sim --days 20 --serve    # no keys needed: synthetic data + dashboard
python -m openbell run      # live sessions, paper only; dashboard on http://127.0.0.1:8000
```

`run` is meant to stay up (systemd, Docker, a small VPS). It sleeps until 25 minutes
before each open. If it restarts mid-session it rebuilds VWAP and the opening range
from today's bars and restores positions, cash and the day's counters from SQLite.

`python -m openbell replay --start 2026-09-01 --end 2026-10-30` runs the same engine
over real historical minute bars. Do this before trusting any setting. A replay uses the
numeric scan only: past headlines and agent decisions are not reproduced, and the universe
is today's most liquid stocks, which flatters the result (survivorship bias).

## What has and has not been verified

- Verified here: the engine, risk limits, calendar, journal, restart recovery, plan
  parsing, the scanner logic, the pre-market flow against stubbed services, the
  simulation and the dashboard (tests plus a rendered run).
- The Alpaca, Finnhub and websocket client code is tested against a local stand-in server
  written from the published protocols (`tests/fake_services.py`).
- **Not run against the real services** from the build environment, which could not reach
  them. `python -m openbell doctor` is the real check: it tests your keys, the market clock,
  the stock list, full-market history, pre-market bars, news, the 30-symbol stream limit,
  Finnhub and the agent model, and prints a report with no keys in it. The build environment could not reach them and had no keys.
  They follow the published protocols, but watch the first live session.
- The simulation uses random-walk prices. It proves the plumbing; its P&L means nothing.

## Known limits

- The free Alpaca stream carries **IEX trades only**, a small share of total volume.
  Prices are real-time but sparse in quiet names, and volume filters see IEX volume.
  The paid `sip` feed fixes this (`feed.feed: sip`). The free stream allows 30 symbols.
- Fills are idealised: last trade price plus fixed slippage, no queue, no partial fills.
- Long only, US equities only, no pre-market or after-hours trading.
- The scan runs once, before the open. Its pre-market picture is 15 minutes old, and it
  does not rescan during the session.
- Simulations and replays have no pre-market data or headlines, so their intraday list
  rests on yesterday's activity only.
- Whether pre-market bars, Alpaca news and Finnhub's earnings calendar are included on
  free keys was not confirmed; `doctor` tells you. If one is refused, the scan carries on
  without it.
- An open swing position is not closed ahead of its earnings date; only new entries are blocked.
- The dashboard has no login. Keep it on localhost or behind a VPN/tunnel.
- Live charts lag 16 minutes (free full-market data) and need internet access for the chart library.
