"""Pre-market scanner: which stocks does the engine watch today?

  universe (weekly)  ->  numeric scan (daily)  ->  news / earnings  ->  agent gate  ->  watchlist

WHAT: Picks up to `max_watchlist` (45) symbols for the day out of the whole US market.
        1. Universe:   `ensure_universe` keeps the ~400 most liquid common stocks,
                       rebuilt once a week (cached in the journal's `universe` table).
        2. Numeric scan: `scan` builds two ranked lists from daily bars, plus optional
                       pre-market prices and news:
                         SWING    - the engine's trend-pullback setup is present and the
                                    stock beat the benchmark over ~3 months;
                         INTRADAY - the stock is "in play" today (gap up, fresh
                                    headline, or unusual activity yesterday);
                       then interleaves them into at most `candidates` (45) Candidates.
        3. Enrichment: done by runner.py (headlines, earnings date, profitability).
        4. Agent gate: `agent.select` approves or rejects each candidate.
        5. Watchlist:  `apply_decisions` keeps held symbols, then fills the free slots
                       with approved candidates in rank order.
HOW:  Steps 2 and 5 are pure functions (data in, list out, no network), which is what
      lets the replay and the simulation reuse them and makes them easy to test. Only
      `ensure_universe` performs I/O.
WHY:  Live data is limited (the free websocket takes 30 symbols; the rest of the
      watchlist is polled, see feeds.py), so the app cannot watch the whole market. The
      scan spends those slots on stocks where the strategies are most likely to fire,
      and its rank order decides which names get the real-time stream. The agent can only remove names (never add any), and being on the
      watchlist triggers nothing by itself: entries still need the engine's rules.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .config import Config
from .indicators import build_daily_context
from .journal import Journal

log = logging.getLogger("openbell.scanner")
# Words in an asset's name that mark it as a fund, leveraged/inverse product or
# non-common-stock security. `\b` = word boundary (so "Unit" does not match "United");
# re.I = case-insensitive. These products move differently from operating companies,
# and the strategies were not designed for them.
FUND_WORDS = re.compile(r"\b(ETF|ETN|Fund|ProShares|Direxion|iShares|SPDR|Index|Trust Series|"
                        r"2X|3X|Leveraged|Inverse|Warrant|Unit|Rights?)\b", re.I)


@dataclass
class Candidate:
    """One stock proposed by the scan, carried through enrichment and the agent gate,
    then saved to the journal's `scan` table."""
    symbol: str
    kind: str                     # SWING | INTRADAY | BOTH
    score: float                  # ranking score within its list (higher = better)
    rank: int = 0                 # 1 = best overall position after interleaving
    metrics: dict = field(default_factory=dict)    # numbers shown to the agent and dashboard
    headlines: list = field(default_factory=list)  # up to 4 recent headlines
    profitable: Optional[bool] = None              # from Finnhub; None = unknown
    earnings_in_days: Optional[int] = None         # sessions until the next report
    decision: str = "approve"     # "approve" or "reject" after the gate
    reason: str = ""              # the agent's (or fallback's) reason
    picked: bool = False          # made it onto today's watchlist


# ---------------------------------------------------------------- universe
def eligible_assets(assets: list) -> list:
    """Common stocks on the two main listing exchanges; funds and odd share classes dropped."""
    out = []
    for a in assets:
        sym = a.get("symbol", "")
        # sym.isalpha(): drops classes like "BRK.B" and odd tickers with digits;
        # len <= 5: US common stock tickers are 1-5 letters.
        if (a.get("tradable") and a.get("status") == "active"
                and a.get("exchange") in ("NYSE", "NASDAQ") and sym.isalpha() and len(sym) <= 5
                and not FUND_WORDS.search(a.get("name") or "")):
            out.append(sym)
    return out


def rank_universe(daily: dict, cfg: Config) -> list:
    """(symbol, price, dollar_volume) for the most liquid names, best first."""
    sc, rows = cfg.scanner, []
    for sym, bars in daily.items():
        if len(bars) < 15:                # too little history to judge liquidity
            continue
        last = bars[-20:]
        # Average daily dollar volume = shares x price, which compares a $10 stock and a
        # $1,000 stock fairly. Liquid names have tight spreads, so simulated fills are realistic.
        dv = sum(b.c * b.v for b in last) / len(last)
        if bars[-1].c >= sc.min_price and dv >= sc.min_dollar_volume:
            rows.append((sym, bars[-1].c, dv))
    rows.sort(key=lambda r: -r[2])        # most dollar volume first
    return rows[:sc.universe_size]


async def ensure_universe(data, journal: Journal, cfg: Config, now: datetime) -> list:
    """Return the cached universe if it is fresh, otherwise rebuild it from Alpaca.

    Rebuilding is expensive (thousands of symbols, many requests), so it happens only
    every `universe_refresh_days`. Liquidity rankings change slowly, so weekly is enough."""
    symbols, updated = journal.load_universe()
    if symbols and updated and now - datetime.fromisoformat(updated) < timedelta(
            days=cfg.scanner.universe_refresh_days):
        return symbols
    names = eligible_assets(await data.assets())
    daily: dict = {}
    # Alpaca accepts a limited number of symbols per request; fetch in chunks of 200.
    # 45 calendar days covers the 20 trading days rank_universe needs. The end is
    # 12 hours ago so only completed daily bars are used, and feed="sip" (whole market)
    # is free for data at least 15 minutes old.
    for i in range(0, len(names), 200):
        daily.update(await data.bars(names[i:i + 200], "1Day", now - timedelta(days=45),
                                     now - timedelta(hours=12), feed="sip"))
    rows = rank_universe(daily, cfg)
    if not rows:
        raise RuntimeError("universe build returned no symbols")
    journal.save_universe(rows, now.isoformat())
    log.info("universe rebuilt: %d of %d eligible stocks", len(rows), len(names))
    return [r[0] for r in rows]


# ---------------------------------------------------------------- numeric scan
def scan(cfg: Config, daily: dict, benchmark_bars: list, premarket: Optional[dict] = None,
         news: Optional[dict] = None, fresh_since: Optional[datetime] = None) -> list:
    """Rank candidates. Pure function, no I/O.

    SWING     the engine's own trend-pullback setup is present and the stock has beaten
              the benchmark over three months; ranked by that relative strength.
    INTRADAY  'in play' today, by any of:
                - gapping up before the open on meaningful pre-market volume,
                - a company headline since the previous close,
                - unusual volume or range yesterday with a close in the upper half.
              Ranked by gap size (in ATRs), pre-market volume, fresh news, then
              yesterday's activity.
    Stocks gapping down more than half an ATR are left out of both lists (long only).

    premarket: symbol -> {"price", "dollar_vol"} from delayed full-market bars, or None.
    news:      symbol -> [(timestamp, headline)] newest first, or None.
    With both None the intraday list rests on yesterday's activity alone."""
    sc = cfg.scanner
    # Benchmark's ~3-month return (63 trading days); each stock is compared against it.
    bench_ret = _ret(benchmark_bars, 63) or 0.0
    swing, intra, info, heads = [], [], {}, {}
    for sym, bars in daily.items():
        # 64 bars = enough for the 63-day return; the benchmark itself is never a candidate.
        if sym == cfg.benchmark or len(bars) < 64:
            continue
        ctx = build_daily_context(sym, bars, cfg.swing)   # same setup logic the engine trades on
        if ctx is None:
            continue
        last, prior = bars[-1], bars[-21:-1]   # yesterday, and the 20 days before it
        dv = sum(b.c * b.v for b in bars[-20:]) / 20
        # Liquidity and price floors again (the universe may be a week old), and skip
        # degenerate bars (high == low) that would divide by zero below.
        if last.c < sc.min_price or dv < sc.min_dollar_volume or last.h <= last.l or ctx.atr <= 0:
            continue
        # ---- yesterday's activity measures ----
        rs = (_ret(bars, 63) or 0.0) - bench_ret       # relative strength vs the benchmark, % points
        avg_v = sum(b.v for b in prior) / len(prior)
        rvol = last.v / avg_v if avg_v > 0 else 0.0     # yesterday's volume vs its 20-day average
        range_atr = (last.h - last.l) / ctx.atr         # yesterday's range in ATRs (>1 = wide day)
        close_pos = (last.c - last.l) / (last.h - last.l)  # 0 = closed at the low, 1 = at the high
        # ---- today's pre-market picture (if available) ----
        pm = (premarket or {}).get(sym)
        gap_pct = gap_atr = pm_share = None
        if pm and pm.get("price", 0) > 0:
            gap_pct = (pm["price"] / last.c - 1) * 100      # gap vs yesterday's close, in %
            gap_atr = (pm["price"] - last.c) / ctx.atr      # the same gap in ATRs (volatility-adjusted)
            pm_share = pm.get("dollar_vol", 0.0) / dv       # pre-market $ volume as a share of a normal day
        # ---- news ----
        items = (news or {}).get(sym) or []
        # "Fresh" = at least one headline since the previous close (older ones are priced in).
        fresh = bool(fresh_since) and any(ts >= fresh_since for ts, _ in items)
        heads[sym] = [h for _, h in items][:4]
        # Everything the agent and the dashboard get to see about this stock.
        info[sym] = {"price": round(last.c, 2), "rs_3m_pct": round(rs, 1), "rvol": round(rvol, 2),
                     "range_atr": round(range_atr, 2), "ret_5d_pct": round(ctx.ret_5d, 1),
                     "below_high_pct": round(ctx.dist_from_high_pct, 1),
                     "dollar_vol_m": round(dv / 1e6),
                     "gap_pct": None if gap_pct is None else round(gap_pct, 2),
                     "premarket_vol_pct_of_day": None if pm_share is None else round(pm_share * 100, 2),
                     "fresh_news": fresh}
        # ---- classification ----
        # A gap only counts if real money traded on it; a 4% gap on a few hundred shares is noise.
        gapping = (gap_pct is not None and gap_pct >= sc.min_gap_pct
                   and pm_share >= sc.min_premarket_share)
        # Unusual yesterday: 1.5x volume or a 1.5-ATR range, AND buyers won (upper-half close).
        active = (rvol >= 1.5 or range_atr >= 1.5) and close_pos >= 0.5
        gapping_down = gap_atr is not None and gap_atr < -0.5
        if ctx.swing_trigger is not None and rs >= sc.min_rel_strength_pct and not gapping_down:
            swing.append((rs, sym))
        if (gapping or fresh or active) and not gapping_down:
            # Intraday score: each term is capped so no single factor dominates.
            #   gap in ATRs, 0..3                          -> up to 3.0
            #   pre-market share of a day's volume, 0..20% -> x10 -> up to 2.0
            #   fresh headline                             -> 1.0
            #   yesterday's relative volume (if active)    -> 0.3 x (up to 3) -> up to 0.9
            score = (min(max(gap_atr or 0.0, 0.0), 3.0) + min(pm_share or 0.0, 0.2) * 10
                     + (1.0 if fresh else 0.0) + (0.3 * min(rvol, 3.0) if active else 0.0))
            intra.append((score, sym))
    # Tuples sort by their first element (the score), best first.
    swing.sort(reverse=True)
    intra.sort(reverse=True)
    # Interleave: SWING #1, INTRADAY #1, SWING #2, INTRADAY #2, ...
    # Why: the two lists use different score scales, so merging by score would be
    # meaningless; alternating gives both strategies a fair share of the slots.
    # A symbol found in both lists keeps its first position and becomes kind "BOTH".
    out, seen = [], {}
    for i in range(max(len(swing), len(intra))):          # interleave the two lists
        for kind, lst in (("SWING", swing), ("INTRADAY", intra)):
            if i >= len(lst):
                continue
            score, sym = lst[i]
            if sym in seen:
                seen[sym].kind = "BOTH"
            elif len(out) < sc.candidates:
                c = Candidate(sym, kind, round(score, 3), rank=len(out) + 1, metrics=info[sym],
                              headlines=heads[sym])
                seen[sym] = c
                out.append(c)
    return out


def _ret(bars: list, n: int) -> Optional[float]:
    """% return over the last `n` bars, or None if the history is too short."""
    return (bars[-1].c / bars[-1 - n].c - 1) * 100 if len(bars) > n else None


# ---------------------------------------------------------------- gate
def apply_decisions(candidates: list, decisions: Optional[dict], slots: int, held: set) -> list:
    """Mark each candidate approved/rejected and pick the best `slots`. Returns picked symbols.

    decisions=None (agent off or failed) means: take the top-ranked."""
    picks = []
    for c in candidates:                         # already in rank order
        if decisions is None:
            c.decision, c.reason = "approve", "top-ranked (agent not used)"
        elif c.symbol in decisions:
            ok, why = decisions[c.symbol]
            c.decision, c.reason = ("approve" if ok else "reject"), why
        else:
            # The agent skipped this ticker. Default to keeping it: a missing answer is
            # not evidence of a problem (the gate is meant to reject on concrete reasons).
            c.decision, c.reason = "approve", "no decision returned; kept on rank"
        if c.decision == "approve" and c.symbol not in held and len(picks) < slots:
            c.picked = True
            picks.append(c.symbol)
        elif c.symbol in held:
            c.picked = True                      # already on the list through a position
    return picks
