"""Pre-market scanner: which stocks does the engine watch today?

  universe (weekly)  ->  numeric scan (daily)  ->  news / earnings  ->  agent gate  ->  watchlist

The scan itself is plain arithmetic on completed daily bars. The agent can only
approve or discard what the scan proposes; entries still come from the engine's rules."""
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
FUND_WORDS = re.compile(r"\b(ETF|ETN|Fund|ProShares|Direxion|iShares|SPDR|Index|Trust Series|"
                        r"2X|3X|Leveraged|Inverse|Warrant|Unit|Rights?)\b", re.I)


@dataclass
class Candidate:
    symbol: str
    kind: str                     # SWING | INTRADAY | BOTH
    score: float
    rank: int = 0
    metrics: dict = field(default_factory=dict)
    headlines: list = field(default_factory=list)
    profitable: Optional[bool] = None
    earnings_in_days: Optional[int] = None
    decision: str = "approve"
    reason: str = ""
    picked: bool = False


# ---------------------------------------------------------------- universe
def eligible_assets(assets: list) -> list:
    """Common stocks on the two main listing exchanges; funds and odd share classes dropped."""
    out = []
    for a in assets:
        sym = a.get("symbol", "")
        if (a.get("tradable") and a.get("status") == "active"
                and a.get("exchange") in ("NYSE", "NASDAQ") and sym.isalpha() and len(sym) <= 5
                and not FUND_WORDS.search(a.get("name") or "")):
            out.append(sym)
    return out


def rank_universe(daily: dict, cfg: Config) -> list:
    """(symbol, price, dollar_volume) for the most liquid names, best first."""
    sc, rows = cfg.scanner, []
    for sym, bars in daily.items():
        if len(bars) < 15:
            continue
        last = bars[-20:]
        dv = sum(b.c * b.v for b in last) / len(last)
        if bars[-1].c >= sc.min_price and dv >= sc.min_dollar_volume:
            rows.append((sym, bars[-1].c, dv))
    rows.sort(key=lambda r: -r[2])
    return rows[:sc.universe_size]


async def ensure_universe(data, journal: Journal, cfg: Config, now: datetime) -> list:
    symbols, updated = journal.load_universe()
    if symbols and updated and now - datetime.fromisoformat(updated) < timedelta(
            days=cfg.scanner.universe_refresh_days):
        return symbols
    names = eligible_assets(await data.assets())
    daily: dict = {}
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
    bench_ret = _ret(benchmark_bars, 63) or 0.0
    swing, intra, info, heads = [], [], {}, {}
    for sym, bars in daily.items():
        if sym == cfg.benchmark or len(bars) < 64:
            continue
        ctx = build_daily_context(sym, bars, cfg.swing)
        if ctx is None:
            continue
        last, prior = bars[-1], bars[-21:-1]
        dv = sum(b.c * b.v for b in bars[-20:]) / 20
        if last.c < sc.min_price or dv < sc.min_dollar_volume or last.h <= last.l or ctx.atr <= 0:
            continue
        rs = (_ret(bars, 63) or 0.0) - bench_ret
        avg_v = sum(b.v for b in prior) / len(prior)
        rvol = last.v / avg_v if avg_v > 0 else 0.0
        range_atr = (last.h - last.l) / ctx.atr
        close_pos = (last.c - last.l) / (last.h - last.l)
        pm = (premarket or {}).get(sym)
        gap_pct = gap_atr = pm_share = None
        if pm and pm.get("price", 0) > 0:
            gap_pct = (pm["price"] / last.c - 1) * 100
            gap_atr = (pm["price"] - last.c) / ctx.atr
            pm_share = pm.get("dollar_vol", 0.0) / dv
        items = (news or {}).get(sym) or []
        fresh = bool(fresh_since) and any(ts >= fresh_since for ts, _ in items)
        heads[sym] = [h for _, h in items][:4]
        info[sym] = {"price": round(last.c, 2), "rs_3m_pct": round(rs, 1), "rvol": round(rvol, 2),
                     "range_atr": round(range_atr, 2), "ret_5d_pct": round(ctx.ret_5d, 1),
                     "below_high_pct": round(ctx.dist_from_high_pct, 1),
                     "dollar_vol_m": round(dv / 1e6),
                     "gap_pct": None if gap_pct is None else round(gap_pct, 2),
                     "premarket_vol_pct_of_day": None if pm_share is None else round(pm_share * 100, 2),
                     "fresh_news": fresh}
        gapping = (gap_pct is not None and gap_pct >= sc.min_gap_pct
                   and pm_share >= sc.min_premarket_share)
        active = (rvol >= 1.5 or range_atr >= 1.5) and close_pos >= 0.5
        gapping_down = gap_atr is not None and gap_atr < -0.5
        if ctx.swing_trigger is not None and rs >= sc.min_rel_strength_pct and not gapping_down:
            swing.append((rs, sym))
        if (gapping or fresh or active) and not gapping_down:
            score = (min(max(gap_atr or 0.0, 0.0), 3.0) + min(pm_share or 0.0, 0.2) * 10
                     + (1.0 if fresh else 0.0) + (0.3 * min(rvol, 3.0) if active else 0.0))
            intra.append((score, sym))
    swing.sort(reverse=True)
    intra.sort(reverse=True)
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
    return (bars[-1].c / bars[-1 - n].c - 1) * 100 if len(bars) > n else None


# ---------------------------------------------------------------- gate
def apply_decisions(candidates: list, decisions: Optional[dict], slots: int, held: set) -> list:
    """Mark each candidate approved/rejected and pick the best `slots`. Returns picked symbols.

    decisions=None (agent off or failed) means: take the top-ranked."""
    picks = []
    for c in candidates:
        if decisions is None:
            c.decision, c.reason = "approve", "top-ranked (agent not used)"
        elif c.symbol in decisions:
            ok, why = decisions[c.symbol]
            c.decision, c.reason = ("approve" if ok else "reject"), why
        else:
            c.decision, c.reason = "approve", "no decision returned; kept on rank"
        if c.decision == "approve" and c.symbol not in held and len(picks) < slots:
            c.picked = True
            picks.append(c.symbol)
        elif c.symbol in held:
            c.picked = True                      # already on the list through a position
    return picks
