"""Technical indicators and the swing-setup detector, computed from daily bars.

WHAT: `ema` (exponential moving average), `atr` (Average True Range) and
      `build_daily_context`, which turns a symbol's daily history into a
      `models.DailyContext`: yesterday's prices, volatility, trend, recent returns and,
      if present, the swing strategy's buy trigger.
HOW:  Pure arithmetic over lists of completed bars; no I/O, no clock, no state.
      The last bar in the list is treated as "yesterday" (the previous session).
WHY:  The same function must give the same answer live, in a historical replay, in the
      simulation and in the scanner. Making it pure (same input -> same output) is what
      guarantees that, and makes it trivial to unit-test.
      Callers are responsible for passing only bars *before* the session being traded
      (see `replay.contexts_for`); otherwise the indicators would "see the future".
"""
from __future__ import annotations

from typing import Optional, Sequence

from .models import Bar, DailyContext
from .config import SwingCfg


def ema(values: Sequence[float], period: int) -> float:
    """Exponential moving average of `values`, returning only the latest value.

    How it works: an EMA is an average that weights recent values more heavily.
      1. Seed it with a simple average of the first `period` values.
      2. For every later value: ema = value * k + previous_ema * (1 - k),
         where k = 2 / (period + 1). For period 20, k is about 0.095, so each new day
         contributes about 9.5% and the past fades out geometrically.
    Why EMA (not a simple average): it reacts faster to a change in trend, while still
    smoothing day-to-day noise."""
    if len(values) < period:
        raise ValueError("not enough values for EMA")
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def atr(bars: Sequence[Bar], period: int) -> float:
    """Wilder ATR over completed bars.

    What it measures: how much the stock typically moves in a day, in dollars. Stops and
    targets are set in ATR units so they scale with each stock's volatility: a 2-ATR
    stop is "wide" for a calm utility and for a jumpy chip stock alike.

    How:
      * True Range of a day = the largest of
          high - low                     (the day's own range),
          |high - previous close|        (a gap up counts),
          |low  - previous close|        (a gap down counts).
        Plain high-low would miss overnight gaps.
      * Seed with the simple average of the first `period` true ranges, then apply
        Wilder's smoothing: atr = (atr * (period - 1) + tr) / period. This is an EMA with
        k = 1/period, the original definition by J. Welles Wilder."""
    if len(bars) < period + 1:          # each TR needs the previous bar, hence the +1
        raise ValueError("not enough bars for ATR")
    trs = []
    # zip(bars, bars[1:]) pairs each bar with the next one: (b0,b1), (b1,b2), ...
    for prev, cur in zip(bars, bars[1:]):
        trs.append(max(cur.h - cur.l, abs(cur.h - prev.c), abs(cur.l - prev.c)))
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def build_daily_context(symbol: str, bars: Sequence[Bar], cfg: SwingCfg) -> Optional[DailyContext]:
    """Facts from completed daily bars (last bar = previous session). None if too short."""
    # Enough history for the slowest indicator (EMA50), ATR (period + 1 bars) and the
    # 20-day return (21 closes), plus one spare. Too-short histories (new listings,
    # data gaps) are skipped rather than producing unreliable numbers.
    need = max(cfg.ema_slow, cfg.atr_period + 1, 21) + 1
    if len(bars) < need:
        return None
    closes = [b.c for b in bars]
    last = bars[-1]                     # yesterday's bar
    a = atr(bars, cfg.atr_period)
    ef = ema(closes, cfg.ema_fast)
    es = ema(closes, cfg.ema_slow)
    hi = max(closes[-252:])             # highest close of roughly the last trading year
    ctx = DailyContext(
        symbol=symbol, prev_close=last.c, prev_high=last.h, prev_low=last.l,
        atr=a, ema_fast=ef, ema_slow=es,
        # closes[-6] is the close 5 sessions before yesterday, closes[-21] is 20 before.
        ret_5d=(closes[-1] / closes[-6] - 1) * 100,
        ret_20d=(closes[-1] / closes[-21] - 1) * 100,
        dist_from_high_pct=(1 - closes[-1] / hi) * 100,
    )
    # Trend-pullback setup: uptrend, yesterday dipped toward the fast EMA and
    # held above the slow EMA. Entry is a buy-stop above yesterday's high.
    #
    # The idea in plain words: in a stock that is trending up, buy a short-term dip, but
    # only once the price proves the dip is over by trading above yesterday's high.
    #   uptrend     - price above the 50-day EMA and the 20-day EMA above the 50-day:
    #                 both the price and the short-term average sit above the long one.
    #   pulled_back - yesterday's low came down to (or within 1 ATR above) the 20-day
    #                 EMA. A stock far above its EMA is "extended", not pulling back.
    #   held        - yesterday still closed above the 50-day EMA, so the dip did not
    #                 break the trend.
    uptrend = last.c > es and ef > es
    pulled_back = last.l <= ef + cfg.pullback_atr * a
    held = last.c > es
    if uptrend and pulled_back and held and a > 0:
        # One cent above yesterday's high: the engine buys only if today trades through it.
        ctx.swing_trigger = round(last.h + 0.01, 2)
        ctx.swing_reason = (f"uptrend (close {last.c:.2f} > EMA{cfg.ema_slow} {es:.2f}), "
                            f"pullback to EMA{cfg.ema_fast} {ef:.2f}; buy-stop {ctx.swing_trigger:.2f}")
    return ctx
