from __future__ import annotations

from typing import Optional, Sequence

from .models import Bar, DailyContext
from .config import SwingCfg


def ema(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        raise ValueError("not enough values for EMA")
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def atr(bars: Sequence[Bar], period: int) -> float:
    """Wilder ATR over completed bars."""
    if len(bars) < period + 1:
        raise ValueError("not enough bars for ATR")
    trs = []
    for prev, cur in zip(bars, bars[1:]):
        trs.append(max(cur.h - cur.l, abs(cur.h - prev.c), abs(cur.l - prev.c)))
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def build_daily_context(symbol: str, bars: Sequence[Bar], cfg: SwingCfg) -> Optional[DailyContext]:
    """Facts from completed daily bars (last bar = previous session). None if too short."""
    need = max(cfg.ema_slow, cfg.atr_period + 1, 21) + 1
    if len(bars) < need:
        return None
    closes = [b.c for b in bars]
    last = bars[-1]
    a = atr(bars, cfg.atr_period)
    ef = ema(closes, cfg.ema_fast)
    es = ema(closes, cfg.ema_slow)
    hi = max(closes[-252:])
    ctx = DailyContext(
        symbol=symbol, prev_close=last.c, prev_high=last.h, prev_low=last.l,
        atr=a, ema_fast=ef, ema_slow=es,
        ret_5d=(closes[-1] / closes[-6] - 1) * 100,
        ret_20d=(closes[-1] / closes[-21] - 1) * 100,
        dist_from_high_pct=(1 - closes[-1] / hi) * 100,
    )
    # Trend-pullback setup: uptrend, yesterday dipped toward the fast EMA and
    # held above the slow EMA. Entry is a buy-stop above yesterday's high.
    uptrend = last.c > es and ef > es
    pulled_back = last.l <= ef + cfg.pullback_atr * a
    held = last.c > es
    if uptrend and pulled_back and held and a > 0:
        ctx.swing_trigger = round(last.h + 0.01, 2)
        ctx.swing_reason = (f"uptrend (close {last.c:.2f} > EMA{cfg.ema_slow} {es:.2f}), "
                            f"pullback to EMA{cfg.ema_fast} {ef:.2f}; buy-stop {ctx.swing_trigger:.2f}")
    return ctx
