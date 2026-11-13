"""Optional Finnhub lookups: next earnings date and a profitability flag.
Everything here degrades to 'unknown' on any error or when no key is set.

WHAT: `Finnhub.earnings()` returns each symbol's next report date in the coming two
      weeks; `Finnhub.profitable()` says whether each candidate is profitable on a
      trailing-twelve-month basis.
HOW:  Plain async HTTP calls to the Finnhub REST API with `httpx`. Every method returns
      an empty dict (meaning "unknown") when there is no key or anything goes wrong.
WHY:  * Earnings dates feed the engine's swing blackout: no new multi-day position just
        before a report, which can gap the stock past any stop.
      * Profitability is extra context for the agent gate (e.g. hype on a loss-maker).
      Both are nice-to-have, never required, so a missing key or a Finnhub outage must
      never stop the scan; hence the "degrade to unknown" design.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Optional

import httpx

log = logging.getLogger("openbell.finnhub")
BASE = "https://finnhub.io/api/v1"           # module-level so tests can redirect it


class Finnhub:
    def __init__(self, key: str) -> None:
        self.key = key

    @property
    def active(self) -> bool:
        return bool(self.key)

    async def earnings(self, start: date, days: int = 14) -> dict:
        """symbol -> next report date within the window."""
        if not self.active:
            return {}
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                # One call returns the whole market's earnings calendar for the window.
                r = await c.get(f"{BASE}/calendar/earnings", params={
                    "from": start.isoformat(), "to": (start + timedelta(days=days)).isoformat(),
                    "token": self.key})
                r.raise_for_status()
                out: dict = {}
                for row in r.json().get("earningsCalendar") or []:
                    try:
                        d = date.fromisoformat(row["date"])
                    except (KeyError, ValueError, TypeError):   # missing or malformed date: skip the row
                        continue
                    # Keep the EARLIEST future date per symbol (the calendar can list
                    # several rows for one company).
                    if d >= start and (row["symbol"] not in out or d < out[row["symbol"]]):
                        out[row["symbol"]] = d
                return out
        except Exception as e:
            log.warning("earnings calendar unavailable (%s)", type(e).__name__)
            return {}

    async def profitable(self, symbols: list) -> dict:
        """symbol -> True/False from trailing net margin (or EPS). Missing = unknown."""
        if not self.active:
            return {}
        out: dict = {}
        async with httpx.AsyncClient(timeout=20) as c:
            for sym in symbols:
                try:
                    r = await c.get(f"{BASE}/stock/metric",
                                    params={"symbol": sym, "metric": "all", "token": self.key})
                    if r.status_code == 429:
                        # Rate limited: pause and move on. This symbol stays "unknown"
                        # rather than delaying the whole pre-market scan.
                        await asyncio.sleep(5)
                        continue
                    r.raise_for_status()
                    m = r.json().get("metric") or {}
                    # Prefer net profit margin; fall back to earnings per share. Either
                    # one above zero means the company made money over the last 12 months.
                    v = m.get("netProfitMarginTTM")
                    if v is None:
                        v = m.get("epsTTM")
                    if v is not None:
                        out[sym] = v > 0
                except Exception as e:
                    log.warning("metric lookup failed for %s (%s)", sym, type(e).__name__)
                await asyncio.sleep(1.1)        # stay under 60 calls a minute
        return out
