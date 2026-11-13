from __future__ import annotations

from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .models import Session

NY = ZoneInfo("America/New_York")


class MarketCalendar:
    """NYSE schedule: weekends, holidays, half-days and DST are all handled here."""

    def __init__(self) -> None:
        self._cal = xcals.get_calendar("XNYS")

    def _session(self, label) -> Session:
        o = self._cal.session_open(label).to_pydatetime()
        c = self._cal.session_close(label).to_pydatetime()
        early = c.astimezone(NY).time() < time(16, 0)
        return Session(day=label.date(), open=o, close=c, early_close=early)

    def current_or_next(self, now: datetime) -> Session:
        """The session in progress at `now`, otherwise the next one to open."""
        ts = pd.Timestamp(now.astimezone(timezone.utc)).floor("min")
        return self._session(self._cal.minute_to_session(ts, direction="next"))

    def is_open(self, now: datetime) -> bool:
        s = self.current_or_next(now)
        return s.open <= now < s.close

    def sessions_between(self, start, end) -> list:
        return [self._session(s) for s in self._cal.sessions_in_range(
            pd.Timestamp(start), pd.Timestamp(end))]
