"""NYSE trading calendar: when does the market open and close?

WHAT: `MarketCalendar` answers three questions: which session is in progress or next
      (`current_or_next`), is the market open right now (`is_open`), and which sessions
      fall between two dates (`sessions_between`). Each answer is a `models.Session`
      with UTC open/close times.
HOW:  It wraps the `exchange_calendars` library's "XNYS" (New York Stock Exchange)
      calendar, which ships the official holiday and half-day schedule. Results are
      converted from pandas Timestamps to plain Python UTC datetimes.
WHY:  Market hours are full of traps: weekends, about ten holidays a year, 13:00
      half-days (e.g. the day after Thanksgiving), and the open moving between 13:30 and
      14:30 UTC with US daylight saving. Putting all of that behind one small class means
      the runner, the replay, the doctor and the tests never do time-zone arithmetic for
      sessions themselves.
"""
from __future__ import annotations

from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .models import Session

# New York time zone. Also imported by runner.py and doctor.py for "4:00 am New York"
# style calculations (the pre-market starts at 04:00 local time).
NY = ZoneInfo("America/New_York")


class MarketCalendar:
    """NYSE schedule: weekends, holidays, half-days and DST are all handled here."""

    def __init__(self) -> None:
        # Loading a calendar takes a moment, so it is done once per instance.
        self._cal = xcals.get_calendar("XNYS")

    def _session(self, label) -> Session:
        """Build a Session from an exchange_calendars session label (a date)."""
        o = self._cal.session_open(label).to_pydatetime()
        c = self._cal.session_close(label).to_pydatetime()
        # A half-day is any session closing before 16:00 New York time. Comparing in NY
        # time (not UTC) keeps this right on both sides of a daylight-saving change.
        early = c.astimezone(NY).time() < time(16, 0)
        return Session(day=label.date(), open=o, close=c, early_close=early)

    def current_or_next(self, now: datetime) -> Session:
        """The session in progress at `now`, otherwise the next one to open."""
        # exchange_calendars works in whole minutes, so round the time down to the minute.
        ts = pd.Timestamp(now.astimezone(timezone.utc)).floor("min")
        # direction="next": if `ts` is outside trading hours (night, weekend, holiday),
        # roll forward to the next session instead of raising an error.
        return self._session(self._cal.minute_to_session(ts, direction="next"))

    def is_open(self, now: datetime) -> bool:
        s = self.current_or_next(now)
        # The close itself counts as closed (half-open interval [open, close)).
        return s.open <= now < s.close

    def sessions_between(self, start, end) -> list:
        """All sessions with a date in [start, end]. Accepts dates, datetimes or strings."""
        return [self._session(s) for s in self._cal.sessions_in_range(
            pd.Timestamp(start), pd.Timestamp(end))]
