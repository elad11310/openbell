from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional

from .models import Position

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, strategy TEXT, qty INTEGER, entry_px REAL, entry_ts TEXT,
    stop REAL, initial_stop REAL, target REAL, high_close REAL, exit_flag TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, symbol TEXT, strategy TEXT, qty INTEGER,
    entry_ts TEXT, entry_px REAL, exit_ts TEXT, exit_px REAL, initial_stop REAL,
    pnl REAL, r_multiple REAL, exit_reason TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, symbol TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, cash REAL, benchmark REAL);
CREATE TABLE IF NOT EXISTS days (
    day TEXT PRIMARY KEY, start_equity REAL, end_equity REAL, pnl REAL, trades INTEGER,
    benchmark_close REAL, plan TEXT, review TEXT);
CREATE TABLE IF NOT EXISTS universe (symbol TEXT PRIMARY KEY, price REAL, dollar_volume REAL, updated TEXT);
CREATE TABLE IF NOT EXISTS scan (
    day TEXT, symbol TEXT, rank INTEGER, kind TEXT, score REAL, metrics TEXT, headlines TEXT,
    profitable INTEGER, earnings_in_days INTEGER, decision TEXT, reason TEXT, source TEXT,
    picked INTEGER, ret_1d REAL, ret_5d REAL, PRIMARY KEY (day, symbol));
"""


class Journal:
    """SQLite record of everything the engine does. Also the engine's durable state."""

    def __init__(self, path: str = "openbell.db") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- account / positions (state that must survive a restart) ---
    def get_account(self, key: str, default: float) -> float:
        r = self.db.execute("SELECT value FROM account WHERE key=?", (key,)).fetchone()
        return float(r["value"]) if r else default

    def set_account(self, key: str, value: float) -> None:
        self.db.execute("INSERT INTO account(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
        self.db.commit()

    def save_position(self, p: Position) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO positions VALUES(?,?,?,?,?,?,?,?,?,?)",
            (p.symbol, p.strategy, p.qty, p.entry_px, p.entry_ts.isoformat(), p.stop,
             p.initial_stop, p.target, p.high_close, p.exit_flag))
        self.db.commit()

    def delete_position(self, symbol: str) -> None:
        self.db.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        self.db.commit()

    def load_positions(self) -> dict:
        out = {}
        for r in self.db.execute("SELECT * FROM positions"):
            out[r["symbol"]] = Position(
                symbol=r["symbol"], strategy=r["strategy"], qty=r["qty"], entry_px=r["entry_px"],
                entry_ts=datetime.fromisoformat(r["entry_ts"]), stop=r["stop"],
                initial_stop=r["initial_stop"], target=r["target"],
                high_close=r["high_close"], exit_flag=r["exit_flag"])
        return out

    # --- history ---
    def add_trade(self, **kw) -> None:
        cols = ",".join(kw)
        self.db.execute(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(kw))})",
                        tuple(kw.values()))
        self.db.commit()

    def event(self, ts: datetime, kind: str, symbol: str, message: str) -> None:
        self.db.execute("INSERT INTO events(ts,kind,symbol,message) VALUES(?,?,?,?)",
                        (ts.isoformat(), kind, symbol, message))
        self.db.commit()

    def add_equity(self, ts: datetime, equity: float, cash: float, benchmark: Optional[float]) -> None:
        self.db.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?)",
                        (ts.isoformat(), equity, cash, benchmark))
        self.db.commit()

    def upsert_day(self, day: str, **kw) -> None:
        self.db.execute("INSERT OR IGNORE INTO days(day) VALUES(?)", (day,))
        for k, v in kw.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            self.db.execute(f"UPDATE days SET {k}=? WHERE day=?", (v, day))
        self.db.commit()

    # --- scanner ---
    def save_universe(self, rows: list, updated: str) -> None:
        self.db.execute("DELETE FROM universe")
        self.db.executemany("INSERT INTO universe VALUES(?,?,?,?)",
                            [(s, p, dv, updated) for s, p, dv in rows])
        self.db.commit()

    def load_universe(self) -> tuple:
        rows = self.rows("SELECT symbol, updated FROM universe ORDER BY dollar_volume DESC")
        return [r["symbol"] for r in rows], (rows[0]["updated"] if rows else None)

    def save_scan(self, day: str, candidates: list, source: str) -> None:
        self.db.execute("DELETE FROM scan WHERE day=?", (day,))
        self.db.executemany(
            "INSERT INTO scan(day,symbol,rank,kind,score,metrics,headlines,profitable,"
            "earnings_in_days,decision,reason,source,picked) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(day, c.symbol, c.rank, c.kind, c.score, json.dumps(c.metrics), json.dumps(c.headlines),
              None if c.profitable is None else int(c.profitable), c.earnings_in_days,
              c.decision, c.reason, source, int(c.picked)) for c in candidates])
        self.db.commit()

    def update_scan_outcomes(self, daily: dict) -> None:
        """Fill in what each past candidate did after it was scanned (close-to-close)."""
        for r in self.rows("SELECT day, symbol FROM scan WHERE ret_5d IS NULL"):
            bars = daily.get(r["symbol"]) or []
            idx = next((i for i, b in enumerate(bars) if b.ts.date().isoformat() == r["day"]), None)
            if not idx:
                continue
            base = bars[idx - 1].c
            r1 = (bars[idx].c / base - 1) * 100
            r5 = (bars[idx + 4].c / base - 1) * 100 if idx + 4 < len(bars) else None
            self.db.execute("UPDATE scan SET ret_1d=?, ret_5d=? WHERE day=? AND symbol=?",
                            (r1, r5, r["day"], r["symbol"]))
        self.db.commit()

    def scan_rows(self, day: Optional[str] = None) -> list:
        if day is None:
            last = self.rows("SELECT MAX(day) d FROM scan")[0]["d"]
            if not last:
                return []
            day = last
        out = self.rows("SELECT * FROM scan WHERE day=? ORDER BY rank", (day,))
        for r in out:
            r["metrics"] = json.loads(r["metrics"] or "{}")
            r["headlines"] = json.loads(r["headlines"] or "[]")
        return out

    def scan_stats(self) -> dict:
        """Average next-day and 5-day move of approved vs rejected candidates."""
        out = {}
        for d in ("approve", "reject"):
            r = self.rows("SELECT COUNT(ret_1d) n, AVG(ret_1d) r1, AVG(ret_5d) r5 FROM scan "
                          "WHERE decision=? AND ret_1d IS NOT NULL", (d,))[0]
            out[d] = {"n": r["n"], "avg_1d": None if r["r1"] is None else round(r["r1"], 2),
                      "avg_5d": None if r["r5"] is None else round(r["r5"], 2)}
        return out

    # --- queries ---
    def rows(self, sql: str, args: tuple = ()) -> list:
        return [dict(r) for r in self.db.execute(sql, args)]

    def trades(self, day: Optional[str] = None, limit: int = 200) -> list:
        if day:
            return self.rows("SELECT * FROM trades WHERE day=? ORDER BY id", (day,))
        return self.rows("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))

    def stats(self) -> dict:
        t = self.rows("SELECT strategy, pnl, r_multiple FROM trades")
        out = {}
        for name in sorted({x["strategy"] for x in t} | {"ALL"}):
            sel = [x for x in t if name == "ALL" or x["strategy"] == name]
            if not sel:
                out[name] = {"trades": 0}
                continue
            wins = [x for x in sel if x["pnl"] > 0]
            gross_w = sum(x["pnl"] for x in wins)
            gross_l = -sum(x["pnl"] for x in sel if x["pnl"] <= 0)
            out[name] = {
                "trades": len(sel),
                "win_rate": round(100 * len(wins) / len(sel), 1),
                "avg_r": round(sum(x["r_multiple"] for x in sel) / len(sel), 2),
                "pnl": round(sum(x["pnl"] for x in sel), 2),
                "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None,
            }
        return out
