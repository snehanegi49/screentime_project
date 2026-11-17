"""
Real usage data: storage, upload parsing and feature building.

Live data comes from (1) the Chrome extension via collector.py, or
(2) a CSV uploaded in the dashboard. Both end up in the live_sessions table
and are converted into the same feature columns the ML models were trained on.
"""
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "data/app.db"
VALID = ["reels", "social", "messaging", "productive", "other"]
SCROLL = ["reels", "social"]


def _conn(path=DB_PATH):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS live_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, duration_min REAL,
        app_category TEXT, app TEXT)""")
    return c


def save_session(start_ms: float, duration_min: float, category: str, app: str = "", path=DB_PATH):
    category = category if category in VALID else "other"
    start = datetime.fromtimestamp(start_ms / 1000).isoformat(timespec="seconds")
    with _conn(path) as c:
        c.execute("INSERT INTO live_sessions (start_time, duration_min, app_category, app) VALUES (?,?,?,?)",
                  (start, round(float(duration_min), 2), category, app))


def save_dataframe(df: pd.DataFrame, path=DB_PATH):
    with _conn(path) as c:
        df[["start_time", "duration_min", "app_category", "app"]].astype(
            {"start_time": str}).to_sql("live_sessions", c, if_exists="append", index=False)


def clear(path=DB_PATH):
    with _conn(path) as c:
        c.execute("DELETE FROM live_sessions")


"""
Real usage data: storage, upload parsing and feature building.

Live data comes from (1) the Chrome extension via collector.py, or
(2) a CSV uploaded in the dashboard. Both end up in the live_sessions table
and are converted into the same feature columns the ML models were trained on.
"""
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "data/app.db"
VALID = ["reels", "social", "messaging", "productive", "other"]
SCROLL = ["reels", "social"]


def _conn(path=DB_PATH):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS live_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, duration_min REAL,
        app_category TEXT, app TEXT)""")
    return c


def save_session(start_ms: float, duration_min: float, category: str, app: str = "", path=DB_PATH):
    category = category if category in VALID else "other"
    start = datetime.fromtimestamp(start_ms / 1000).isoformat(timespec="seconds")
    with _conn(path) as c:
        c.execute("INSERT INTO live_sessions (start_time, duration_min, app_category, app) VALUES (?,?,?,?)",
                  (start, round(float(duration_min), 2), category, app))


def save_dataframe(df: pd.DataFrame, path=DB_PATH):
    with _conn(path) as c:
        out = df[["start_time", "duration_min", "app_category", "app"]].copy()
        out["start_time"] = pd.to_datetime(out["start_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        out.to_sql("live_sessions", c, if_exists="append", index=False)


def clear(path=DB_PATH):
    with _conn(path) as c:
        c.execute("DELETE FROM live_sessions")


def load_live(path=DB_PATH) -> pd.DataFrame:
    with _conn(path) as c:
        df = pd.read_sql_query("SELECT start_time, duration_min, app_category, app FROM live_sessions", c)
    df["start_time"] = pd.to_datetime([datetime.fromisoformat(str(x)) for x in df["start_time"]])
    return df


def guess_category(app: str) -> str:
    a = str(app).lower()
    if any(k in a for k in ["reel", "short", "tiktok"]): return "reels"
    if any(k in a for k in ["instagram", "facebook", "snapchat", "twitter", "x.com", "reddit", "linkedin"]): return "social"
    if any(k in a for k in ["whatsapp", "telegram", "message", "sms", "signal"]): return "messaging"
    if any(k in a for k in ["duolingo", "coursera", "kindle", "notion", "classroom", "udemy", "leetcode"]): return "productive"
    return "other"


def parse_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Accepts CSV with start_time, duration_min and either app_category or app."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    missing = {"start_time", "duration_min"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}. Need start_time, duration_min and app or app_category.")
    out = pd.DataFrame({"start_time": pd.to_datetime(df["start_time"], errors="coerce"),
                        "duration_min": pd.to_numeric(df["duration_min"], errors="coerce")})
    app = df["app"] if "app" in df.columns else pd.Series([""] * len(df))
    out["app"] = app.fillna("").astype(str)
    if "app_category" in df.columns:
        cat = df["app_category"].astype(str).str.lower()
        out["app_category"] = [c if c in VALID else guess_category(a) for c, a in zip(cat, out["app"])]
    else:
        out["app_category"] = out["app"].map(guess_category)
    return out.dropna(subset=["start_time", "duration_min"]).query("duration_min > 0")


def enrich(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the same context features the models were trained on."""
    d = raw.sort_values("start_time").reset_index(drop=True).copy()
    d["date"] = d["start_time"].dt.date
    d["start_hour"] = d["start_time"].dt.hour
    d["day_of_week"] = d["start_time"].dt.dayofweek
    prev_end = (d["start_time"] + pd.to_timedelta(d["duration_min"], unit="m")).shift(1)
    gap = ((d["start_time"] - prev_end).dt.total_seconds() / 60).clip(lower=1)
    first = d.groupby("date").cumcount() == 0
    d["gap_since_last_min"] = gap.where(~first, 600.0).fillna(600.0).round(1)
    d["sessions_so_far_today"] = d.groupby("date").cumcount()
    scroll = d["duration_min"].where(d["app_category"].isin(SCROLL), 0.0)
    d["reels_min_so_far_today"] = (scroll.groupby(d["date"]).cumsum() - scroll).round(1)
    d["prev_session_min"] = d.groupby("date")["duration_min"].shift(1).fillna(0.0)
    d["user_id"] = 0
    d["high_risk"] = ((d["duration_min"] >= 30) & d["app_category"].isin(SCROLL)).astype(int)
    return d


def current_context(path=DB_PATH) -> dict:
    """State right now, from today's live sessions (used for the risk check and nudges)."""
    now = datetime.now()
    raw = load_live(path)
    base = dict(sessions_so_far_today=0, reels_min_so_far_today=0.0, prev_session_min=0.0, gap_since_last_min=600.0)
    if raw.empty:
        return base
    today = raw[raw["start_time"].dt.date == now.date()]
    if today.empty:
        return base
    last = today.sort_values("start_time").iloc[-1]
    end = last["start_time"] + pd.Timedelta(minutes=float(last["duration_min"]))
    return dict(
        sessions_so_far_today=int(len(today)),
        reels_min_so_far_today=float(today[today.app_category.isin(SCROLL)].duration_min.sum()),
        prev_session_min=float(last["duration_min"]),
        gap_since_last_min=float(max((now - end).total_seconds() / 60, 1.0)),
    )