"""SQLite storage for feedback, interventions and streaks."""
import sqlite3
from datetime import date, datetime, timedelta

DB_PATH = "data/app.db"


def _conn(path=DB_PATH):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, ts TEXT,
        risk REAL, activity_id TEXT, accepted INTEGER)""")
    return c


def log_feedback(user_id: int, risk: float, activity_id: str, accepted: bool, path=DB_PATH):
    with _conn(path) as c:
        c.execute("INSERT INTO interventions (user_id, ts, risk, activity_id, accepted) VALUES (?,?,?,?,?)",
                  (user_id, datetime.now().isoformat(timespec="seconds"), risk, activity_id, int(accepted)))


def activity_weights(user_id: int, path=DB_PATH) -> dict:
    """Feedback-weighted scoring: +0.15 per accept, -0.10 per skip, clipped to [0.5, 1.8]."""
    w = {}
    with _conn(path) as c:
        for act, acc in c.execute("SELECT activity_id, accepted FROM interventions WHERE user_id=?", (user_id,)):
            w[act] = w.get(act, 1.0) + (0.15 if acc else -0.10)
    return {k: min(max(v, 0.5), 1.8) for k, v in w.items()}


def history(user_id: int, path=DB_PATH):
    with _conn(path) as c:
        return c.execute("SELECT ts, risk, activity_id, accepted FROM interventions WHERE user_id=? ORDER BY id DESC",
                         (user_id,)).fetchall()


def stats(user_id: int, path=DB_PATH) -> dict:
    """Swaps = accepted suggestions. Streak = consecutive days (up to today) with >=1 swap."""
    with _conn(path) as c:
        days = {r[0] for r in c.execute(
            "SELECT DISTINCT substr(ts,1,10) FROM interventions WHERE user_id=? AND accepted=1", (user_id,))}
        total = c.execute("SELECT COUNT(*) FROM interventions WHERE user_id=?", (user_id,)).fetchone()[0]
    swaps = len(history_accepted(user_id, path))
    streak, d = 0, date.today()
    if d.isoformat() not in days:
        d -= timedelta(days=1)
    while d.isoformat() in days:
        streak += 1; d -= timedelta(days=1)
    return {"swaps": swaps, "suggestions": total, "streak": streak}


def history_accepted(user_id: int, path=DB_PATH):
    with _conn(path) as c:
        return c.execute("SELECT 1 FROM interventions WHERE user_id=? AND accepted=1", (user_id,)).fetchall()
