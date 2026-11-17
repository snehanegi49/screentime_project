"""
Step 1 - Synthetic data generation.

Real screen-time logs are privacy-restricted, so we simulate them.
Each simulated user belongs to a hidden "archetype" (late-night scroller,
weekend binger, etc.). Sessions are generated from that archetype, with
noise, so that the ML models have real patterns to discover. The archetype
is NOT given to the models - clustering has to rediscover it.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42

# hour weights, session-length (minutes) and reel preference per archetype
ARCHETYPES = {
    "late_night":   {"hours": [22, 23, 0, 1, 2], "reel_p": 0.75, "len_mu": 3.1, "sessions": (3, 6), "weekend_boost": 1.1},
    "weekend":      {"hours": [11, 14, 16, 20, 21], "reel_p": 0.65, "len_mu": 3.0, "sessions": (2, 4), "weekend_boost": 2.0},
    "frequent":     {"hours": [8, 10, 12, 15, 17, 19, 21], "reel_p": 0.50, "len_mu": 1.7, "sessions": (8, 15), "weekend_boost": 1.0},
    "balanced":     {"hours": [9, 13, 18, 20], "reel_p": 0.25, "len_mu": 2.0, "sessions": (2, 5), "weekend_boost": 1.0},
}
CATEGORIES = ["reels", "social", "messaging", "productive", "other"]


def generate_sessions(n_users: int = 60, n_days: int = 30, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES)
    rows = []
    start_date = pd.Timestamp("2026-08-01")

    for uid in range(1, n_users + 1):
        arch = names[(uid - 1) % len(names)]
        cfg = ARCHETYPES[arch]
        for d in range(n_days):
            date = start_date + pd.Timedelta(days=d)
            weekend = date.dayofweek >= 5
            lo, hi = cfg["sessions"]
            n_sessions = int(rng.integers(lo, hi + 1) * (cfg["weekend_boost"] if weekend else 1.0))
            hours = np.sort(rng.choice(cfg["hours"], size=n_sessions, replace=True))
            last_end, reels_so_far, prev_dur = None, 0.0, 0.0

            for i, h in enumerate(hours):
                minute = int(rng.integers(0, 60))
                start = date + pd.Timedelta(hours=int(h), minutes=minute)
                if h < 5:                       # after-midnight sessions belong to same "night"
                    start += pd.Timedelta(days=1)
                p_reel = cfg["reel_p"]
                cat = rng.choice(
                    CATEGORIES,
                    p=_cat_probs(p_reel),
                )
                gap = 600.0 if last_end is None else max((start - last_end).total_seconds() / 60, 1.0)

                # duration: depends on category, time of day, fatigue (cumulative reels) and short gaps
                base = np.exp(rng.normal(cfg["len_mu"], 0.5))
                if cat in ("reels", "social"):
                    base *= 1.5
                if cat == "productive":
                    base *= 1.2
                if h >= 22 or h < 3:
                    base *= 1.5                 # late night -> longer scrolls
                if reels_so_far > 60:
                    base *= 1.3                 # already scrolling a lot today
                if gap < 30:
                    base *= 1.25                # re-opened quickly -> habit loop
                duration = float(np.clip(base, 1, 180))

                rows.append(
                    dict(
                        user_id=uid, date=date.date(), start_time=start, start_hour=int(h),
                        day_of_week=int(date.dayofweek), app_category=cat,
                        gap_since_last_min=round(gap, 1), sessions_so_far_today=i,
                        reels_min_so_far_today=round(reels_so_far, 1),
                        prev_session_min=round(prev_dur, 1), duration_min=round(duration, 1),
                        true_archetype=arch,
                    )
                )
                last_end = start + pd.Timedelta(minutes=duration)
                prev_dur = duration
                if cat in ("reels", "social"):
                    reels_so_far += duration
    df = pd.DataFrame(rows)
    # binge label: a long scroll on a reels/social app
    df["high_risk"] = ((df["duration_min"] >= 30) & df["app_category"].isin(["reels", "social"])).astype(int)
    return df


def _cat_probs(p_reel):
    rest = 1 - p_reel
    p = np.array([p_reel * 0.7, p_reel * 0.3 + 0.0, rest * 0.4, rest * 0.3, rest * 0.3])
    return p / p.sum()


if __name__ == "__main__":
    df = generate_sessions()
    import os; os.makedirs("data", exist_ok=True)
    df.to_csv("data/sessions.csv", index=False)
    print(f"Saved {len(df)} sessions for {df.user_id.nunique()} users -> data/sessions.csv")
    print(f"High-risk share: {df.high_risk.mean():.1%}")
