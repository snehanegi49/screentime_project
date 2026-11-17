"""
Step 5 - Content-based activity recommender with feedback-weighted scoring.

Each activity and the user's current context are turned into vectors over the
same features; cosine similarity says how well they match. Past feedback
(accepted / skipped) then scales each activity's score.
"""
import numpy as np

TIMES = ["morning", "afternoon", "evening", "night"]
ENERGY = ["low", "medium", "high"]
LENGTH = ["short", "medium", "long"]            # <=10, 11-30, 30+ minutes
TAGS = ["calm", "creative", "active", "learning", "social"]
FEATURES = TIMES + ENERGY + LENGTH + TAGS

# id, name, minutes, energy, good-times, tags
_ACTIVITIES = [
    ("read",      "Read 10 pages of a book",              15, "low",    ["afternoon", "evening", "night"], ["calm", "learning"]),
    ("journal",   "Write a 5-line journal entry",          8, "low",    ["evening", "night"],              ["calm", "creative"]),
    ("breathe",   "2-minute breathing exercise",           3, "low",    TIMES,                             ["calm"]),
    ("stretch",   "Quick stretching routine",              7, "medium", ["morning", "afternoon", "evening"], ["active", "calm"]),
    ("walk",      "Take a 15-minute walk outside",        15, "medium", ["morning", "afternoon", "evening"], ["active", "calm"]),
    ("workout",   "20-minute home workout",               20, "high",   ["morning", "afternoon", "evening"], ["active"]),
    ("code",      "Solve one coding problem",             30, "high",   ["morning", "afternoon", "evening"], ["learning"]),
    ("course",    "Watch one lesson of an online course", 25, "medium", ["morning", "afternoon", "evening"], ["learning"]),
    ("sketch",    "Sketch or doodle something",           15, "low",    ["afternoon", "evening", "night"], ["creative", "calm"]),
    ("music",     "Practice an instrument / listen to an album", 20, "medium", ["afternoon", "evening", "night"], ["creative", "calm"]),
    ("callfriend","Call a friend or family member",       15, "medium", ["afternoon", "evening"],          ["social"]),
    ("plan",      "Plan tomorrow's top 3 tasks",           8, "low",    ["evening", "night"],              ["calm", "learning"]),
    ("tidy",      "Tidy your desk / room for 10 minutes", 10, "medium", ["morning", "afternoon", "evening"], ["active", "calm"]),
    ("water",     "Drink water and rest your eyes",        3, "low",    TIMES,                             ["calm"]),
    ("cook",      "Prepare a healthy snack or meal",      25, "medium", ["morning", "afternoon", "evening"], ["creative", "active"]),
]
ACTIVITIES = {a[0]: dict(id=a[0], name=a[1], minutes=a[2], energy=a[3], times=a[4], tags=a[5]) for a in _ACTIVITIES}

# keyword -> (tags to boost, preferred energy or None)
MOOD_RULES = {
    "stress": (["calm"], "low"), "anx": (["calm"], "low"), "overwhelm": (["calm"], "low"),
    "tired": (["calm"], "low"), "sleep": (["calm"], "low"), "sad": (["social", "calm"], "low"),
    "lonely": (["social"], None), "bored": (["creative", "active", "learning"], "medium"),
    "restless": (["active"], "high"), "energetic": (["active", "learning"], "high"),
    "focus": (["learning"], "medium"), "creative": (["creative"], None),
}


def time_of_day(hour: int) -> str:
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 22: return "evening"
    return "night"


def _length_bucket(minutes: int) -> str:
    return "short" if minutes <= 10 else ("medium" if minutes <= 30 else "long")


def _vec(time_, energy, length, tags):
    v = np.zeros(len(FEATURES))
    for t in ([time_] if isinstance(time_, str) else time_): v[FEATURES.index(t)] = 1
    v[FEATURES.index(energy)] = 1
    v[FEATURES.index(length)] = 1
    for t in tags: v[FEATURES.index(t)] = 1
    return v


def parse_mood(text: str):
    tags, energy = set(), None
    for key, (t, e) in MOOD_RULES.items():
        if key in (text or "").lower():
            tags.update(t); energy = energy or e
    return sorted(tags), energy


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def recommend(hour: int, energy: str, minutes_free: int, mood_text: str = "",
              weights: dict | None = None, top_n: int = 3, exclude: set | None = None):
    """Return the top_n activities ranked by cosine similarity x feedback weight."""
    weights, exclude = weights or {}, exclude or set()
    mood_tags, mood_energy = parse_mood(mood_text)
    energy = mood_energy or energy
    ctx = _vec(time_of_day(hour), energy, _length_bucket(minutes_free), mood_tags)
    ranked = []
    for a in ACTIVITIES.values():
        if a["id"] in exclude or a["minutes"] > max(minutes_free, 3):
            continue
        av = _vec(a["times"], a["energy"], _length_bucket(a["minutes"]), a["tags"])
        score = _cos(ctx, av) * weights.get(a["id"], 1.0)
        ranked.append({**a, "similarity": round(_cos(ctx, av), 3), "score": round(score, 3)})
    return sorted(ranked, key=lambda r: r["score"], reverse=True)[:top_n]
