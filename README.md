# ML-Based Screen Time and Activity Recommender

Detects risky (doomscrolling) sessions from usage behaviour and recommends a productive activity instead.

## Run
```
pip install -r requirements.txt
python data_generator.py     # 1. synthetic usage logs  -> data/sessions.csv
python models.py             # 2. clustering + risk classifier -> models/
streamlit run app.py         # 3. dashboard
```
(`app.py` also generates data and trains models automatically on first run.)

## Files
| File | Purpose |
|---|---|
| data_generator.py | Synthetic session logs for 60 users x 30 days, built from 4 hidden archetypes |
| models.py | Feature engineering, K-Means personas, Logistic Regression vs Random Forest risk model |
| recommender.py | Content-based recommender (cosine similarity) + mood keywords |
| db.py | SQLite: feedback, feedback-weighted scores, streaks |
| app.py | Streamlit dashboard (5 tabs) |

## Pipeline
1. **Data generation** - session start hour, day, app category, gap since last session, reels minutes so far today, duration.
2. **Feature engineering** - weekend/night flags, one-hot app category; per-user features (daily minutes, reels share, night share, weekend ratio, session length, sessions/day).
3. **Persona clustering** - K-Means (k=4): Late-Night Scroller, Weekend Binge-Watcher, Frequent Short-Check Scroller, Balanced User.
4. **Risk classification** - target = session of 30+ min on reels/social. Only features known at session *start* are used (no leakage). Class-weighted LR vs RF, compared on precision, recall, F1, ROC-AUC.
5. **Recommendation** - context vector (time of day, energy, minutes free, mood) vs activity vectors; score = cosine similarity x feedback weight.
6. **Feedback loop** - accept +0.15, skip -0.10 per activity (clipped 0.5-1.8). Streak = consecutive days with a accepted suggestion.

## Honest notes (for viva)
- Data is **synthetic**; results (AUC ~0.98, perfect cluster recovery) are optimistic and indicative only. Real validation is future scope.
- The feedback loop is weighted scoring, not full reinforcement learning.
