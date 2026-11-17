"""
ML-Based Screen Time and Activity Recommender - Streamlit dashboard.
Run:  streamlit run app.py
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import db
from models import predict_risk
from recommender import recommend, time_of_day

st.set_page_config(page_title="Screen Time & Activity Recommender", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------- bootstrap
def ensure_artifacts():
    """Generate data and train models on first run so the app works out of the box."""
    if not os.path.exists("data/sessions.csv"):
        from data_generator import generate_sessions
        os.makedirs("data", exist_ok=True)
        generate_sessions().to_csv("data/sessions.csv", index=False)
    if not os.path.exists("models/classifier.joblib"):
        import models
        models.main()


@st.cache_resource
def load_all():
    ensure_artifacts()
    sessions = pd.read_csv("data/sessions.csv", parse_dates=["start_time"])
    personas = pd.read_csv("data/user_personas.csv", index_col=0)
    centers = pd.read_csv("data/persona_centers.csv")
    clf = joblib.load("models/classifier.joblib")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return sessions, personas, centers, clf, metrics


sessions, personas, centers, clf, metrics = load_all()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🧭 Screen Time Recommender")
user_id = st.sidebar.selectbox("Select user", sorted(personas.index.tolist()))
persona = personas.loc[user_id, "persona"]
st.sidebar.markdown(f"**Persona:** {persona}")
stats = db.stats(user_id)
st.sidebar.metric("🔥 Focus streak (days)", stats["streak"])
st.sidebar.metric("✅ Productive swaps", stats["swaps"])
st.sidebar.caption("Demo uses synthetic usage data (real logs are privacy-restricted).")

user_df = sessions[sessions.user_id == user_id]
tab_dash, tab_risk, tab_rec, tab_progress, tab_model = st.tabs(
    ["📊 Dashboard", "⚠️ Risk Check", "💡 Activity Suggestions", "🏆 Progress", "🔬 Model Report"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    st.header(f"User {user_id} — {persona}")
    daily = user_df.groupby("date")["duration_min"].sum()
    scroll = user_df[user_df.app_category.isin(["reels", "social"])]["duration_min"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg daily screen time", f"{daily.mean():.0f} min")
    c2.metric("Reels/social share", f"{scroll / user_df.duration_min.sum():.0%}")
    c3.metric("High-risk sessions", f"{user_df.high_risk.mean():.0%}")
    focus = max(0, 100 - int(100 * scroll / user_df.duration_min.sum()) + min(stats["swaps"] * 2, 20))
    c4.metric("Focus score", f"{min(focus, 100)}/100")

    left, right = st.columns(2)
    with left:
        st.subheader("Daily screen time (min)")
        st.line_chart(daily)
    with right:
        st.subheader("When do you scroll? (minutes by hour)")
        st.bar_chart(user_df.groupby("start_hour")["duration_min"].sum().reindex(range(24), fill_value=0))
    st.subheader("Minutes by app category")
    st.bar_chart(user_df.groupby("app_category")["duration_min"].sum())

# ---------------------------------------------------------------- risk check
with tab_risk:
    st.header("Will this session turn into a long scroll?")
    st.caption("Enter the situation at the moment the app is opened. The model estimates the chance it becomes a 30+ minute session.")
    a, b, c = st.columns(3)
    hour = a.slider("Hour of day", 0, 23, 23)
    dow = b.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=5)
    cat = c.selectbox("App category", ["reels", "social", "messaging", "productive", "other"])
    d, e, f, g = st.columns(4)
    gap = d.number_input("Minutes since last session", 1, 600, 15)
    n_sess = e.number_input("Sessions so far today", 0, 30, 4)
    reel_min = f.number_input("Reels/social minutes today", 0, 600, 70)
    prev = g.number_input("Last session length (min)", 0, 180, 25)

    if st.button("Check risk", type="primary"):
        risk = predict_risk(clf, dict(day_of_week=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow),
                                      start_hour=hour, app_category=cat, gap_since_last_min=gap,
                                      sessions_so_far_today=n_sess, reels_min_so_far_today=reel_min,
                                      prev_session_min=prev))
        st.session_state["risk"] = risk
        st.session_state["hour"] = hour
    if "risk" in st.session_state:
        risk = st.session_state["risk"]
        level = "HIGH" if risk >= 0.5 else ("MEDIUM" if risk >= 0.25 else "LOW")
        st.progress(min(risk, 1.0), text=f"Binge risk: {risk:.0%} ({level})")
        if risk >= 0.5:
            st.error("High risk of a long scroll. Head to **Activity Suggestions** for a better use of this time.")
        elif risk >= 0.25:
            st.warning("Moderate risk. Consider setting a 10-minute limit.")
        else:
            st.success("Low risk. Enjoy, but stay mindful.")

# ---------------------------------------------------------------- recommendations
with tab_rec:
    st.header("Do something better with this time")
    r1, r2, r3 = st.columns(3)
    h = r1.slider("Current hour", 0, 23, st.session_state.get("hour", 22), key="rec_hour")
    energy = r2.select_slider("Your energy", ["low", "medium", "high"], value="medium")
    free = r3.slider("Minutes available", 5, 60, 20, step=5)
    mood = st.text_input("How are you feeling? (optional)", placeholder="e.g. stressed, bored, tired")
    st.caption(f"Time of day: **{time_of_day(h)}** · past feedback shapes the ranking.")

    if "suggestions" not in st.session_state or st.button("🔄 Get suggestions"):
        st.session_state["suggestions"] = recommend(h, energy, free, mood, weights=db.activity_weights(user_id))
        st.session_state["answered"] = set()

    for s in st.session_state.get("suggestions", []):
        with st.container(border=True):
            st.subheader(s["name"])
            st.caption(f"{s['minutes']} min · {s['energy']} energy · match {s['similarity']:.0%} · {', '.join(s['tags'])}")
            if s["id"] in st.session_state["answered"]:
                st.write("Thanks, feedback saved.")
                continue
            ok, no = st.columns(2)
            risk_now = st.session_state.get("risk", 0.0)
            if ok.button("👍 I'll do it", key=f"y_{s['id']}"):
                db.log_feedback(user_id, risk_now, s["id"], True)
                st.session_state["answered"].add(s["id"]); st.rerun()
            if no.button("👎 Not now", key=f"n_{s['id']}"):
                db.log_feedback(user_id, risk_now, s["id"], False)
                st.session_state["answered"].add(s["id"]); st.rerun()

# ---------------------------------------------------------------- progress
with tab_progress:
    st.header("Your progress")
    p1, p2, p3 = st.columns(3)
    p1.metric("Suggestions shown", stats["suggestions"])
    p2.metric("Productive swaps", stats["swaps"])
    p3.metric("Streak", f"{stats['streak']} days")
    hist = db.history(user_id)
    if hist:
        st.dataframe(pd.DataFrame(hist, columns=["Time", "Risk", "Activity", "Accepted"]))
    else:
        st.info("No activity yet. Accept a suggestion to start your streak.")

# ---------------------------------------------------------------- model report
with tab_model:
    st.header("Model report")
    st.subheader("Risk classifier comparison (20% hold-out test set)")
    res = pd.DataFrame(metrics["results"]).T.drop(columns="confusion_matrix")
    st.dataframe(res.style.format("{:.3f}"))
    st.success(f"Selected model: **{metrics['best_model']}** (highest ROC-AUC). Recall is prioritised because missing a risky session is the costly error.")
    if metrics.get("feature_importance"):
        imp = pd.Series(metrics["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4)); imp.plot.barh(ax=ax); ax.set_title("Feature importance"); st.pyplot(fig)
    st.subheader("Persona profiles (K-Means cluster centres)")
    st.dataframe(centers.set_index("persona").round(2))
    st.caption("Results are on synthetic data, so scores are indicative only (proof of concept).")
