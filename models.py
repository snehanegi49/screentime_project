"""
Step 2-4 - Feature engineering, persona clustering and risk classification.

Run:  python models.py
Creates: models/*.joblib and models/metrics.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORIES = ["reels", "social", "messaging", "productive", "other"]
SESSION_FEATURES = [
    "start_hour", "is_weekend", "is_night", "gap_since_last_min", "sessions_so_far_today",
    "reels_min_so_far_today", "prev_session_min",
] + [f"cat_{c}" for c in CATEGORIES]

USER_FEATURES = ["avg_daily_min", "reels_share", "night_share", "weekend_ratio", "avg_session_min", "sessions_per_day"]


# ---------------------------------------------------------------- features
def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_night"] = ((out["start_hour"] >= 22) | (out["start_hour"] < 4)).astype(int)
    for c in CATEGORIES:
        out[f"cat_{c}"] = (out["app_category"] == c).astype(int)
    return out


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_session_features(df)
    d["scroll_min"] = np.where(d["app_category"].isin(["reels", "social"]), d["duration_min"], 0)
    d["weekend_min"] = d["duration_min"] * d["is_weekend"]
    d["night_min"] = d["duration_min"] * d["is_night"]
    g = d.groupby("user_id")
    n_days = g["date"].nunique()
    total = g["duration_min"].sum()
    feats = pd.DataFrame({
        "avg_daily_min": total / n_days,
        "reels_share": g["scroll_min"].sum() / total,
        "night_share": g["night_min"].sum() / total,
        "weekend_ratio": (g["weekend_min"].sum() / total),
        "avg_session_min": g["duration_min"].mean(),
        "sessions_per_day": g.size() / n_days,
    })
    return feats[USER_FEATURES]


# ---------------------------------------------------------------- clustering
def name_clusters(centers: pd.DataFrame) -> dict:
    """Give each cluster a human-friendly persona name from its dominant trait."""
    names, used = {}, set()
    rules = [
        ("Frequent Short-Check Scroller", "sessions_per_day"),
        ("Late-Night Scroller", "night_share"),
        ("Weekend Binge-Watcher", "weekend_ratio"),
    ]
    z = (centers - centers.mean()) / (centers.std(ddof=0) + 1e-9)
    for persona, col in rules:
        best = z[col].drop(index=list(used)).idxmax()
        names[best] = persona
        used.add(best)
    for c in centers.index:
        if c not in names:
            names[c] = "Balanced User"
    return names


def train_clustering(df: pd.DataFrame, k: int = 4):
    uf = build_user_features(df)
    scaler = StandardScaler().fit(uf)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaler.transform(uf))
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=USER_FEATURES)
    names = name_clusters(centers)
    uf = uf.assign(cluster=km.labels_)
    uf["persona"] = uf["cluster"].map(names)
    return {"scaler": scaler, "kmeans": km, "names": names}, uf, centers.assign(persona=pd.Series(names))


# ---------------------------------------------------------------- classification
def train_classifiers(df: pd.DataFrame):
    d = add_session_features(df)
    X, y = d[SESSION_FEATURES], d["high_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    }
    scaler = StandardScaler().fit(X_tr)
    results, fitted = {}, {}
    for name, model in candidates.items():
        Xa, Xb = (scaler.transform(X_tr), scaler.transform(X_te)) if name == "Logistic Regression" else (X_tr, X_te)
        model.fit(Xa, y_tr)
        prob = model.predict_proba(Xb)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results[name] = {
            "accuracy": float((pred == y_te).mean()),
            "precision": float(precision_score(y_te, pred)),
            "recall": float(recall_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, prob)),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        fitted[name] = model
    best = max(results, key=lambda n: results[n]["roc_auc"])
    importances = None
    if best == "Random Forest":
        importances = dict(zip(SESSION_FEATURES, map(float, fitted[best].feature_importances_)))
    elif best == "Logistic Regression":
        importances = dict(zip(SESSION_FEATURES, map(float, np.abs(fitted[best].coef_[0]))))
    return {"best_name": best, "model": fitted[best], "scaler": scaler,
            "uses_scaler": best == "Logistic Regression"}, results, importances


def predict_risk(bundle: dict, row: dict) -> float:
    """Probability that a session starting now becomes a long (30+ min) scroll."""
    r = {"day_of_week": row["day_of_week"], "start_hour": row["start_hour"], "app_category": row["app_category"],
         "gap_since_last_min": row["gap_since_last_min"], "sessions_so_far_today": row["sessions_so_far_today"],
         "reels_min_so_far_today": row["reels_min_so_far_today"], "prev_session_min": row["prev_session_min"]}
    X = add_session_features(pd.DataFrame([r]))[SESSION_FEATURES]
    if bundle["uses_scaler"]:
        X = bundle["scaler"].transform(X)
    return float(bundle["model"].predict_proba(X)[0, 1])


def main():
    df = pd.read_csv("data/sessions.csv", parse_dates=["start_time"])
    os.makedirs("models", exist_ok=True)

    cluster_bundle, user_table, centers = train_clustering(df)
    clf_bundle, results, importances = train_classifiers(df)

    # how well did unsupervised clusters recover the hidden archetypes?
    truth = df.groupby("user_id")["true_archetype"].first()
    check = pd.crosstab(user_table["persona"], truth)

    joblib.dump(cluster_bundle, "models/clustering.joblib")
    joblib.dump(clf_bundle, "models/classifier.joblib")
    user_table.to_csv("data/user_personas.csv")
    centers.to_csv("data/persona_centers.csv", index=False)
    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": clf_bundle["best_name"], "results": results, "feature_importance": importances}, f, indent=2)

    print("=== Persona clustering (K-Means, k=4) ===")
    print(user_table["persona"].value_counts().to_string(), "\n")
    print("Personas vs hidden archetypes (sanity check):\n", check, "\n")
    print("=== Risk classification (test set) ===")
    for n, r in results.items():
        print(f"{n:20s} acc={r['accuracy']:.3f} prec={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")
    print("Best model:", clf_bundle["best_name"])


if __name__ == "__main__":
    main()
