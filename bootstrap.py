"""Generate demo data and train models if they do not exist yet."""
import os


def ensure_artifacts():
    if not os.path.exists("data/sessions.csv"):
        from data_generator import generate_sessions
        os.makedirs("data", exist_ok=True)
        generate_sessions().to_csv("data/sessions.csv", index=False)
    if not os.path.exists("models/classifier.joblib"):
        import models
        models.main()
