"""
Local collector: receives usage from the Chrome extension and answers nudge requests.

Run:  py collector.py        (keep this window open while browsing)
Listens on http://localhost:8765 - data never leaves your laptop.
"""
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import joblib  # noqa: E402

import db  # noqa: E402
import live  # noqa: E402
from bootstrap import ensure_artifacts  # noqa: E402
from models import predict_risk  # noqa: E402
from recommender import recommend  # noqa: E402

PORT = 8765
ensure_artifacts()
CLF = joblib.load("models/classifier.joblib")
LIVE_USER = 0


def nudge(category: str) -> dict:
    now = datetime.now()
    ctx = live.current_context()
    risk = predict_risk(CLF, dict(day_of_week=now.weekday(), start_hour=now.hour, app_category=category, **ctx))
    energy = "low" if (now.hour >= 22 or now.hour < 6) else "medium"
    picks = recommend(now.hour, energy, 20, weights=db.activity_weights(LIVE_USER), top_n=1)
    return {"risk": round(risk, 3), "suggestion": picks[0] if picks else None, "reels_today": ctx["reels_min_so_far_today"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code=200, body=None):
        data = json.dumps(body or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/ping":
            return self._send(200, {"ok": True})
        if u.path == "/nudge":
            cat = parse_qs(u.query).get("category", ["reels"])[0]
            return self._send(200, nudge(cat))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/session":
                live.save_session(body["start_ms"], body["duration_min"], body.get("category", "other"), body.get("app", ""))
                print(f"  + {body.get('category'):10s} {body['duration_min']:.1f} min  ({body.get('app', '')})")
                return self._send(200, {"saved": True})
            if self.path == "/feedback":
                db.log_feedback(LIVE_USER, float(body.get("risk", 0)), body["activity_id"], bool(body["accepted"]))
                return self._send(200, {"saved": True})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):  # keep console quiet
        pass


if __name__ == "__main__":
    print(f"Collector running on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
