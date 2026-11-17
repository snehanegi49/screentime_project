// Tracks active time per category on social sites and shows a gentle nudge banner.
const IDLE_MS = 60 * 1000;        // no mouse/scroll/key for 60s => session ends
const MIN_SESSION_SEC = 10;       // ignore very short blips
const LONG_SCROLL_MIN = 15;       // nudge after this many minutes in one reels/social session

function categoryFor() {
  const h = location.hostname, p = location.pathname;
  if (h.includes("instagram.com")) return p.startsWith("/reel") ? "reels" : "social";
  if (h.includes("youtube.com")) return p.startsWith("/shorts") ? "reels" : "other";
  if (h.includes("facebook.com")) return p.startsWith("/reel") || p.startsWith("/watch") ? "reels" : "social";
  if (h.includes("tiktok.com")) return "reels";
  return "social"; // x.com, twitter.com, snapchat.com
}

let session = null;               // { start, category, nudged }
let lastInput = Date.now();
["mousemove", "scroll", "wheel", "keydown", "touchstart", "click"].forEach((ev) =>
  window.addEventListener(ev, () => (lastInput = Date.now()), { passive: true, capture: true })
);

function isActive() {
  return document.visibilityState === "visible" && document.hasFocus() && Date.now() - lastInput < IDLE_MS;
}

function endSession() {
  if (!session) return;
  const seconds = (Date.now() - session.start) / 1000;
  if (seconds >= MIN_SESSION_SEC) {
    chrome.runtime.sendMessage({
      type: "session",
      data: { start_ms: session.start, duration_min: seconds / 60, category: session.category, app: location.hostname.replace("www.", "") },
    });
  }
  session = null;
}

function startSession() {
  session = { start: Date.now(), category: categoryFor(), nudged: false };
  if (session.category === "reels" || session.category === "social") {
    chrome.runtime.sendMessage({ type: "nudge", category: session.category }, (r) => {
      if (r && !r.error && r.risk >= 0.5 && r.suggestion) {
        showBanner(`High chance this turns into a long scroll (${Math.round(r.risk * 100)}%).`, r);
      }
    });
  }
}

function showBanner(text, info) {
  if (document.getElementById("stg-banner")) return;
  const s = info.suggestion;
  const box = document.createElement("div");
  box.id = "stg-banner";
  box.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:2147483647;max-width:320px;padding:16px;" +
    "background:#05386B;color:#fff;border-radius:12px;font:14px/1.4 system-ui,sans-serif;box-shadow:0 6px 24px rgba(0,0,0,.35)";
  box.innerHTML = `<div style="font-weight:600;margin-bottom:6px">🧭 Screen Time Guardian</div>
    <div style="margin-bottom:8px">${text}</div>
    <div style="margin-bottom:12px">Try instead: <b>${s.name}</b> (${s.minutes} min)</div>`;
  const mk = (label, accepted) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "margin-right:8px;padding:6px 12px;border:0;border-radius:8px;cursor:pointer;font-weight:600;" +
      (accepted ? "background:#02C39A;color:#05386B" : "background:#2d5a8c;color:#fff");
    b.onclick = () => {
      chrome.runtime.sendMessage({ type: "feedback", data: { activity_id: s.id, accepted, risk: info.risk } });
      box.remove();
    };
    return b;
  };
  box.append(mk("👍 I'll do it", true), mk("Not now", false));
  document.body.appendChild(box);
  setTimeout(() => box.remove(), 30000);
}

setInterval(() => {
  const active = isActive();
  if (active && !session) startSession();
  else if (!active && session) endSession();
  else if (active && session) {
    const cat = categoryFor();
    if (cat !== session.category) { endSession(); startSession(); return; }   // e.g. feed -> reels
    const minutes = (Date.now() - session.start) / 60000;
    if (!session.nudged && minutes >= LONG_SCROLL_MIN && (cat === "reels" || cat === "social")) {
      session.nudged = true;
      chrome.runtime.sendMessage({ type: "nudge", category: cat }, (r) => {
        if (r && !r.error && r.suggestion) showBanner(`You've been scrolling for ${Math.round(minutes)} minutes.`, r);
      });
    }
  }
}, 1000);

window.addEventListener("pagehide", endSession);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") endSession(); });
