// Service worker: talks to the local collector (content scripts cannot, due to page CORS).
const API = "http://localhost:8765";

async function call(path, options) {
  try {
    const res = await fetch(API + path, options);
    return await res.json();
  } catch (e) {
    return { error: "collector not running" };
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "session") {
    call("/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(msg.data) })
      .then(sendResponse);
  } else if (msg.type === "nudge") {
    call("/nudge?category=" + encodeURIComponent(msg.category)).then(sendResponse);
  } else if (msg.type === "feedback") {
    call("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(msg.data) })
      .then(sendResponse);
  }
  return true; // keep the channel open for the async response
});
