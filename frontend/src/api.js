// Backend base URL — override with VITE_API in an .env file if the API runs elsewhere.
const BASE = import.meta.env.VITE_API || "http://127.0.0.1:8000";

let _healthCache = null;
export async function health() {
  if (_healthCache) return _healthCache;
  try {
    const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2500) });
    _healthCache = await r.json();
  } catch {
    _healthCache = { ok: false, offline: true };
  }
  return _healthCache;
}

export async function interpret(q) {
  const r = await fetch(`${BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
    signal: AbortSignal.timeout(20000),
  });
  return r.json();
}

export async function computeChange(site, modeId) {
  const bbox = bboxOf(site);
  const d = site.dates || {};
  const r = await fetch(`${BASE}/api/change`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lat: site.lat,
      lng: site.lng,
      bbox,
      mode: modeId,
      index: site.index,
      direction: site.direction,
      date_before_from: d.b0,
      date_before: d.b1,
      date_after_from: d.a0,
      date_after: d.a1,
    }),
    signal: AbortSignal.timeout(120000),
  });
  return r.json();
}

// VLM chatbot: send 1-2 images + a query to the fine-tuned Qwen3-VL + RAG pipeline.
// The first call loads the model, so allow a long timeout.
export async function vlmQuery({ image1, image2, query, mod1 = "auto", mod2 = "auto" }) {
  const fd = new FormData();
  fd.append("image1", image1);
  if (image2) fd.append("image2", image2);
  fd.append("query", query || "");
  fd.append("mod1", mod1);
  fd.append("mod2", mod2);
  const r = await fetch(`${BASE}/api/vlm`, {
    method: "POST",
    body: fd,
    signal: AbortSignal.timeout(300000),
  });
  if (!r.ok) throw new Error(`VLM request failed (${r.status})`);
  return r.json();
}

export function bboxOf(site) {
  try {
    const ring = site.geo.coordinates[0];
    const xs = ring.map((p) => p[0]);
    const ys = ring.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  } catch {
    const d = 0.06;
    return [site.lng - d, site.lat - d, site.lng + d, site.lat + d];
  }
}
