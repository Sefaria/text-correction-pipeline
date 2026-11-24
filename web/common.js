// Shared helpers for the dashboard and the review page.

export async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...opts });
  if (res.status === 401) {
    location.href = "/login";
    throw new Error("sign in required");
  }
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("json") ? res.json() : res.text();
}

export const postJSON = (path, body, method = "POST") =>
  api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Text a reader sees: HTML tags removed. */
export function plain(s) {
  return String(s ?? "").replace(/&lt;.*?&gt;|<[^>]*>|&nbsp;/g, " ").replace(/\s+/g, " ").trim();
}

let toastTimer;
export function toast(msg, ms = 2600) {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    el.setAttribute("role", "status");
    document.body.append(el);
  }
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

/** Still being processed (queued or running)? */
export const isActive = (p) => p?.status === "running" || p?.status === "queued";

export function fmtEta(s) {
  if (s == null) return "";
  if (s < 60) return "<1 min left";
  if (s < 3600) return `~${Math.round(s / 60)} min left`;
  return `~${(s / 3600).toFixed(1)} h left`;
}

const STAGES = { queued: "Queued", fetch: "Fetching text", detect: "Detecting", align: "Comparing with reference", llm: "LLM check",
  ready: "Ready" };
const UNITS = { detect: "windows", align: "candidates", llm: "candidates" };

/** Progress bar + label for a project's progress object. */
export function progressHTML(p) {
  if (p.status === "ready") return `<span class="pill p-ok">Ready</span>`;
  if (p.status === "failed") return `<span class="pill p-bad" title="${esc(p.error || "")}">Failed</span>`;
  if (p.status === "cancelled") return `<span class="pill p-muted">Cancelled</span>`;
  if (p.status === "queued") {
    return `<span class="pill p-info">Queued${p.queue_position ? ` · #${p.queue_position}` : ""}</span>`;
  }
  const pct = p.total ? Math.min(100, (100 * p.done) / p.total) : 0;
  const count = p.total ? ` ${p.done.toLocaleString()} / ${p.total.toLocaleString()} ${UNITS[p.stage] || ""}` : "";
  const eta = fmtEta(p.eta_seconds);
  return `<div class="small">${esc(STAGES[p.stage] || p.stage)}${count}${eta ? " · " + eta : ""}</div>
    <div class="bar" role="progressbar" aria-valuenow="${Math.round(pct)}" aria-valuemin="0" aria-valuemax="100"><div style="width:${pct}%"></div></div>`;
}

/** Word-level diff of two strings (tags stripped for display). Returns HTML with <del>/<ins>. */
export function diffHTML(a, b) {
  const A = plain(a).split(" ").filter(Boolean), B = plain(b).split(" ").filter(Boolean);
  // trim common prefix/suffix, then LCS on the middle (segments are short; the middle is tiny)
  let s = 0;
  while (s < A.length && s < B.length && A[s] === B[s]) s++;
  let e = 0;
  while (e < A.length - s && e < B.length - s && A[A.length - 1 - e] === B[B.length - 1 - e]) e++;
  const a1 = A.slice(s, A.length - e), b1 = B.slice(s, B.length - e);
  const L = Array.from({ length: a1.length + 1 }, () => new Array(b1.length + 1).fill(0));
  for (let i = a1.length - 1; i >= 0; i--)
    for (let j = b1.length - 1; j >= 0; j--)
      L[i][j] = a1[i] === b1[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const mid = [];
  let i = 0, j = 0;
  while (i < a1.length || j < b1.length) {
    if (i < a1.length && j < b1.length && a1[i] === b1[j]) { mid.push(esc(a1[i])); i++; j++; }
    else if (j < b1.length && (i >= a1.length || L[i][j + 1] >= L[i + 1][j])) { mid.push(`<ins>${esc(b1[j])}</ins>`); j++; }
    else { mid.push(`<del>${esc(a1[i])}</del>`); i++; }
  }
  const ctx = 25;
  const pre = A.slice(Math.max(0, s - ctx), s).map(esc), post = A.slice(A.length - e, A.length - e + ctx).map(esc);
  return [(s > ctx ? "… " : "") + pre.join(" "), mid.join(" "), post.join(" ") + (e > ctx ? " …" : "")].filter(Boolean).join(" ");
}

export function sefariaLink(ref) {
  return `https://www.sefaria.org/${encodeURIComponent(ref.replace(/ /g, "_")).replace(/%3A/g, ":").replace(/%2C/g, ",")}?lang=he`;
}
