// Review page: one candidate at a time, keyboard first. Every action is saved on the server immediately.
import { api, diffHTML, esc, plain, postJSON, progressHTML, sefariaLink, toast } from "./common.js";

const $ = (s) => document.querySelector(s);
const pid = +new URLSearchParams(location.search).get("id");
const FILTERS = [...$("#filter").options].map((o) => o.value);

let project, list = [], idx = 0, cur = null, editing = null, busy = false;

// ---- loading -------------------------------------------------------------------------------------------------------------------------
async function loadProject() {
  project = await api(`/api/projects/${pid}`);
  $("#pname").textContent = project.name;
  $("#pversion").textContent = `${project.title} · ${project.version_title}${project.ref_filename ? " · reference: " + project.ref_filename : ""}`;
  document.title = `${project.name} · Review`;
}

async function loadList(keepId) {
  list = await api(`/api/projects/${pid}/candidates?filter=${$("#filter").value}`);
  idx = keepId != null ? Math.max(0, list.findIndex((c) => c.id === keepId)) : 0;
  await show();
}

async function show() {
  cancelEdit();
  updateNav();
  if (!list.length) {
    cur = null;
    $("#cand").hidden = true;
    const p = project?.progress;
    const running = p && (p.status === "running" || p.status === "queued");
    $("#empty").hidden = false;
    $("#empty").textContent = running
      ? "Nothing here yet. Candidates appear as the detector works through the text; this list refreshes automatically."
      : $("#filter").value === "todo" ? "All candidates reviewed. Export the CSV when you're ready." : "No candidates in this view.";
    return;
  }
  $("#empty").hidden = true;
  cur = await api(`/api/projects/${pid}/candidates/${list[idx].id}`);
  render();
}

function updateNav() {
  $("#pos").textContent = list.length ? `${idx + 1} of ${list.length}` : "";
  $("#prev").disabled = idx <= 0;
  $("#next").disabled = idx >= list.length - 1;
}

// ---- rendering -----------------------------------------------------------------------------------------------------------------------
const BUCKET = {
  likely_error: ["Likely error", "p-bad"], likely_ok: ["Likely OK", "p-ok"], dismissed: ["Reference agrees", "p-muted"],
  unchecked: ["Not checked yet", "p-info"],
};
const DECISION = { pending: ["Pending", "p-muted"], accepted: ["Accepted", "p-ok"], rejected: ["Rejected", "p-warn"],
  overridden: ["Edited", "p-info"] };
const pill = ([t, c]) => `<span class="pill ${c}">${esc(t)}</span>`;

function markExcerpt(text) {
  // reference excerpt marks the aligned words ⟦like this⟧
  return esc(text).replace(/⟦(.*?)⟧/g, (_, w) => `<mark>${w || "∅"}</mark>`);
}

function render() {
  const c = cur;
  $("#cand").hidden = false;
  $("#ref").textContent = c.ref;
  $("#ref").href = sefariaLink(c.ref);
  $("#badges").innerHTML = `${pill(BUCKET[c.bucket] || [c.bucket, "p-muted"])} <span class="small muted">detector ${c.score?.toFixed(2)}</span>`;
  $("#context").innerHTML = `<span class="dim">${esc(c.before.split(" ").slice(-30).join(" "))}</span> <mark>${esc(plain(c.words))}</mark> <span class="dim">${esc(c.after.split(" ").slice(0, 30).join(" "))}</span>`;

  const why = [];
  if (c.llm_verdict) why.push(`LLM: <b>${c.llm_verdict}</b>${c.llm_confidence != null ? ` (${(+c.llm_confidence).toFixed(2)})` : ""}${c.llm_reason ? " — " + esc(c.llm_reason) : ""}`);
  if (c.llm_error) why.push(`<span class="pill p-warn">LLM check failed</span> ${esc(c.llm_error)}`);
  if (c.align_status === "agrees" && c.llm_verdict === "error") why.push("The reference agrees with Sefaria here, but the LLM thinks it is an error.");
  $("#why").innerHTML = why.join("<br>");

  // proposed change
  let diff, src = "";
  if (c.decision === "pending") {
    if (c.proposed_text != null) {
      diff = diffHTML(c.current, c.proposed_text);
      src = { reference: "from the reference", llm: "from the LLM", "reference+llm": "reference and LLM agree" }[c.fix_source] || "";
    } else {
      diff = `<span class="muted" dir="ltr" style="font-family:var(--sans);font-size:14px">No proposed fix. Press <kbd>E</kbd> to write one, or <kbd>R</kbd> if it's not an error.</span>`;
    }
  } else {
    diff = c.current !== c.original ? diffHTML(c.original, c.current)
      : `<span class="muted" dir="ltr" style="font-family:var(--sans);font-size:14px">Segment unchanged.</span>`;
    src = c.decision === "pending" ? "" : "segment as it stands now";
  }
  $("#diff").innerHTML = diff;
  $("#source").textContent = src;

  // reference
  if (!project.ref_filename) {
    $("#refstatus").textContent = "";
    $("#refctx").innerHTML = `<span class="muted" dir="ltr" style="font-family:var(--sans);font-size:14px">No reference text was uploaded for this project.</span>`;
  } else if (c.align_status === "not_found" || !c.ref_excerpt) {
    $("#refstatus").textContent = "passage not found";
    $("#refctx").innerHTML = `<span class="muted" dir="ltr" style="font-family:var(--sans);font-size:14px">This passage could not be found in the reference.</span>`;
  } else {
    $("#refstatus").textContent = c.align_status === "agrees" ? "same as Sefaria" : "differs from Sefaria";
    $("#refctx").innerHTML = markExcerpt(c.ref_excerpt);
  }

  $("#decision").innerHTML = pill(DECISION[c.decision]) + (c.decided_by && c.decision !== "pending" ? ` <span class="small muted">${esc(c.decided_by)}</span>` : "");
  $("#accept").disabled = c.decision !== "pending" || c.proposed_text == null;
  $("#reject").disabled = c.decision === "rejected";
  $("#reset").hidden = c.decision === "pending";
  if (c.overridden && c.decision === "pending") {
    $("#why").innerHTML += `<br><span class="small">This segment was edited by hand; accepting patches the edited text.</span>`;
  }
}

// ---- actions -------------------------------------------------------------------------------------------------------------------------
async function act(action, text) {
  if (!cur || busy) return;
  busy = true;
  try {
    cur = await postJSON(`/api/projects/${pid}/candidates/${cur.id}/decision`, { action, text });
    list[idx].decision = cur.decision;
    loadHistory();
    loadCounts();
    if (action === "reset") return render();
    // move to the next candidate still pending in this view
    const next = list.findIndex((c, i) => i > idx && c.decision === "pending" && !c.gone);
    if (next >= 0) { idx = next; await show(); }
    else { render(); updateNav(); toast(idx >= list.length - 1 ? "End of the list" : "Saved"); }
  } catch (e) {
    toast(e.message, 5000);
  } finally {
    busy = false;
  }
}

async function undo() {
  try {
    const c = await api(`/api/projects/${pid}/undo`, { method: "POST" });
    toast(`Undid change at ${c.ref}`);
    loadHistory(); loadCounts();
    const i = list.findIndex((x) => x.id === c.id);
    if (i >= 0) { list[i].decision = c.decision; list[i].gone = false; idx = i; await show(); }
    else await jumpTo(c.id);
  } catch (e) { toast(e.message); }
}

async function jumpTo(cid) {
  let i = list.findIndex((x) => x.id === cid && !x.gone);
  if (i < 0) {                                   // not in the cached list: refresh this view, else fall back to "All"
    list = await api(`/api/projects/${pid}/candidates?filter=${$("#filter").value}`);
    i = list.findIndex((x) => x.id === cid);
    if (i < 0) { $("#filter").value = "all"; await loadList(cid); return; }
  }
  idx = i; await show();
}

function go(d) {
  let n = idx + d;
  while (n >= 0 && n < list.length && list[n].gone) n += d;     // skip items that have left this view (e.g. LLM said OK)
  if (n < 0 || n >= list.length) return;
  idx = n; show();
}

function startEdit(whole = false) {
  if (!cur) return;
  editing = whole ? "seg" : "span";
  $("#edit").hidden = false;
  $("#edit-span").hidden = whole;
  $("#edit-seg").hidden = !whole;
  $("#toggle-seg").textContent = whole ? "Edit just the marked words" : "Edit whole segment";
  if (whole) {
    const t = $("#seg-input");
    t.value = cur.proposed_text ?? cur.current;
    t.focus();
  } else {
    const inp = $("#fix-input");
    inp.value = cur.applied_fix ?? cur.proposed_fix ?? plain(cur.words);
    inp.focus(); inp.select();
  }
}
function cancelEdit() { editing = null; $("#edit").hidden = true; }
function saveEdit() {
  if (editing === "seg") act("override", $("#seg-input").value);
  else if (editing === "span") act("accept", $("#fix-input").value);
}

// ---- side panels ---------------------------------------------------------------------------------------------------------------------
async function loadHistory() {
  const h = await api(`/api/projects/${pid}/history?limit=20`);
  $("#no-history").hidden = h.length > 0;
  const VERB = { accept: "Accepted", reject: "Rejected", override: "Edited", reset: "Reset" };
  $("#history").innerHTML = h.map((e) => `<li data-cid="${e.cand_id}" class="${e.undone ? "undone" : ""}" tabindex="0">
      <div class="row small"><b>${VERB[e.action] || e.action}</b><span class="muted">${esc(e.ref)}</span><span class="spacer"></span>
        <span class="muted">${esc((e.user || "").split("@")[0])}</span></div>
      ${e.old_text !== e.new_text ? `<div class="he" lang="he">${diffHTML(e.old_text, e.new_text)}</div>` : ""}</li>`).join("");
}
$("#history").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) jumpTo(+li.dataset.cid); });
$("#history").addEventListener("keydown", (e) => { if (e.key === "Enter") { const li = e.target.closest("li"); if (li) jumpTo(+li.dataset.cid); } });

async function loadCounts() {
  const p = await api(`/api/projects/${pid}/progress`);
  const c = p.counts;
  $("#counts").textContent = `${c.todo} to review · ${c.accepted} accepted · ${c.rejected} rejected · ${c.overridden} edited · ${c.changed_segments} segments changed`;
  $("#pprogress").innerHTML = p.status === "ready" ? "" : progressHTML(p);
  if (project) project.progress = p;
  return p;
}

// poll while the pipeline is still running: merge new candidates and fresh LLM verdicts into the list without moving the reviewer
async function poll() {
  const p = await loadCounts().catch(() => null);
  const running = p && (p.status === "running" || p.status === "queued");
  if (running && !editing) {
    const fresh = await api(`/api/projects/${pid}/candidates?filter=${$("#filter").value}`).catch(() => null);
    if (fresh) mergeList(fresh);
  }
  setTimeout(poll, running ? 3000 : 20000);
}

function mergeList(fresh) {
  const had = list.length, keep = list[idx]?.id;
  const byId = new Map(fresh.map((c) => [c.id, c]));
  for (const c of list) {                      // update the ones we have; keep ones that left the view (marked) so idx stays put
    const f = byId.get(c.id);
    if (f) Object.assign(c, f); else c.gone = true;
  }
  const known = new Set(list.map((c) => c.id));
  const added = fresh.filter((c) => !known.has(c.id));
  if (added.length) {
    list = [...list, ...added].sort((a, b) => a.seg - b.seg || a.id - b.id);
    idx = Math.max(0, list.findIndex((c) => c.id === keep));
  }
  if (!had && list.length) show(); else updateNav();
}

// ---- wiring --------------------------------------------------------------------------------------------------------------------------
$("#prev").onclick = () => go(-1);
$("#next").onclick = () => go(1);
$("#accept").onclick = () => act("accept");
$("#reject").onclick = () => act("reject");
$("#editbtn").onclick = () => startEdit(false);
$("#reset").onclick = () => act("reset");
$("#undo").onclick = undo;
$("#save-edit").onclick = saveEdit;
$("#cancel-edit").onclick = cancelEdit;
$("#toggle-seg").onclick = () => startEdit(editing !== "seg");
$("#filter").onchange = () => loadList();
$("#export").onclick = exportCsv;

function exportCsv() {
  location.href = `/api/projects/${pid}/export.csv`;
}

document.addEventListener("keydown", (e) => {
  const inField = e.target.matches("input, textarea, select");
  if (editing) {
    if (e.key === "Escape") { e.preventDefault(); cancelEdit(); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveEdit(); }
    else if (e.key === "Enter" && editing === "span" && e.target.id === "fix-input") { e.preventDefault(); saveEdit(); }
    return;
  }
  if (inField || e.altKey) return;
  // physical keys (e.code), so shortcuts work with a Hebrew keyboard layout too
  const letter = e.code.startsWith("Key") ? e.code.slice(3).toLowerCase() : null;
  if ((e.ctrlKey || e.metaKey) && letter === "z") { e.preventDefault(); undo(); return; }
  if (e.ctrlKey || e.metaKey) return;
  const k = e.key === "Enter" ? "Enter" : e.key === "?" || (e.code === "Slash" && e.shiftKey) ? "?"
    : letter ? (e.shiftKey ? letter.toUpperCase() : letter) : e.key;
  const handlers = {
    j: () => go(1), k: () => go(-1),
    a: () => { if (!$("#accept").disabled) act("accept"); }, Enter: () => { if (!$("#accept").disabled) act("accept"); },
    r: () => { if (!$("#reject").disabled) act("reject"); }, x: () => { if (!$("#reject").disabled) act("reject"); },
    e: () => startEdit(false), s: () => startEdit(true),
    u: undo,
    f: () => { const i = FILTERS.indexOf($("#filter").value); $("#filter").value = FILTERS[(i + 1) % FILTERS.length]; loadList(); },
    E: exportCsv,
    "?": () => { const a = document.querySelector("aside"); a.hidden = !a.hidden; },
  };
  const h = handlers[k];
  if (h) { e.preventDefault(); h(); }
});

(async () => {
  if (!pid) { location.href = "/"; return; }
  await loadProject();
  await loadCounts();
  await loadList();
  loadHistory();
  poll();
})();
